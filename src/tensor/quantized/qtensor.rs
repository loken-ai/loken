//! The quantised tensor the LLM loaders hold.
//!
//! It carries the host blocks, the device blob, and a `OnceLock` over the kernel matmul
//! built from them - the hot weight path, built once and shared.

use super::*;

/// Facade-shaped quantized tensor: host blocks ([`QHostTensor`]) + the
/// device-resident blob + a lazily-built [`QKernelMatMul`] sharing that blob.

pub struct QTensor {
    inner: Arc<QHostTensor>,
    shape: Shape,
    device: Device,
    storage: QStorage,
    qmm: std::sync::OnceLock<QKernelMatMul>,
    /// The room a placement on a counting device gives these blocks - the padded blob
    /// the CUDA arm above uploads, charged where that upload happens and released by
    /// the drop that would have freed it. `None` on every real placement.
    ///
    /// Held, never read: the entry is made by its constructor and withdrawn by its
    /// drop, so the value carries nothing a caller would want.
    #[allow(dead_code)]
    dry: Option<crate::tensor::dry::DryStorage>,
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QTensor")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape.dims())
            .field("device", &self.device)
            .finish()
    }
}

impl QTensor {
    /// Wrap a host-side [`QHostTensor`], placing its blob on `device`.
    pub(crate) fn from_native(inner: Arc<QHostTensor>, device: &Device) -> Result<Self> {
        let storage = match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => QStorage::Cuda(QCudaStorage::upload(
                &CudaDevice(d.clone()),
                inner.data(),
                inner.dtype,
            )?),
            _ => QStorage::Cpu,
        };
        let dry = dry_blob_room(&inner, device);
        Ok(Self {
            shape: Shape::from(inner.dims.clone()),
            inner,
            device: device.clone(),
            storage,
            qmm: std::sync::OnceLock::new(),
            dry,
        })
    }

    /// What identifies these weights. Stable across `to_device` - the same weights on another
    /// card are the same weights - and never reused, so a cache keyed by it cannot answer a
    /// new tensor with a dropped one's entry.
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// The same weights, resident on another device.
    ///
    /// The host tensor is kept beside the device blob precisely so this is possible:
    /// a weight can be re-materialised anywhere without going back to the checkpoint,
    /// which is what lets a model that was pushed onto the host under pressure climb
    /// back onto a card once one frees up - rather than staying there for the rest of
    /// its residency while VRAM sits unused.
    ///
    /// Same device in, same tensor out (a fresh handle over the same host blocks), so
    /// callers can ask unconditionally.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Self::from_native(self.inner.clone(), device)
    }

    /// legacy-shaped: build from a `QStorage::from_data` result.
    pub fn new(storage: QStorageWithHost, shape: Shape) -> Result<Self> {
        let inner = QHostTensor::from_bytes(&storage.host, storage.dtype, shape.dims().to_vec())?;
        let dry = dry_blob_room(&inner, &storage.device);
        Ok(Self {
            shape,
            inner: Arc::new(inner),
            device: storage.device.clone(),
            storage: storage.storage,
            qmm: std::sync::OnceLock::new(),
            dry,
        })
    }

    /// Quantize an f32/f16 tensor's values into GGML blocks (host-side;
    /// result lives on the input tensor's device).
    pub fn quantize(t: &Tensor, dtype: GgmlDType) -> Result<Self> {
        Self::quantize_onto(t, dtype, &t.device())
    }

    /// Quantize onto an explicit device (fork extension the loaders use).
    /// Host-side quantize + padded upload - correctness-identical to the
    /// fork's device kernels.
    pub fn quantize_onto(t: &Tensor, dtype: GgmlDType, device: &Device) -> Result<Self> {
        let f32s: Vec<f32> = t
            .to_dtype(crate::tensor::DType::F32)?
            .flatten_all()?
            .to_device(&tensor::Device::Cpu)?
            .to_vec1::<f32>()?;
        let bytes = tensor::quant_cpu::from_float_bytes(dtype, &f32s)?;
        let inner = QHostTensor::from_bytes(&bytes, dtype, t.dims().to_vec())?;
        Self::from_native(Arc::new(inner), device)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.inner.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn storage(&self) -> &QStorage {
        &self.storage
    }

    /// Raw device pointer of the CUDA blob (fork-shaped raw-launch path).
    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            #[cfg(feature = "cuda")]
            QStorage::Cuda(c) => c.device_ptr(),
            _ => Err(Error::msg("QTensor::device_ptr: not on CUDA")),
        }
    }

    /// The host-side [`QHostTensor`] (quant_view expert slicing).
    pub(crate) fn native_qtensor(&self) -> &Arc<QHostTensor> {
        &self.inner
    }

    /// Host block bytes (always available: the native QTensor keeps the
    /// host copy that built the device blob).
    pub fn data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
        Ok(std::borrow::Cow::Borrowed(self.inner.data()))
    }

    /// Dequantize to a dense tensor on `device`. Mirrors the fork: F16
    /// blocks stay F16 on CPU; everything else lands F32.
    pub fn dequantize(&self, device: &tensor::Device) -> tensor::Result<Tensor> {
        // Counted, not decoded: the result is a dense buffer of this shape, and a dry
        // run wants its size rather than its values. The half-carrier arm below is a
        // host special case, so what a counting device stands in for lands F32.
        if device.is_dry() {
            return Tensor::dry(device, crate::tensor::DType::F32, self.shape.clone());
        }
        let f = self.inner.dequantize_f32()?;
        let t = Tensor::from_vec(f, self.shape.clone(), &tensor::Device::Cpu)?;
        let t = match (self.dtype(), device) {
            (GgmlDType::F16, tensor::Device::Cpu) => t.to_dtype(crate::tensor::DType::F16)?,
            (GgmlDType::BF16, tensor::Device::Cpu) => t.to_dtype(crate::tensor::DType::BF16)?,
            _ => t,
        };
        t.to_device(device)
    }

    /// Dequantize to F16 on `device` (fork extension).
    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // The host staging the real path takes is not part of what the card holds, and
        // a counting device must not build it: what lands there is the F16 result.
        if device.is_dry() {
            return Tensor::dry(device, crate::tensor::DType::F16, self.shape.clone());
        }
        self.dequantize(&tensor::Device::Cpu)?
            .to_dtype(crate::tensor::DType::F16)?
            .to_device(device)
    }

    /// The production-kernel matmul over this tensor's shared blob
    /// (built once; QMatMul::forward delegates here).
    pub(crate) fn native_qmm(&self) -> Result<&QKernelMatMul> {
        if let Some(q) = self.qmm.get() {
            return Ok(q);
        }
        #[cfg(feature = "cuda")]
        let q = {
            let blob = match &self.storage {
                QStorage::Cuda(c) => Some(c.blob.clone()),
                QStorage::Cpu => None,
            };
            QKernelMatMul::from_qtensor_on_with_blob(self.inner.clone(), &self.device, blob)?
        };
        #[cfg(not(feature = "cuda"))]
        let q = QKernelMatMul::from_qtensor_on(self.inner.clone(), &self.device)?;
        let _ = self.qmm.set(q);
        Ok(self.qmm.get().unwrap())
    }
}

/// legacy-shaped quantized matmul enum (frozen sites construct/match all
/// three variants).

impl QTensor {
    /// Build one from raw GGML block bytes. The legacy container loader's entry point; the
    /// bytes are a block layout, so there is nothing to parse beyond knowing the dtype.
    pub fn from_ggml_bytes(
        ggml_dtype: GgmlDType,
        raw_data: &[u8],
        dims: Vec<usize>,
        device: &Device,
    ) -> Result<QTensor> {
        QTensor::from_native(
            Arc::new(QHostTensor::from_bytes(raw_data, ggml_dtype, dims)?),
            device,
        )
    }
}

#[cfg(test)]
mod identity {
    //! What a cache may key a weight by.
    //!
    //! Two process-global caches in the mixture-of-experts path store a REPACKED COPY of a
    //! weight - an entry holds nothing of the tensor it came from. Keyed by address, a tensor
    //! that is dropped hands its address to the next allocation, which is then answered with
    //! the previous one's repack: no error, no warning, wrong weights. That is what these
    //! assert cannot happen.

    use super::*;
    use crate::tensor::{Device, Tensor};

    fn weight(seed: usize) -> QTensor {
        let v: Vec<f32> = (0..256)
            .map(|i| ((i * seed) % 61) as f32 * 0.03 - 0.9)
            .collect();
        let t = Tensor::from_vec(v, (1, 256), &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::Q8_0).unwrap()
    }

    #[test]
    fn a_dropped_weight_does_not_lend_its_identity_to_the_next_one() {
        // Force the reuse the address key could not survive: build, record, drop, build again.
        let first = weight(7);
        let id = first.id();
        let addr = &first as *const QTensor as usize;
        drop(first);

        let mut collided = false;
        for s in 1..64 {
            let next = weight(s);
            if &next as *const QTensor as usize == addr {
                collided = true;
            }
            assert_ne!(
                next.id(),
                id,
                "a new weight took a dropped one's identity - a cache keyed by it would \
                 answer this tensor with the dropped one's data"
            );
        }
        // The address reuse is what makes this test worth having; say so if it never happened.
        if !collided {
            eprintln!("note: no address was reused in this run; the identity check still held");
        }
    }

    #[test]
    fn the_same_weights_on_another_device_are_the_same_weights() {
        let w = weight(11);
        let same = w.to_device(&Device::Cpu).unwrap();
        assert_eq!(
            w.id(),
            same.id(),
            "moving a weight must not change what identifies it, or every move misses the cache"
        );
    }

    #[test]
    fn two_weights_are_two_identities() {
        let (a, b) = (weight(3), weight(5));
        assert_ne!(a.id(), b.id());
    }
}
