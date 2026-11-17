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
            #[cfg(feature = "gpu")]
            Device::Cuda(d) => QStorage::Cuda(QCudaStorage::upload(
                &CudaDevice(d.clone()),
                inner.data(),
                inner.dtype,
            )?),
            _ => QStorage::Cpu,
        };
        let dry = dry_blob_room(&inner, device);
        // Bytes this tensor owns, now on the card, are let go: nothing on a card placement
        // reads the host copy, and the mapping of a file (a view) costs only page cache.
        // The shell keeps the weight's identity; a download serves the rare host reader.
        #[cfg(feature = "gpu")]
        let inner = if matches!(storage, QStorage::Cuda(_))
            && matches!(inner.storage, super::host::QHostStorage::Owned(_))
        {
            Arc::new(inner.card_only())
        } else {
            inner
        };
        Ok(Self {
            shape: Shape::from(inner.dims.clone()),
            inner,
            device: device.clone(),
            storage,
            qmm: std::sync::OnceLock::new(),
            dry,
        })
    }

    /// The tensor `inner` describes, held by `storage` on the card in place of an upload of its
    /// blocks: the storage is filled by the caller, as often as it likes.
    #[cfg(feature = "gpu")]
    pub(crate) fn over_card(
        inner: Arc<QHostTensor>,
        device: &Device,
        storage: QCudaStorage,
    ) -> Self {
        Self {
            shape: Shape::from(inner.dims.clone()),
            inner,
            device: device.clone(),
            storage: QStorage::Cuda(storage),
            qmm: std::sync::OnceLock::new(),
            dry: None,
        }
    }

    /// The first `n` rows of this 2-D weight on a card, sharing its blocks on the host and on
    /// the card: rows are whole blocks, laid out one after the other. `None` off a card or
    /// for a shape that is not rows of blocks.
    #[cfg(feature = "gpu")]
    pub(crate) fn row_prefix(&self, n: usize) -> Result<Option<Self>> {
        let dims = self.shape.dims();
        let QStorage::Cuda(card) = &self.storage else {
            return Ok(None);
        };
        if dims.len() != 2 || n == 0 || n > dims[0] {
            return Ok(None);
        }
        let row_bytes = self.inner.data().len() / dims[0];
        let len = n * row_bytes;
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(self.inner.clone());
        let view = unsafe {
            QHostTensor::view(
                owner,
                self.inner.data().as_ptr(),
                0,
                len,
                self.dtype(),
                vec![n, dims[1]],
            )
        }
        .map_err(|e| Error::msg(e.0))?;
        let storage = QCudaStorage::sharing(card.blob.clone(), len, self.dtype(), card.device());
        Ok(Some(Self::over_card(Arc::new(view), &self.device, storage)))
    }

    /// [`Self::row_prefix`] on a build with no cards.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn row_prefix(&self, _n: usize) -> Result<Option<Self>> {
        Ok(None)
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
        #[cfg(feature = "gpu")]
        if self.inner.is_card_only() {
            if let (QStorage::Cuda(c), true) = (&self.storage, self.device.same_device(device)) {
                return Ok(Self {
                    inner: self.inner.clone(),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                    storage: QStorage::Cuda(c.share()),
                    qmm: std::sync::OnceLock::new(),
                    dry: None,
                });
            }
            let bytes = self.host_blocks()?;
            let host = QHostTensor::from_bytes(&bytes, self.inner.dtype, self.inner.dims.clone())?;
            return Self::from_native(Arc::new(host), device);
        }
        Self::from_native(self.inner.clone(), device)
    }

    /// The block bytes on the host: borrowed when the host holds them, downloaded from the
    /// card when the card holds the only copy.
    fn host_blocks(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
        #[cfg(feature = "gpu")]
        if self.inner.is_card_only() {
            return match &self.storage {
                QStorage::Cuda(c) => Ok(std::borrow::Cow::Owned(c.download()?)),
                _ => Err(Error::msg("a card-only weight has no card storage")),
            };
        }
        Ok(std::borrow::Cow::Borrowed(self.inner.data()))
    }

    /// legacy-shaped: build from a `QStorage::from_data` result.
    pub fn new(storage: QStorageWithHost, shape: Shape) -> Result<Self> {
        let inner = QHostTensor::from_bytes(&storage.host, storage.dtype, shape.dims().to_vec())?;
        let dry = dry_blob_room(&inner, &storage.device);
        // The card holds the blocks `from_data` uploaded; the host copy is let go.
        #[cfg(feature = "gpu")]
        let inner = if matches!(storage.storage, QStorage::Cuda(_)) {
            inner.card_only()
        } else {
            inner
        };
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
        // CONVERSION: quantisation of a float tensor is the constructor; the result is the resident form.
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
            #[cfg(feature = "gpu")]
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
        self.host_blocks()
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
        // A 16-bit float tensor decoded on the host keeps its width.
        // Its words are copied once, with no F32 buffer in between.
        if matches!(device, tensor::Device::Cpu) {
            let words = match self.dtype() {
                GgmlDType::F16 => Some(Tensor::from_vec(
                    halves(&self.host_blocks()?, half::f16::from_bits),
                    self.shape.clone(),
                    device,
                )?),
                GgmlDType::BF16 => Some(Tensor::from_vec(
                    halves(&self.host_blocks()?, half::bf16::from_bits),
                    self.shape.clone(),
                    device,
                )?),
                _ => None,
            };
            if let Some(t) = words {
                return Ok(t);
            }
        }
        let f = if self.inner.is_card_only() {
            let mut out = vec![0f32; self.inner.elem_count()];
            crate::tensor::quant_cpu::to_float_bytes(
                self.inner.dtype,
                &self.host_blocks()?,
                &mut out,
            )?;
            out
        } else {
            // CONVERSION: a dense decode is what dequantize returns.
            self.inner.dequantize_f32()?
        };
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
        // A 16-bit float tensor turns into F16 word by word.
        // The host holds 1 F16 buffer and no F32 one.
        let words = match self.dtype() {
            GgmlDType::F16 => Some(halves(&self.host_blocks()?, half::f16::from_bits)),
            GgmlDType::BF16 => Some(halves(&self.host_blocks()?, |b| {
                half::f16::from_f32(half::bf16::from_bits(b).to_f32())
            })),
            _ => None,
        };
        if let Some(words) = words {
            return Tensor::from_vec(words, self.shape.clone(), &tensor::Device::Cpu)?
                .to_device(device);
        }
        // CONVERSION: F16 dequantisation goes through the host decode.
        self.dequantize(&tensor::Device::Cpu)?
            .to_dtype(crate::tensor::DType::F16)?
            .to_device(device)
    }

    /// [`Self::dequantize_f16`] with the host holding at most `piece_bytes` of F16 words.
    ///
    /// A 16-bit float tensor bound for a card is converted and copied by blocks of whole rows.
    /// The card buffer is allocated once and each block lands at its offset.
    /// A block is never smaller than 1 row.
    /// Every other case takes [`Self::dequantize_f16`].
    pub fn dequantize_f16_in_pieces(&self, device: &Device, piece_bytes: usize) -> Result<Tensor> {
        #[cfg(feature = "gpu")]
        if let tensor::Device::Cuda(dev) = device {
            let from_bits: Option<fn(u16) -> half::f16> = match self.dtype() {
                GgmlDType::F16 => Some(half::f16::from_bits),
                GgmlDType::BF16 => Some(|b| half::f16::from_f32(half::bf16::from_bits(b).to_f32())),
                _ => None,
            };
            if let Some(from_bits) = from_bits {
                return self.f16_onto_card_in_pieces(dev, piece_bytes, from_bits);
            }
        }
        let _ = piece_bytes;
        // CONVERSION: F16 is what the caller asked for; off a card or from a block format, whole.
        self.dequantize_f16(device)
    }

    /// The 16-bit words of this tensor turned into F16 by `from_bits`, written to a card buffer
    /// by blocks of whole rows of at most `piece_bytes`.
    #[cfg(feature = "gpu")]
    fn f16_onto_card_in_pieces(
        &self,
        dev: &Arc<crate::tensor::cuda::CudaDevice>,
        piece_bytes: usize,
        from_bits: fn(u16) -> half::f16,
    ) -> Result<Tensor> {
        use crate::tensor::cuda::CudaStorage;
        let n = self.shape.elem_count();
        let row = self.shape.dims().last().copied().unwrap_or(n).max(1);
        let piece = (piece_bytes / (row * 2)).max(1) * row;
        let blocks = self.host_blocks()?;
        let mut out = CudaStorage::alloc_uninit(dev, crate::tensor::DType::F16, n)?;
        let CudaStorage::F16(slice) = &mut out else {
            return Err(Error::msg("an F16 card buffer of another type"));
        };
        let stream = dev.stream();
        let mut words: Vec<half::f16> = Vec::with_capacity(piece.min(n));
        for (i, chunk) in blocks.chunks(piece * 2).enumerate() {
            words.clear();
            {
                use rayon::prelude::*;
                chunk
                    .par_chunks_exact(2)
                    .map(|w| from_bits(u16::from_le_bytes([w[0], w[1]])))
                    .collect_into_vec(&mut words);
            }
            let at = i * piece;
            stream
                .memcpy_htod(&words[..], &mut slice.slice_mut(at..at + words.len()))
                .map_err(|e| Error::msg(format!("F16 upload by pieces: {e}")))?;
            // The next block refills the host buffer this copy reads.
            stream
                .synchronize()
                .map_err(|e| Error::msg(format!("F16 upload by pieces: {e}")))?;
        }
        Tensor::from_cuda_storage(out, dev.clone(), self.shape.clone())
    }

    /// The rows of `parts`, one after the other, as one `[rows, cols]` weight on the card
    /// `device`, written there part by part with no host copy of the whole.
    /// `None` off a card.
    pub fn rows_onto_card(
        parts: &[&[u8]],
        dtype: GgmlDType,
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> Result<Option<Self>> {
        #[cfg(feature = "gpu")]
        if let tensor::Device::Cuda(d) = device {
            let storage = QCudaStorage::upload_parts(&CudaDevice(d.clone()), parts, dtype)?;
            let inner = Arc::new(QHostTensor::on_card(dtype, vec![rows, cols]));
            return Ok(Some(Self::over_card(inner, device, storage)));
        }
        let _ = (parts, dtype, rows, cols, device);
        Ok(None)
    }

    /// The production-kernel matmul over this tensor's shared blob
    /// (built once; QMatMul::forward delegates here).
    pub(crate) fn native_qmm(&self) -> Result<&QKernelMatMul> {
        if let Some(q) = self.qmm.get() {
            return Ok(q);
        }
        #[cfg(feature = "gpu")]
        let q = {
            let blob = match &self.storage {
                QStorage::Cuda(c) => Some(c.blob.clone()),
                QStorage::Cpu => None,
            };
            QKernelMatMul::from_qtensor_on_with_blob(self.inner.clone(), &self.device, blob)?
        };
        #[cfg(not(feature = "gpu"))]
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

    /// A host tensor of `dims` in `ggml_dtype` whose block bytes `fill` writes, held once.
    pub fn host_filled(
        ggml_dtype: GgmlDType,
        dims: Vec<usize>,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<QTensor> {
        QTensor::from_native(
            Arc::new(QHostTensor::filled(ggml_dtype, dims, fill)?),
            &Device::Cpu,
        )
    }
}

/// The little-endian 16-bit words of `bytes`, each turned into a `T` by `from_bits`.
/// The only buffer is the result, filled in parallel.
fn halves<T: Copy + Send>(bytes: &[u8], from_bits: fn(u16) -> T) -> Vec<T> {
    use rayon::prelude::*;
    bytes
        .par_chunks_exact(2)
        .map(|w| from_bits(u16::from_le_bytes([w[0], w[1]])))
        .collect()
}

#[cfg(test)]
mod half_width {
    use super::*;
    use crate::tensor::{DType, Device, Tensor};

    /// Values a weight can hold: signs, a subnormal of F16, a large and an exact one.
    const VALUES: [f32; 8] = [0.0, -0.0, 1.5, -3.25e-6, 6.0e-8, 1024.5, -65504.0, 0.333];

    fn stored(dtype: GgmlDType) -> QTensor {
        let bytes: Vec<u8> = VALUES
            .iter()
            .flat_map(|&v| match dtype {
                GgmlDType::F16 => half::f16::from_f32(v).to_bits().to_le_bytes(),
                _ => half::bf16::from_f32(v).to_bits().to_le_bytes(),
            })
            .collect();
        QTensor::from_ggml_bytes(dtype, &bytes, vec![2, 4], &Device::Cpu).unwrap()
    }

    /// The decode through F32 the host took before: every value widened, then narrowed.
    fn through_f32(q: &QTensor, to: DType) -> Vec<u16> {
        let wide = q.native_qtensor().dequantize_f32().unwrap();
        bits(
            &Tensor::from_vec(wide, (2, 4), &Device::Cpu)
                .unwrap()
                .to_dtype(to)
                .unwrap(),
        )
    }

    fn bits(t: &Tensor) -> Vec<u16> {
        let flat = t.flatten_all().unwrap();
        match t.dtype() {
            DType::F16 => flat
                .to_vec1::<half::f16>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            DType::BF16 => flat
                .to_vec1::<half::bf16>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            other => panic!("not a 16-bit float: {other:?}"),
        }
    }

    /// A 16-bit float tensor decoded on the host stays in its width, with the bits of the
    /// decode through F32.
    #[test]
    fn a_half_tensor_decodes_on_the_host_in_its_own_width() {
        for (dtype, held) in [(GgmlDType::F16, DType::F16), (GgmlDType::BF16, DType::BF16)] {
            let q = stored(dtype);
            let t = q.dequantize(&Device::Cpu).unwrap();
            assert_eq!(t.dtype(), held, "{dtype:?}");
            assert_eq!(bits(&t), through_f32(&q, held), "{dtype:?}");
        }
    }

    /// A BF16 or F16 tensor turned into F16 has the bits of the decode through F32.
    #[test]
    fn a_half_tensor_turns_into_f16_as_the_decode_through_f32_does() {
        for dtype in [GgmlDType::F16, GgmlDType::BF16] {
            let q = stored(dtype);
            let t = q.dequantize_f16(&Device::Cpu).unwrap();
            assert_eq!(t.dtype(), DType::F16, "{dtype:?}");
            assert_eq!(bits(&t), through_f32(&q, DType::F16), "{dtype:?}");
        }
    }

    /// A `rows` by `cols` half weight of `dtype` as a view of a buffer another owner holds, as a
    /// file mapping is held, and the same weight copied into the tensor.
    fn viewed_and_owned(dtype: GgmlDType, rows: usize, cols: usize) -> (QTensor, QTensor) {
        let words: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let v = ((i * 37) % 23) as f32 * 0.125 - 1.4;
                match dtype {
                    GgmlDType::F16 => half::f16::from_f32(v).to_bits(),
                    _ => half::bf16::from_f32(v).to_bits(),
                }
            })
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // 8-byte words keep the view aligned as a mapped tensor is.
        let mut backing = vec![0u64; bytes.len().div_ceil(8)];
        // SAFETY: `backing` holds at least `bytes.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                backing.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(backing);
        let base = owner.downcast_ref::<Vec<u64>>().unwrap().as_ptr() as *const u8;
        // SAFETY: `base` points at `bytes.len()` bytes `owner` keeps alive.
        let view = unsafe {
            QHostTensor::view(owner.clone(), base, 0, bytes.len(), dtype, vec![rows, cols])
        }
        .unwrap();
        let viewed = QTensor::from_native(Arc::new(view), &Device::Cpu).unwrap();
        let owned =
            QTensor::from_ggml_bytes(dtype, &bytes, vec![rows, cols], &Device::Cpu).unwrap();
        (viewed, owned)
    }

    /// A half weight the host views in its file is read in place, and its products have the
    /// bits of the same weight held as stored in a copy.
    #[test]
    fn a_viewed_half_weight_runs_in_place_with_the_products_of_a_held_copy() {
        use crate::tensor::quantized::QMatMul;
        let (rows, cols) = (37usize, 64usize);
        for dtype in [GgmlDType::F16, GgmlDType::BF16] {
            let (viewed, owned) = viewed_and_owned(dtype, rows, cols);
            let in_place = QMatMul::from_qtensor(viewed).unwrap();
            assert!(matches!(in_place, QMatMul::QTensor(_)), "{dtype:?}");
            assert_eq!(in_place.float_format(), Some(dtype));
            assert_eq!(in_place.out_dim().unwrap(), rows);
            let held = QMatMul::from_qtensor(owned).unwrap();
            assert!(!matches!(held, QMatMul::QTensor(_)), "{dtype:?}");
            for n in [1usize, 3] {
                let x: Vec<f32> = (0..n * cols).map(|i| (i as f32 * 0.37).sin()).collect();
                let x = Tensor::from_vec(x, (n, cols), &Device::Cpu).unwrap();
                let got = in_place.forward(&x).unwrap().to_vec_f32();
                let want = held.forward(&x).unwrap().to_vec_f32();
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "{dtype:?} {n} rows");
            }
        }
    }

    /// A half weight uploaded to a card by pieces of a few rows has the bits of the whole upload.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "needs a card"]
    fn a_half_weight_uploaded_by_pieces_matches_the_whole_upload() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let (rows, cols) = (37usize, 64usize);
        for dtype in [GgmlDType::F16, GgmlDType::BF16] {
            let (viewed, _) = viewed_and_owned(dtype, rows, cols);
            let whole = viewed.dequantize_f16(&dev).unwrap();
            // 5 rows a piece leaves a last piece of 2 rows.
            for piece in [1usize, 5 * cols * 2, rows * cols * 2, usize::MAX] {
                let pieces = viewed.dequantize_f16_in_pieces(&dev, piece).unwrap();
                assert_eq!(pieces.dims(), whole.dims());
                assert_eq!(
                    bits(&pieces.to_device(&Device::Cpu).unwrap()),
                    bits(&whole.to_device(&Device::Cpu).unwrap()),
                    "{dtype:?} pieces of {piece} bytes"
                );
            }
        }
    }

    /// The words of a buffer come back in order, one per pair of bytes.
    #[test]
    fn the_words_of_a_buffer_come_back_in_order() {
        let bytes = [0x01, 0x02, 0xff, 0x7f, 0x00, 0x80];
        assert_eq!(halves(&bytes, |b| b), vec![0x0201, 0x7fff, 0x8000]);
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
