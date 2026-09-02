//! Reading quantised weights out of a GGUF file by name.

use super::*;
use crate::tensor::prefix::Prefix;

/// Prefix-walking builder over a GGUF file for quantized models: hands out
/// [`QKernelMatMul`]s for the projection weights and dequantized F32 [`crate::tensor::Tensor`]s
/// for the small mixed-precision entries (norm scales, biases). All tensors
/// are read once into host memory at open; uploads happen per `get_*`.
///
/// The path walk is [`Prefix`], the same one the dense loader walks: the names are the
/// checkpoint's, and they do not depend on whether the blob under them is quantized.
#[derive(Clone)]
pub struct QVarBuilder {
    inner: std::sync::Arc<QvbInner>,
    prefix: Prefix,
    device: crate::tensor::Device,
}

struct QvbInner {
    tensors: HashMap<String, std::sync::Arc<QHostTensor>>,
}

impl QVarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P,
        device: &crate::tensor::Device,
    ) -> Result<Self> {
        let f = std::fs::File::open(path.as_ref())
            .map_err(|e| Error(format!("QVarBuilder open: {e}")))?;
        // Zero-copy: mmap the file and build every tensor as a VIEW pinned by the
        // shared Arc<Mmap>. The pages are file-backed and kernel-reclaimable, so a
        // 12 GB checkpoint no longer pins 12 GB of owned host RAM for its lifetime
        // (the previous read_exact path did). Misaligned tensors (never produced by
        // our 32-byte-aligned writer, possible in third-party files) fall back to an
        // owned copy inside `read_slice_owned`.
        // SAFETY: standard mmap contract - the file must not be truncated while
        // mapped (same contract as the GGUF model loader's mmap).
        let mmap = std::sync::Arc::new(
            unsafe { memmap2::Mmap::map(&f) }
                .map_err(|e| Error(format!("QVarBuilder mmap: {e}")))?,
        );
        let content =
            gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())?;
        let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> = mmap;
        let mmap_bytes: &[u8] = owner
            .downcast_ref::<memmap2::Mmap>()
            .expect("owner is the Mmap just created");
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let info = &content.tensor_infos[&name];
            let n = info.size_in_bytes();
            let start = (content.tensor_data_offset + info.offset) as usize;
            let end = start
                .checked_add(n)
                .ok_or_else(|| Error("gguf tensor slice overflow".into()))?;
            let raw = mmap_bytes.get(start..end).ok_or_else(|| {
                Error(format!("gguf tensor out of mmap bounds: [{start}..{end}]"))
            })?;
            let dims = info.shape.dims().to_vec();
            // SAFETY: the `mmap_bytes.get(start..end)` above proved the range is in
            // bounds of this mapping, and `owner` pins it for the view's lifetime.
            let qt = match unsafe {
                QHostTensor::view(
                    owner.clone(),
                    mmap_bytes.as_ptr(),
                    start,
                    n,
                    info.ggml_dtype,
                    dims.clone(),
                )
            } {
                Ok(v) => v,
                // Misaligned in-file tensor: owned copy (correct, just not zero-copy).
                Err(_) => QHostTensor::from_bytes(raw, info.ggml_dtype, dims)?,
            };
            tensors.insert(name, std::sync::Arc::new(qt));
        }
        Ok(Self {
            inner: std::sync::Arc::new(QvbInner { tensors }),
            prefix: Prefix::root(),
            device: device.clone(),
        })
    }

    /// Build a builder from an in-memory `name -> dense native tensor` map,
    /// the way the fp8-scaled safetensors bridge produces it (Ray / Boogu style
    /// checkpoints: fp8 e4m3/e5m2 matmul weights decoded to F32 with their
    /// per-tensor `weight_scale` already folded in). Additive: the GGUF
    /// [`from_gguf`] path is unchanged.
    ///
    /// Each 2-D projection weight whose input dim (`dims[1]`) is a multiple of
    /// `weight_dtype`'s block size is quantized to `weight_dtype` -- pass
    /// `GgmlDType::Q8_0` for full CPU+GPU kernel coverage (matches how the GGUF
    /// path serves projections), or `GgmlDType::F32` for an exact dense weight
    /// (e.g. the Qwen-Image DiT, whose activation magnitudes overflow the
    /// quantized MMQ path). Everything else -- norm scales, biases, embeddings,
    /// and any non-block-aligned weight -- is kept dense F32 and served by
    /// [`get_f32`]. Values are preserved exactly for the F32 path; the Q8_0 path
    /// adds one block-quant step on top of the (already 8-bit) fp8 source.
    pub fn from_named_tensors(
        tensors: impl IntoIterator<Item = (String, crate::tensor::Tensor)>,
        weight_dtype: GgmlDType,
        device: &crate::tensor::Device,
    ) -> Result<Self> {
        // Quantize each tensor in parallel: fp8/bf16->F32 (already done upstream) -> Q8_0 blocks is
        // pure per-tensor CPU work over ~10-19 B params for a diffusion checkpoint; a serial loop
        // dominates load time. rayon over the tensor list cuts it to ~1/cores.
        use rayon::prelude::*;
        let items: Vec<(String, crate::tensor::Tensor)> = tensors.into_iter().collect();
        let map: HashMap<String, std::sync::Arc<QHostTensor>> = items
            .into_par_iter()
            .map(
                |(name, t)| -> Result<(String, std::sync::Arc<QHostTensor>)> {
                    let dims = t.dims().to_vec();
                    let f32s: Vec<f32> = t
                        .to_dtype(crate::tensor::DType::F32)?
                        .flatten_all()?
                        .to_device(&crate::tensor::Device::Cpu)?
                        .to_vec1::<f32>()?;
                    // 2-D projection weights go to the requested block dtype when the input dim tiles
                    // cleanly; everything else stays dense F32.
                    let block = weight_dtype.block_size();
                    let use_weight_dtype = dims.len() == 2
                        && weight_dtype != GgmlDType::F32
                        && block > 0
                        && dims[1] % block == 0;
                    let dtype = if use_weight_dtype {
                        weight_dtype
                    } else {
                        GgmlDType::F32
                    };
                    let bytes = crate::tensor::quant_cpu::from_float_bytes(dtype, &f32s)?;
                    let qht = QHostTensor::from_bytes(&bytes, dtype, dims)?;
                    Ok((name, std::sync::Arc::new(qht)))
                },
            )
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            inner: std::sync::Arc::new(QvbInner { tensors: map }),
            prefix: Prefix::root(),
            device: device.clone(),
        })
    }

    /// Build from already-quantized GGUF-shaped entries (the fallback when a sidecar cache
    /// cannot be written to disk - same bytes, kept in memory instead).
    pub fn from_quantized_entries(
        entries: Vec<crate::tensor::gguf_write::GgufEntry>,
        device: &crate::tensor::Device,
    ) -> Result<Self> {
        let mut map: HashMap<String, std::sync::Arc<QHostTensor>> =
            HashMap::with_capacity(entries.len());
        for e in entries {
            let qht = QHostTensor::from_bytes(&e.data, e.dtype, e.dims)?;
            map.insert(e.name, std::sync::Arc::new(qht));
        }
        Ok(Self {
            inner: std::sync::Arc::new(QvbInner { tensors: map }),
            prefix: Prefix::root(),
            device: device.clone(),
        })
    }

    /// Cheap re-homing: same shared host tensors, a different placement device. The host
    /// blobs are device-independent; `device` only steers where later `get_*`/`qmatmul`
    /// calls upload.
    pub fn clone_for_device(&self, device: &crate::tensor::Device) -> Self {
        Self {
            inner: self.inner.clone(),
            prefix: self.prefix.clone(),
            device: device.clone(),
        }
    }

    /// Total host bytes held by the underlying tensors (cache accounting).
    pub fn host_bytes(&self) -> u64 {
        self.inner
            .tensors
            .values()
            .map(|t| t.data().len() as u64)
            .sum()
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        Self {
            inner: self.inner.clone(),
            prefix: self.prefix.join(s),
            device: self.device.clone(),
        }
    }

    pub fn device(&self) -> &crate::tensor::Device {
        &self.device
    }

    fn full(&self, name: &str) -> String {
        self.prefix.path(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.inner.tensors.contains_key(&self.full(name))
    }

    /// The dense form of a stored entry, on `device`.
    ///
    /// A counting device gets the shape and the dtype and stops there. The decode
    /// would otherwise run for real - a vocabulary-sized table is gigabytes of host
    /// f32 - to produce values a dry run has no use for, and a dry run allocates
    /// nothing ANYWHERE, host included.
    fn dense_on(qt: &QHostTensor, device: &crate::tensor::Device) -> Result<crate::tensor::Tensor> {
        if device.is_dry() {
            return crate::tensor::Tensor::dry(device, crate::tensor::DType::F32, qt.dims.clone());
        }
        let data = qt.dequantize_f32()?;
        crate::tensor::Tensor::from_vec_f32(data, qt.dims.clone())?.to_device(device)
    }

    fn fetch(&self, name: &str) -> Result<std::sync::Arc<QHostTensor>> {
        let path = self.full(name);
        self.inner
            .tensors
            .get(&path)
            .cloned()
            .ok_or_else(|| Error(format!("QVarBuilder: missing `{path}`")))
    }

    /// Projection weight `[out, in]` -> quantized matmul on the builder's device.
    pub fn qmatmul(&self, in_dim: usize, out_dim: usize, name: &str) -> Result<QKernelMatMul> {
        self.qmatmul_on(in_dim, out_dim, name, &self.device)
    }

    /// Like `qmatmul`, but places the weight on `device` instead of the builder's
    /// device. The parsed `QHostTensor` blob is device-independent, so this enables
    /// per-layer placement (HeteroPlan segments: some layers GPU, some CPU/other GPU)
    /// from a single GGUF parse - no re-load per device.
    pub fn qmatmul_on(
        &self,
        in_dim: usize,
        out_dim: usize,
        name: &str,
        device: &crate::tensor::Device,
    ) -> Result<QKernelMatMul> {
        let qt = self.fetch(name)?;
        if qt.dims != [out_dim, in_dim] {
            return Err(Error(format!(
                "QVarBuilder `{}`: expected [{out_dim}, {in_dim}], got {:?}",
                self.full(name),
                qt.dims
            )));
        }
        QKernelMatMul::from_qtensor_on(qt, device)
    }

    /// Like `qmatmul_on`, but reads the stored `[out, in]` shape from the checkpoint
    /// instead of requiring the caller to pass dims. For loaders that don't carry an
    /// explicit per-weight dim table (e.g. the Boogu DiT, whose many distinct shapes
    /// would be error-prone to hardcode); the stored shape IS the source of truth.
    pub fn qmatmul_auto(
        &self,
        name: &str,
        device: &crate::tensor::Device,
    ) -> Result<QKernelMatMul> {
        let qt = self.fetch(name)?;
        if qt.dims.len() != 2 {
            return Err(Error(format!(
                "QVarBuilder `{}`: qmatmul_auto expects a 2-D weight, got {:?}",
                self.full(name),
                qt.dims
            )));
        }
        QKernelMatMul::from_qtensor_on(qt, device)
    }

    /// Like `get_f32`, but reads the stored shape from the checkpoint instead of
    /// requiring the caller to pass it (the auto-dim counterpart of `qmatmul_auto`).
    pub fn get_f32_auto(&self, name: &str) -> Result<crate::tensor::Tensor> {
        self.get_f32_auto_on(name, &self.device)
    }

    /// Like `get_f32_auto`, but lands the dense tensor on `device` instead of the builder's default
    /// device. Needed for per-block hetero placement: a block's norms/biases/embedders must follow
    /// the (possibly different) device its matmuls (`qmatmul_auto(.., device)`) were placed on.
    pub fn get_f32_auto_on(
        &self,
        name: &str,
        device: &crate::tensor::Device,
    ) -> Result<crate::tensor::Tensor> {
        let qt = self.fetch(name)?;
        Self::dense_on(&qt, device)
    }

    /// Small tensor (norm scale, bias, embedding) dequantized to F32 on the
    /// builder's device.
    pub fn get_f32<S: Into<crate::tensor::Shape>>(
        &self,
        shape: S,
        name: &str,
    ) -> Result<crate::tensor::Tensor> {
        let qt = self.fetch(name)?;
        let shape: crate::tensor::Shape = shape.into();
        if qt.dims != shape.dims() {
            return Err(Error(format!(
                "QVarBuilder `{}`: expected {:?}, got {:?}",
                self.full(name),
                shape.dims(),
                qt.dims
            )));
        }
        Self::dense_on(&qt, &self.device)
    }
}
