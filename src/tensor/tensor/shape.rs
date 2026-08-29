//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    pub fn dim<I: Dim>(&self, d: I) -> Result<usize> {
        let i = d.to_index(&self.shape, "dim")?;
        Ok(self.shape.dims()[i])
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn dtype(&self) -> DType {
        self.storage_raw.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage_raw.device()
    }

    /// Cheap identity for memoization: the address of the shared storage
    /// allocation, mixed with the view offset. Two tensors report the same id
    /// iff they share the same `Arc<Storage>` AT the same element offset
    /// (clones and metadata-only views like `reshape` / `unsqueeze` of one
    /// another). Two different narrow VIEWS over one storage are DIFFERENT
    /// tensors for cache keys, so the offset participates (hashed so small
    /// offsets don't collide with neighbouring allocations). ⚠️ Addresses are
    /// RECYCLED once the last Arc drops - a cache keyed on this id must keep a
    /// clone of the keyed tensor alive inside the entry to pin the allocation
    /// (see `inference/flux_native.rs` for the pattern).
    pub fn storage_ptr_id(&self) -> usize {
        let ptr = Arc::as_ptr(&self.storage_raw) as *const () as usize;
        // offset AND span participate: a PREFIX view (offset 0, shorter shape)
        // must not collide with its parent. reshape/unsqueeze keep both, so
        // metadata-only views of one another still share the id.
        ptr ^ self.offset.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ self
                .shape
                .elem_count()
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                .rotate_left(17)
    }

    /// Legacy-shaped tensor identity (the storage pointer) for cache keys. Same pin caveat
    /// as `storage_ptr_id`. (compat->native union.)
    pub fn id(&self) -> crate::tensor::ops::traits::TensorId {
        crate::tensor::ops::traits::TensorId(self.storage_ptr_id())
    }

    /// Borrow the shared storage (compat-shim projection).
    /// Views return the materialized packed copy (offset-0 data), so every
    /// external consumer stays correct without offset awareness.
    pub(crate) fn storage_arc(&self) -> &Arc<Storage> {
        self.storage()
    }

    /// Build a tensor over an existing shared storage allocation (compat
    /// shim: re-wrapping a projected storage without copying).
    pub(crate) fn from_storage_arc<S: Into<Shape>>(
        storage: Arc<Storage>,
        shape: S,
    ) -> Result<Self> {
        let shape = shape.into();
        let len = match storage.as_ref() {
            Storage::Cpu(c) => c.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda { data, .. } => data.len(),
            Storage::Dry(d) => d.len(),
        };
        if len != shape.elem_count() {
            return Err(Error(format!(
                "from_storage_arc: {len} elements for shape {:?}",
                shape.dims()
            )));
        }
        Ok(Self::from_packed(storage, shape))
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        match self.storage().as_ref() {
            Storage::Cpu(c) => c.to_f32_vec(),
            // A dry tensor has no values. Zeros keep the SHAPES downstream right -
            // a positional-id table read back to gather rows of a cached rotary
            // table produces the same shapes whatever the ids are - and the read is
            // recorded so a caller that cannot accept the risk can refuse the
            // measurement rather than trust a forward that may have branched on a
            // number that was never computed.
            Storage::Dry(d) => {
                d.device().note_blind_read();
                vec![0f32; self.elem_count()]
            }
            #[cfg(feature = "cuda")]
            Storage::Cuda { data, dev } => data
                .download(dev)
                .map(|c| c.to_f32_vec())
                .unwrap_or_default(),
        }
    }

    /// Move to a device (CPU↔CUDA copies; same-device is a cheap Arc clone).
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device().same_device(device) {
            return Ok(self.clone());
        }
        match (self.storage().as_ref(), device) {
            // Onto a dry device: what the upload WOULD cost, which is how a weight
            // built on the host and moved to the card gets counted at all.
            (_, Device::Dry(dev)) => Ok(Self::from_packed(
                Arc::new(Storage::Dry(crate::tensor::dry::DryStorage::new(
                    dev.clone(),
                    self.dtype(),
                    self.elem_count(),
                ))),
                self.shape.clone(),
            )),
            // Off one: there is nothing to copy. A forward that tries has left the
            // dry run, and silently returning an empty buffer would let it keep
            // going and report a peak for a computation that never happened.
            (Storage::Dry(_), _) => Err(Error(
                "a dry tensor cannot be moved to a real device".into(),
            )),
            #[cfg(feature = "cuda")]
            (Storage::Cpu(c), Device::Cuda(dev)) => {
                let data = crate::tensor::cuda::CudaStorage::upload_host_safe(dev, c)?;
                Ok(Self::from_packed(
                    Arc::new(Storage::Cuda {
                        data,
                        dev: dev.clone(),
                    }),
                    self.shape.clone(),
                ))
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda { data, dev }, Device::Cpu) => {
                let c = data.download(dev)?;
                Ok(Self::from_packed(
                    Arc::new(Storage::Cpu(c)),
                    self.shape.clone(),
                ))
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda { data, dev }, Device::Cuda(target)) => {
                // cross-GPU peer copy: enqueued on the DESTINATION stream, which will
                // not otherwise wait for the in-flight source compute that produces
                // `data` (the TP-race lesson: a naive `to_device` has exactly this
                // hazard). The ordering is established with an event rather than by
                // stopping the host - same guarantee, no pipeline drain per boundary
                // per token, and legal inside a graph capture where a stream
                // synchronize is not.
                target.wait_for(dev)?;
                let d2 = crate::tensor::cuda::CudaStorage::peer_copy(data, dev, target)?;
                Ok(Self::from_packed(
                    Arc::new(Storage::Cuda {
                        data: d2,
                        dev: target.clone(),
                    }),
                    self.shape.clone(),
                ))
            }
            (Storage::Cpu(_), Device::Cpu) => Ok(self.clone()),
        }
    }

    /// Wrap an existing device buffer (kernel outputs become tensors this way).
    #[cfg(feature = "cuda")]
    pub fn from_cuda_storage<S: Into<Shape>>(
        data: crate::tensor::cuda::CudaStorage,
        dev: std::sync::Arc<crate::tensor::cuda::CudaDevice>,
        shape: S,
    ) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error(format!(
                "from_cuda_storage: {} elements for shape {:?}",
                data.len(),
                shape.dims()
            )));
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cuda { data, dev }),
            shape,
        ))
    }

    /// Borrow the f32 device slice + its device (the kernel-launch boundary).
    #[cfg(feature = "cuda")]
    pub fn cuda_f32_slice(
        &self,
    ) -> Result<(
        &cudarc::driver::CudaSlice<f32>,
        &std::sync::Arc<crate::tensor::cuda::CudaDevice>,
    )> {
        match self.storage().as_ref() {
            Storage::Cuda { data, dev } => Ok((data.as_f32_slice()?, dev)),
            Storage::Cpu(_) => Err(Error("cuda_f32_slice on a CPU tensor".into())),
            Storage::Dry(_) => Err(Error("cuda_f32_slice on a dry tensor".into())),
        }
    }

    /// Borrow the f16 device slice + its device.
    #[cfg(feature = "cuda")]
    pub fn cuda_f16_slice(
        &self,
    ) -> Result<(
        &cudarc::driver::CudaSlice<half::f16>,
        &std::sync::Arc<crate::tensor::cuda::CudaDevice>,
    )> {
        match self.storage().as_ref() {
            Storage::Cuda { data, dev } => Ok((data.as_f16_slice()?, dev)),
            Storage::Cpu(_) => Err(Error("cuda_f16_slice on a CPU tensor".into())),
            Storage::Dry(_) => Err(Error("cuda_f16_slice on a dry tensor".into())),
        }
    }

    /// Borrow the bf16 device slice + its device.
    #[cfg(feature = "cuda")]
    pub fn cuda_bf16_slice(
        &self,
    ) -> Result<(
        &cudarc::driver::CudaSlice<half::bf16>,
        &std::sync::Arc<crate::tensor::cuda::CudaDevice>,
    )> {
        match self.storage().as_ref() {
            Storage::Cuda { data, dev } => Ok((data.as_bf16_slice()?, dev)),
            Storage::Cpu(_) => Err(Error("cuda_bf16_slice on a CPU tensor".into())),
            Storage::Dry(_) => Err(Error("cuda_bf16_slice on a dry tensor".into())),
        }
    }

    /// Run a CPU-only op on a CUDA tensor by bouncing through host memory.
    /// Correctness-first fallback; ops earn dedicated GPU kernels when the
    /// decode path needs them hot.
    #[cfg(feature = "cuda")]
    #[track_caller]
    pub(super) fn host_bounce(&self, f: impl FnOnce(&Self) -> Result<Self>) -> Result<Self> {
        let _b = crate::tensor::bounce::start("host_bounce");
        let dev = self.device();
        let cpu = self.to_device(&Device::Cpu)?;
        f(&cpu)?.to_device(&dev)
    }

    pub(super) fn is_on_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Half-precision (f16/bf16) CUDA tensors run the f32 kernels via
    /// on-device casts (f32 accumulation - the standard mixed-precision
    /// recipe; dedicated half kernels arrive only where a hot path needs them).
    #[cfg(feature = "cuda")]
    #[track_caller]
    pub(super) fn f16_unary_via_f32(&self, f: impl Fn(&Self) -> Result<Self>) -> Result<Self> {
        let _b = crate::tensor::bounce::start("f16_via_f32");
        let back = self.dtype();
        f(&self.to_dtype(DType::F32)?)?.to_dtype(back)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn is_cuda_f16(&self) -> bool {
        self.is_on_cuda() && matches!(self.dtype(), DType::F16 | DType::BF16)
    }

    /// Multiply by a scalar.
    pub fn scale(&self, alpha: f32) -> Result<Self> {
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::F16 {
            // half-precision arithmetic, like the reference f16 affine kernel
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::affine_f16(
                    dev,
                    data.as_f16_slice()?,
                    alpha,
                    0.0,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::BF16 {
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::affine_bf16(
                    dev,
                    data.as_bf16_slice()?,
                    alpha,
                    0.0,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.scale(alpha));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::scale_f32(
                dev,
                data.as_f32_slice()?,
                alpha,
                self.elem_count(),
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        self.unary_f32(|x| x * alpha)
    }

    pub fn exp(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.exp());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_exp_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::exp)
    }

    /// `x * mul + add` elementwise.
    /// `x * mul + add`.
    ///
    /// Under memory pressure this falls back to the host instead of failing. That net
    /// existed only on the conv family, which is why a VAE decode survived a full card
    /// while a denoise step did not: an `affine` mid-sampling hit `cuda OOM in scale`,
    /// the retry found no more memory either, and the whole request died with a 500.
    /// A bounced element-wise op makes that step slow; a 500 makes the render nothing.
    pub fn affine(&self, mul: f32, add: f32) -> Result<Self> {
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            return match self.affine_on_device(mul, add) {
                Err(e) if e.is_oom() => {
                    crate::tensor::bounce::note_pressure_bounce();
                    self.host_bounce(|cpu| cpu.affine(mul, add))
                }
                r => r,
            };
        }
        self.affine_on_device(mul, add)
    }

    pub(super) fn affine_on_device(&self, mul: f32, add: f32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::F16 {
            // half-precision arithmetic, like the reference f16 affine kernel
            //
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::affine_f16(
                    dev,
                    data.as_f16_slice()?,
                    mul,
                    add,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::BF16 {
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::affine_bf16(
                    dev,
                    data.as_bf16_slice()?,
                    mul,
                    add,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.affine(mul, add));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::affine_f32(
                dev,
                data.as_f32_slice()?,
                mul,
                add,
                self.elem_count(),
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        self.unary_f32(|x| x * mul + add)
    }
}
