//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    /// Per-channel bias over `[b, c, ...spatial]` (the conv-bias pattern  - 
    /// the channel axis is dim 1, which tail-aligned broadcast can't express).
    pub fn add_channel_bias(&self, bias: &Self) -> Result<Self> {
        let dims = self.dims().to_vec();
        if dims.len() < 2 {
            return Err(Error("add_channel_bias needs [b, c, ...]".into()));
        }
        let c = dims[1];
        if bias.elem_count() != c {
            return Err(Error(format!(
                "add_channel_bias: bias {} != channels {c}",
                bias.elem_count()
            )));
        }
        let spatial: usize = dims[2..].iter().product();
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let b = bias.to_dtype(DType::F32)?;
            return self.f16_unary_via_f32(|t| t.add_channel_bias(&b));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let (b_slice, b_dev) = bias.cuda_f32_slice().map_err(|_| {
                Error("add_channel_bias: bias must be on the same cuda device".into())
            })?;
            if b_dev.ordinal() != dev.ordinal() {
                return Err(Error(
                    "add_channel_bias: bias on a different cuda device".into(),
                ));
            }
            let out = crate::tensor::cuda::bias_chw_f32(
                dev,
                data.as_f32_slice()?,
                b_slice,
                self.elem_count(),
                spatial,
                c,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        let x = self.f32_data()?;
        let bv = bias.f32_data()?;
        let mut out = vec![0f32; x.len()];
        if x.len() >= PAR_CPU_MIN {
            use rayon::prelude::*;
            out.par_chunks_mut(spatial)
                .zip(x.par_chunks(spatial))
                .enumerate()
                .for_each(|(row, (d, s))| {
                    let bb = bv[row % c];
                    for (dd, &sv) in d.iter_mut().zip(s) {
                        *dd = sv + bb;
                    }
                });
        } else {
            for (i, (d, &s)) in out.iter_mut().zip(x).enumerate() {
                *d = s + bv[(i / spatial) % c];
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            return Err(Error(format!(
                "reshape: {:?} -> {:?} changes element count",
                self.dims(),
                shape.dims()
            )));
        }
        // Metadata-only: views reshape freely (offset unchanged, packed copy
        // shared - the data span is identical).
        Ok(Self {
            storage_raw: self.storage_raw.clone(),
            shape,
            offset: self.offset,
            packed: self.packed.clone(),
        })
    }

    /// Borrow the f16 CPU elements in place. Same rationale as `f32_data`.
    pub(crate) fn f16_data(&self) -> Result<&[half::f16]> {
        match self.storage().as_ref() {
            Storage::Cpu(CpuStorage::F16(v)) => Ok(v),
            other => Err(Error(format!(
                "op needs f16 CPU storage, got {} on {:?}",
                other.dtype(),
                self.device()
            ))),
        }
    }

    /// Borrow the f32 CPU elements in place. Callers that only read the values
    /// should prefer this over the `to_vec*` family, which copies the whole
    /// tensor (and, for the rank-2 form, allocates one buffer per row).
    pub(crate) fn f32_data(&self) -> Result<&[f32]> {
        match self.storage().as_ref() {
            Storage::Cpu(CpuStorage::F32(v)) => Ok(v),
            other => Err(Error(format!(
                "op needs f32 CPU storage, got {} on {:?}",
                other.dtype(),
                other.device()
            ))),
        }
    }

    /// Borrow contiguous CPU f32 data without copying (quantized-matmul
    /// decode fast path); errors on device tensors / other dtypes.
    pub(in crate::tensor) fn cpu_f32_data(&self) -> Result<&[f32]> {
        self.f32_data()
    }

    /// Borrow contiguous CPU f16 data without copying (half-carrier models'
    /// quantized-matmul fast path - skips the f32 detour buffers).
    pub(in crate::tensor) fn cpu_f16_data(&self) -> Result<&[half::f16]> {
        match self.storage().as_ref() {
            Storage::Cpu(CpuStorage::F16(v)) => Ok(v),
            other => Err(Error(format!(
                "op needs f16 CPU storage, got {} on {:?}",
                other.dtype(),
                other.device()
            ))),
        }
    }

    /// Borrow the CPU storage (dtype-generic ops dispatch on it via
    /// `map_cpu!`); errors on device tensors.
    pub(super) fn cpu_storage_ref(&self) -> Result<&CpuStorage> {
        match self.storage().as_ref() {
            Storage::Cpu(c) => Ok(c),
            other => Err(Error(format!(
                "op needs CPU storage, got {} on {:?}",
                other.dtype(),
                other.device()
            ))),
        }
    }

    /// The tensor's data as host storage (downloads when on CUDA).
    pub(super) fn host_storage(&self) -> Result<std::borrow::Cow<'_, CpuStorage>> {
        match self.storage().as_ref() {
            Storage::Cpu(c) => Ok(std::borrow::Cow::Borrowed(c)),
            Storage::Dry(_) => Err(Error("a dry tensor has no host data to read".into())),
            #[cfg(feature = "cuda")]
            Storage::Cuda { data, dev } => Ok(std::borrow::Cow::Owned(data.download(dev)?)),
        }
    }

    /// Read a CPU index tensor (u32 or i64) as usize values.
    pub(super) fn index_values(ids: &Self) -> Result<Vec<usize>> {
        match ids.cpu_storage_ref()? {
            CpuStorage::U32(v) => Ok(v.iter().map(|&x| x as usize).collect()),
            CpuStorage::I64(v) => {
                if let Some(neg) = v.iter().find(|&&x| x < 0) {
                    return Err(Error(format!("negative index {neg}")));
                }
                Ok(v.iter().map(|&x| x as usize).collect())
            }
            other => Err(Error(format!(
                "index tensor must be u32/i64, got {}",
                other.dtype()
            ))),
        }
    }

    pub(super) fn binary_f32(
        &self,
        rhs: &Self,
        op: &'static str,
        f: impl Fn(f32, f32) -> f32 + Sync,
    ) -> Result<Self> {
        if self.dims() != rhs.dims() {
            return Err(Error(format!(
                "{op}: shape mismatch {:?} vs {:?}",
                self.dims(),
                rhs.dims()
            )));
        }
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        if let Some(t) = rhs.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        // CPU half-precision route:
        // compute in f32 and round back - identical to the `half` crate's
        // own arithmetic (it converts through f32 per op).
        if matches!(self.dtype(), DType::F16 | DType::BF16) && rhs.dtype() == self.dtype() {
            let dt = self.dtype();
            return self
                .to_dtype(DType::F32)?
                .binary_f32(&rhs.to_dtype(DType::F32)?, op, f)?
                .to_dtype(dt);
        }
        let a = self.f32_data()?;
        let b = rhs.f32_data()?;
        let out: Vec<f32> = if a.len() >= PAR_CPU_MIN {
            // Spin-pool, not rayon (see unary_f32).
            const CH: usize = 1 << 16;
            let mut out = vec![0f32; a.len()];
            crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, CH, &|c, d| {
                let sa = &a[c * CH..c * CH + d.len()];
                let sb = &b[c * CH..c * CH + d.len()];
                for ((dd, &x), &y) in d.iter_mut().zip(sa).zip(sb) {
                    *dd = f(x, y);
                }
            });
            out
        } else {
            a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
        };
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn binary_cuda(&self, rhs: &Self, kernel: &'static str) -> Result<Option<Self>> {
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() != rdev.ordinal() {
                return Err(Error(format!(
                    "{kernel}: operands on different cuda devices"
                )));
            }
            if self.dims() != rhs.dims() {
                return Err(Error(format!(
                    "{kernel}: shape mismatch {:?} vs {:?}",
                    self.dims(),
                    rhs.dims()
                )));
            }
            let out = crate::tensor::cuda::binary_f32(
                dev,
                kernel,
                data.as_f32_slice()?,
                rd.as_f32_slice()?,
                self.elem_count(),
            )?;
            return Ok(Some(Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            )?));
        }
        Ok(None)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_f16_direct(rhs, "native_add_f16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_bf16_direct(rhs, "native_add_bf16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() && rhs.is_cuda_f16() {
            let r = rhs.to_dtype(DType::F32)?;
            return self.f16_unary_via_f32(|t| t.add(&r));
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_cuda(rhs, "native_add_f32")? {
            return Ok(t);
        }
        self.binary_f32(rhs, "add", |x, y| x + y)
    }

    /// Direct half-precision same-shape binary kernel (f16 only).
    /// `Ok(None)` = not applicable.
    #[cfg(feature = "cuda")]
    pub(super) fn binary_f16_direct(
        &self,
        rhs: &Self,
        kernel: &'static str,
    ) -> Result<Option<Self>> {
        if self.dtype() != DType::F16 || rhs.dtype() != DType::F16 {
            return Ok(None);
        }
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() == rdev.ordinal() && self.dims() == rhs.dims() {
                let out = crate::tensor::cuda::binary_f16(
                    dev,
                    kernel,
                    data.as_f16_slice()?,
                    rd.as_f16_slice()?,
                    self.elem_count(),
                )?;
                return Ok(Some(Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                )?));
            }
        }
        Ok(None)
    }

    /// Direct bf16 same-shape binary kernel (single-kernel; bit-identical to
    /// the f32 detour for add/mul/div, minus 2 casts + allocs per op).
    /// `Ok(None)` = not applicable.
    #[cfg(feature = "cuda")]
    pub(super) fn binary_bf16_direct(
        &self,
        rhs: &Self,
        kernel: &'static str,
    ) -> Result<Option<Self>> {
        if self.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
            return Ok(None);
        }
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() == rdev.ordinal() && self.dims() == rhs.dims() {
                let out = crate::tensor::cuda::binary_bf16(
                    dev,
                    kernel,
                    data.as_bf16_slice()?,
                    rd.as_bf16_slice()?,
                    self.elem_count(),
                )?;
                return Ok(Some(Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(out),
                    dev.clone(),
                    self.shape.clone(),
                )?));
            }
        }
        Ok(None)
    }

    /// Elementwise division - DIRECT a/b kernels (single rounding, unlike the
    /// `bdiv` semantics; a recip-then-mul chain rounds twice).
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::F16 && rhs.dtype() == DType::F16 {
            if let (
                Storage::Cuda { data, dev },
                Storage::Cuda {
                    data: rd,
                    dev: rdev,
                },
            ) = (self.storage().as_ref(), rhs.storage().as_ref())
            {
                if dev.ordinal() == rdev.ordinal() && self.dims() == rhs.dims() {
                    let out = crate::tensor::cuda::binary_f16(
                        dev,
                        "native_div_f16",
                        data.as_f16_slice()?,
                        rd.as_f16_slice()?,
                        self.elem_count(),
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F16(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_bf16_direct(rhs, "native_div_bf16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() && rhs.is_cuda_f16() {
            let r = rhs.to_dtype(DType::F32)?;
            return self.f16_unary_via_f32(|t| t.div(&r));
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_cuda(rhs, "native_div_f32")? {
            return Ok(t);
        }
        self.binary_f32(rhs, "div", |x, y| x / y)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_f16_direct(rhs, "native_mul_f16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_bf16_direct(rhs, "native_mul_bf16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() && rhs.is_cuda_f16() {
            let r = rhs.to_dtype(DType::F32)?;
            return self.f16_unary_via_f32(|t| t.mul(&r));
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.binary_cuda(rhs, "native_mul_f32")? {
            return Ok(t);
        }
        self.binary_f32(rhs, "mul", |x, y| x * y)
    }

    /// Swap two axes. When the swap only moves size-1 axes (the decode-path
    /// `[1, 1, h, d] -> [1, h, 1, d]` head reshapes), the packed memory order
    /// is unchanged and this is a zero-cost metadata reshape - the same cases
    /// a stride-view substrate serves for free. Other swaps materialize a
    /// contiguous copy.
    #[track_caller]
    pub fn transpose<A: Dim, B: Dim>(&self, d1: A, d2: B) -> Result<Self> {
        let a = d1.to_index(&self.shape, "transpose")?;
        let b = d2.to_index(&self.shape, "transpose")?;
        if a == b {
            return Ok(self.clone());
        }
        {
            let (lo, hi) = (a.min(b), a.max(b));
            let dims = self.dims();
            // The relative order of the non-1 dims is preserved iff both
            // swapped dims are 1, or one is 1 and nothing non-1 sits between.
            let middle_unit = dims[lo + 1..hi].iter().all(|&d| d == 1);
            if (dims[lo] == 1 && dims[hi] == 1) || (middle_unit && (dims[lo] == 1 || dims[hi] == 1))
            {
                let mut odims = dims.to_vec();
                odims.swap(lo, hi);
                return self.reshape(odims);
            }
        }
        // A transpose is a COPY on this substrate, not a view - the attention's
        // q/k/v swap allocates its operands again, per block. Counted here for the
        // same reason: what the forward allocates is what a placement must reserve.
        if self.is_dry() {
            let mut odims = self.dims().to_vec();
            odims.swap(a, b);
            if let Some(t) = self.dry_out(odims, self.dtype()) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            // width-generic device gather - every dtype, no f16-cast detour
            //
            let _b = crate::tensor::bounce::start("dev_transpose_copy");
            let dims = self.dims().to_vec();
            let mut odims = dims.clone();
            odims.swap(a, b);
            let in_stride = self.shape.stride_contiguous();
            let mut perm: Vec<usize> = (0..dims.len()).collect();
            perm.swap(a, b);
            let odims_i: Vec<i32> = odims.iter().map(|&d| d as i32).collect();
            let strides_i: Vec<i32> = perm.iter().map(|&ax| in_stride[ax] as i32).collect();
            let out = crate::tensor::cuda::permute_storage(
                dev,
                data,
                &odims_i,
                &strides_i,
                self.elem_count(),
            )?;
            return Self::from_cuda_storage(out, dev.clone(), odims);
        }
        let dims = self.dims().to_vec();
        let mut odims = dims.clone();
        odims.swap(a, b);
        let in_stride = self.shape.stride_contiguous();
        let oshape = Shape::from(odims.clone());
        let out_stride = oshape.stride_contiguous();
        // walk output indices; map back to input (output axis i = input axis
        // perm[i]). Dtype-generic.
        let mut perm: Vec<usize> = (0..dims.len()).collect();
        perm.swap(a, b);
        let ndim = dims.len();
        let inner = *dims.last().unwrap_or(&1);
        // Fast path for the attention q/k/v layout swap `transpose(1,2)` and any
        // swap that leaves the innermost (contiguous) axis in place: the last
        // axis is a contiguous run in BOTH layouts, so copy it as a block and
        // decompose only the outer indices - one div/mod per block instead of per
        // element, and a vectorized `copy_from_slice` for the run. The general
        // per-element gather below still covers swaps that touch the last axis.
        let last_axis_kept = ndim >= 2 && inner > 0 && a != ndim - 1 && b != ndim - 1;
        let out_outer: Vec<usize> = out_stride[..ndim.saturating_sub(1)].to_vec();
        let in_outer: Vec<usize> = (0..ndim.saturating_sub(1))
            .map(|ax| in_stride[perm[ax]])
            .collect();
        let c = self.cpu_storage_ref()?;
        let storage = map_cpu!(c, |v, wrap| {
            let mut out = v.clone();
            if last_axis_kept {
                let block = |first_block: usize, chunk: &mut [_]| {
                    for (bi, blk) in chunk.chunks_mut(inner).enumerate() {
                        let mut rem = (first_block + bi) * inner;
                        let mut ii = 0usize;
                        for ax in 0..ndim - 1 {
                            let idx = rem / out_outer[ax];
                            rem %= out_outer[ax];
                            ii += idx * in_outer[ax];
                        }
                        blk.copy_from_slice(&v[ii..ii + inner]);
                    }
                };
                if out.len() >= PAR_CPU_MIN {
                    use rayon::prelude::*;
                    // Chunk on an inner-block boundary so each task owns whole runs.
                    let per = ((1usize << 14) / inner).max(1) * inner;
                    out.par_chunks_mut(per)
                        .enumerate()
                        .for_each(|(ci, chunk)| block(ci * (per / inner), chunk));
                } else {
                    block(0, &mut out);
                }
            } else {
                let gather = |chunk_base: usize, chunk: &mut [_]| {
                    for (off, slot) in chunk.iter_mut().enumerate() {
                        let mut rem = chunk_base + off;
                        let mut ii = 0usize;
                        for (ax, &os) in out_stride.iter().enumerate() {
                            let idx = rem / os;
                            rem %= os;
                            ii += idx * in_stride[perm[ax]];
                        }
                        *slot = v[ii];
                    }
                };
                if out.len() >= PAR_CPU_MIN {
                    use rayon::prelude::*;
                    out.par_chunks_mut(1 << 14)
                        .enumerate()
                        .for_each(|(ci, chunk)| gather(ci << 14, chunk));
                } else {
                    gather(0, &mut out);
                }
            }
            wrap(out)
        });
        Ok(Self::from_packed(Arc::new(Storage::Cpu(storage)), oshape))
    }

    /// Slice `len` elements starting at `start` along `dim`. Zero-copy VIEW
    /// when every dim before `dim` is 1; strided copy otherwise
    /// (`#[track_caller]`: NATIVE_BOUNCE charges the copy to the call site).
    #[track_caller]
    pub fn narrow<I: Dim>(&self, dim: I, start: usize, len: usize) -> Result<Self> {
        let d = dim.to_index(&self.shape, "narrow")?;
        let dims = self.dims();
        if start + len > dims[d] {
            return Err(Error(format!(
                "narrow: {start}+{len} > dim {d} of {:?}",
                dims
            )));
        }
        let inner: usize = dims[d + 1..].iter().product();
        // Zero-copy VIEW when the selected range is contiguous in the packed
        // layout - every dim BEFORE `d` is 1 (includes d == 0). A narrow ON a
        // view composes: the offsets add (invariant: offset + elem_count() <=
        // raw span, preserved because the parent's span ends at
        // offset + dims[d].inner here).
        if dims[..d].iter().all(|&x| x == 1) {
            let mut odims = dims.to_vec();
            odims[d] = len;
            return Ok(Self {
                storage_raw: self.storage_raw.clone(),
                shape: Shape::from(odims),
                offset: self.offset + start * inner,
                packed: Arc::new(OnceLock::new()),
            });
        }
        let outer: usize = dims[..d].iter().product();
        // Past the zero-copy case a narrow is a strided COPY, on every device.
        if self.is_dry() {
            let mut odims = dims.to_vec();
            odims[d] = len;
            if let Some(t) = self.dry_out(odims, self.dtype()) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage_raw.as_ref() {
            // device-side byte copy - the old host_bounce round-tripped the
            // whole tensor through the CPU per narrow. Works directly on the
            // RAW storage of a view: the view's base offset adds to the
            // per-row start offset.
            let _b = crate::tensor::bounce::start("dev_narrow_copy");
            let esize = self.dtype().size_in_bytes();
            let out = crate::tensor::cuda::narrow_storage(
                dev,
                data,
                outer.max(1),
                dims[d] * inner * esize,
                (self.offset + start * inner) * esize,
                len * inner * esize,
            )?;
            let mut odims = dims.to_vec();
            odims[d] = len;
            return Self::from_cuda_storage(out, dev.clone(), Shape::from(odims));
        }
        let c = match self.storage_raw.as_ref() {
            Storage::Cpu(c) => c,
            other => {
                return Err(Error(format!(
                    "narrow needs CPU storage, got {} on {:?}",
                    other.dtype(),
                    other.device()
                )))
            }
        };
        let storage = map_cpu!(c, |v, wrap| {
            let mut out = Vec::with_capacity(outer * len * inner);
            for o in 0..outer {
                let base = self.offset + (o * dims[d] + start) * inner;
                out.extend_from_slice(&v[base..base + len * inner]);
            }
            wrap(out)
        });
        let mut odims = dims.to_vec();
        odims[d] = len;
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(storage)),
            Shape::from(odims),
        ))
    }

    /// Split into `n` equal narrows along `dim`.
    pub fn chunk<I: Dim>(&self, n: usize, dim: I) -> Result<Vec<Self>> {
        let d = dim.to_index(&self.shape, "chunk")?;
        let size = self.dims()[d];
        if size % n != 0 {
            return Err(Error(format!(
                "chunk: dim {d} of {size} not divisible by {n}"
            )));
        }
        let len = size / n;
        (0..n).map(|i| self.narrow(d, i * len, len)).collect()
    }

    /// Drop a size-1 dim (contiguous storage: pure reshape).
    pub fn squeeze<I: Dim>(&self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "squeeze")?;
        if self.dims()[d] != 1 {
            return Err(Error(format!(
                "squeeze: dim {d} is {} != 1",
                self.dims()[d]
            )));
        }
        let mut odims = self.dims().to_vec();
        odims.remove(d);
        self.reshape(odims)
    }

    /// Insert a size-1 dim at `dim` (contiguous storage: pure reshape).
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        if dim > self.rank() {
            return Err(Error(format!(
                "unsqueeze: dim {dim} > rank {}",
                self.rank()
            )));
        }
        let mut odims = self.dims().to_vec();
        odims.insert(dim, 1);
        self.reshape(odims)
    }

    /// Collapse dims `[dim..]` into one (contiguous storage: pure reshape).
    pub fn flatten_from<I: Dim>(&self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "flatten_from")?;
        let mut odims: Vec<usize> = self.dims()[..d].to_vec();
        odims.push(self.dims()[d..].iter().product());
        self.reshape(odims)
    }

    /// Zero-pad `dim` with `left`/`right` extra slots.
    pub fn pad_with_zeros<I: Dim>(&self, dim: I, left: usize, right: usize) -> Result<Self> {
        let d = dim.to_index(&self.shape, "pad_with_zeros")?;
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let dims = self.dims().to_vec();
        let outer: usize = dims[..d].iter().product();
        let inner: usize = dims[d + 1..].iter().product();
        let d_in = dims[d];
        let mut odims = dims.clone();
        odims[d] = d_in + left + right;
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.pad_with_zeros(d, left, right));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::pad_dim_f32(
                dev,
                data.as_f32_slice()?,
                outer,
                d_in,
                left,
                right,
                inner,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                odims,
            );
        }
        let src = self.f32_data()?;
        let d_out = d_in + left + right;
        let mut out = vec![0f32; outer * d_out * inner];
        for o in 0..outer {
            let dst = &mut out[(o * d_out + left) * inner..][..d_in * inner];
            dst.copy_from_slice(&src[o * d_in * inner..][..d_in * inner]);
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(odims),
        ))
    }

    /// Nearest-neighbor upsample of the last two dims of `[b, c, h, w]`.
    pub fn upsample_nearest2d(&self, oh: usize, ow: usize) -> Result<Self> {
        let (b, c, h, w) = self.shape.dims4()?;
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.upsample_nearest2d(oh, ow));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::upsample2d_f32(
                dev,
                data.as_f32_slice()?,
                b * c,
                h,
                w,
                oh,
                ow,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                vec![b, c, oh, ow],
            );
        }
        let src = self.f32_data()?;
        let mut out = vec![0f32; b * c * oh * ow];
        for bc in 0..b * c {
            let s = &src[bc * h * w..][..h * w];
            let dst = &mut out[bc * oh * ow..][..oh * ow];
            for ho in 0..oh {
                let hi = ho * h / oh;
                for wo in 0..ow {
                    dst[ho * ow + wo] = s[hi * w + wo * w / ow];
                }
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c, oh, ow]),
        ))
    }

    /// Transposed convolution (`[b, ci, h, w]` -> `[b, co, h*stride - ..., ...]`).
    ///
    /// The weight is `[in, out, kh, kw]` - the INPUT channel leads, which is the
    /// transposed layout and the opposite of `conv2d`. Reading it as `[out, in, ..]`
    /// produces a correctly-shaped, entirely wrong result.
    ///
    /// Scatter formulation: every input pixel contributes its kernel, scaled, to a
    /// stride-spaced window of the output. That is the definition, and it avoids the
    /// index gymnastics of expressing it as a padded forward convolution.
    /// The same transposed convolution, built from operations that HAVE a device path.
    ///
    /// The scalar scatter below is host-only and unconditional: it copies the input AND
    /// the weights down, loops in one thread, and everything computed after it stays on
    /// the host too. Measured on the face-swap generator, whose upsampling path is two of
    /// these, that was the WHOLE network running on one core - 1.9 s a face, with the GPU
    /// idle, while the log reported the weights placed on a card.
    ///
    /// A transposed convolution is a normal one on a dilated input with a flipped kernel:
    /// insert `stride - 1` zeros between the input's samples, pad by `k - 1 - padding`,
    /// reverse the kernel along both spatial axes, exchange its in/out channel axes, and
    /// convolve at stride 1. `conv2d`, `cat`, `narrow` and `index_select` all have device
    /// paths, so the whole thing stays where the weights are.
    ///
    /// Returns `None` when the shape is one this rewrite does not cover, and the caller
    /// falls back to the scalar loop rather than producing something subtly different.
    pub(super) fn conv_transpose2d_via_conv2d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
    ) -> Result<Option<Self>> {
        let (b, ci, h, w) = self.shape.dims4()?;
        let (wi, _co, kh, kw) = weight.shape().dims4()?;
        // `conv2d` takes ONE padding for both axes, and the identity below needs the
        // kernel to reach at least as far as the padding it undoes.
        if wi != ci || kh != kw || kh == 0 || kh <= padding || stride == 0 {
            return Ok(None);
        }
        let dev = self.device().clone();
        let dt = self.dtype();

        // Zero-dilate: [b,ci,h,w] -> [b,ci,(h-1)*s+1,(w-1)*s+1]. Built by widening each
        // sample into a run of `stride` along a fresh axis, filled with zeros after the
        // first, then folding that axis back into the spatial one.
        let x = if stride == 1 {
            self.contiguous()?
        } else {
            let pad_w = Self::zeros_on(vec![b, ci, h, w, stride - 1], dt, &dev)?;
            let rowed = Self::cat(&[&self.reshape(vec![b, ci, h, w, 1])?, &pad_w], 4)?
                .reshape(vec![b, ci, h, w * stride])?;
            let pad_h = Self::zeros_on(vec![b, ci, h, stride - 1, w * stride], dt, &dev)?;
            let full = Self::cat(&[&rowed.reshape(vec![b, ci, h, 1, w * stride])?, &pad_h], 3)?
                .reshape(vec![b, ci, h * stride, w * stride])?;
            // The trailing zeros of the last sample are not part of the signal.
            full.narrow(2, 0, (h - 1) * stride + 1)?
                .narrow(3, 0, (w - 1) * stride + 1)?
                .contiguous()?
        };

        // Reverse both spatial axes, then exchange the channel axes: the stored layout is
        // [in, out, kh, kw] and `conv2d` reads [out, in, kh, kw]. Reading it either way
        // without this produces a correctly-shaped and entirely wrong result.
        let rev: Vec<u32> = (0..kh as u32).rev().collect();
        let idx = Self::from_vec_u32(rev, kh)?.to_device(&dev)?;
        let k = weight
            .index_select(&idx, 2)?
            .index_select(&idx, 3)?
            .permute(&[1, 0, 2, 3])?
            .contiguous()?;

        Ok(Some(x.conv2d(&k, kh - 1 - padding, 1, 1, 1)?))
    }

    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let (b, ci, h, w) = self.shape.dims4()?;
        let (wi, co, kh, kw) = weight.shape().dims4()?;
        if wi != ci {
            return Err(Error(format!(
                "conv_transpose2d: input has {ci} channels, weight expects {wi}"
            )));
        }
        let oh = (h - 1) * stride + kh - 2 * padding;
        let ow = (w - 1) * stride + kw - 2 * padding;
        // On a device, do it THERE. See `conv_transpose2d_via_conv2d`: the loop below
        // drags the whole forward onto the host, and everything after it stays.
        if self.is_on_cuda() {
            if let Some(out) = self.conv_transpose2d_via_conv2d(weight, stride, padding)? {
                return match bias {
                    Some(bs) => out.broadcast_add(&bs.reshape(vec![1, co, 1, 1])?),
                    None => Ok(out),
                };
            }
        }
        let x = self.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let k = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let xs = x.f32_data()?;
        let ks = k.f32_data()?;
        let mut out = vec![0f32; b * co * oh * ow];
        for bi in 0..b {
            for c in 0..ci {
                let plane = &xs[((bi * ci) + c) * h * w..][..h * w];
                for y in 0..h {
                    for xx in 0..w {
                        let v = plane[y * w + xx];
                        if v == 0.0 {
                            continue;
                        }
                        for o in 0..co {
                            let kern = &ks[((c * co) + o) * kh * kw..][..kh * kw];
                            let dst = &mut out[((bi * co) + o) * oh * ow..][..oh * ow];
                            for ky in 0..kh {
                                let oy = y * stride + ky;
                                if oy < padding || oy - padding >= oh {
                                    continue;
                                }
                                let oy = oy - padding;
                                for kx in 0..kw {
                                    let ox = xx * stride + kx;
                                    if ox < padding || ox - padding >= ow {
                                        continue;
                                    }
                                    dst[oy * ow + (ox - padding)] += v * kern[ky * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(bias) = bias {
            let bs = bias.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let bv = bs.f32_data()?;
            for bi in 0..b {
                for o in 0..co {
                    let add = bv[o];
                    for v in &mut out[((bi * co) + o) * oh * ow..][..oh * ow] {
                        *v += add;
                    }
                }
            }
        }
        let t = Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, co, oh, ow]),
        );
        t.to_device(&self.device())
    }

    /// Max pooling over `k x k` windows with stride `k`.
    pub fn max_pool2d(&self, k: usize) -> Result<Self> {
        let (b, c, h, w) = self.shape.dims4()?;
        let (oh, ow) = (h / k, w / k);
        // On a device, do it THERE. Pooling is a max over each k-by-k block, and folding
        // those blocks onto their own axes turns it into two reductions that already have
        // a device path. Without this the loop below copies the whole feature map to the
        // host, and - as with `conv_transpose2d` - everything computed after it stays on
        // the host too: the face-swap generator's encoder pools once per scale, which was
        // enough to drag the network off the card it had been placed on.
        if self.is_on_cuda() && k > 0 && oh > 0 && ow > 0 {
            let folded = self
                .narrow(2, 0, oh * k)?
                .narrow(3, 0, ow * k)?
                .contiguous()?
                .reshape(vec![b, c, oh, k, ow, k])?;
            // Innermost axis first, so the second reduction reads a contiguous result.
            let out = folded.max_keepdim(5)?.max_keepdim(3)?;
            return out.contiguous()?.reshape(vec![b, c, oh, ow]);
        }
        let x = self.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let xs = x.f32_data()?;
        let mut out = vec![0f32; b * c * oh * ow];
        for bc in 0..b * c {
            let src = &xs[bc * h * w..][..h * w];
            let dst = &mut out[bc * oh * ow..][..oh * ow];
            for y in 0..oh {
                for xx in 0..ow {
                    let mut m = f32::NEG_INFINITY;
                    for dy in 0..k {
                        for dx in 0..k {
                            m = m.max(src[(y * k + dy) * w + xx * k + dx]);
                        }
                    }
                    dst[y * ow + xx] = m;
                }
            }
        }
        let t = Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c, oh, ow]),
        );
        t.to_device(&self.device())
    }

    /// Bilinear resample to `[b, c, oh, ow]`, HALF-PIXEL centres.
    ///
    /// Matches PyTorch's `align_corners=false` and ONNX's `pytorch_half_pixel`. The
    /// corner-aligned variant differs by half an output pixel, which is a visible
    /// drift once the result is warped back into a photograph.
    pub fn upsample_bilinear2d(&self, oh: usize, ow: usize) -> Result<Self> {
        let (b, c, h, w) = self.shape.dims4()?;
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.upsample_bilinear2d(oh, ow));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::upsample_bilinear2d_f32(
                dev,
                data.as_f32_slice()?,
                b * c,
                h,
                w,
                oh,
                ow,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                vec![b, c, oh, ow],
            );
        }
        let src = self.f32_data()?;
        let mut out = vec![0f32; b * c * oh * ow];
        let (sy_r, sx_r) = (h as f32 / oh as f32, w as f32 / ow as f32);
        for bc in 0..b * c {
            let s = &src[bc * h * w..][..h * w];
            let dst = &mut out[bc * oh * ow..][..oh * ow];
            for ho in 0..oh {
                let sy = (((ho as f32 + 0.5) * sy_r - 0.5).max(0.0)).min((h - 1) as f32);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(h - 1);
                let fy = sy - y0 as f32;
                for wo in 0..ow {
                    let sx = (((wo as f32 + 0.5) * sx_r - 0.5).max(0.0)).min((w - 1) as f32);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let fx = sx - x0 as f32;
                    let top = s[y0 * w + x0] * (1.0 - fx) + s[y0 * w + x1] * fx;
                    let bot = s[y1 * w + x0] * (1.0 - fx) + s[y1 * w + x1] * fx;
                    dst[ho * ow + wo] = top * (1.0 - fy) + bot * fy;
                }
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c, oh, ow]),
        ))
    }

    /// Concatenate along `dim`; all other dims must match.
    pub fn cat<I: Dim>(tensors: &[&Self], dim: I) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| Error("cat: empty input".into()))?;
        let d = dim.to_index(&first.shape, "cat")?;
        if tensors.iter().any(|t| t.is_dry()) {
            let mut odims = first.dims().to_vec();
            odims[d] = tensors.iter().map(|t| t.dims()[d]).sum();
            let dtype = first.dtype();
            for t in tensors {
                if let Some(out) = t.dry_out(odims.clone(), dtype) {
                    return Ok(out);
                }
            }
        }
        #[cfg(feature = "cuda")]
        if first.is_on_cuda() {
            // device cat for ANY same-dtype/same-device operands (cat is pure
            // data movement - one width-generic strided-copy kernel per
            // operand; f16/bf16/int cats previously host-bounced).
            let all_same_dev = tensors.iter().all(|t| {
                matches!(t.storage().as_ref(), Storage::Cuda { data, dev }
                    if data.dtype() == first.dtype()
                        && data.dtype() != DType::F64
                        && matches!(first.storage().as_ref(), Storage::Cuda { dev: fd, .. }
                            if fd.ordinal() == dev.ordinal()))
            });
            if all_same_dev {
                let _b = crate::tensor::bounce::start("dev_cat_copy");
                let dims = first.dims();
                let mut cat_len = 0usize;
                for t in tensors {
                    let td = t.dims();
                    if td.len() != dims.len()
                        || td
                            .iter()
                            .zip(dims)
                            .enumerate()
                            .any(|(i, (x, y))| i != d && x != y)
                    {
                        return Err(Error(format!("cat: shape mismatch {td:?} vs {dims:?}")));
                    }
                    cat_len += td[d];
                }
                let outer: usize = dims[..d].iter().product();
                let inner: usize = dims[d + 1..].iter().product();
                let Storage::Cuda { dev, .. } = first.storage().as_ref() else {
                    unreachable!()
                };
                // every element is written by exactly one cat_copy launch
                let mut out = crate::tensor::cuda::CudaStorage::alloc_uninit(
                    dev,
                    first.dtype(),
                    outer * cat_len * inner,
                )?;
                let mut off_d = 0usize;
                for t in tensors {
                    let Storage::Cuda { data, .. } = t.storage().as_ref() else {
                        unreachable!()
                    };
                    let td = t.dims()[d];
                    // one strided-copy kernel per operand (the old per-outer
                    // dtod loop issued `outer` copies per tensor - 10⁵ tiny
                    // memcpys per forward on the flux single-block cats)
                    crate::tensor::cuda::cat_copy_storage(
                        dev,
                        data,
                        &mut out,
                        outer * td * inner,
                        td * inner,
                        cat_len * inner,
                        off_d * inner,
                    )?;
                    off_d += td;
                }
                let mut odims = dims.to_vec();
                odims[d] = cat_len;
                return Self::from_cuda_storage(out, dev.clone(), odims);
            }
            let _b = crate::tensor::bounce::start("host_cat");
            let dev = first.device();
            let cpus: Vec<Self> = tensors
                .iter()
                .map(|t| t.to_device(&Device::Cpu))
                .collect::<Result<_>>()?;
            let refs: Vec<&Self> = cpus.iter().collect();
            return Self::cat(&refs, d)?.to_device(&dev);
        }
        let dims = first.dims();
        let mut cat_len = 0usize;
        for t in tensors {
            let td = t.dims();
            if td.len() != dims.len()
                || td
                    .iter()
                    .zip(dims)
                    .enumerate()
                    .any(|(i, (x, y))| i != d && x != y)
            {
                return Err(Error(format!("cat: shape mismatch {td:?} vs {dims:?}")));
            }
            cat_len += td[d];
        }
        let outer: usize = dims[..d].iter().product();
        let inner: usize = dims[d + 1..].iter().product();
        // dtype-generic byte interleave (u32 token ids / i64 positions /
        // halves concatenate through the same row copies as f32)
        let dtype = first.dtype();
        let esize = dtype.size_in_bytes();
        let mut out = Vec::with_capacity(outer * cat_len * inner * esize);
        for o in 0..outer {
            for t in tensors {
                if t.dtype() != dtype {
                    return Err(Error(format!(
                        "cat: dtype mismatch {} vs {dtype}",
                        t.dtype()
                    )));
                }
                let bytes = cpu_storage_bytes(t.cpu_storage_ref()?);
                let td = t.dims()[d];
                let base = o * td * inner * esize;
                out.extend_from_slice(&bytes[base..base + td * inner * esize]);
            }
        }
        let mut odims = dims.to_vec();
        odims[d] = cat_len;
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(cpu_storage_from_bytes(dtype, &out)?)),
            Shape::from(odims),
        ))
    }

    /// Numpy broadcast layout of `self` x `rhs`: output shape + per-axis
    /// effective strides into each operand (0 on broadcast axes), padded to
    /// the output rank. Shared by the CPU loop and the CUDA N-d kernel.
    pub(super) fn broadcast_layout(
        &self,
        rhs: &Self,
        op: &'static str,
    ) -> Result<(Shape, Vec<usize>, Vec<usize>)> {
        let ld = self.dims();
        let rd = rhs.dims();
        let rank = ld.len().max(rd.len());
        let pad = |d: &[usize]| -> Vec<usize> {
            let mut v = vec![1usize; rank - d.len()];
            v.extend_from_slice(d);
            v
        };
        let (lp, rp) = (pad(ld), pad(rd));
        let mut odims = vec![0usize; rank];
        for i in 0..rank {
            odims[i] = match (lp[i], rp[i]) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                (a, b) => {
                    return Err(Error(format!(
                        "{op}: cannot broadcast {ld:?} with {rd:?} (axis {i}: {a} vs {b})"
                    )))
                }
            };
        }
        // effective strides: 0 on broadcast axes
        let eff = |padded: &[usize]| -> Vec<usize> {
            let s = Shape::from(padded.to_vec()).stride_contiguous();
            padded
                .iter()
                .zip(&s)
                .map(|(&d, &st)| if d == 1 { 0 } else { st })
                .collect()
        };
        let (ls, rs) = (eff(&lp), eff(&rp));
        Ok((Shape::from(odims), ls, rs))
    }

    /// Numpy-style broadcasting of `rhs` against `self` for a binary op.
    pub(super) fn broadcast_binary(
        &self,
        rhs: &Self,
        op: &'static str,
        f: impl Fn(f32, f32) -> f32 + Sync,
    ) -> Result<Self> {
        // CPU half-precision route: f32 compute + round back (see binary_f32).
        if matches!(self.dtype(), DType::F16 | DType::BF16) && rhs.dtype() == self.dtype() {
            let dt = self.dtype();
            return self
                .to_dtype(DType::F32)?
                .broadcast_binary(&rhs.to_dtype(DType::F32)?, op, f)?
                .to_dtype(dt);
        }
        let (oshape, ls, rs) = self.broadcast_layout(rhs, op)?;
        let ostride = oshape.stride_contiguous();
        let a = self.f32_data()?;
        let b = rhs.f32_data()?;
        let n = oshape.elem_count();
        let mut out = vec![0.0f32; n];
        let elem = |oi: usize| -> f32 {
            let mut rem = oi;
            let (mut ia, mut ib) = (0usize, 0usize);
            for (ax, &os) in ostride.iter().enumerate() {
                let idx = rem / os;
                rem %= os;
                ia += idx * ls[ax];
                ib += idx * rs[ax];
            }
            f(a[ia], b[ib])
        };
        // Serial small ops run as-is; large broadcasts (e.g. attention's
        // scale/mask over [heads, 1, kv_len] at long context) ran single-
        // threaded while the rest of the cores idled - a dominant wall-time
        // cost of long-context CPU decode. Parallelize those over the output.
        if n >= (1 << 15) {
            // Spin-pool, not rayon (see unary_f32).
            const CH: usize = 1 << 14;
            crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, CH, &|c, d| {
                let base = c * CH;
                for (j, slot) in d.iter_mut().enumerate() {
                    *slot = elem(base + j);
                }
            });
        } else {
            for (oi, slot) in out.iter_mut().enumerate() {
                *slot = elem(oi);
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            oshape,
        ))
    }

    /// True iff padded-rhs is a pure TRAILING block of lhs (i.e. `i % rhs_len`
    /// indexing is exact): leading 1s, then dims equal to lhs's tail.
    pub(super) fn rhs_is_tail_aligned(&self, rhs: &Self) -> bool {
        let ld = self.dims();
        let rd = rhs.dims();
        if rd.len() > ld.len() {
            return false;
        }
        let pad = ld.len() - rd.len();
        let mut seen_non1 = false;
        for i in 0..rd.len() {
            if rd[i] != 1 {
                seen_non1 = true;
            }
            if seen_non1 && rd[i] != ld[pad + i] {
                return false;
            }
        }
        true
    }

    pub(super) fn broadcast_dispatch(
        &self,
        rhs: &Self,
        op: &'static str,
        kernel: &'static str,
        nd_op: i32,
        f: impl Fn(f32, f32) -> f32 + Sync,
    ) -> Result<Self> {
        // The broadcast SHAPE comes from the same helper the device kernel uses, so
        // the counted output is the buffer that kernel would have written.
        if self.is_dry() || rhs.is_dry() {
            let (oshape, _, _) = self.broadcast_layout(rhs, op)?;
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(oshape.clone(), dtype) {
                return Ok(t);
            }
            if let Some(t) = rhs.dry_out(oshape, dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() || rhs.is_on_cuda() {
            if let (
                Storage::Cuda { data, dev },
                Storage::Cuda {
                    data: rd,
                    dev: rdev,
                },
            ) = (self.storage().as_ref(), rhs.storage().as_ref())
            {
                if dev.ordinal() == rdev.ordinal()
                    && self.dtype() == DType::F32
                    && rhs.dtype() == DType::F32
                {
                    if self.rhs_is_tail_aligned(rhs) {
                        let out = crate::tensor::cuda::broadcast_tail_f32(
                            dev,
                            kernel,
                            data.as_f32_slice()?,
                            rd.as_f32_slice()?,
                            self.elem_count(),
                            rhs.elem_count(),
                        )?;
                        return Self::from_cuda_storage(
                            crate::tensor::cuda::CudaStorage::F32(out),
                            dev.clone(),
                            self.shape.clone(),
                        );
                    }
                    // general broadcast (middle axes / lhs-broadcast): the
                    // N-d stride-gather kernel - no host roundtrip.
                    let (oshape, ls, rs) = self.broadcast_layout(rhs, op)?;
                    let odims: Vec<i32> = oshape.dims().iter().map(|&d| d as i32).collect();
                    let lsi: Vec<i32> = ls.iter().map(|&s| s as i32).collect();
                    let rsi: Vec<i32> = rs.iter().map(|&s| s as i32).collect();
                    let out = crate::tensor::cuda::broadcast_nd_f32(
                        dev,
                        data.as_f32_slice()?,
                        rd.as_f32_slice()?,
                        &odims,
                        &lsi,
                        &rsi,
                        oshape.elem_count(),
                        nd_op,
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        oshape,
                    );
                }
            }
            // mixed devices / non-f32 pairs: host bounce
            let rhs_cpu = rhs.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.broadcast_binary(&rhs_cpu, op, f));
        }
        self.broadcast_binary(rhs, op, f)
    }

    /// Direct half tail-aligned broadcast (f16 lhs+rhs, rhs a trailing block
    /// of lhs - the rope/mask/norm-weight decode shapes). `Ok(None)` = not
    /// applicable, caller falls back to the f32 detour.
    #[cfg(feature = "cuda")]
    pub(super) fn broadcast_tail_f16_direct(
        &self,
        rhs: &Self,
        kernel: &'static str,
    ) -> Result<Option<Self>> {
        if self.dtype() != DType::F16 || rhs.dtype() != DType::F16 {
            return Ok(None);
        }
        if !self.rhs_is_tail_aligned(rhs) {
            return Ok(None);
        }
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() == rdev.ordinal() {
                let out = crate::tensor::cuda::broadcast_tail_f16(
                    dev,
                    kernel,
                    data.as_f16_slice()?,
                    rd.as_f16_slice()?,
                    self.elem_count(),
                    rhs.elem_count(),
                )?;
                return Ok(Some(Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                )?));
            }
        }
        Ok(None)
    }

    /// Direct bf16 tail-aligned broadcast (single-kernel, bit-identical to
    /// the f32 detour for add/mul). `Ok(None)` = not applicable.
    #[cfg(feature = "cuda")]
    pub(super) fn broadcast_tail_bf16_direct(
        &self,
        rhs: &Self,
        kernel: &'static str,
    ) -> Result<Option<Self>> {
        if self.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
            return Ok(None);
        }
        if !self.rhs_is_tail_aligned(rhs) {
            return Ok(None);
        }
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() == rdev.ordinal() {
                let out = crate::tensor::cuda::broadcast_tail_bf16(
                    dev,
                    kernel,
                    data.as_bf16_slice()?,
                    rd.as_bf16_slice()?,
                    self.elem_count(),
                    rhs.elem_count(),
                )?;
                return Ok(Some(Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(out),
                    dev.clone(),
                    self.shape.clone(),
                )?));
            }
        }
        Ok(None)
    }

    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Some(t) = self.broadcast_tail_f16_direct(rhs, "native_badd_tail_f16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.broadcast_tail_bf16_direct(rhs, "native_badd_tail_bf16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let r = if matches!(rhs.dtype(), DType::F16 | DType::BF16) {
                rhs.to_dtype(DType::F32)?
            } else {
                rhs.clone()
            };
            return self.f16_unary_via_f32(|t| t.broadcast_add(&r));
        }
        self.broadcast_dispatch(rhs, "broadcast_add", "native_badd_tail_f32", 0, |x, y| {
            x + y
        })
    }

    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Some(t) = self.broadcast_tail_f16_direct(rhs, "native_bmul_tail_f16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.broadcast_tail_bf16_direct(rhs, "native_bmul_tail_bf16")? {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let r = if matches!(rhs.dtype(), DType::F16 | DType::BF16) {
                rhs.to_dtype(DType::F32)?
            } else {
                rhs.clone()
            };
            return self.f16_unary_via_f32(|t| t.broadcast_mul(&r));
        }
        self.broadcast_dispatch(rhs, "broadcast_mul", "native_bmul_tail_f32", 1, |x, y| {
            x * y
        })
    }

    /// Gather rows along `dim` using u32 indices (embedding lookup is
    /// `index_select(ids, 0)` on the `[vocab, dim]` table).
    pub fn index_select<I: Dim>(&self, ids: &Self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "index_select")?;
        if self.is_dry() {
            let mut odims = self.dims().to_vec();
            odims[d] = ids.elem_count();
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(odims, dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            // Device path: the token-embedding lookup runs through
            // here - the host bounce downloaded the WHOLE table per call
            // (qwen3.5's 248320-row table ≈ 2 GB D2H per decoded token).
            if matches!(ids.dtype(), DType::U32 | DType::I64) {
                let ids_dev = ids.to_device(&self.device())?;
                let ids_storage = match ids_dev.storage().as_ref() {
                    Storage::Cuda { data, .. } => data,
                    _ => unreachable!("just moved to cuda"),
                };
                let dims = self.dims();
                let idx_len = ids.elem_count();
                let inner: usize = dims[d + 1..].iter().product();
                let outer: usize = dims[..d].iter().product();
                let n_out = outer * idx_len * inner;
                let out = crate::tensor::cuda::index_select_storage(
                    dev,
                    data,
                    ids_storage,
                    n_out,
                    inner,
                    idx_len,
                    dims[d],
                )?;
                let mut odims = dims.to_vec();
                odims[d] = idx_len;
                return Self::from_cuda_storage(out, dev.clone(), odims);
            }
            let ids_cpu = ids.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.index_select(&ids_cpu, d));
        }
        let idx = Self::index_values(ids)?;
        let c = self.cpu_storage_ref()?;
        let dims = self.dims();
        let outer: usize = dims[..d].iter().product();
        let inner: usize = dims[d + 1..].iter().product();
        let storage = map_cpu!(c, |v, wrap| {
            let mut out = Vec::with_capacity(outer * idx.len() * inner);
            for o in 0..outer {
                for &i in &idx {
                    if i >= dims[d] {
                        return Err(Error(format!(
                            "index_select: index {i} out of range {}",
                            dims[d]
                        )));
                    }
                    let base = (o * dims[d] + i) * inner;
                    out.extend_from_slice(&v[base..base + inner]);
                }
            }
            wrap(out)
        });
        let mut odims = dims.to_vec();
        odims[d] = idx.len();
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(storage)),
            Shape::from(odims),
        ))
    }

    pub(super) fn unary_f32(&self, f: impl Fn(f32) -> f32 + Sync) -> Result<Self> {
        // Every elementwise unary - silu, tanh, exp, the trigonometry the rotary
        // tables are built from - ends here once the device paths have declined, so
        // one arm covers all of them.
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        // CPU half-precision route: f32 compute + round back (== `half` crate
        // per-op arithmetic; see binary_f32).
        if matches!(self.dtype(), DType::F16 | DType::BF16) && !self.is_on_cuda() {
            let dt = self.dtype();
            return self.to_dtype(DType::F32)?.unary_f32(f)?.to_dtype(dt);
        }
        let a = self.f32_data()?;
        let out: Vec<f32> = if a.len() >= PAR_CPU_MIN {
            // Spin-pool, not rayon: on small models the matmul spin-pool and rayon
            // both keep ~N threads, and rayon's idle workers steal-spin through the
            // matmuls - running elementwise on the same pool keeps one pool active.
            const CH: usize = 1 << 16;
            let mut out = vec![0f32; a.len()];
            crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, CH, &|c, d| {
                let s = &a[c * CH..c * CH + d.len()];
                for (dd, &sv) in d.iter_mut().zip(s) {
                    *dd = f(sv);
                }
            });
            out
        } else {
            a.iter().map(|&x| f(x)).collect()
        };
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn unary_cuda(&self, kernel: &'static str) -> Result<Option<Self>> {
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out = crate::tensor::cuda::unary_f32(
                dev,
                kernel,
                data.as_f32_slice()?,
                self.elem_count(),
            )?;
            return Ok(Some(Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            )?));
        }
        Ok(None)
    }

    pub fn sigmoid(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.sigmoid());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_sigmoid_f32")? {
            return Ok(t);
        }
        self.unary_f32(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn sin(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.sin());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_sin_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::sin)
    }

    pub fn cos(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.cos());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_cos_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::cos)
    }

    /// Packed tensors (offset 0): identity, FREE (qwen35_moe/gptoss rely on
    /// this). Narrow views: materialize the packed copy (once, cached) and
    /// return a tensor over it, so callers get offset-0 data.
    #[track_caller] // NATIVE_BOUNCE attribution: charge the materialization to our caller
    pub fn contiguous(&self) -> Result<Self> {
        if self.spans_whole() {
            return Ok(self.clone());
        }
        Ok(Self::from_packed(
            self.storage().clone(),
            self.shape.clone(),
        ))
    }
}
