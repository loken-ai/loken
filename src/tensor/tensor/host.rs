//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    /// Contiguous layout for the kernel-FFI extraction pattern (carries the
    /// view offset, like `storage_and_layout`).
    pub fn layout(&self) -> crate::tensor::kernel_ffi::Layout {
        crate::tensor::kernel_ffi::Layout::contiguous_with_offset(self.shape.clone(), self.offset)
    }

    /// True iff both tensors share one storage allocation at the same offset.
    pub fn shares_storage(&self, other: &Self) -> bool {
        self.storage_ptr_id() == other.storage_ptr_id()
    }

    // -- constructors --
    pub fn from_slice<S: Into<Shape>, T: crate::tensor::kernel_ffi::WithDType>(
        data: &[T],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_storage(T::into_cpu_storage(data.to_vec()), shape)?.to_device(device)
    }

    pub fn full<S: Into<Shape>, T: crate::tensor::kernel_ffi::WithDType>(
        value: T,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        Self::from_storage(T::into_cpu_storage(vec![value; n]), shape)?.to_device(device)
    }

    /// the old-substrate `Tensor::new` for the array-ish inputs the call sites use.
    pub fn new<A: NewArg>(array: A, device: &Device) -> Result<Self> {
        array.into_native_tensor(device)
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        let storage = match dtype {
            DType::F32 => CpuStorage::F32(vec![1.0; n]),
            DType::F16 => CpuStorage::F16(vec![half::f16::ONE; n]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::ONE; n]),
            DType::U8 => CpuStorage::U8(vec![1; n]),
            DType::U32 => CpuStorage::U32(vec![1; n]),
            DType::I16 => CpuStorage::I16(vec![1; n]),
            DType::I32 => CpuStorage::I32(vec![1; n]),
            DType::I64 => CpuStorage::I64(vec![1; n]),
            DType::F64 => CpuStorage::F64(vec![1.0; n]),
        };
        Self::from_storage(storage, shape)?.to_device(device)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros_on(self.shape.clone(), self.dtype(), &self.device())
    }

    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape.clone(), self.dtype(), &self.device())
    }

    /// Uninitialized-allocation constructor (fork-shaped `unsafe fn`). True
    /// uninit on CUDA (caller must overwrite before reading); CPU stays zeroed.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn empty<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = device {
            let shape: Shape = shape.into();
            let data =
                crate::tensor::cuda::CudaStorage::alloc_uninit(dev, dtype, shape.elem_count())?;
            return Self::from_cuda_storage(data, dev.clone(), shape);
        }
        Self::zeros_on(shape, dtype, device)
    }

    /// Normally-distributed random tensor (host RNG).
    pub fn randn<S: Into<Shape>>(mean: f32, std: f32, shape: S, device: &Device) -> Result<Self> {
        let shape: Shape = shape.into();
        let z = Self::zeros_on(shape, DType::F32, &Device::Cpu)?;
        z.randn_like()?.affine(std, mean)?.to_device(device)
    }

    // -- shape ops --
    #[track_caller]
    pub fn t(&self) -> Result<Self> {
        if self.rank() < 2 {
            return Err(Error::msg("t() on rank<2 tensor"));
        }
        self.transpose(self.rank() - 2, self.rank() - 1)
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    pub fn flatten_to<I: Dim>(&self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "flatten_to")?;
        let dims = self.dims();
        let mut odims = vec![dims[..=d].iter().product::<usize>()];
        odims.extend_from_slice(&dims[d + 1..]);
        self.reshape(odims)
    }

    /// Broadcast by prepending `left_shape` dims (the old-substrate helper).
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left: Shape = left_shape.into();
        let mut dims = left.dims().to_vec();
        dims.extend_from_slice(self.dims());
        self.broadcast_as(dims)
    }

    pub fn permute<S: PermuteArg>(&self, perm: S) -> Result<Self> {
        // express as successive transposes (native has no n-d permute);
        // selection-sort the axis order, swapping one pair at a time.
        let perm = perm.to_dims_vec();
        if perm.len() != self.rank() {
            return Err(Error::msg(format!(
                "permute: {} axes for rank {}",
                perm.len(),
                self.rank()
            )));
        }
        // Zero-cost case: a permutation that only moves size-1 axes keeps the
        // packed memory order of the non-1 dims - pure metadata reshape.
        {
            let dims = self.dims();
            let mut prev_non1: Option<usize> = None;
            let mut order_preserved = true;
            for &p in &perm {
                if dims[p] != 1 {
                    if let Some(q) = prev_non1 {
                        if p < q {
                            order_preserved = false;
                            break;
                        }
                    }
                    prev_non1 = Some(p);
                }
            }
            if order_preserved {
                let odims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
                return self.reshape(odims);
            }
        }
        let mut cur: Vec<usize> = (0..self.rank()).collect();
        let mut t = self.clone();
        for out_ax in 0..perm.len() {
            let want = perm[out_ax];
            let at = cur.iter().position(|&c| c == want).unwrap();
            if at != out_ax {
                t = t.transpose(out_ax, at)?;
                cur.swap(out_ax, at);
            }
        }
        Ok(t)
    }

    pub fn repeat<S: Into<Shape>>(&self, reps: S) -> Result<Self> {
        let reps: Shape = reps.into();
        let mut t = self.clone();
        // pad rank as the reference does (repeat may extend leading dims)
        while t.rank() < reps.rank() {
            t = t.unsqueeze(0)?;
        }
        for (ax, &r) in reps.dims().iter().enumerate() {
            if r > 1 {
                let copies: Vec<Tensor> = std::iter::repeat_n(t.clone(), r).collect();
                let refs: Vec<&Tensor> = copies.iter().collect();
                t = Tensor::cat(&refs, ax)?;
            }
        }
        Ok(t)
    }

    pub fn stack<A: AsRef<Tensor>, I: Dim>(tensors: &[A], dim: I) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| Error::msg("stack: empty input"))?;
        // dim may equal rank (append) - resolve against a rank+1 padded shape.
        let d = {
            let s = first.as_ref().shape();
            let padded = Shape::from(
                s.dims()
                    .iter()
                    .copied()
                    .chain(std::iter::once(1))
                    .collect::<Vec<_>>(),
            );
            dim.to_index(&padded, "stack")?
        };
        let unsq: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.as_ref().unsqueeze(d))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = unsq.iter().collect();
        Tensor::cat(&refs, d)
    }

    // -- elementwise / math --
    pub fn neg(&self) -> Result<Self> {
        self.affine(-1.0, 0.0)
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.add(&rhs.neg()?)
    }
    pub fn sqr(&self) -> Result<Self> {
        self.mul(self)
    }
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_add(&rhs.neg()?)
    }

    /// Elementwise max/min (host fallback; equal shapes only).
    pub(super) fn zip_host_f32(
        &self,
        rhs: &Self,
        f: impl Fn(f32, f32) -> f32 + Sync,
    ) -> Result<Self> {
        if self.dims() != rhs.dims() {
            return Err(Error::msg(format!(
                "zip op: shape mismatch {:?} vs {:?}",
                self.dims(),
                rhs.dims()
            )));
        }
        let dt = self.dtype();
        let a = self
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec_f32();
        let b = rhs
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec_f32();
        let v: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
        Self::from_storage(CpuStorage::F32(v), self.shape.clone())?
            .to_dtype(dt)?
            .to_device(&self.device())
    }

    pub fn maximum(&self, rhs: &Self) -> Result<Self> {
        self.zip_host_f32(rhs, f32::max)
    }
    pub fn minimum(&self, rhs: &Self) -> Result<Self> {
        self.zip_host_f32(rhs, f32::min)
    }

    pub fn broadcast_maximum(&self, rhs: &Self) -> Result<Self> {
        let oshape = broadcast_dims(self.dims(), rhs.dims())?;
        self.broadcast_as(oshape.clone())?
            .maximum(&rhs.broadcast_as(oshape)?)
    }
    pub fn broadcast_minimum(&self, rhs: &Self) -> Result<Self> {
        let oshape = broadcast_dims(self.dims(), rhs.dims())?;
        self.broadcast_as(oshape.clone())?
            .minimum(&rhs.broadcast_as(oshape)?)
    }

    /// Natural log (host fallback - no native ln kernel yet; cold paths only).
    pub fn log(&self) -> Result<Self> {
        let dt = self.dtype();
        let cpu_f32 = self.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let v: Vec<f32> = cpu_f32.to_vec_f32().into_iter().map(f32::ln).collect();
        Self::from_storage(CpuStorage::F32(v), self.shape.clone())?
            .to_dtype(dt)?
            .to_device(&self.device())
    }

    /// `self` + rows of `source` accumulated at `indexes` along `dim`
    /// (host fallback - MoE scatter-accumulate load paths, dim 0 only).
    pub fn index_add<I: Dim>(&self, indexes: &Self, source: &Self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "index_add")?;
        if d != 0 {
            return Err(Error::msg("index_add: only dim 0 supported"));
        }
        let idx: Vec<u32> = indexes
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let dt = self.dtype();
        let dims = self.dims().to_vec();
        let row: usize = dims[1..].iter().product();
        let mut acc = self
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let src = source
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (si, &di) in idx.iter().enumerate() {
            let (di, si) = (di as usize * row, si * row);
            for j in 0..row {
                acc[di + j] += src[si + j];
            }
        }
        Self::from_storage(CpuStorage::F32(acc), dims)?
            .to_dtype(dt)?
            .to_device(&self.device())
    }

    // -- module application --
    pub fn apply<M: crate::tensor::Module>(&self, m: &M) -> Result<Self> {
        m.forward(self)
    }

    // -- generic host accessors --
    pub fn to_scalar<T: crate::tensor::kernel_ffi::WithDType>(&self) -> Result<T> {
        if self.elem_count() != 1 {
            return Err(Error::msg(format!(
                "to_scalar on {:?} (need 1 element)",
                self.dims()
            )));
        }
        Ok(self.to_vec_host_generic::<T>()?.remove(0))
    }

    pub fn to_vec1<T: crate::tensor::kernel_ffi::WithDType>(&self) -> Result<Vec<T>> {
        if self.rank() != 1 {
            return Err(Error::msg(format!(
                "to_vec1 on rank-{} tensor",
                self.rank()
            )));
        }
        self.to_vec_host_generic::<T>()
    }

    pub fn to_vec2<T: crate::tensor::kernel_ffi::WithDType>(&self) -> Result<Vec<Vec<T>>> {
        let (a, b) = self.dims2()?;
        let flat = self.to_vec_host_generic::<T>()?;
        Ok((0..a).map(|i| flat[i * b..(i + 1) * b].to_vec()).collect())
    }

    /// Flat host copy of the tensor's elements (downloads when on CUDA).
    pub(super) fn to_vec_host_generic<T: crate::tensor::kernel_ffi::WithDType>(
        &self,
    ) -> Result<Vec<T>> {
        let cpu = self.to_device(&Device::Cpu)?;
        match cpu.storage_arc().as_ref() {
            Storage::Cpu(c) => Ok(T::cpu_storage_as_slice(c)?.to_vec()),
            _ => Err(Error::msg("to_vec_host: download failed")),
        }
    }
}
