//! The native tensor container + first CPU ops (f32).
//!
//! Deliberately minimal: contiguous storage only, ops added in the order the
//! model forwards need them, each parity-tested against the current substrate.
//! CUDA storage and the kernel-launch boundary live beside it.

use super::{CpuStorage, DType, Device, Dim, Error, Result, Shape, Storage};
use std::sync::{Arc, OnceLock};

/// Bulk f32 -> half-precision narrowing.
///
/// Mapping element-by-element through a parallel iterator and collecting routes
/// every value through a producer/consumer, which prevents the compiler from
/// vectorizing the convert: it emits a load/convert/store per element. Handing
/// whole contiguous chunks to the slice converter lets the conversion run a full
/// vector width at a time. The rounding is the same, so the result is
/// bit-identical to the per-element map.
fn narrow_from_f32<T>(src: &[f32]) -> Vec<T>
where
    T: Copy + Send + Sync,
    [T]: half::slice::HalfFloatSliceExt,
{
    use half::slice::HalfFloatSliceExt;
    // The conversion below fully overwrites every element, so skip the zero-fill.
    let mut out: Vec<T> = Vec::with_capacity(src.len());
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(src.len())
    };
    // Chunk large enough to amortize task dispatch, small enough to stay parallel.
    // Spin-pool (not rayon): these f16/bf16 casts run 2x per elementwise op on F16
    // models, so keeping them off rayon matters for small-model prefill.
    const CHUNK: usize = 8192;
    super::quant_cpu::pool_par_chunks_mut_t(&mut out, CHUNK, &|c, o| {
        o.convert_from_f32_slice(&src[c * CHUNK..c * CHUNK + o.len()]);
    });
    out
}

/// Packed tensor OR zero-copy narrow view.
///
/// `offset == 0` ⇒ the tensor IS the whole `storage_raw` allocation (the
/// historical invariant). `offset > 0` ⇒ a contiguous narrow VIEW over
/// `storage_raw` starting at element `offset` (spanning `shape.elem_count()`
/// elements). Views are produced only by [`Tensor::narrow`] when the selected
/// range is contiguous in the packed layout; every data-reading op goes
/// through [`Tensor::storage`], which materializes a packed copy ONCE (shared
/// between clones via the `packed` cell), while the kernel-FFI boundary
/// ([`Tensor::storage_and_layout`]) projects the offset zero-copy through
/// `Layout::start_offset()`.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage_raw: Arc<Storage>,
    shape: Shape,
    /// Element offset into `storage_raw` (0 = whole allocation).
    offset: usize,
    /// Lazily-materialized packed copy for ops that need offset-0 data.
    /// `Arc` so `Clone` stays cheap and materialization is shared.
    packed: Arc<OnceLock<Arc<Storage>>>,
}

/// Dispatch a typed CPU-storage op across every dtype variant: binds the
/// typed slice and the matching variant constructor (`$wrap`), so copy/compare
/// ops are written once and cover U8/U32/I64/halves/floats uniformly.
macro_rules! map_cpu {
    ($s:expr, |$v:ident, $wrap:ident| $body:expr) => {
        match $s {
            CpuStorage::U8($v) => {
                let $wrap = CpuStorage::U8;
                $body
            }
            CpuStorage::U32($v) => {
                let $wrap = CpuStorage::U32;
                $body
            }
            CpuStorage::I16($v) => {
                let $wrap = CpuStorage::I16;
                $body
            }
            CpuStorage::I32($v) => {
                let $wrap = CpuStorage::I32;
                $body
            }
            CpuStorage::I64($v) => {
                let $wrap = CpuStorage::I64;
                $body
            }
            CpuStorage::BF16($v) => {
                let $wrap = CpuStorage::BF16;
                $body
            }
            CpuStorage::F16($v) => {
                let $wrap = CpuStorage::F16;
                $body
            }
            CpuStorage::F32($v) => {
                let $wrap = CpuStorage::F32;
                $body
            }
            CpuStorage::F64($v) => {
                let $wrap = CpuStorage::F64;
                $body
            }
        }
    };
}

/// Same-dtype pair dispatch (copy/select ops over two storages).
macro_rules! map_cpu2 {
    ($op:expr, $a:expr, $b:expr, |$va:ident, $vb:ident, $wrap:ident| $body:expr) => {
        match ($a, $b) {
            (CpuStorage::U8($va), CpuStorage::U8($vb)) => {
                let $wrap = CpuStorage::U8;
                $body
            }
            (CpuStorage::U32($va), CpuStorage::U32($vb)) => {
                let $wrap = CpuStorage::U32;
                $body
            }
            (CpuStorage::I16($va), CpuStorage::I16($vb)) => {
                let $wrap = CpuStorage::I16;
                $body
            }
            (CpuStorage::I32($va), CpuStorage::I32($vb)) => {
                let $wrap = CpuStorage::I32;
                $body
            }
            (CpuStorage::I64($va), CpuStorage::I64($vb)) => {
                let $wrap = CpuStorage::I64;
                $body
            }
            (CpuStorage::BF16($va), CpuStorage::BF16($vb)) => {
                let $wrap = CpuStorage::BF16;
                $body
            }
            (CpuStorage::F16($va), CpuStorage::F16($vb)) => {
                let $wrap = CpuStorage::F16;
                $body
            }
            (CpuStorage::F32($va), CpuStorage::F32($vb)) => {
                let $wrap = CpuStorage::F32;
                $body
            }
            (CpuStorage::F64($va), CpuStorage::F64($vb)) => {
                let $wrap = CpuStorage::F64;
                $body
            }
            _ => return Err(Error(format!("{}: dtype mismatch", $op))),
        }
    };
}

/// View CPU storage as raw bytes (plain-old-data slices; little-endian host  - 
/// matches `cpu_storage_from_bytes` below).
fn cpu_storage_bytes(c: &CpuStorage) -> &[u8] {
    fn b<T>(v: &[T]) -> &[u8] {
        // SAFETY: all storage element types are plain-old-data with no padding.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    match c {
        CpuStorage::U8(v) => v,
        CpuStorage::U32(v) => b(v),
        CpuStorage::I16(v) => b(v),
        CpuStorage::I32(v) => b(v),
        CpuStorage::I64(v) => b(v),
        CpuStorage::BF16(v) => b(v),
        CpuStorage::F16(v) => b(v),
        CpuStorage::F32(v) => b(v),
        CpuStorage::F64(v) => b(v),
    }
}

/// Rebuild typed CPU storage from raw little-endian bytes.
fn cpu_storage_from_bytes(dtype: DType, bytes: &[u8]) -> Result<CpuStorage> {
    let es = dtype.size_in_bytes();
    if bytes.len() % es != 0 {
        return Err(Error(format!(
            "storage_from_bytes: {} bytes not a multiple of {es}",
            bytes.len()
        )));
    }
    Ok(match dtype {
        DType::U8 => CpuStorage::U8(bytes.to_vec()),
        DType::U32 => CpuStorage::U32(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::I16 => CpuStorage::I16(
            bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::I32 => CpuStorage::I32(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::I64 => CpuStorage::I64(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::BF16 => CpuStorage::BF16(
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        DType::F16 => CpuStorage::F16(
            bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        DType::F32 => CpuStorage::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::F64 => CpuStorage::F64(
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
    })
}

impl Tensor {
    /// Wrap a whole (packed, offset-0) storage allocation.
    fn from_packed<S: Into<Shape>>(storage: Arc<Storage>, shape: S) -> Self {
        Self {
            storage_raw: storage,
            shape: shape.into(),
            offset: 0,
            packed: Arc::new(OnceLock::new()),
        }
    }

    /// The result of an op that would allocate `shape` at `dtype`, when this tensor
    /// lives on a device that counts instead of allocating.
    ///
    /// `None` on a real tensor, so an op opens with
    /// `if let Some(t) = self.dry_out(shape, dtype)? { return Ok(t) }` and is
    /// otherwise untouched. The shape passed in is the one the op has ALREADY
    /// computed for its real output - never a second derivation of it, which is the
    /// whole point: a dry run that computed its own shapes would be the parallel
    /// description this exists to remove.
    fn dry_out<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Option<Self> {
        let Storage::Dry(d) = self.storage_raw.as_ref() else {
            return None;
        };
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        Some(Self::from_packed(
            Arc::new(Storage::Dry(super::dry::DryStorage::new(
                d.device().clone(),
                dtype,
                n,
            ))),
            shape,
        ))
    }

    /// Whether this tensor is being counted rather than computed.
    fn is_dry(&self) -> bool {
        matches!(self.storage_raw.as_ref(), Storage::Dry(_))
    }

    /// True iff the tensor covers its raw allocation exactly (offset 0 AND
    /// same element count - a PREFIX view has offset 0 but a SHORTER shape,
    /// and ops that read `storage.len()` instead of `elem_count()` would see
    /// the excess tail).
    fn spans_whole(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let raw_len = match self.storage_raw.as_ref() {
            Storage::Cpu(c) => c.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda { data, .. } => data.len(),
            Storage::Dry(s) => s.len(),
        };
        raw_len == self.shape.elem_count()
    }

    /// The tensor's data as a whole (exact-span) storage. Packed tensors
    /// return their storage untouched; narrow views materialize a packed copy
    /// ONCE (cached in `packed`, shared between clones) and return that. Every
    /// data-reading op routes through here, which keeps all ops correct on
    /// views with zero per-op changes; the kernel-FFI hot path bypasses this
    /// via `storage_and_layout` (zero-copy offset projection).
    ///
    /// Panics only if the device->device materialization copy itself fails
    /// (device lost / OOM - already fatal for the forward pass).
    /// `#[track_caller]` so the NATIVE_BOUNCE profile attributes each
    /// materialization to the op that forced it.
    #[track_caller]
    fn storage(&self) -> &Arc<Storage> {
        if self.spans_whole() {
            return &self.storage_raw;
        }
        if self.packed.get().is_none() {
            // counted with the narrow copies: this is the "materialize once"
            // cost a view pays when a non-FFI op reads it.
            let _b = super::bounce::start("dev_narrow_copy");
            let s = self
                .materialize_packed()
                .expect("narrow-view materialization (device copy) failed");
            let _ = self.packed.set(s); // racing set loses harmlessly
        }
        self.packed.get().expect("just set")
    }

    /// Copy the view's element range `[offset, offset + elem_count)` out of the
    /// raw storage into a fresh packed storage.
    fn materialize_packed(&self) -> Result<Arc<Storage>> {
        let n = self.shape.elem_count();
        // A view spanning the whole allocation from element 0 is already packed:
        // materialising it copies a buffer onto itself, and on CUDA that copy is a kernel
        // launch. Sharing the storage is bit-identical by construction - same bytes, same
        // allocation - and costs an Arc bump instead of a submission.
        if self.offset == 0 && n == self.storage_raw.elem_count() {
            return Ok(Arc::clone(&self.storage_raw));
        }
        match self.storage_raw.as_ref() {
            Storage::Cpu(c) => {
                let s = map_cpu!(c, |v, wrap| wrap(v[self.offset..self.offset + n].to_vec()));
                Ok(Arc::new(Storage::Cpu(s)))
            }
            // A view being made contiguous is a real copy on every device, and it
            // is one of the larger transients a denoise holds - the transposed
            // attention operands alone are tens of megabytes per block. Charged
            // here so it is charged once, at the point every op that needs a packed
            // buffer goes through.
            Storage::Dry(d) => Ok(Arc::new(Storage::Dry(super::dry::DryStorage::new(
                d.device().clone(),
                d.dtype(),
                n,
            )))),
            #[cfg(feature = "cuda")]
            Storage::Cuda { data, dev } => {
                // device-side range copy (timed by the caller, `storage()`).
                let es = self.dtype().size_in_bytes();
                let out = super::cuda::narrow_storage(
                    dev,
                    data,
                    1,
                    (self.offset + n) * es,
                    self.offset * es,
                    n * es,
                )?;
                Ok(Arc::new(Storage::Cuda {
                    data: out,
                    dev: dev.clone(),
                }))
            }
        }
    }

    pub fn from_vec_f32<S: Into<Shape>>(data: Vec<f32>, shape: S) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error(format!(
                "from_vec_f32: {} elements for shape {:?}",
                data.len(),
                shape.dims()
            )));
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(data))),
            shape,
        ))
    }

    /// 3-arg dtype-generic constructor (the `from_vec(data, shape, device)`
    /// call-site shape). Builds a CPU storage from any `WithDType` then places
    /// it on `device`.
    pub fn from_vec<S: Into<Shape>, T: super::kernel_ffi::WithDType>(
        data: Vec<T>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_storage(T::into_cpu_storage(data), shape)?.to_device(device)
    }

    pub fn from_vec_u32<S: Into<Shape>>(data: Vec<u32>, shape: S) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error(format!(
                "from_vec_u32: {} elements for shape {:?}",
                data.len(),
                shape.dims()
            )));
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::U32(data))),
            shape,
        ))
    }

    pub fn from_vec_i64<S: Into<Shape>>(data: Vec<i64>, shape: S) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error(format!(
                "from_vec_i64: {} elements for shape {:?}",
                data.len(),
                shape.dims()
            )));
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::I64(data))),
            shape,
        ))
    }

    /// `[start, end)` with unit step, f32 (the RoPE position constructor).
    pub fn arange(start: f32, end: f32) -> Result<Self> {
        let mut v = Vec::new();
        let mut x = start;
        while x < end {
            v.push(x);
            x += 1.0;
        }
        let n = v.len();
        Self::from_vec_f32(v, n)
    }

    pub fn arange_u32(start: u32, end: u32) -> Result<Self> {
        let v: Vec<u32> = (start..end).collect();
        let n = v.len();
        Self::from_vec_u32(v, n)
    }

    pub fn arange_i64(start: i64, end: i64) -> Result<Self> {
        let v: Vec<i64> = (start..end).collect();
        let n = v.len();
        Self::from_vec_i64(v, n)
    }

    /// Wrap host storage directly (weight loaders build tensors this way).
    pub fn from_storage<S: Into<Shape>>(storage: CpuStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        if storage.len() != shape.elem_count() {
            return Err(Error(format!(
                "from_storage: {} elements for shape {:?}",
                storage.len(),
                shape.dims()
            )));
        }
        Ok(Self::from_packed(Arc::new(Storage::Cpu(storage)), shape))
    }

    /// A tensor of this shape and dtype on a device that counts instead of
    /// allocating. Errors on any other device: a dry tensor holds no values, and
    /// handing one to a real forward would compute on nothing.
    pub fn dry<S: Into<Shape>>(device: &Device, dtype: DType, shape: S) -> Result<Self> {
        let Device::Dry(dev) = device else {
            return Err(Error("Tensor::dry needs a dry device".into()));
        };
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        Ok(Self::from_packed(
            Arc::new(Storage::Dry(super::dry::DryStorage::new(
                dev.clone(),
                dtype,
                n,
            ))),
            shape,
        ))
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let n = shape.elem_count();
        let storage = match dtype {
            DType::F32 => CpuStorage::F32(vec![0.0; n]),
            DType::F16 => CpuStorage::F16(vec![half::f16::ZERO; n]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::ZERO; n]),
            DType::U8 => CpuStorage::U8(vec![0; n]),
            DType::U32 => CpuStorage::U32(vec![0; n]),
            DType::I16 => CpuStorage::I16(vec![0; n]),
            DType::I32 => CpuStorage::I32(vec![0; n]),
            DType::I64 => CpuStorage::I64(vec![0; n]),
            DType::F64 => CpuStorage::F64(vec![0.0; n]),
        };
        Ok(Self::from_packed(Arc::new(Storage::Cpu(storage)), shape))
    }

    /// Zeros allocated directly on `device` (CUDA: async `alloc_zeros` on the
    /// device stream; CPU: the host vec path). `Tensor::zeros` + `to_device`
    /// forced a host alloc + H2D memcpy per kernel-output buffer on the
    /// decode hot path - this constructor keeps it device-resident.
    pub fn zeros_on<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        match device {
            Device::Cpu => Self::zeros(shape, dtype),
            Device::Dry(_) => Self::dry(device, dtype, shape),
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                let shape = shape.into();
                let data = super::cuda::CudaStorage::zeros(dev, dtype, shape.elem_count())?;
                Ok(Self::from_packed(
                    Arc::new(Storage::Cuda {
                        data,
                        dev: dev.clone(),
                    }),
                    shape,
                ))
            }
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
}

/// Numpy broadcast of two dim lists -> output shape (ex-compat helper).
fn broadcast_dims(ld: &[usize], rd: &[usize]) -> Result<Shape> {
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
                return Err(Error::msg(format!(
                    "cannot broadcast {ld:?} with {rd:?} (axis {i}: {a} vs {b})"
                )))
            }
        };
    }
    Ok(Shape::from(odims))
}

// -- compat->native union: trait surface for ex-wrapper `new`/`permute`/`stack` --

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

/// Array-ish inputs for `Tensor::new`.
pub trait NewArg {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor>;
}

impl<T: super::kernel_ffi::WithDType> NewArg for &[T] {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor> {
        let n = self.len();
        Tensor::from_slice(self, n, device)
    }
}

impl<T: super::kernel_ffi::WithDType, const N: usize> NewArg for &[T; N] {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor> {
        Tensor::from_slice(self.as_slice(), N, device)
    }
}

impl<T: super::kernel_ffi::WithDType> NewArg for Vec<T> {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor> {
        let n = self.len();
        Tensor::from_storage(T::into_cpu_storage(self), n)?.to_device(device)
    }
}

impl<T: super::kernel_ffi::WithDType> NewArg for Vec<Vec<T>> {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor> {
        let rows = self.len();
        let cols = self.first().map(|r| r.len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(rows * cols);
        for r in self {
            if r.len() != cols {
                return Err(Error::msg("Tensor::new: ragged rows"));
            }
            flat.extend(r);
        }
        Tensor::from_storage(T::into_cpu_storage(flat), (rows, cols))?.to_device(device)
    }
}

impl<T: super::kernel_ffi::WithDType> NewArg for T {
    fn into_native_tensor(self, device: &Device) -> Result<Tensor> {
        Tensor::from_storage(
            T::into_cpu_storage(vec![self]),
            Shape::from(Vec::<usize>::new()),
        )?
        .to_device(device)
    }
}

/// Axis-list argument for `permute` (tuples / arrays / slices / vecs) - ported
/// from compat's `DimsArg`.
pub trait PermuteArg {
    fn to_dims_vec(self) -> Vec<usize>;
}

impl PermuteArg for Vec<usize> {
    fn to_dims_vec(self) -> Vec<usize> {
        self
    }
}
impl PermuteArg for &[usize] {
    fn to_dims_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
impl<const N: usize> PermuteArg for [usize; N] {
    fn to_dims_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
impl<const N: usize> PermuteArg for &[usize; N] {
    fn to_dims_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

macro_rules! permute_dims_tuple {
    ($($n:ident),+) => {
        #[allow(non_snake_case)]
        impl PermuteArg for ($(permute_dims_tuple!(@usize $n),)+) {
            fn to_dims_vec(self) -> Vec<usize> {
                let ($($n,)+) = self;
                vec![$($n,)+]
            }
        }
    };
    (@usize $n:ident) => { usize };
}

permute_dims_tuple!(A, B);
permute_dims_tuple!(A, B, C);
permute_dims_tuple!(A, B, C, D2);
permute_dims_tuple!(A, B, C, D2, E);
permute_dims_tuple!(A, B, C, D2, E, F);

// -- arithmetic operators ----------------------------------------------------
// The inference layer writes `&a + &b`, `a * b`, `t * 2.0f64` and `res? + &b`,
// so every combination of value, reference and `Result` on either side has to
// exist, and all of them return `Result<Tensor>`.
macro_rules! native_bin_op_impl {
    ($trait:ident, $fn:ident, $method:ident) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$method(self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$method(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$method(self, &rhs)
            }
        }
        impl std::ops::$trait<Result<Tensor>> for Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: Result<Tensor>) -> Result<Tensor> {
                Tensor::$method(&self, &rhs?)
            }
        }
        impl std::ops::$trait<Result<Tensor>> for &Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: Result<Tensor>) -> Result<Tensor> {
                Tensor::$method(self, &rhs?)
            }
        }
    };
}

macro_rules! native_result_lhs_op_impl {
    ($trait:ident, $fn:ident, $method:ident) => {
        impl std::ops::$trait<&Tensor> for Result<Tensor> {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$method(&self?, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Result<Tensor> {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$method(&self?, &rhs)
            }
        }
    };
}

native_result_lhs_op_impl!(Add, add, add);
native_result_lhs_op_impl!(Sub, sub, sub);
native_result_lhs_op_impl!(Mul, mul, mul);
native_result_lhs_op_impl!(Div, div, div);

native_bin_op_impl!(Add, add, add);
native_bin_op_impl!(Sub, sub, sub);
native_bin_op_impl!(Mul, mul, mul);
native_bin_op_impl!(Div, div, div);

macro_rules! native_scalar_op_impl {
    ($trait:ident, $fn:ident, $mul:expr, $add:expr) => {
        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: f64) -> Result<Tensor> {
                self.affine($mul(rhs) as f32, $add(rhs) as f32)
            }
        }
        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;
            fn $fn(self, rhs: f64) -> Result<Tensor> {
                self.affine($mul(rhs) as f32, $add(rhs) as f32)
            }
        }
    };
}

native_scalar_op_impl!(Add, add, |_r| 1.0, |r| r);
native_scalar_op_impl!(Sub, sub, |_r| 1.0, |r: f64| -r);
native_scalar_op_impl!(Mul, mul, |r| r, |_r| 0.0);
native_scalar_op_impl!(Div, div, |r: f64| 1.0 / r, |_r| 0.0);

/// Below this element count, CPU ops stay serial - the rayon fan-out costs
/// more than it saves on decode-sized tensors.
const PAR_CPU_MIN: usize = 1 << 17;

/// NT (dot-product) f32 gemm: `out[m,n] = a[m,k] . b[n,k]ᵀ` with `b` given
/// UNtransposed - both operand rows are contiguous, so the `x @ wᵀ` linear
/// pattern needs no transpose materialization. Parallelized over output
/// columns (decode has m=1, so row-parallelism would serialize).
pub(super) fn gemm_nt_f32_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    let dot = |x: &[f32], y: &[f32]| -> f32 {
        // 4 independent accumulators so the loop autovectorizes
        let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
        let c4 = k / 4 * 4;
        let mut i = 0;
        while i < c4 {
            s0 += x[i] * y[i];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            i += 4;
        }
        while i < k {
            s0 += x[i] * y[i];
            i += 1;
        }
        (s0 + s1) + (s2 + s3)
    };
    let par = m * n * k >= PAR_CPU_MIN;
    if !par {
        for r in 0..m {
            let ar = &a[r * k..][..k];
            let orow = &mut out[r * n..][..n];
            for (j, o) in orow.iter_mut().enumerate() {
                *o = dot(ar, &b[j * k..][..k]);
            }
        }
        return;
    }
    // Column-sharded on the persistent GEMV pool (one short region per call;
    // a parked work-stealing pool pays a wake cascade on each).
    let pool = super::quant_cpu::gemv_pool::pool();
    let (chunk_cols, n_chunks) = super::quant_cpu::gemv_grid(m, n, pool.threads);
    let chunks_per_row = n.div_ceil(chunk_cols);
    struct MutPtr(*mut f32);
    unsafe impl Send for MutPtr {}
    unsafe impl Sync for MutPtr {}
    let out_ptr = MutPtr(out.as_mut_ptr());
    pool.run(n_chunks, &|chunk| {
        let out_ptr = &out_ptr;
        let r = chunk / chunks_per_row;
        let col0 = (chunk % chunks_per_row) * chunk_cols;
        let col1 = (col0 + chunk_cols).min(n);
        let ar = &a[r * k..][..k];
        // SAFETY: chunks address disjoint [r, col0..col1] ranges of `out`.
        let orow =
            unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add(r * n + col0), col1 - col0) };
        for (jj, o) in orow.iter_mut().enumerate() {
            *o = dot(ar, &b[(col0 + jj) * k..][..k]);
        }
    });
}

/// Blocked, rayon-parallel row-major f32 gemm: `out[m,n] = a[m,k] . b[k,n]`
/// (`out` fully overwritten). Threads own disjoint column stripes through a
/// local tile, and the k loop is blocked so each streamed B stripe stays
/// cache-resident - the naive row-rescan pattern re-reads B from DRAM
/// `m` times and collapses on megapixel conv/attention shapes.
pub(super) fn gemm_f32_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    const NT: usize = 512; // column-stripe width
    const KB: usize = 256; // k block: B k-tile = KB*NT*4B = 512 KB, L2-resident

    let compute_tile = |j0: usize| -> (usize, Vec<f32>) {
        let nt = NT.min(n - j0);
        let mut tile = vec![0f32; m * nt];
        let mut kb = 0;
        while kb < k {
            let kbl = KB.min(k - kb);
            // 4-row register blocking: each streamed B row feeds 4 output
            // rows, quartering the cache traffic per FMA. (m-blocking was
            // tried on top and measured SLOWER - it trades the L2-resident
            // B k-tile for accumulator locality and loses.)
            let mut i = 0;
            while i + 4 <= m {
                let (t0, rest) = tile[i * nt..].split_at_mut(nt);
                let (t1, rest) = rest.split_at_mut(nt);
                let (t2, rest) = rest.split_at_mut(nt);
                let t3 = &mut rest[..nt];
                for kk in 0..kbl {
                    let a0 = a[i * k + kb + kk];
                    let a1 = a[(i + 1) * k + kb + kk];
                    let a2 = a[(i + 2) * k + kb + kk];
                    let a3 = a[(i + 3) * k + kb + kk];
                    let brow = &b[(kb + kk) * n + j0..][..nt];
                    for (j, &bv) in brow.iter().enumerate() {
                        t0[j] += a0 * bv;
                        t1[j] += a1 * bv;
                        t2[j] += a2 * bv;
                        t3[j] += a3 * bv;
                    }
                }
                i += 4;
            }
            while i < m {
                let orow = &mut tile[i * nt..][..nt];
                for kk in 0..kbl {
                    let av = a[i * k + kb + kk];
                    if av == 0.0 {
                        continue;
                    }
                    let brow = &b[(kb + kk) * n + j0..][..nt];
                    for (o, &bv) in orow.iter_mut().zip(brow) {
                        *o += av * bv;
                    }
                }
                i += 1;
            }
            kb += kbl;
        }
        (j0, tile)
    };

    let tiles: Vec<(usize, Vec<f32>)> = if m * k * n >= (1 << 20) && n > NT {
        use rayon::prelude::*;
        (0..n.div_ceil(NT))
            .into_par_iter()
            .map(|t| compute_tile(t * NT))
            .collect()
    } else {
        (0..n.div_ceil(NT)).map(|t| compute_tile(t * NT)).collect()
    };
    for (j0, tile) in tiles {
        let nt = tile.len() / m.max(1);
        for i in 0..m {
            out[i * n + j0..][..nt].copy_from_slice(&tile[i * nt..][..nt]);
        }
    }
}

/// `gemm_f32_cpu` with an F16 `b` operand streamed through a per-(stripe,
/// k-block) L2-resident f32 scratch. Same tile geometry and accumulation
/// order as `gemm_f32_cpu`, and f16->f32 conversion is exact, so the result
/// is bit-identical to upcasting `b` wholesale first - while reading the
/// (large, per-call) rhs at 2 B/elem from memory instead of materializing a
/// 4 B/elem f32 copy per call. Each `b` element is converted exactly once.
pub(super) fn gemm_f16w_f32_cpu(
    a: &[f32],
    b: &[half::f16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    use half::slice::HalfFloatSliceExt;
    const NT: usize = 512; // column-stripe width (matches gemm_f32_cpu)
    const KB: usize = 256; // k block: f32 b k-tile = KB*NT*4B = 512 KB, L2-resident

    let compute_tile = |j0: usize| -> (usize, Vec<f32>) {
        let nt = NT.min(n - j0);
        let mut tile = vec![0f32; m * nt];
        let mut bblk = vec![0f32; KB * nt];
        let mut kb = 0;
        while kb < k {
            let kbl = KB.min(k - kb);
            // convert this b k-tile once (exact, so accumulation below sees
            // the very same f32 values the upcast-first route fed the gemm)
            for kk in 0..kbl {
                b[(kb + kk) * n + j0..][..nt].convert_to_f32_slice(&mut bblk[kk * nt..][..nt]);
            }
            let mut i = 0;
            while i + 4 <= m {
                let (t0, rest) = tile[i * nt..].split_at_mut(nt);
                let (t1, rest) = rest.split_at_mut(nt);
                let (t2, rest) = rest.split_at_mut(nt);
                let t3 = &mut rest[..nt];
                for kk in 0..kbl {
                    let a0 = a[i * k + kb + kk];
                    let a1 = a[(i + 1) * k + kb + kk];
                    let a2 = a[(i + 2) * k + kb + kk];
                    let a3 = a[(i + 3) * k + kb + kk];
                    let brow = &bblk[kk * nt..][..nt];
                    for (j, &bv) in brow.iter().enumerate() {
                        t0[j] += a0 * bv;
                        t1[j] += a1 * bv;
                        t2[j] += a2 * bv;
                        t3[j] += a3 * bv;
                    }
                }
                i += 4;
            }
            while i < m {
                let orow = &mut tile[i * nt..][..nt];
                for kk in 0..kbl {
                    let av = a[i * k + kb + kk];
                    if av == 0.0 {
                        continue;
                    }
                    let brow = &bblk[kk * nt..][..nt];
                    for (o, &bv) in orow.iter_mut().zip(brow) {
                        *o += av * bv;
                    }
                }
                i += 1;
            }
            kb += kbl;
        }
        (j0, tile)
    };

    let tiles: Vec<(usize, Vec<f32>)> = if m * k * n >= (1 << 20) && n > NT {
        use rayon::prelude::*;
        (0..n.div_ceil(NT))
            .into_par_iter()
            .map(|t| compute_tile(t * NT))
            .collect()
    } else {
        (0..n.div_ceil(NT)).map(|t| compute_tile(t * NT)).collect()
    };
    for (j0, tile) in tiles {
        let nt = tile.len() / m.max(1);
        for i in 0..m {
            out[i * n + j0..][..nt].copy_from_slice(&tile[i * nt..][..nt]);
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod conv_transpose_rewrite_tests {
    use super::*;

    /// Deterministic data, so a failure is reproducible and a pass is not luck.
    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((t >> 8) as f32 / 65536.0).sin()
            })
            .collect()
    }

    /// The rewrite must agree with the scalar scatter it replaces, everywhere.
    ///
    /// This is the whole safety of the change: the device path is a DIFFERENT algorithm
    /// - dilate, flip, convolve - and a transposed convolution is easy to get subtly
    /// wrong in a way that still produces a plausible picture. Reading the stored weight
    /// as [out, in] instead of [in, out], or forgetting to reverse the kernel, both give
    /// correctly-shaped output and a wrong result. Run on the HOST, so it needs no GPU
    /// and covers the arithmetic rather than the dispatch.
    #[test]
    fn the_device_rewrite_matches_the_scalar_scatter() {
        // (b, ci, co, h, w, k, stride, padding) - including the two the face-swap
        // generator actually uses: stride 2 with padding 1, and stride 2 with padding 0.
        let cases = [
            (
                1usize, 2usize, 3usize, 5usize, 4usize, 3usize, 2usize, 1usize,
            ),
            (1, 3, 2, 4, 4, 3, 2, 0),
            (2, 2, 2, 3, 5, 4, 2, 1),
            (1, 1, 1, 6, 6, 3, 1, 1),
            (1, 2, 2, 4, 4, 2, 2, 0),
        ];
        for (b, ci, co, h, w, k, stride, padding) in cases {
            let x = Tensor::from_vec_f32(ramp(b * ci * h * w, 1), vec![b, ci, h, w]).unwrap();
            let we = Tensor::from_vec_f32(ramp(ci * co * k * k, 7), vec![ci, co, k, k]).unwrap();
            let reference = x.conv_transpose2d(&we, None, stride, padding).unwrap();
            let rewritten = x
                .conv_transpose2d_via_conv2d(&we, stride, padding)
                .unwrap()
                .expect("this shape is covered");
            assert_eq!(
                reference.shape().dims(),
                rewritten.shape().dims(),
                "shape differs for {:?}",
                (b, ci, co, h, w, k, stride, padding)
            );
            let a = reference.to_vec_f32();
            let c = rewritten.to_vec_f32();
            let worst = a
                .iter()
                .zip(&c)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            assert!(
                worst / scale < 1e-5,
                "{:?}: worst absolute difference {worst} against a peak of {scale}",
                (b, ci, co, h, w, k, stride, padding)
            );
        }
    }

    /// A shape the rewrite does not cover declines, so the caller keeps the scalar path
    /// instead of receiving something subtly different.
    #[test]
    fn an_uncovered_shape_declines_rather_than_guessing() {
        let x = Tensor::from_vec_f32(ramp(1 * 2 * 4 * 4, 1), vec![1, 2, 4, 4]).unwrap();
        // Padding beyond the kernel's reach: the identity has no non-negative padding.
        let we = Tensor::from_vec_f32(ramp(2 * 2 * 2 * 2, 3), vec![2, 2, 2, 2]).unwrap();
        assert!(x.conv_transpose2d_via_conv2d(&we, 2, 3).unwrap().is_none());
        // A non-square kernel: conv2d takes one padding for both axes.
        let ns = Tensor::from_vec_f32(ramp(2 * 2 * 3 * 2, 5), vec![2, 2, 3, 2]).unwrap();
        assert!(x.conv_transpose2d_via_conv2d(&ns, 2, 0).unwrap().is_none());
    }
}

#[cfg(test)]
mod max_pool_rewrite_tests {
    use super::*;

    /// The folded reduction must agree with the scalar loop it replaces.
    ///
    /// Run on the HOST by folding explicitly, so it checks the arithmetic rather than the
    /// dispatch: a reshape that groups the wrong axes still returns the right SHAPE and a
    /// silently wrong pooling, which downstream shows up as a slightly soft picture and
    /// nothing else.
    #[test]
    fn folding_the_blocks_matches_the_scalar_pool() {
        for (b, c, h, w, k) in [
            (1usize, 2usize, 8usize, 8usize, 2usize),
            (2, 3, 9, 7, 2), // a remainder on both axes
            (1, 1, 12, 12, 3),
            (1, 4, 6, 10, 2),
        ] {
            let n = b * c * h * w;
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    let t = (i as u32).wrapping_mul(2246822519);
                    ((t >> 9) as f32 / 32768.0).sin()
                })
                .collect();
            let x = Tensor::from_vec_f32(data, vec![b, c, h, w]).unwrap();
            let scalar = x.max_pool2d(k).unwrap(); // host path: this build has no device
            let (oh, ow) = (h / k, w / k);
            let folded = x
                .narrow(2, 0, oh * k)
                .unwrap()
                .narrow(3, 0, ow * k)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape(vec![b, c, oh, k, ow, k])
                .unwrap()
                .max_keepdim(5)
                .unwrap()
                .max_keepdim(3)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape(vec![b, c, oh, ow])
                .unwrap();
            assert_eq!(scalar.shape().dims(), folded.shape().dims());
            let (a, d) = (scalar.to_vec_f32(), folded.to_vec_f32());
            let worst = a
                .iter()
                .zip(&d)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            assert!(worst == 0.0, "({b},{c},{h},{w},k={k}): differs by {worst}");
        }
    }
}

#[cfg(test)]
mod host_bounce_gate;

mod conv;
mod elementwise;
mod host;
mod index;
mod norm;
mod shape;
