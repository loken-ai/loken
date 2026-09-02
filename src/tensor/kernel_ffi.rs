//! The kernel boundary: everything a launch site reads off a tensor, in one module.
//!
//! A launch needs three things and no more - where the buffer is, how to walk it, and which
//! device and function to hand it to. That is [`StorageHandle`] lending a [`StorageView`],
//! a [`Layout`], and [`CudaDevice`] / [`CudaFunc`]. Around four hundred sites take the first
//! two from `Tensor::storage_and_layout`.
//!
//! The narrow surface is deliberate. A launch site that could see the whole tensor could also
//! see a strided view, and the kernels here index raw device memory: the projection admits
//! only packed buffers, at an offset.

use super::Shape;

/// Contiguous layout projection. Native tensors are packed (contiguous strides); a
/// zero-copy narrow view projects the SAME contiguous strides at a non-zero `start_offset`.
pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
    offset: usize,
}

impl Layout {
    /// The strides are not an argument because they are not a choice: a packed buffer's
    /// strides follow from its shape, and the shape is read here before it is moved in.
    pub(crate) fn contiguous_with_offset(shape: Shape, offset: usize) -> Self {
        Self {
            stride: shape.stride_contiguous(),
            shape,
            offset,
        }
    }

    pub fn start_offset(&self) -> usize {
        self.offset
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Borrowed strides (call sites bind the layout first: `let l = t.layout(); l.stride()`).
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// The half-open range this view covers in the buffer behind it.
    ///
    /// Total BY CONSTRUCTION, not optional: the projection only ever describes packed
    /// tensors, so there is always a range. A `Option` here had one caller, whose `None`
    /// arm could not be reached, and an `is_contiguous()` that returned a constant `true`
    /// had none at all.
    pub fn contiguous_offsets(&self) -> (usize, usize) {
        (self.offset, self.offset + self.shape.elem_count())
    }
}

use super::{CpuStorage, DType, Error, Result};

/// Host dtypes: `Vec<T>` ↔ `CpuStorage` projection (the typed CPU-slice extraction the
/// inference layer reads for CPU kernels).
pub trait WithDType: Sized + Clone + 'static {
    const DTYPE: DType;
    fn into_cpu_storage(v: Vec<Self>) -> CpuStorage;
    fn cpu_storage_as_slice(c: &CpuStorage) -> Result<&[Self]>;
    fn from_f64(v: f64) -> Self;
}

macro_rules! with_dtype {
    ($ty:ty, $dt:ident, $from_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dt;
            fn into_cpu_storage(v: Vec<Self>) -> CpuStorage {
                CpuStorage::$dt(v)
            }
            fn cpu_storage_as_slice(c: &CpuStorage) -> Result<&[Self]> {
                match c {
                    CpuStorage::$dt(v) => Ok(v),
                    other => Err(Error::msg(format!(
                        "expected {} storage, got {}",
                        stringify!($dt),
                        other.dtype()
                    ))),
                }
            }
            fn from_f64(v: f64) -> Self {
                #[allow(clippy::redundant_closure_call)]
                ($from_f64)(v)
            }
        }
    };
}

with_dtype!(u8, U8, |v: f64| v as u8);
with_dtype!(u32, U32, |v: f64| v as u32);
with_dtype!(i16, I16, |v: f64| v as i16);
with_dtype!(i32, I32, |v: f64| v as i32);
with_dtype!(i64, I64, |v: f64| v as i64);
with_dtype!(half::f16, F16, |v: f64| half::f16::from_f64(v));
with_dtype!(half::bf16, BF16, |v: f64| half::bf16::from_f64(v));
with_dtype!(f32, F32, |v: f64| v as f32);
with_dtype!(f64, F64, |v: f64| v);

/// Device dtypes the CUDA-slice projection covers (kernel-FFI extraction).
#[cfg(feature = "cuda")]
pub trait CudaDType: cudarc::driver::DeviceRepr + Sized {
    fn as_cuda_slice(cs: &super::cuda::CudaStorage) -> Result<&cudarc::driver::CudaSlice<Self>>;
    fn wrap_cuda_slice(slice: cudarc::driver::CudaSlice<Self>) -> super::cuda::CudaStorage;
}

#[cfg(feature = "cuda")]
macro_rules! cuda_dtype {
    ($ty:ty, $variant:ident, $as:ident) => {
        impl CudaDType for $ty {
            fn as_cuda_slice(
                cs: &super::cuda::CudaStorage,
            ) -> Result<&cudarc::driver::CudaSlice<Self>> {
                cs.$as()
            }
            fn wrap_cuda_slice(slice: cudarc::driver::CudaSlice<Self>) -> super::cuda::CudaStorage {
                super::cuda::CudaStorage::$variant(slice)
            }
        }
    };
}

#[cfg(feature = "cuda")]
cuda_dtype!(f32, F32, as_f32_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(half::f16, F16, as_f16_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(half::bf16, BF16, as_bf16_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(u32, U32, as_u32_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(u8, U8, as_u8_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(i32, I32, as_i32_slice);
#[cfg(feature = "cuda")]
cuda_dtype!(i64, I64, as_i64_slice);

// -- storage projection: (StorageHandle, Layout) for kernel launches --
use std::sync::Arc;

/// CUDA storage view: legacy-shaped `as_cuda_slice::<T>()` over the tensor's
/// shared native storage (kept alive by the Arc clone inside).
#[cfg(feature = "cuda")]
pub struct CudaStorage {
    pub(crate) storage: Arc<super::Storage>,
    /// Public field like the reference CudaStorage (one opencl-path site reads
    /// `cs.device` directly).
    pub device: CudaDevice,
}

#[cfg(feature = "cuda")]
impl CudaStorage {
    pub fn as_cuda_slice<T: CudaDType>(&self) -> Result<&cudarc::driver::CudaSlice<T>> {
        match self.storage.as_ref() {
            super::Storage::Cuda { data, .. } => T::as_cuda_slice(data),
            _ => Err(Error::msg("as_cuda_slice on non-CUDA storage")),
        }
    }

    pub fn dtype(&self) -> Result<DType> {
        match self.storage.as_ref() {
            super::Storage::Cuda { data, .. } => Ok(data.dtype()),
            _ => Err(Error::msg("dtype on non-CUDA storage")),
        }
    }

    pub fn wrap_cuda_slice<T: CudaDType>(
        slice: cudarc::driver::CudaSlice<T>,
        dev: CudaDevice,
    ) -> Self {
        let data = T::wrap_cuda_slice(slice);
        Self {
            storage: Arc::new(super::Storage::Cuda {
                data,
                dev: dev.0.clone(),
            }),
            device: dev,
        }
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

/// CPU storage view (typed slice projection).
pub struct CpuStorageRef {
    pub(crate) storage: Arc<super::Storage>,
}

impl CpuStorageRef {
    pub fn as_slice<T: WithDType>(&self) -> Result<&[T]> {
        match self.storage.as_ref() {
            super::Storage::Cpu(c) => T::cpu_storage_as_slice(c),
            _ => Err(Error::msg("as_slice on non-CPU storage")),
        }
    }
}

/// What a kernel-launch site needs to see of a tensor's storage: the device
/// slice, or the host bytes. A projection of [`Device`'s](super::Device) own
/// `Storage`, not a second copy of it - that one owns the allocation and
/// carries the dry-run variant, neither of which a launch site can use.
pub enum StorageView {
    Cpu(CpuStorageRef),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

/// Owns a projection and lends it: `match &*handle { StorageView::Cuda(c) => ... }`.
/// The view borrows nothing, so the handle exists to give the call sites one
/// binding whose lifetime covers the launch.
pub struct StorageHandle(pub(crate) StorageView);

impl std::ops::Deref for StorageHandle {
    type Target = StorageView;
    fn deref(&self) -> &StorageView {
        &self.0
    }
}

// ------------------------------------------------------------
// CUDA device + function handles (ex-`native/cuda_facade.rs`): the
// kernel-launch sites read these together with the storage projection above.
// The `tensor::kernel_ffi::...` paths keep resolving via the shim in
// `native/mod.rs`.
// ------------------------------------------------------------
use crate::tensor;

/// legacy-shaped CUDA device handle: a cheap-clone wrapper over the shared
/// native per-ordinal device (context + default stream + kernel modules).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaDevice(pub(crate) Arc<tensor::cuda::CudaDevice>);

/// Non-CUDA builds: a stub so `Device::Cuda` still exists (the type is kept so the
/// variant with a stub device too); constructors error at runtime.
#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct CudaDevice;

#[cfg(not(feature = "cuda"))]
impl CudaDevice {
    pub fn new(_ordinal: usize) -> Result<Self> {
        Err(Error::msg("CUDA support not compiled in"))
    }

    pub fn new_with_stream(_ordinal: usize) -> Result<Self> {
        Err(Error::msg("CUDA support not compiled in"))
    }

    pub fn ordinal(&self) -> usize {
        0
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    pub fn same_device(&self, _other: &Self) -> bool {
        true
    }

    /// Substrate `Device` for this handle. The stub can never be constructed
    /// (both constructors error), so this is unreachable at runtime; it exists
    /// so device-selection call sites compile without `#[cfg]` forks.
    pub fn native_device(&self) -> tensor::Device {
        tensor::Device::Cpu
    }
}

/// Alias for the cpu-build cuda_ext stub's return type.
pub type NoCudaDevice = CudaDevice;

#[cfg(feature = "cuda")]
impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        Ok(Self(tensor::cuda::CudaDevice::get(ordinal)?))
    }

    /// One device - context and default stream - per ordinal, shared by every caller, so all
    /// work stays on one stream.
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    pub fn ordinal(&self) -> usize {
        self.0.ordinal()
    }

    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.0.stream().clone()
    }

    /// Substrate `Device` owning this handle (cheap: clones the inner Arc).
    pub fn native_device(&self) -> tensor::Device {
        tensor::Device::Cuda(self.0.clone())
    }

    /// See [`tensor::cuda::CudaDevice::has_ampere_tensor_cores`]. Repeated here because the
    /// kernel call sites hold this handle, and a capability question they cannot ask is a
    /// capability they will assume.
    pub fn has_ampere_tensor_cores(&self) -> bool {
        self.native().has_ampere_tensor_cores()
    }

    pub fn native(&self) -> &Arc<tensor::cuda::CudaDevice> {
        &self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }

    pub fn same_device(&self, other: &Self) -> bool {
        self.0.ordinal() == other.0.ordinal()
    }

    /// Zero-filled typed device allocation (substrate CudaDevice surface used by
    /// kernel-launch helpers and tests).
    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.0
            .stream()
            .alloc_zeros::<T>(len)
            .map_err(|e| Error::msg(format!("alloc_zeros: {e}")))
    }

    pub fn memcpy_dtod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.0
            .stream()
            .memcpy_dtod(src, dst)
            .map_err(|e| Error::msg(format!("memcpy_dtod: {e}")))
    }

    pub fn memcpy_stod<T: cudarc::driver::DeviceRepr + Clone>(
        &self,
        src: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.0
            .stream()
            .clone_htod(src)
            .map_err(|e| Error::msg(format!("memcpy_stod: {e}")))
    }

    pub fn memcpy_dtov<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &cudarc::driver::CudaSlice<T>,
    ) -> Result<Vec<T>> {
        self.0
            .stream()
            .clone_dtoh(src)
            .map_err(|e| Error::msg(format!("memcpy_dtov: {e}")))
    }

    /// Uninitialized typed device allocation (fork-shaped; the unsafety
    /// contract is the caller's: every element must be written before read).
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        unsafe { self.0.stream().alloc::<T>(len) }.map_err(|e| Error::msg(format!("alloc: {e}")))
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.0
            .stream()
            .memcpy_htod(src, dst)
            .map_err(|e| Error::msg(format!("memcpy_htod: {e}")))
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.0
            .stream()
            .clone_dtoh(src)
            .map_err(|e| Error::msg(format!("clone_dtoh: {e}")))
    }

    /// SM count (adaptive grid sizing in quantize/GEMV launchers).
    pub fn multiprocessor_count(&self) -> usize {
        // A device that will not report its geometry still has at least one multiprocessor,
        // and a grid sized for one is slow rather than wrong.
        self.0.mmq_device_info().map_or(1, |i| i.nsm as usize)
    }

    /// fork-compat: load `fn_name` from an NVRTC-compiled module cached by
    /// `module_name` (PTX compiled once per device).
    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        // The module is compiled on the first call for this device and cached under
        // `module_name`; every later call for the same PTX only looks the function up.
        let func = self
            .0
            .custom_fn(module_name, ptx, fn_name)
            .map_err(|e| Error::msg(e.0))?;
        Ok(CudaFunc {
            func,
            stream: self.0.stream().clone(),
        })
    }

    /// The device's cached secondary stream (frozen overlap sites call this
    /// directly on the device).
    pub fn alt_cuda_stream(&self) -> Result<Arc<cudarc::driver::CudaStream>> {
        self.0.alt_stream().map_err(|e| Error::msg(e.0))
    }

    /// fork-compat: disable cudarc's event tracking on the context (graph
    /// capture). Same vendored cudarc -> same semantics.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn disable_event_tracking(&self) {
        unsafe { self.0.context().disable_event_tracking() }
    }
}

/// fork-shaped function handle: a `CudaFunction` + the device's stream
/// (derefs to the function for `stream.launch_builder(&func)` call sites).
#[cfg(feature = "cuda")]
pub struct CudaFunc {
    pub(crate) func: cudarc::driver::CudaFunction,
    pub(crate) stream: Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl std::ops::Deref for CudaFunc {
    type Target = cudarc::driver::CudaFunction;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

#[cfg(feature = "cuda")]
impl CudaFunc {
    pub fn cuda_function(&self) -> &cudarc::driver::CudaFunction {
        &self.func
    }

    pub fn stream(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.stream
    }

    /// Launch builder on the device's compute stream (fork-shaped).
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}
