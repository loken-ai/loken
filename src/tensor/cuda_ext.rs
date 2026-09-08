//! The CUDA boundary.
//!
//! Substrate-neutral helpers so model/inference files never spell out the
//! substrate's storage/cudarc paths directly for the handful of raw-CUDA
//! things the LLM decode path needs:
//!
//!   - raw `CudaSlice`/`CudaView` access to a tensor's storage (kernel FFI),
//!   - building a `Tensor` back from a kernel-produced `CudaSlice`,
//!   - the quantized weight blob of a `QTensor`,
//!   - stream / event / alt-stream (overlap) handles,
//!   - CUDA-graph capture status + begin/end + capture-arena wrappers
//!     (the cudarc-fork APIs the gptoss/lfm2 graph decode uses).
//!
//! Everything CUDA-typed is gated on the `cuda` feature; `capture_active`
//! also exists as a `false` stub on non-CUDA builds so call sites can drop
//! their own `#[cfg]` forks.

#[cfg(feature = "cuda")]
pub use cuda::*;

/// Non-CUDA builds: graph capture can never be active.
#[cfg(not(feature = "cuda"))]
pub fn capture_active(_device: &crate::tensor::Device) -> bool {
    false
}

/// Non-CUDA builds: no CUDA device type exists - error at runtime, keeping
/// engine call sites cfg-free.
#[cfg(not(feature = "cuda"))]
pub fn new_device_with_stream(
    ordinal: usize,
) -> crate::tensor::Result<crate::tensor::kernel_ffi::NoCudaDevice> {
    let _ = ordinal;
    Err(crate::tensor::Error::msg("CUDA support not compiled in"))
}

/// Non-CUDA builds: no cuBLAS to configure.
#[cfg(not(feature = "cuda"))]
pub fn set_gemm_reduced_precision(_enable: bool) {}

/// Non-CUDA builds: no device registry to drop.
#[cfg(not(feature = "cuda"))]
pub fn clear_device_cache() {}

/// The CUDA backing: the substrate's CudaDevice/Storage over the vendored
/// cudarc fork - the raw handle types (`CudaSlice`/`CudaStream`/...) are exactly
/// what every custom-kernel launcher takes.
#[cfg(feature = "cuda")]
mod cuda {
    use crate::tensor;
    use cudarc::driver::sys::{
        CUgraphInstantiate_flags_enum, CUstreamCaptureMode, CUstreamCaptureStatus,
    };
    use std::sync::Arc;
    use tensor::kernel_ffi::CudaDType;
    use tensor::kernel_ffi::CudaDevice;
    use tensor::{Device, Error, Result, Tensor};

    // Re-export the raw handle types under the boundary's namespace so call sites
    // can name them without spelling out the cudarc path.
    pub use crate::tensor::kernel_ffi::CudaDevice as RawCudaDevice;
    pub use cudarc::driver::{CudaEvent, CudaGraph, CudaSlice, CudaStream, CudaView};

    // Raw launch surface for the FFI kernel-wrapper files ( Wave 2): the
    // launch-config / kernel-arg / device-pointer traits every custom-kernel
    // launcher uses, the NVRTC compiler, and the CudaStorage wrapper for
    // kernel-produced slices.
    pub use crate::tensor::kernel_ffi::CudaStorage;
    pub use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    pub use cudarc::nvrtc::safe::compile_ptx;

    // Graph-introspection types for the gptoss/lfm2/engine CUDA-graph paths:
    // node-type histogram diagnostics + the cheap-update-or-
    // reinstantiate recapture outcome.
    pub use cudarc::driver::sys::CUgraphNodeType;
    pub use cudarc::driver::UpdateOutcome;

    fn msg<E: std::fmt::Debug>(ctx: &str) -> impl FnOnce(E) -> Error + '_ {
        move |e| Error::msg(format!("cuda_ext::{ctx}: {e:?}"))
    }

    // -- device / stream handles -------------------------------------------

    /// The compute stream of `device` - the one every facade op and FFI
    /// kernel launch of this device enqueues on.
    pub fn stream_of(device: &Device) -> Result<Arc<CudaStream>> {
        Ok(device.as_cuda_device()?.native().stream().clone())
    }

    /// The device's cached secondary ("alt") stream, used to overlap work
    /// with the default stream (e.g. the FFN-up GEMV during decode). One alt
    /// stream per device, created lazily and reused - NOT a fresh fork per
    /// call. Synchronisation with the default stream is the caller's job via
    /// [`record_event`]/[`wait_event`].
    pub fn fork_stream(device: &Device) -> Result<Arc<CudaStream>> {
        device.as_cuda_device()?.native().alt_stream()
    }

    /// Block the host until all submitted work on `device` completed.
    pub fn sync_device(device: &Device) -> Result<()> {
        device.synchronize()
    }

    /// Free + total device memory in bytes.
    pub fn mem_get_info(device: &Device) -> Result<(usize, usize)> {
        device
            .as_cuda_device()?
            .native()
            .context()
            .mem_get_info()
            .map_err(msg("mem_get_info"))
    }

    // -- events ------------------------------------------------------------

    /// Create a fresh event (timing disabled) with no work recorded.
    pub fn new_event(device: &Device) -> Result<CudaEvent> {
        device
            .as_cuda_device()?
            .native()
            .context()
            .new_event(None)
            .map_err(msg("new_event"))
    }

    /// Create an event capturing all work currently enqueued on `stream`.
    pub fn record_event(stream: &Arc<CudaStream>) -> Result<CudaEvent> {
        stream.record_event(None).map_err(msg("record_event"))
    }

    /// Make `stream` wait (device-side, host does not block) for the work
    /// recorded in `event`.
    pub fn wait_event(stream: &Arc<CudaStream>, event: &CudaEvent) -> Result<()> {
        stream.wait(event).map_err(msg("wait_event"))
    }

    // -- CUDA-graph capture ------------------------------------------------

    /// True while `device`'s compute stream is capturing a CUDA graph.
    /// Returns `false` for non-CUDA devices and on driver query errors
    /// (matching the unwrap_or(false) pattern of the existing call sites).
    pub fn capture_active(device: &Device) -> bool {
        device
            .as_cuda_device()
            .ok()
            .map(|d| capture_active_on(d.native().stream()))
            .unwrap_or(false)
    }

    /// Stream-level form of [`capture_active`].
    pub fn capture_active_on(stream: &Arc<CudaStream>) -> bool {
        stream
            .capture_status()
            .ok()
            .map(|st| !matches!(st, CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE))
            .unwrap_or(false)
    }

    /// Begin capturing `stream` into a CUDA graph (relaxed mode - the mode
    /// every production capture in loken uses).
    pub fn begin_capture(stream: &Arc<CudaStream>) -> Result<()> {
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(msg("begin_capture"))
    }

    /// End the capture begun by [`begin_capture`] and instantiate the graph
    /// (default instantiation flags). `Ok(None)` mirrors cudarc: the driver
    /// returned a null graph.
    pub fn end_capture(stream: &Arc<CudaStream>) -> Result<Option<CudaGraph>> {
        // Default (empty) instantiate flags. The sys enum has no 0 variant to
        // name, hence the transmute - same as the existing gptoss/lfm2 sites.
        let flags: CUgraphInstantiate_flags_enum = unsafe { std::mem::transmute(0u32) };
        stream.end_capture(flags).map_err(msg("end_capture"))
    }

    /// Route transient allocations of the upcoming capture into a pre-sized
    /// bump arena (cudarc-fork extension). Allocations inside a captured
    /// region must not create MEM_ALLOC nodes - the arena guarantees stable
    /// replay addresses. Pair with [`end_capture_arena`].
    pub fn begin_capture_arena(device: &Device, size_bytes: usize) -> Result<()> {
        device
            .as_cuda_device()?
            .native()
            .context()
            .begin_capture_arena(size_bytes)
            .map_err(msg("begin_capture_arena"))
    }

    /// Re-arm an already-allocated capture arena WITHOUT freeing/resetting it, so
    /// multiple captured graphs share one persistent arena (each at a distinct
    /// offset range, all valid across replays). Returns false if no arena exists.
    pub fn resume_capture_arena(device: &Device) -> Result<bool> {
        Ok(device
            .as_cuda_device()?
            .native()
            .context()
            .resume_capture_arena())
    }

    /// Stop arena routing; returns `(peak_bytes_used, overflow_bytes)`.
    /// `overflow > 0` means the capture spilled past the arena into real
    /// allocations (-> relocatable MEM_ALLOC nodes): grow and recapture.
    pub fn end_capture_arena(device: &Device) -> Result<(usize, usize)> {
        Ok(device
            .as_cuda_device()?
            .native()
            .context()
            .end_capture_arena())
    }

    /// Release the capture arena's backing allocation (e.g. when a graph is
    /// discarded or the model resets at position 0).
    pub fn free_capture_arena(device: &Device) -> Result<()> {
        device
            .as_cuda_device()?
            .native()
            .context()
            .free_capture_arena();
        Ok(())
    }

    /// Pin a dedicated cuBLAS workspace of `bytes` for `device`'s handle so
    /// cuBLAS stops allocating per-call workspaces - required before graph
    /// capture (in-capture cuBLAS allocs invalidate the capture). Leaks the
    /// workspace by design (lives for the process).
    pub fn pin_cublas_workspace(device: &Device, bytes: usize) -> Result<()> {
        use cudarc::cublas::sys::cublasSetWorkspace_v2;
        use cudarc::driver::result::malloc_sync;
        let cd = device.as_cuda_device()?;
        let raw = *cd.native().blas()?.handle();
        let ws = unsafe { malloc_sync(bytes) }.map_err(msg("pin_cublas_workspace/malloc"))?;
        unsafe { cublasSetWorkspace_v2(raw, ws as *mut std::ffi::c_void, bytes) }
            .result()
            .map_err(msg("pin_cublas_workspace/set"))?;
        Ok(())
    }

    // -- global CUDA-backend switches --------------------------------------

    /// Construct a facade CUDA device (engine load paths). The substrate
    /// shares one device/stream per ordinal (same-stream discipline).
    pub fn new_device_with_stream(ordinal: usize) -> Result<CudaDevice> {
        CudaDevice::new_with_stream(ordinal)
    }

    /// Page-locked host allocation (hetero pipeline staging buffers).
    /// Safety contract is the caller's (raw pointer lifecycle).
    pub fn malloc_host(bytes: usize) -> std::result::Result<*mut std::ffi::c_void, String> {
        unsafe { cudarc::driver::result::malloc_host(bytes, 0).map_err(|e| format!("{e}")) }
    }

    /// # Safety
    /// `ptr` must come from [`malloc_host`] and must not have been freed already;
    /// nothing may alias it afterwards.
    pub unsafe fn free_host(ptr: *mut std::ffi::c_void) {
        unsafe {
            let _ = cudarc::driver::result::free_host(ptr);
        }
    }

    /// Reduced-precision GEMM switch, plumbed into the substrate's cuBLAS
    /// wrappers: TF32 for F32, reduced-precision accumulation for F16/BF16
    /// (all default-off) - the model-load throughput switch the engine flips
    /// on CUDA devices.
    pub fn set_gemm_reduced_precision(enable: bool) {
        tensor::cuda::set_gemm_reduced_precision_f32(enable);
        tensor::cuda::set_gemm_reduced_precision_f16(enable);
        tensor::cuda::set_gemm_reduced_precision_bf16(enable);
    }

    /// Drop the per-ordinal device registry. Each entry holds an
    /// `Arc<CudaContext>`; the primary context (and its pool reservation)
    /// cannot be released back to the OS while any clone is alive, so model
    /// unload clears the cache after dropping tensors/workspaces.
    pub fn clear_device_cache() {
        tensor::cuda::clear_device_cache();
    }

    // -- tensor ⇄ raw slice ------------------------------------------------

    /// Borrowed view of a CUDA tensor's storage (substrate tensors are always
    /// contiguous with offset 0). Pins the shared storage via an Arc clone
    /// and carries the device's compute stream so FFI launch sites get both
    /// from one call.
    pub struct TensorCudaSlice<T: CudaDType> {
        storage: Arc<tensor::Storage>,
        len: usize,
        stream: Arc<CudaStream>,
        _t: std::marker::PhantomData<T>,
    }

    impl<T: CudaDType> TensorCudaSlice<T> {
        /// The tensor's elements as a `CudaView<T>` (zero-copy).
        pub fn view(&self) -> Result<CudaView<'_, T>> {
            match self.storage.as_ref() {
                tensor::Storage::Cuda { data, .. } => {
                    Ok(T::as_cuda_slice(data)?.slice(0..self.len))
                }
                _ => Err(Error::msg("cuda_ext: tensor storage is not CUDA")),
            }
        }

        /// Number of elements covered.
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// The compute stream of the tensor's device.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }
    }

    fn slice_of<T: CudaDType>(t: &Tensor, what: &str) -> Result<TensorCudaSlice<T>> {
        let storage = t.storage_arc().clone();
        let stream = match storage.as_ref() {
            tensor::Storage::Cuda { dev, .. } => dev.stream().clone(),
            _ => {
                return Err(Error::msg(format!(
                    "cuda_ext::{what}: tensor is not on CUDA"
                )))
            }
        };
        Ok(TensorCudaSlice {
            storage,
            len: t.elem_count(),
            stream,
            _t: std::marker::PhantomData,
        })
    }

    /// Raw F32 CUDA view of a contiguous tensor (+ its stream). The view
    /// pins the storage - drop it before mutating the tensor.
    pub fn f32_slice_of(t: &Tensor) -> Result<TensorCudaSlice<f32>> {
        slice_of::<f32>(t, "f32_slice_of")
    }

    /// Raw F16 CUDA view of a contiguous tensor (+ its stream).
    pub fn f16_slice_of(t: &Tensor) -> Result<TensorCudaSlice<half::f16>> {
        slice_of::<half::f16>(t, "f16_slice_of")
    }

    /// Wrap a kernel-produced `CudaSlice<f32>` into a facade `Tensor` of
    /// `shape` (contiguous) - the common tail of every kernel launcher.
    pub fn tensor_from_f32_slice<S: Into<crate::tensor::Shape>>(
        slice: CudaSlice<f32>,
        shape: S,
        device: &Device,
    ) -> Result<Tensor> {
        tensor_from_slice_t(slice, shape, device)
    }

    /// F16 variant of [`tensor_from_f32_slice`].
    pub fn tensor_from_f16_slice<S: Into<crate::tensor::Shape>>(
        slice: CudaSlice<half::f16>,
        shape: S,
        device: &Device,
    ) -> Result<Tensor> {
        tensor_from_slice_t(slice, shape, device)
    }

    /// U32 variant of [`tensor_from_f32_slice`] (token ids - e.g. the fused
    /// penalty-argmax sampler's device-resident sampled token).
    pub fn tensor_from_u32_slice<S: Into<crate::tensor::Shape>>(
        slice: CudaSlice<u32>,
        shape: S,
        device: &Device,
    ) -> Result<Tensor> {
        tensor_from_slice_t(slice, shape, device)
    }

    fn tensor_from_slice_t<T: CudaDType, S: Into<crate::tensor::Shape>>(
        slice: CudaSlice<T>,
        shape: S,
        device: &Device,
    ) -> Result<Tensor> {
        let dev = device.as_cuda_device()?;
        let storage = CudaStorage::wrap_cuda_slice(slice, dev.clone());
        tensor_from_cuda_storage(storage, shape)
    }

    /// Wrap an already-built `CudaStorage` (any dtype) into a `Tensor`.
    pub fn tensor_from_cuda_storage<S: Into<crate::tensor::Shape>>(
        storage: CudaStorage,
        shape: S,
    ) -> Result<Tensor> {
        tensor::Tensor::from_storage_arc(storage.storage, shape)
    }

    /// The raw row-padded quantized weight blob of a CUDA `QTensor`, for
    /// launchers that drive the quantized GEMV/GEMM kernels themselves.
    pub fn qtensor_blob(qt: &crate::tensor::quantized::QTensor) -> Result<&CudaSlice<u8>> {
        match qt.storage() {
            crate::tensor::quantized::QStorage::Cuda(s) => Ok(s.weight_cuda_slice()),
            _ => Err(Error::msg("cuda_ext::qtensor_blob: QTensor is not on CUDA")),
        }
    }

    // -- tests ------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::tensor::DType;

        fn try_open_cuda() -> Option<Device> {
            for idx in 0..4 {
                if let Ok(dev) = Device::new_cuda(idx) {
                    if Tensor::zeros_on((1,), DType::F32, &dev).is_ok() {
                        return Some(dev);
                    }
                }
            }
            None
        }

        #[test]
        fn capture_active_false_outside_capture_and_cpu() {
            assert!(!capture_active(&Device::Cpu));
            if let Some(dev) = try_open_cuda() {
                assert!(!capture_active(&dev));
                let stream = stream_of(&dev).unwrap();
                assert!(!capture_active_on(&stream));
            }
        }

        #[test]
        fn f32_roundtrip_slice_and_back() {
            let Some(dev) = try_open_cuda() else {
                eprintln!("skipping: no usable CUDA device");
                return;
            };
            let vals: Vec<f32> = (0..96).map(|i| i as f32 * 0.5 - 3.0).collect();
            let t = Tensor::from_vec(vals.clone(), (4, 24), &dev).unwrap();

            // Borrow the raw view and copy it into a fresh owned slice (as a
            // kernel would), then wrap that back into a tensor.
            let cd = dev.as_cuda_device().unwrap();
            let owned = {
                let s = f32_slice_of(&t).unwrap();
                assert_eq!(s.len(), 96);
                let view = s.view().unwrap();
                let mut owned = cd.alloc_zeros::<f32>(96).unwrap();
                cd.memcpy_dtod(&view, &mut owned).unwrap();
                owned
            };
            let t2 = tensor_from_f32_slice(owned, (4, 24), &dev).unwrap();
            sync_device(&dev).unwrap();
            let back: Vec<f32> = t2.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(back, vals);
        }

        #[test]
        fn events_order_alt_stream_work() {
            let Some(dev) = try_open_cuda() else {
                eprintln!("skipping: no usable CUDA device");
                return;
            };
            let default = stream_of(&dev).unwrap();
            let alt = fork_stream(&dev).unwrap();
            assert!(!Arc::ptr_eq(&default, &alt));
            // default -> event -> alt waits -> alt records -> default waits.
            let e1 = record_event(&default).unwrap();
            wait_event(&alt, &e1).unwrap();
            let e2 = record_event(&alt).unwrap();
            wait_event(&default, &e2).unwrap();
            sync_device(&dev).unwrap();
        }

        #[test]
        fn mem_get_info_sane() {
            let Some(dev) = try_open_cuda() else {
                eprintln!("skipping: no usable CUDA device");
                return;
            };
            let (free, total) = mem_get_info(&dev).unwrap();
            assert!(total > 0 && free <= total);
        }
    }
}
