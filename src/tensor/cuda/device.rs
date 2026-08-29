//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Emergency VRAM reclaim for one device, called between an allocation OOM
/// and its single retry: drop the static quantized-GEMM workspaces (held
/// CudaSlices pin pool memory), drain the stream so pending frees land in
/// the pool, then trim the default mempool (saving/restoring the release
/// threshold so graph-capture mode's u64::MAX setting survives).
///
/// LOCKING INVARIANT: this runs on the OOM'd thread ITSELF, potentially
/// inside a forward that still holds a workspace mutex (mmq_forward keeps
/// its guard across kernel setup). Every lock this function - or anything
/// it calls - acquires MUST be try_lock-with-skip, never a blocking lock:
/// a blocking acquire of a mutex the caller holds is a self-deadlock that
/// freezes the entire server (an incident: a Flux OOM inside MMQ
/// wedged the process for two hours). Skipping a contended workspace is
/// also semantically required - the holder is using it.
pub(super) fn emergency_reclaim(dev: &CudaDevice) {
    crate::tensor::quantized::fast_mmvq::release_workspaces();
    crate::tensor::quantized::release_mmq_workspaces();
    let _ = dev.stream().synchronize();
    unsafe {
        use cudarc::driver::sys::{
            cuDeviceGetDefaultMemPool, cuMemPoolGetAttribute, cuMemPoolSetAttribute,
            cuMemPoolTrimTo, cudaError_enum, CUmemPool_attribute, CUmemoryPool,
        };
        let ordinal = dev.ordinal() as i32;
        let mut pool: CUmemoryPool = std::ptr::null_mut();
        if cuDeviceGetDefaultMemPool(&mut pool, ordinal) != cudaError_enum::CUDA_SUCCESS
            || pool.is_null()
        {
            return;
        }
        let mut threshold: u64 = 0;
        let _ = cuMemPoolGetAttribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &mut threshold as *mut u64 as *mut std::ffi::c_void,
        );
        let zero: u64 = 0;
        let _ = cuMemPoolSetAttribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &zero as *const u64 as *mut std::ffi::c_void,
        );
        let _ = cuMemPoolTrimTo(pool, 0);
        let _ = cuMemPoolSetAttribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *mut std::ffi::c_void,
        );
    }
}

/// Whether the device's stream is inside a CUDA-graph capture (reclaim's
/// synchronize would abort the capture - skip the retry path then).
pub(super) fn stream_capturing(dev: &CudaDevice) -> bool {
    unsafe {
        use cudarc::driver::sys::{cuStreamIsCapturing, CUstreamCaptureStatus};
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let _ = cuStreamIsCapturing(dev.stream().cu_stream(), &mut status);
        status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    }
}

/// Run a device allocation with one reclaim-and-retry on OOM. Every
/// storage-level allocation funnels through this so a transient landing on
/// a pool full of freed-but-retained memory degrades to a warn + retry
/// instead of a hard failure.
pub(crate) fn with_oom_retry<T>(
    dev: &CudaDevice,
    ctx: &str,
    mut f: impl FnMut() -> std::result::Result<T, cudarc::driver::DriverError>,
) -> Result<T> {
    match f() {
        Ok(v) => Ok(v),
        Err(e)
            if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
                && !stream_capturing(dev) =>
        {
            tracing::warn!(
                "cuda OOM in {ctx} on GPU{}; reclaiming pool and retrying once",
                dev.ordinal()
            );
            emergency_reclaim(dev);
            f().map_err(|e2| alloc_err(ctx, e2))
        }
        Err(e) => Err(alloc_err(ctx, e)),
    }
}

/// Map a cuBLAS error into a substrate `Error`, tagging the memory-pressure
/// class (`ALLOC_FAILED`, `NOT_INITIALIZED` - a failed workspace/handle
/// allocation) with the stable `[oom]` marker so `Error::is_oom()` routes the
/// op through the reclaim/retry + CPU-bounce nets instead of failing the
/// request. `INTERNAL_ERROR` is deliberately NOT in the class: it can leave
/// the CUDA context poisoned (every later launch fails), so "recovering" from
/// it mid-graph only defers a worse failure - it must surface as fatal.
pub(crate) fn cublas_err(ctx: &str, e: cudarc::cublas::result::CublasError) -> Error {
    use cudarc::cublas::sys::cublasStatus_t as S;
    match e.0 {
        S::CUBLAS_STATUS_ALLOC_FAILED | S::CUBLAS_STATUS_NOT_INITIALIZED => {
            Error(format!("[oom] cublas {ctx}: {e:?}"))
        }
        _ => Error(format!("cublas {ctx}: {e:?}")),
    }
}

/// A CUDA device handle: primary context + its default stream.
pub struct CudaDevice {
    pub(super) ctx: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) ordinal: usize,
    /// The production fused-kernel module (NVRTC-compiled once per device).
    pub(super) fused_module: OnceLock<Arc<CudaModule>>,
    /// Native elementwise kernels (small ops with no production equivalent).
    pub(super) elementwise_module: OnceLock<Arc<CudaModule>>,
    /// cuBLAS handle for matmul (the cuBLAS handle is thread-safe per NVIDIA).
    pub(super) blas: OnceLock<cudarc::cublas::CudaBlas>,
    /// The production mmvq quantized-GEMV module (same PTX the live decode
    /// path compiles from cuda/mmvq_gguf.cu).
    pub(super) mmvq_module: OnceLock<Arc<CudaModule>>,
    /// The full relocated quantized.cu module (KV-staging quantize kernels +
    /// dequant; compat shim).
    pub(super) quantized_module: OnceLock<Arc<CudaModule>>,
    /// Device attributes for the tiled quantized-matmul dispatch.
    pub(super) mmq_info: OnceLock<MmqDeviceInfo>,
    /// Generic NVRTC custom-module cache keyed by module name (compat
    /// `get_or_load_custom_func` - same per-device-once semantics as the
    /// fork's custom_modules map).
    pub(super) custom_modules: std::sync::Mutex<std::collections::HashMap<String, Arc<CudaModule>>>,
    /// Cached secondary stream for overlap (cuda_ext::fork_stream - one alt
    /// stream per device, lazily created and reused).
    pub(super) alt_stream: OnceLock<Arc<CudaStream>>,
    /// The pool's reservation chunk, measured the first time it can be (see
    /// [`CudaDevice::pool_chunk`]). Zero until then, so a measurement that could not be
    /// taken is retried rather than remembered as an answer.
    pub(super) pool_chunk: std::sync::atomic::AtomicU64,
    /// Marks this device's queued work so another device's stream can wait for it.
    /// One per device and re-recorded in place: a boundary crossing happens once per
    /// token, and creating a driver object that often is a cost on the hot path.
    pub(super) xfer_event: OnceLock<CudaEvent>,
}

/// Compute capability / SM count / opt-in smem / warp size - what the MMQ
/// launchers select tile configurations on.
#[derive(Clone, Copy, Debug)]
pub struct MmqDeviceInfo {
    pub cc: i32,
    pub nsm: i32,
    pub smpbo: i64,
    pub warp_size: i32,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({})", self.ordinal)
    }
}

/// Per-ordinal device registry - `CudaDevice` carries per-device kernel-module
/// caches, so everything should share one instance per GPU.
pub(super) static DEVICES: OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, Arc<CudaDevice>>>,
> = OnceLock::new();

impl CudaDevice {
    /// The shared device handle for `ordinal` (created on first use).
    pub fn get(ordinal: usize) -> Result<Arc<Self>> {
        let map = DEVICES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = g.get(&ordinal) {
            return Ok(d.clone());
        }
        let d = Self::new(ordinal)?;
        g.insert(ordinal, d.clone());
        Ok(d)
    }

    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        tracing::debug!("cuda ctx: creating a context on ordinal {ordinal}");
        let ctx =
            CudaContext::new(ordinal).map_err(|e| Error(format!("cuda context {ordinal}: {e}")))?;
        // Per-slice event tracking OFF, here and not at each loader: with it on,
        // every slice records a read/write event on each use so the driver can
        // order it against a SECOND stream. There is none. A device owns exactly
        // one working stream (below), so within a device the ops are already
        // stream-ordered, and a cross-device transfer records and waits its own
        // event explicitly in `wait_for`. The events order a stream against
        // itself.
        //
        // It also has to happen BEFORE any slice exists, which only the
        // constructor can guarantee: the flip affects slices created after it.
        // Stating it once here rather than at each device-creating loader is the
        // point - the previous arrangement disabled it at eight call sites and
        // still missed the architecture backends, so a model that spans cards
        // spent about a third of its CUDA API time recording and awaiting events
        // nothing could consume. Capturing a decode graph needs it off too: the
        // capture would otherwise await events created outside the capture.
        unsafe { ctx.disable_event_tracking() };
        // A REAL stream, not `ctx.default_stream()`: the legacy NULL stream
        // cannot be captured (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED broke the
        // gpt-oss/lfm2 graph decode under the flip). Matches the
        // facade: the reference CudaDevice also runs on a `new_stream()`.
        let stream = ctx
            .new_stream()
            .map_err(|e| Error(format!("cuda stream {ordinal}: {e}")))?;
        let dev = Arc::new(Self {
            ctx,
            stream,
            ordinal,
            fused_module: OnceLock::new(),
            elementwise_module: OnceLock::new(),
            blas: OnceLock::new(),
            mmvq_module: OnceLock::new(),
            quantized_module: OnceLock::new(),
            mmq_info: OnceLock::new(),
            custom_modules: std::sync::Mutex::new(std::collections::HashMap::new()),
            alt_stream: OnceLock::new(),
            pool_chunk: std::sync::atomic::AtomicU64::new(0),
            xfer_event: OnceLock::new(),
        });
        // Taken HERE because it can only be taken against a pool with nothing spare,
        // and this is the moment that is true: the device has just been opened and
        // nothing has allocated on it yet. One allocation of one byte, two attribute
        // reads, and every later caller gets the answer from the cache.
        let _ = dev.pool_chunk();
        Ok(dev)
    }

    /// The two numbers that decide what an allocation COSTS this card, as opposed to
    /// what it asks for: the page the driver hands out physical memory in, and the
    /// boundary it rounds a request up to.
    ///
    /// BOTH ARE READ FROM THE DEVICE, and that is the point of the function. A block
    /// smaller than a page still pins one; a block that is not a multiple of the
    /// alignment still occupies one. The waste that follows is not a property of this
    /// machine, of this model or of this checkpoint - it is a property of the card and
    /// its driver, and asking them is the only way to get it right on a card nobody
    /// has measured. `cuMemGetAllocationGranularity` reports the first; the texture
    /// alignment - which is what the driver guarantees a `cuMemAlloc` pointer meets -
    /// reports the second.
    ///
    /// Falls back to the values the CUDA programming guide states as universal minima
    /// when the driver declines to answer, so a caller always gets a usable pair.
    ///
    /// NOTE ON WHICH ALLOCATOR THIS DESCRIBES. `cuMemGetAllocationGranularity` answers
    /// for a plain `cuMemAlloc`. Allocations here go through the stream-ordered pool
    /// (`CudaStream::alloc` issues `cuMemAllocAsync` whenever the device has one), and
    /// that pool reserves in a much larger unit - measured, not documented, which is
    /// what [`Self::pool_chunk_bytes`] is for. Use this pair for the alignment and for
    /// a device with no pool; use the pool's own figure to account for what a card
    /// reports.
    pub fn alloc_shape(&self) -> (u64, u64) {
        use cudarc::driver::{result, sys};
        let cu_device = self.stream.context().cu_device();
        let align = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
            )
        }
        .unwrap_or(512)
        .max(1) as u64;
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.__bindgen_anon_1.id = self.ordinal as i32;
        let mut page: usize = 0;
        let ok = unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut page,
                &prop,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        };
        let page = if ok == sys::cudaError_enum::CUDA_SUCCESS && page > 0 {
            page as u64
        } else {
            2 << 20
        };
        (page, align)
    }

    /// The unit the STREAM-ORDERED POOL reserves in, measured by asking it for one
    /// byte and watching what it takes from the card.
    ///
    /// WHY MEASURED AND NOT READ. There is no attribute for it: the pool reports what
    /// it has reserved and what is in use, never the step between them. But the step
    /// is the whole difference between the memory a forward holds and the memory a
    /// card reports holding for it, so it cannot be left as a guess or as a constant
    /// somebody measured once on their own hardware. One allocation and two attribute
    /// reads settle it on the card in hand, in microseconds.
    ///
    /// `None` when the pool already had room for the byte - the answer is only visible
    /// against a pool with nothing spare, which in practice means asking early, before
    /// the first model is resident. A caller that gets `None` has learnt something
    /// true (this allocation cost the card nothing) and should keep whatever figure it
    /// had.
    pub fn pool_chunk_bytes(&self) -> Option<u64> {
        use cudarc::driver::sys::{
            cuDeviceGetDefaultMemPool, cuMemPoolGetAttribute, cudaError_enum, CUmemPool_attribute,
            CUmemoryPool,
        };
        let reserved = |pool: CUmemoryPool| -> u64 {
            let mut v: u64 = 0;
            unsafe {
                let _ = cuMemPoolGetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                    &mut v as *mut u64 as *mut std::ffi::c_void,
                );
            }
            v
        };
        let mut pool: CUmemoryPool = std::ptr::null_mut();
        unsafe {
            if cuDeviceGetDefaultMemPool(&mut pool, self.ordinal as i32)
                != cudaError_enum::CUDA_SUCCESS
                || pool.is_null()
            {
                return None;
            }
        }
        let before = reserved(pool);
        // Through the same call the whole substrate allocates through, so what is
        // measured is the allocator that will be paying.
        let probe = unsafe { self.stream.alloc::<u8>(1) }.ok()?;
        let after = reserved(pool);
        drop(probe);
        after.checked_sub(before).filter(|d| *d > 0)
    }

    /// The pool's reservation chunk, remembered once it has been possible to measure it.
    ///
    /// [`Self::pool_chunk_bytes`] only answers against a pool with nothing spare, which
    /// in a live process means early. So the answer is kept the first time it comes, and
    /// `None` afterwards means it has never come - which is a fact the caller must act
    /// on rather than paper over: charging a load at the driver's plain granularity when
    /// the pool reserves sixteen times that under-states what the card gives up, and
    /// under-stating is the direction that admits a placement which does not fit.
    pub fn pool_chunk(&self) -> Option<u64> {
        use std::sync::atomic::Ordering;
        let known = self.pool_chunk.load(Ordering::Relaxed);
        if known > 0 {
            return Some(known);
        }
        let measured = self.pool_chunk_bytes()?;
        self.pool_chunk.store(measured, Ordering::Relaxed);
        Some(measured)
    }

    /// Device attributes the tiled quantized-matmul launchers dispatch on
    /// (compute capability, SM count, opt-in smem, warp size). Cached.
    /// Whether this card has the Ampere tensor-core instructions a few kernel families are
    /// built on: bf16 WMMA fragments and the `m16n8k32` integer MMA. Neither exists below
    /// sm_80, so those families ship no code for a Turing card and their callers have to
    /// decline the same way they decline an unsupported dtype - a launch would fail.
    pub fn has_ampere_tensor_cores(&self) -> bool {
        self.mmq_device_info().is_ok_and(|i| i.cc >= 800)
    }

    pub fn mmq_device_info(&self) -> Result<MmqDeviceInfo> {
        if let Some(i) = self.mmq_info.get() {
            return Ok(*i);
        }
        use cudarc::driver::{result, sys};
        let cu_device = self.stream.context().cu_device();
        let attr = |a: sys::CUdevice_attribute, default: i32| -> i32 {
            unsafe { result::device::get_attribute(cu_device, a) }.unwrap_or(default)
        };
        let major = attr(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            8,
        );
        let minor = attr(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0,
        );
        let info = MmqDeviceInfo {
            cc: major * 100 + minor * 10,
            nsm: attr(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                1,
            ),
            smpbo: attr(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                49152,
            ) as i64,
            warp_size: attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE, 32),
        };
        let _ = self.mmq_info.set(info);
        Ok(info)
    }

    /// cuBLAS handle (created once per device; thread-safe per NVIDIA docs).
    pub fn blas(&self) -> Result<&cudarc::cublas::CudaBlas> {
        if self.blas.get().is_none() {
            let b = cudarc::cublas::CudaBlas::new(self.stream.clone())
                .map_err(|e| cublas_err("init", e))?;
            let _ = self.blas.set(b);
        }
        Ok(self.blas.get().unwrap())
    }

    /// Load a function from the native elementwise-kernel module.
    pub fn elementwise_fn(&self, name: &str) -> Result<CudaFunction> {
        if self.elementwise_module.get().is_none() {
            let opts = cudarc::nvrtc::CompileOptions {
                include_paths: crate::inference::quantized_cuda::cuda_include_paths(),
                arch: crate::inference::quantized_cuda::nvrtc_arch_of_ordinal(self.ordinal()),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(ELEMENTWISE_SRC, opts)
                .map_err(|e| Error(format!("elementwise nvrtc: {e}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| Error(format!("elementwise module load: {e}")))?;
            let _ = self.elementwise_module.set(module);
        }
        self.elementwise_module
            .get()
            .unwrap()
            .load_function(name)
            .map_err(|e| Error(format!("elementwise fn `{name}`: {e}")))
    }

    /// Load a function from the production fused-kernel module (compiled once
    /// per device from the same in-crate source the live decode path uses).
    pub fn fused_fn(&self, name: &str) -> Result<CudaFunction> {
        if self.fused_module.get().is_none() {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: crate::inference::quantized_cuda::nvrtc_arch_of_ordinal(self.ordinal()),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                crate::inference::kernel::fused::fused_cuda_src(),
                opts,
            )
            .map_err(|e| Error(format!("fused kernel nvrtc: {e}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| Error(format!("fused module load: {e}")))?;
            let _ = self.fused_module.set(module);
        }
        self.fused_module
            .get()
            .unwrap()
            .load_function(name)
            .map_err(|e| Error(format!("fused fn `{name}`: {e}")))
    }

    /// Load a function from the production mmvq module (quantized GEMV - the
    /// exact kernels the live decode path runs for GGUF weights).
    pub fn mmvq_fn(&self, name: &str) -> Result<CudaFunction> {
        if self.mmvq_module.get().is_none() {
            let ptx_src =
                crate::inference::quantized_cuda::get_mmvq_ptx_for_ordinal(self.ordinal())
                    .map_err(|e| Error(format!("mmvq ptx: {e}")))?;
            let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| Error(format!("mmvq module load: {e}")))?;
            let _ = self.mmvq_module.set(module);
        }
        self.mmvq_module
            .get()
            .unwrap()
            .load_function(name)
            .map_err(|e| Error(format!("mmvq fn `{name}`: {e}")))
    }

    /// Load a function from the full relocated quantized.cu module (the
    /// KV-staging quantize kernels the compat QCudaStorage drives;
    /// same cached PTX the inference launchers compile).
    pub fn quantized_fn(&self, name: &str) -> Result<CudaFunction> {
        if self.quantized_module.get().is_none() {
            let ptx_src =
                crate::inference::quantized_cuda::get_quantized_ptx_for_ordinal(self.ordinal())
                    .map_err(|e| Error(format!("quantized ptx: {e}")))?;
            let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| Error(format!("quantized module load: {e}")))?;
            let _ = self.quantized_module.set(module);
        }
        self.quantized_module
            .get()
            .unwrap()
            .load_function(name)
            .map_err(|e| Error(format!("quantized fn `{name}`: {e}")))
    }

    /// The device's cached secondary ("alt") stream for overlap work.
    /// Created lazily, reused for the device's lifetime - synchronisation
    /// with the default stream is the caller's job (events).
    pub fn alt_stream(&self) -> Result<Arc<CudaStream>> {
        if let Some(s) = self.alt_stream.get() {
            return Ok(s.clone());
        }
        let s = self
            .ctx
            .new_stream()
            .map_err(|e| Error(format!("alt stream: {e}")))?;
        let _ = self.alt_stream.set(s);
        Ok(self.alt_stream.get().unwrap().clone())
    }

    /// Load `fn_name` from the custom module `module_name`, compiling/loading
    /// `ptx` once per device (compat boundary for the fork's
    /// `get_or_load_custom_func` call sites).
    pub fn custom_fn(&self, module_name: &str, ptx: &str, fn_name: &str) -> Result<CudaFunction> {
        let mut g = self
            .custom_modules
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !g.contains_key(module_name) {
            let module = self
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
                .map_err(|e| Error(format!("custom module `{module_name}` load: {e}")))?;
            g.insert(module_name.to_string(), module);
        }
        g.get(module_name)
            .unwrap()
            .load_function(fn_name)
            .map_err(|e| Error(format!("custom fn `{fn_name}`: {e}")))
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The stream all native allocations/copies/launches go through. Existing
    /// kernel launchers take exactly this type.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| Error(format!("cuda sync: {e}")))
    }

    /// A fresh event on this device's context, for a caller that owns a shared resource and
    /// has to order the next user of it behind its own queued work.
    ///
    /// The one this device keeps for cross-device copies is re-recorded in place and cannot
    /// be borrowed for a second purpose: two users recording the same event would each wait
    /// for the other's work rather than their own.
    pub fn new_event(&self) -> Result<CudaEvent> {
        self.ctx
            .new_event(None)
            .map_err(|e| Error(format!("cuda event create: {e}")))
    }

    /// Hold this device's stream until everything queued on `src` has finished, without
    /// stopping the host.
    ///
    /// This is the ordering a cross-device copy needs, and the only part of it that was
    /// ever needed: the copy itself is already enqueued on the destination stream, so
    /// what has to be established is that the source's kernels finished producing the
    /// data first. Blocking the host to get that guarantee costs a full pipeline drain
    /// at every boundary, every token - and it also makes the placement uncapturable,
    /// because a stream synchronize is illegal inside a graph capture while an event
    /// record and an event wait are not. Every model spread over more than one card was
    /// paying both prices.
    pub fn wait_for(&self, src: &CudaDevice) -> Result<()> {
        let ev = match src.xfer_event.get() {
            Some(e) => e,
            None => {
                let e = src
                    .ctx
                    .new_event(None)
                    .map_err(|e| Error(format!("cuda event create: {e}")))?;
                let _ = src.xfer_event.set(e);
                src.xfer_event
                    .get()
                    .ok_or_else(|| Error("cuda event create: lost the race".into()))?
            }
        };
        ev.record(&src.stream)
            .map_err(|e| Error(format!("cuda event record: {e}")))?;
        self.stream
            .wait(ev)
            .map_err(|e| Error(format!("cuda event wait: {e}")))
    }
}

/// Drop the per-ordinal device registry (model unload: each entry pins an
/// `Arc<CudaContext>`; clearing lets the last user release it).
pub fn clear_device_cache() {
    if let Some(m) = DEVICES.get() {
        m.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    // The device-constant cache pins slices (-> streams -> contexts) too.
    if let Some(m) = DEV_CONST_I32.get() {
        m.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Per-device cache of small immutable i32 meta arrays (permute dims/strides,
/// broadcast shape/stride tables), keyed by content.
///
/// Two reasons this exists instead of a per-call `memcpy_stod`:
///  - CUDA-graph replay correctness: a pageable H2D memcpy captured into a
///    graph records the HOST pointer; the temporary Vec is freed right after
///    capture, so every replay re-reads freed host memory and feeds garbage
///    strides to the gather kernels (out-of-bounds device reads). A cached
///    constant is uploaded once OUTSIDE capture; captured kernels then bake a
///    stable device pointer and replay never touches host memory.
///  - decode-loop overhead: these arrays are shape-derived, so a decode loop
///    re-uploads identical contents every token; the cache replaces an
///    alloc + H2D per call with a map lookup.
pub(super) static DEV_CONST_I32: OnceLock<
    std::sync::Mutex<std::collections::HashMap<(usize, Vec<i32>), Arc<CudaSlice<i32>>>>,
> = OnceLock::new();

/// Device-resident copy of `vals` for `dev` (cached; see [`DEV_CONST_I32`]).
pub(crate) fn dev_const_i32(dev: &CudaDevice, vals: &[i32]) -> Result<Arc<CudaSlice<i32>>> {
    let map = DEV_CONST_I32.get_or_init(|| std::sync::Mutex::new(Default::default()));
    let key = (dev.ordinal(), vals.to_vec());
    if let Some(s) = map.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(s.clone());
    }
    let stream = dev.stream();
    // Graph capture in flight (or its bump arena armed): a fresh upload here
    // would land in the capture arena (whose lifetime is the graph's, not the
    // process's) and record an H2D node. Keep it OUT of the persistent cache
    // and leak the host buffer so the captured memcpy node's source stays
    // valid for every replay. Pre-capture warmup makes this path cold.
    let capturing = stream.context().arena_armed()
        || stream
            .capture_status()
            .ok()
            .map(|st| {
                !matches!(
                    st,
                    cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                )
            })
            .unwrap_or(false);
    if capturing {
        let host: &'static [i32] = Box::leak(vals.to_vec().into_boxed_slice());
        let mut d = with_oom_retry(dev, "dev_const", || unsafe {
            stream.alloc::<i32>(host.len())
        })?;
        stream
            .memcpy_htod(host, &mut d)
            .map_err(|e| Error(format!("dev_const upload: {e}")))?;
        return Ok(Arc::new(d));
    }
    let d = stream
        .clone_htod(vals)
        .map_err(|e| Error(format!("dev_const upload: {e}")))?;
    let arc = Arc::new(d);
    map.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, arc.clone());
    Ok(arc)
}

/// Typed device buffer. Mirrors `CpuStorage`, holding vendored-`cudarc` slices
/// so existing launch sites can consume them directly.
#[derive(Debug)]
pub enum CudaStorage {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I16(CudaSlice<i16>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    F16(CudaSlice<half::f16>),
    BF16(CudaSlice<half::bf16>),
    F32(CudaSlice<f32>),
}
