//! The card a bench runs on, through the crate's own device: its streams and events, page-locked
//! host memory, and a kernel that holds the compute stream for a stated time.

use crate::tensor::cuda::CudaDevice;
use crate::tensor::gpu_runtime::driver::{
    result, sys, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

/// The device opened for a run and the name its driver reports.
pub struct Card {
    pub dev: Arc<CudaDevice>,
    pub name: String,
}

/// Open card `ordinal` with a context and a stream of its own, as a served model's device is.
pub fn open(ordinal: usize) -> Result<Card, String> {
    let dev = CudaDevice::new(ordinal).map_err(|e| format!("card {ordinal}: {e:?}"))?;
    let name = dev
        .context()
        .name()
        .map_err(|e| format!("card {ordinal} name: {e:?}"))?;
    Ok(Card { dev, name })
}

/// A copy stream of the card, beside its compute stream, as the expert uploader opens one.
pub fn copy_stream(card: &Card) -> Result<Arc<CudaStream>, String> {
    card.dev
        .context()
        .new_stream()
        .map_err(|e| format!("copy stream: {e:?}"))
}

/// `len` bytes on the card, allocated on `stream`.
pub fn device_bytes(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<u8>, String> {
    // Safety: every reader of the buffer runs after a copy into it on the same stream order.
    unsafe { stream.alloc::<u8>(len.max(1)) }.map_err(|e| format!("card memory: {e:?}"))
}

/// Microseconds between two recorded events that have both completed.
pub fn elapsed_us(start: &CudaEvent, end: &CudaEvent) -> Result<f64, String> {
    start
        .elapsed_ms(end)
        .map(|ms| ms as f64 * 1e3)
        .map_err(|e| format!("event elapsed: {e:?}"))
}

/// Page-locked host memory of the card's context, zero-filled.
pub struct Pinned {
    ptr: *mut u8,
    len: usize,
}

// Safety: the memory is owned here alone and freed once, in `drop`.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Pinned {
    pub fn new(card: &Card, len: usize) -> Result<Self, String> {
        card.dev
            .context()
            .bind_to_thread()
            .map_err(|e| format!("pinned memory: {e:?}"))?;
        // Safety: a fresh allocation, written before it is read.
        let ptr = unsafe { result::malloc_host(len.max(1), 0) }
            .map_err(|e| format!("pinned memory of {len} bytes: {e:?}"))?
            as *mut u8;
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        // Safety: `len` bytes were allocated at `ptr` and stay until `drop`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: as `as_slice`, with the exclusive borrow of `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // Safety: allocated by `malloc_host` and freed here once.
        let _ = unsafe { result::free_host(self.ptr as *mut std::ffi::c_void) };
    }
}

/// The kernel that holds the compute stream: one warp per multiprocessor, each spinning on the
/// card's clock for the cycles asked.
const SPIN_SRC: &str = r#"
extern "C" __global__ void microbench_spin(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
    }
}
"#;

/// A kernel that keeps the compute stream busy for about the time it was built for.
pub struct Spin {
    func: CudaFunction,
    cfg: LaunchConfig,
    cycles: i64,
    /// The time asked for, in microseconds.
    pub us: f64,
}

impl Spin {
    /// A spin of about `us` microseconds, from the card's reported clock rate.
    pub fn lasting(card: &Card, us: f64) -> Result<Self, String> {
        let ctx = card.dev.context();
        let attribute = |a: sys::CUdevice_attribute| {
            ctx.attribute(a)
                .map_err(|e| format!("card attribute {a:?}: {e:?}"))
        };
        let khz = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
        let sms = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let warp = attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
        let opts = crate::tensor::gpu_runtime::nvrtc::CompileOptions {
            arch: crate::inference::quantized_cuda::nvrtc_arch_of_ordinal(card.dev.ordinal()),
            include_paths: crate::inference::quantized_cuda::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = crate::tensor::cuda::compile::compile(SPIN_SRC, opts)
            .map_err(|e| format!("spin kernel: {e}"))?;
        let func = card
            .dev
            .custom_fn("microbench_spin", &ptx, "microbench_spin")
            .map_err(|e| format!("spin kernel: {e:?}"))?;
        Ok(Self {
            func,
            cfg: LaunchConfig {
                grid_dim: (sms.max(1) as u32, 1, 1),
                block_dim: (warp.max(1) as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            cycles: (us * khz as f64 / 1e3) as i64,
            us,
        })
    }

    /// Queue one spin on `stream`.
    pub fn launch(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
        let mut b = stream.launch_builder(&self.func);
        b.arg(&self.cycles);
        // Safety: the kernel reads its one argument and no memory.
        unsafe { b.launch(self.cfg) }
            .map(|_| ())
            .map_err(|e| format!("spin launch: {e:?}"))
    }
}
