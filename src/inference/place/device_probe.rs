//! Single CUDA device probe. One NVML-based enumeration shared by both the
//! generic LLM engine (`llm_engine::from_gguf`) and the ACE-Step native stack
//! (LM / DiT / VAE / FSQ-encoder placement via `vram_manager::probe_under_pressure`).
//!
//! Before this module each path had its own probe: the generic one was the mature,
//! process-aware NVML reader; ACE-Step used a bespoke raw `mem_get_info` 0..8 loop.
//! They diverged. Now the NVML enumeration + stable-free + fraction logic lives here
//! once; ACE-Step's degrade glue is a thin wrapper around `probe_cuda_devices`.
//!
//! NEVER panics. NVML init failure (driver absent / acestep_render on a CPU box) ->
//! empty list -> caller falls back to CPU placement.

use tracing::{info, warn};

/// ~0.5 GB driver/display reserve per GPU. Using raw `mem.free` produces
/// non-deterministic layer plans across reloads because driver caches from the
/// previous model linger and shrink the observed free pool. Anchoring to
/// `total - reservation` gives a stable high-watermark that matches the
/// steady-state available VRAM after the driver releases its caches.
const GPU_SYSTEM_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// One probed CUDA GPU. `stable_free` is the process-aware free VRAM (see
/// `probe_cuda_gpus`); `available` is that figure scaled by the caller's
/// `max_gpu_memory_fraction`. GPUs are returned sorted by `sm_clock` descending
/// (fastest first) so the placer packs the fastest card first.
#[derive(Debug, Clone, Copy)]
pub struct CudaGpuInfo {
    pub index: usize,
    /// Process-aware free VRAM (bytes): trusts NVML's `mem.free` when another
    /// process - or our own loaded model - holds VRAM; otherwise floors at
    /// `total - GPU_SYSTEM_RESERVE_BYTES` to recover the driver's lingering cache.
    pub stable_free: u64,
    /// `stable_free * max_gpu_memory_fraction` (the usable budget for placement).
    pub available: u64,
    pub sm_clock: u32,
    /// CUDA core count (NVML `nvmlDeviceGetNumGpuCores`), 0 if unavailable.
    pub num_cores: u32,
    /// Compute-throughput proxy used to rank GPUs fastest-first:
    /// `num_cores x sm_clock`, falling back to `sm_clock` alone when the core
    /// count is unavailable. Clock alone misranks a high-boost / low-core card
    /// (e.g. a xx60 whose boost clock exceeds a faster xx70's) as the fastest.
    pub perf_score: u64,
    /// Physical capacity (NVML `mem.total`). Unlike the free-VRAM fields this does
    /// not move, so it is the ceiling on what waiting for a gap could ever deliver.
    pub total: u64,
    /// Memory bandwidth in GB/s (bus width x DDR memory clock), 0.0 when NVML
    /// lacks either figure. The hardware term a decode ceiling derives from.
    pub mem_bw_gbs: f64,
}

/// Enumerate every visible CUDA GPU via NVML, process-aware, sorted by compute
/// throughput (CUDA cores x SM clock)
/// (fastest first). This is the SINGLE source of the stable-free logic; both the
/// generic LLM engine and ACE-Step consume it. Returns empty (-> CPU placement)
/// when NVML init fails - never panics.
///
/// `max_gpu_memory_fraction` scales `stable_free` into `available` (the generic
/// engine's per-GPU budget knob). ACE-Step passes `1.0` (it applies its own
/// reserve via `probe_cuda_devices`).
/// Memory bandwidth of one CUDA device in GB/s, from the driver's own attributes
/// (DDR memory clock x bus width). NVML lacks the bus width on recent cards, the
/// driver never does. 0.0 when CUDA is absent or the query fails.
#[cfg(feature = "cuda")]
fn cuda_mem_bw_gbs(index: usize) -> f64 {
    use cudarc::driver::result as cu;
    use cudarc::driver::sys::CUdevice_attribute as A;
    let _ = cu::init();
    let Ok(d) = cu::device::get(index as i32) else {
        return 0.0;
    };
    let clk_khz = unsafe { cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) }
        .unwrap_or(0) as f64;
    let bus_bits =
        unsafe { cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) }
            .unwrap_or(0) as f64;
    clk_khz * 1e3 * (bus_bits / 8.0) * 2.0 / 1e9
}

#[cfg(not(feature = "cuda"))]
fn cuda_mem_bw_gbs(_index: usize) -> f64 {
    0.0
}

/// Per-card VRAM snapshot at debug level, tagged with the load phase that asked.
/// The instrument task #64 calls for: a context appearing on a card that holds no
/// layer is invisible in any log until the phase that created it is bracketed.
pub fn debug_vram_by_card(tag: &str) {
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        let n = nvml.device_count().unwrap_or(0);
        for i in 0..n {
            if let Ok(d) = nvml.device_by_index(i) {
                if let Ok(m) = d.memory_info() {
                    let active = primary_ctx_active(i as usize);
                    tracing::debug!(
                        "vram[{tag}] gpu{i}: {:.0} MB used, primary ctx {}",
                        m.used as f64 / 1e6,
                        if active { "ACTIVE" } else { "inactive" }
                    );
                }
            }
        }
    }
}

/// Whether a device's primary context is currently alive - the ground truth on who won a
/// retain/release argument, straight from the driver.
#[cfg(feature = "cuda")]
fn primary_ctx_active(ordinal: usize) -> bool {
    use cudarc::driver::sys::{cuDevicePrimaryCtxGetState, cudaError_enum};
    let _ = cudarc::driver::result::init();
    let Ok(d) = cudarc::driver::result::device::get(ordinal as i32) else {
        return false;
    };
    let mut flags: u32 = 0;
    let mut active: i32 = 0;
    let ok = unsafe {
        cuDevicePrimaryCtxGetState(d, &mut flags, &mut active) == cudaError_enum::CUDA_SUCCESS
    };
    ok && active != 0
}

#[cfg(not(feature = "cuda"))]
fn primary_ctx_active(_ordinal: usize) -> bool {
    false
}

pub fn probe_cuda_gpus(max_gpu_memory_fraction: f64) -> Vec<CudaGpuInfo> {
    let mut gpu_info: Vec<CudaGpuInfo> = Vec::new();
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            use nvml_wrapper::enum_wrappers::device::Clock;
            let gpu_count = nvml.device_count().unwrap_or(0);
            let our_pid = std::process::id();
            for i in 0..gpu_count {
                if let Ok(gpu) = nvml.device_by_index(i) {
                    if let Ok(mem) = gpu.memory_info() {
                        let name = gpu.name().unwrap_or_else(|_| format!("GPU {}", i));
                        // Trust nvml's free figure UNLESS there are no other
                        // compute processes holding memory - in which case the
                        // floor (total - reserve) recovers VRAM that the driver
                        // still caches briefly after our own model unloads.
                        //
                        // Two exclusions to the floor:
                        //  * `other_compute` != 0 - some other process holds VRAM;
                        //    we can't reclaim it.
                        //  * `our_holding` > 0 - WE still hold VRAM on this GPU
                        //    (e.g. mid-reload between models, or a model is loaded).
                        //    The driver cache is part of our footprint here, so the
                        //    "post-unload recovery" justification doesn't apply;
                        //    trust real mem.free.
                        let (other_compute, our_holding) = gpu
                            .running_compute_processes()
                            .map(|procs| {
                                let mut other = false;
                                let mut ours: u64 = 0;
                                for p in procs {
                                    if p.pid == our_pid {
                                        if let nvml_wrapper::enums::device::UsedGpuMemory::Used(m) =
                                            p.used_gpu_memory
                                        {
                                            ours = ours.saturating_add(m);
                                        }
                                    } else {
                                        other = true;
                                    }
                                }
                                (other, ours)
                            })
                            .unwrap_or((false, 0));
                        let stable_free = if other_compute || our_holding > 0 {
                            mem.free
                        } else {
                            mem.free
                                .max(mem.total.saturating_sub(GPU_SYSTEM_RESERVE_BYTES))
                        };
                        let available = (stable_free as f64 * max_gpu_memory_fraction) as u64;
                        let sm_clock = gpu.max_clock_info(Clock::SM).unwrap_or(0);
                        let mem_bw_gbs = cuda_mem_bw_gbs(i as usize);
                        let num_cores = gpu.num_cores().unwrap_or(0);
                        // Throughput proxy: cores x clock. Falls back to clock
                        // alone if the core count is unavailable.
                        let perf_score = if num_cores > 0 {
                            num_cores as u64 * sm_clock as u64
                        } else {
                            sm_clock as u64
                        };
                        info!("✓ GPU {}: {} - {:.1} GB free / {:.1} GB total, stable {:.1} GB, using {:.0}% = {:.1} GB, {} cores @ {} MHz (perf {})",
                            i, name,
                            mem.free as f64 / 1e9,
                            mem.total as f64 / 1e9,
                            stable_free as f64 / 1e9,
                            max_gpu_memory_fraction * 100.0,
                            available as f64 / 1e9,
                            num_cores, sm_clock, perf_score);
                        gpu_info.push(CudaGpuInfo {
                            index: i as usize,
                            stable_free,
                            available,
                            sm_clock,
                            num_cores,
                            perf_score,
                            total: mem.total,
                            mem_bw_gbs,
                        });
                    } else {
                        warn!("⚠️  NVML memory_info() failed for device {}", i);
                    }
                }
            }
            // Rank by compute throughput (CUDA cores x SM clock), fastest first,
            // with the GPU index as a stable tiebreaker. SM clock alone misranks
            // a high-boost but low-core card as fastest.
            gpu_info.sort_by(|a, b| b.perf_score.cmp(&a.perf_score).then(a.index.cmp(&b.index)));
        }
        Err(e) => {
            warn!("⚠️  NVML init failed: {}", e);
        }
    }
    gpu_info
}

/// ACE-Step placement probe: every visible GPU with its usable free VRAM and a
/// native `Device` handle - `(index, usable_free_bytes, Device)` - for the
/// HeteroPlan placement of the LM / DiT / VAE / encoders.
///
/// Built on the shared `probe_cuda_gpus` (process-aware NVML, fraction 1.0).
/// `reserve` is subtracted from each card's `stable_free`; cards left with no
/// usable budget are dropped. Cards are offered fastest-first (SM-clock order).
///
/// Reserve accounting: `reserve` is the ONLY headroom subtracted here. ACE-Step's
/// callers already pass a per-stage headroom (plus the degrade boost added by
/// `probe_under_pressure`), so the generic engine's 768 MB loader-overhead is NOT
/// applied on this path - that overhead lives only in `llm_engine`.
///
/// Returns empty (-> CPU) when CUDA is unavailable; never panics.
#[cfg(feature = "cuda")]
pub fn probe_cuda_devices(reserve: u64) -> Vec<(usize, u64, crate::tensor::Device)> {
    use crate::tensor::Device;
    let mut out = Vec::new();
    for g in probe_cuda_gpus(1.0) {
        let usable = g.stable_free.saturating_sub(reserve);
        if usable == 0 {
            continue;
        }
        // Materialize a context on the card so the planner has a Device handle.
        if let Ok(cd) = crate::tensor::cuda::CudaDevice::get(g.index) {
            out.push((g.index, usable, Device::Cuda(cd)));
        }
    }
    out
}

/// No-CUDA build: no GPUs -> CPU placement.
#[cfg(not(feature = "cuda"))]
pub fn probe_cuda_devices(_reserve: u64) -> Vec<(usize, u64, crate::tensor::Device)> {
    Vec::new()
}

/// Subtract the VRAM other subsystems have DECLARED but not yet allocated.
///
/// NVML reports what is allocated now, so a card that another subsystem is in the
/// middle of filling - or is about to fill with a render's activation peak - reads as
/// free, and a planner that trusts it places on top of work it cannot see. That is one
/// of the two ways concurrent requests OOM each other; the load-admission lock in
/// `vram_manager` is the other.
///
/// Taken GREEDILY from the fastest card down, because that is the order every planner
/// in this process fills cards in: the declaring subsystem's hot component sits on the
/// fastest card that holds it, so that is where its peak lands.
fn apply_pending_demand(owner: &str, cards: &mut [CudaGpuInfo], fraction: f64) {
    let mut pending = crate::inference::place::vram_manager::pending_demand_excluding(owner);
    if pending == 0 {
        return;
    }
    for g in cards.iter_mut() {
        let take = g.stable_free.min(pending);
        g.stable_free -= take;
        pending -= take;
        g.available = (g.stable_free as f64 * fraction) as u64;
        if pending == 0 {
            break;
        }
    }
}

/// [`probe_cuda_gpus`], minus what other subsystems have declared they are about to use.
pub fn probe_cuda_gpus_for(owner: &str, max_gpu_memory_fraction: f64) -> Vec<CudaGpuInfo> {
    let mut cards = probe_cuda_gpus(max_gpu_memory_fraction);
    apply_pending_demand(owner, &mut cards, max_gpu_memory_fraction);
    cards
}

/// [`probe_cuda_devices`], minus what other subsystems have declared they are about to use.
#[cfg(feature = "cuda")]
pub fn probe_cuda_devices_for(
    owner: &str,
    reserve: u64,
) -> Vec<(usize, u64, crate::tensor::Device)> {
    use crate::tensor::Device;
    let mut out = Vec::new();
    for g in probe_cuda_gpus_for(owner, 1.0) {
        let usable = g.stable_free.saturating_sub(reserve);
        if usable == 0 {
            continue;
        }
        if let Ok(cd) = crate::tensor::cuda::CudaDevice::get(g.index) {
            out.push((g.index, usable, Device::Cuda(cd)));
        }
    }
    out
}

/// No-CUDA build: no GPUs, so CPU placement.
#[cfg(not(feature = "cuda"))]
pub fn probe_cuda_devices_for(
    _owner: &str,
    _reserve: u64,
) -> Vec<(usize, u64, crate::tensor::Device)> {
    Vec::new()
}

#[cfg(all(test, feature = "cuda"))]
mod primary_ctx_tests {
    use super::primary_ctx_active;

    /// The proven mechanics behind an idle card that never cools (#64): a driver
    /// context releases cleanly with its last handle, but ONE cuBLAS init latches the
    /// primary context inside the CUDA runtime for the life of the process - no Rust
    /// drop, cache clear or pool release can undo it. This is why a card warmed with a
    /// matmul keeps ~150 MB and ~10 W forever. If a future driver changes either
    /// behaviour, this test says so.
    #[test]
    #[ignore = "hardware probe: needs a second CUDA card and owns its context state"]
    fn a_context_releases_cleanly_until_cublas_latches_it() {
        let ord = 1usize;
        if cudarc::driver::result::init().is_err() {
            return;
        }
        let count = cudarc::driver::result::device::get_count().unwrap_or(0) as usize;
        if count <= ord {
            return;
        }
        assert!(
            !primary_ctx_active(ord),
            "context active before anything ran"
        );
        {
            let _d = crate::tensor::Device::new_cuda(ord).unwrap();
        }
        assert!(
            !primary_ctx_active(ord),
            "a bare create+drop must release the context"
        );
        {
            let d = crate::tensor::Device::new_cuda(ord).unwrap();
            let _a =
                crate::tensor::Tensor::zeros_on((16usize, 16usize), crate::tensor::DType::F32, &d)
                    .unwrap();
        }
        assert!(
            !primary_ctx_active(ord),
            "an allocation must not outlive its device"
        );
        {
            let d = crate::tensor::Device::new_cuda(ord).unwrap();
            if let crate::tensor::Device::Cuda(cd) = &d {
                let _ = cd.blas().unwrap();
            }
        }
        assert!(
            primary_ctx_active(ord),
            "cuBLAS no longer latches the primary context - the warm-up policy in \
             llm_engine (WARM_CTX) can be revisited"
        );
    }
}
