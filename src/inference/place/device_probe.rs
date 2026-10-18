//! Single CUDA device probe. One NVML-based enumeration shared by both the
//! generic LLM engine (`llm_engine::from_gguf`) and the ACE-Step native stack
//! (LM / DiT / VAE / FSQ-encoder placement via `vram_manager::probe_under_pressure`).
//!
//! Before this module each path had its own probe: the generic one was the mature,
//! process-aware NVML reader; ACE-Step used a bespoke raw `mem_get_info` 0..8 loop.
//! They diverged. Now the NVML enumeration + stable-free logic lives here
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
/// `probe_cuda_gpus`); `available` is what placement may plan into: that figure less
/// what other subsystems declared they are about to use. GPUs are returned sorted by `sm_clock` descending
/// (fastest first) so the placer packs the fastest card first.
#[derive(Debug, Clone, Copy)]
pub struct CudaGpuInfo {
    pub index: usize,
    /// Process-aware free VRAM (bytes): trusts NVML's `mem.free` when another
    /// process - or our own loaded model - holds VRAM; otherwise floors at
    /// `total - GPU_SYSTEM_RESERVE_BYTES` to recover the driver's lingering cache.
    pub stable_free: u64,
    /// The usable budget for placement. Headroom for the load itself is the caller's
    /// runtime reserve and the out-of-memory ladder, both derived, not a share of the card.
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
    /// Bytes per second, in GB/s, the card's link to the host carries: the generation the
    /// platform allows at the width the slot negotiated. 0.0 when NVML lacks either.
    pub host_link_gbs: f64,
}

/// GB/s one PCIe lane of generation `generation` carries, after encoding.
fn pcie_lane_gbs(generation: u32) -> f64 {
    match generation {
        1 => 0.25,
        2 => 0.5,
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        6 => 7.563,
        _ => 0.0,
    }
}

/// Enumerate every visible CUDA GPU via NVML, process-aware, sorted by compute
/// throughput (CUDA cores x SM clock)
/// (fastest first). This is the SINGLE source of the stable-free logic; both the
/// generic LLM engine and ACE-Step consume it. Returns empty (-> CPU placement)
/// when NVML init fails - never panics.
///
/// Memory bandwidth of one CUDA device in GB/s, from the driver's own attributes
/// (DDR memory clock x bus width). NVML lacks the bus width on recent cards, the
/// driver never does. 0.0 when CUDA is absent or the query fails.
#[cfg(feature = "gpu")]
fn cuda_mem_bw_gbs(index: usize) -> f64 {
    use crate::tensor::gpu_runtime::driver::result as cu;
    use crate::tensor::gpu_runtime::driver::sys::CUdevice_attribute as A;
    if crate::tensor::cuda::runtime::require().is_err() {
        return 0.0;
    }
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

#[cfg(not(feature = "gpu"))]
fn cuda_mem_bw_gbs(_index: usize) -> f64 {
    0.0
}

/// The cards this process can use, as `(CUDA ordinal, NVML index)` in ordinal order.
///
/// The driver numbers the cards after `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`, NVML
/// numbers every card of the machine; the two meet on the PCI bus id. Without CUDA, or when a
/// card's bus id does not resolve, the NVML index stands for the ordinal.
pub fn visible_cards(nvml: &nvml_wrapper::Nvml) -> Vec<(usize, u32)> {
    let all = nvml.device_count().unwrap_or(0);
    // A driver without the toolkit libraries lists cards through NVML that CUDA cannot open:
    // the machine has no usable card.
    #[cfg(feature = "gpu")]
    if !crate::tensor::cuda::runtime::usable() {
        return Vec::new();
    }
    #[cfg(feature = "gpu")]
    {
        use crate::tensor::gpu_runtime::driver::sys::{cuDeviceGetPCIBusId, cudaError_enum};
        let _ = crate::tensor::gpu_runtime::driver::result::init();
        if let Ok(n) = crate::tensor::gpu_runtime::driver::result::device::get_count() {
            let mut out = Vec::new();
            for ordinal in 0..n {
                let mut id = [0 as std::os::raw::c_char; 64];
                let ok = unsafe { cuDeviceGetPCIBusId(id.as_mut_ptr(), id.len() as i32, ordinal) }
                    == cudaError_enum::CUDA_SUCCESS;
                let bus = ok.then(|| {
                    let bytes: Vec<u8> = id
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| c as u8)
                        .collect();
                    String::from_utf8_lossy(&bytes).to_uppercase()
                });
                let index = bus.and_then(|b| {
                    (0..all).find(|&i| {
                        nvml.device_by_index(i)
                            .and_then(|d| d.pci_info())
                            .is_ok_and(|p| {
                                p.bus_id
                                    .to_uppercase()
                                    .ends_with(&b[b.len().saturating_sub(12)..])
                            })
                    })
                });
                out.push((ordinal as usize, index.unwrap_or(ordinal as u32)));
            }
            return out;
        }
    }
    (0..all).map(|i| (i as usize, i)).collect()
}

/// How many cards this process can open: the ones NVML lists and the process sees on an
/// NVIDIA runtime, the runtime's own count otherwise.
pub fn card_count() -> usize {
    #[cfg(feature = "gpu")]
    if !crate::tensor::gpu_runtime::IS_NVIDIA {
        use crate::tensor::gpu_runtime::driver::result as cu;
        if crate::tensor::cuda::runtime::require().is_err() || cu::init().is_err() {
            return 0;
        }
        return cu::device::get_count().unwrap_or(0).max(0) as usize;
    }
    nvml_wrapper::Nvml::init().map_or(0, |nvml| visible_cards(&nvml).len())
}

/// Per-card VRAM snapshot at debug level, tagged with the load phase that asked.
/// The instrument task #64 calls for: a context appearing on a card that holds no
/// layer is invisible in any log until the phase that created it is bracketed.
pub fn debug_vram_by_card(tag: &str) {
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        for (i, nvml_index) in visible_cards(&nvml) {
            if let Ok(d) = nvml.device_by_index(nvml_index) {
                if let Ok(m) = d.memory_info() {
                    let active = primary_ctx_active(i);
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
#[cfg(feature = "gpu")]
fn primary_ctx_active(ordinal: usize) -> bool {
    use crate::tensor::gpu_runtime::driver::sys::{cuDevicePrimaryCtxGetState, cudaError_enum};
    if crate::tensor::cuda::runtime::require().is_err() {
        return false;
    }
    let _ = crate::tensor::gpu_runtime::driver::result::init();
    let Ok(d) = crate::tensor::gpu_runtime::driver::result::device::get(ordinal as i32) else {
        return false;
    };
    let mut flags: u32 = 0;
    let mut active: i32 = 0;
    let ok = unsafe {
        cuDevicePrimaryCtxGetState(d, &mut flags, &mut active) == cudaError_enum::CUDA_SUCCESS
    };
    ok && active != 0
}

#[cfg(not(feature = "gpu"))]
fn primary_ctx_active(_ordinal: usize) -> bool {
    false
}

/// What a message from card `from` to card `to` costs on the path tensors take between them (a
/// peer copy, else through host memory): microseconds for a small one and bytes per microsecond
/// beyond it, measured once a pair. `None` when either card cannot be opened.
#[cfg(feature = "gpu")]
pub fn card_pair_link(from: usize, to: usize) -> Option<(f64, f64)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static LINKS: OnceLock<Mutex<HashMap<(usize, usize), Option<(f64, f64)>>>> = OnceLock::new();
    let links = LINKS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(known) = links
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(from, to))
    {
        return *known;
    }
    let measured = (|| -> Option<(f64, f64)> {
        let (src, dst) = (
            crate::tensor::Device::new_cuda(from).ok()?,
            crate::tensor::Device::new_cuda(to).ok()?,
        );
        // A row of activations and a block far larger than any latency.
        let (small, large) = (4096usize, 8usize << 20);
        let time = |elems: usize, times: usize| -> Option<f64> {
            let t = crate::tensor::Tensor::from_vec(vec![0f32; elems], elems, &src).ok()?;
            src.synchronize().ok()?;
            let mut runs: Vec<f64> = (0..times)
                .filter_map(|_| {
                    let start = std::time::Instant::now();
                    t.to_device(&dst).ok()?;
                    dst.synchronize().ok()?;
                    Some(start.elapsed().as_secs_f64() * 1e6)
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            runs.get(runs.len() / 2).copied()
        };
        // The first copy of a pair opens its path; it is not timed.
        time(small, 1)?;
        let small_us = time(small, 9)?;
        let large_us = time(large, 3)?;
        let bytes = ((large - small) * 4) as f64;
        Some((
            small_us,
            bytes / (large_us - small_us).max(f64::MIN_POSITIVE),
        ))
    })();
    if let Some((us, rate)) = measured {
        tracing::info!(
            "card link {from} -> {to}: {us:.1} us a small message, {:.1} GB/s",
            rate * 1e6 / 1e9
        );
    }
    links
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((from, to), measured);
    measured
}

/// No-CUDA build: no card, so no link between two.
#[cfg(not(feature = "gpu"))]
pub fn card_pair_link(_from: usize, _to: usize) -> Option<(f64, f64)> {
    None
}

/// What summing a message onto every card of `cards` costs through NCCL, the path split layers
/// take: microseconds for a small one and bytes per microsecond beyond it, measured once a group.
/// `None` when a card cannot be opened or NCCL does not take the group.
#[cfg(feature = "gpu")]
pub fn group_sum(cards: &[usize]) -> Option<(f64, f64)> {
    use crate::tensor::gpu_runtime::nccl::safe::{group_end, group_start, Comm, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static GROUPS: OnceLock<Mutex<HashMap<Vec<usize>, Option<(f64, f64)>>>> = OnceLock::new();
    let groups = GROUPS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(known) = groups.lock().unwrap_or_else(|e| e.into_inner()).get(cards) {
        return *known;
    }
    let measured = (|| -> Option<(f64, f64)> {
        if cards.len() < 2 || !crate::tensor::cuda::runtime::nccl_present() {
            return None;
        }
        let streams = cards
            .iter()
            .map(|&c| {
                let dev = crate::tensor::Device::new_cuda(c).ok()?;
                dev.as_cuda_device().ok().map(|d| d.cuda_stream())
            })
            .collect::<Option<Vec<_>>>()?;
        let comms = Comm::from_devices(streams.clone()).ok()?;
        // A row of activations and a block far larger than any latency.
        let (small, large) = (4096usize, 8usize << 20);
        let time = |elems: usize, times: usize| -> Option<f64> {
            let mut bufs = streams
                .iter()
                .map(|s| s.alloc_zeros::<f32>(elems).ok())
                .collect::<Option<Vec<_>>>()?;
            let settled = || streams.iter().all(|s| s.synchronize().is_ok());
            settled().then_some(())?;
            let mut runs: Vec<f64> = (0..times)
                .filter_map(|_| {
                    let start = std::time::Instant::now();
                    group_start().ok()?;
                    for (comm, buf) in comms.iter().zip(bufs.iter_mut()) {
                        comm.all_reduce_in_place(buf, &ReduceOp::Sum).ok()?;
                    }
                    group_end().ok()?;
                    settled().then_some(())?;
                    Some(start.elapsed().as_secs_f64() * 1e6)
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            runs.get(runs.len() / 2).copied()
        };
        // The first sum of a group opens its paths; it is not timed.
        time(small, 1)?;
        let small_us = time(small, 9)?;
        let large_us = time(large, 3)?;
        let bytes = ((large - small) * 4) as f64;
        Some((
            small_us,
            bytes / (large_us - small_us).max(f64::MIN_POSITIVE),
        ))
    })();
    if let Some((us, rate)) = measured {
        tracing::info!(
            "cards {cards:?} sum through NCCL: {us:.1} us a small message, {:.1} GB/s",
            rate * 1e6 / 1e9
        );
    }
    groups
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(cards.to_vec(), measured);
    measured
}

/// Floating point operations per microsecond card `card` runs a block of prompt rows through a
/// projection at: a BF16 product as wide as a layer's, measured once a card. `None` when the card
/// cannot be opened.
#[cfg(feature = "gpu")]
pub fn card_matmul_rate(card: usize) -> Option<f64> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static RATES: OnceLock<Mutex<HashMap<usize, Option<f64>>>> = OnceLock::new();
    let rates = RATES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(known) = rates.lock().unwrap_or_else(|e| e.into_inner()).get(&card) {
        return *known;
    }
    let measured = (|| -> Option<f64> {
        use crate::tensor::{DType, Tensor};
        let dev = crate::tensor::Device::new_cuda(card).ok()?;
        // Rows of a prompt block through a square projection of a mid-sized model.
        let (rows, width) = (512usize, 4096usize);
        let x = Tensor::zeros_on((rows, width), DType::BF16, &dev).ok()?;
        let w = Tensor::zeros_on((width, width), DType::BF16, &dev).ok()?;
        let time = || -> Option<f64> {
            dev.synchronize().ok()?;
            let start = std::time::Instant::now();
            x.matmul(&w).ok()?;
            dev.synchronize().ok()?;
            Some(start.elapsed().as_secs_f64() * 1e6)
        };
        // The first product loads the library's kernels; it is not timed.
        time()?;
        let mut runs: Vec<f64> = (0..5).filter_map(|_| time()).collect();
        runs.sort_by(f64::total_cmp);
        let us = runs.get(runs.len() / 2).copied()?;
        Some(2.0 * (rows * width * width) as f64 / us.max(f64::MIN_POSITIVE))
    })();
    if let Some(rate) = measured {
        tracing::info!(
            "card {card} projection rate: {:.1} TFLOP/s",
            rate * 1e6 / 1e12
        );
    }
    rates
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(card, measured);
    measured
}

/// No-CUDA build: no card to measure.
#[cfg(not(feature = "gpu"))]
pub fn card_matmul_rate(_card: usize) -> Option<f64> {
    None
}

/// No-CUDA build: no cards to sum across.
#[cfg(not(feature = "gpu"))]
pub fn group_sum(_cards: &[usize]) -> Option<(f64, f64)> {
    None
}

pub fn probe_cuda_gpus() -> Vec<CudaGpuInfo> {
    #[cfg(feature = "gpu")]
    if !crate::tensor::gpu_runtime::IS_NVIDIA {
        return probe_runtime_gpus();
    }
    let mut gpu_info: Vec<CudaGpuInfo> = Vec::new();
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            use nvml_wrapper::enum_wrappers::device::Clock;
            let cards = visible_cards(&nvml);
            let machine_cards = nvml.device_count().unwrap_or(0) as usize;
            if cards.len() != machine_cards {
                info!(
                    "{} of the machine's {} GPU(s) visible to this process (CUDA_VISIBLE_DEVICES)",
                    cards.len(),
                    machine_cards
                );
            }
            let our_pid = std::process::id();
            for (i, nvml_index) in cards {
                if let Ok(gpu) = nvml.device_by_index(nvml_index) {
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
                        let available = stable_free;
                        let sm_clock = gpu.max_clock_info(Clock::SM).unwrap_or(0);
                        let mem_bw_gbs = cuda_mem_bw_gbs(i);
                        let host_link_gbs = match (
                            gpu.max_pcie_link_gen().ok(),
                            gpu.current_pcie_link_width().ok(),
                        ) {
                            (Some(generation), Some(width)) => {
                                pcie_lane_gbs(generation) * width as f64
                            }
                            _ => 0.0,
                        };
                        let num_cores = gpu.num_cores().unwrap_or(0);
                        // Throughput proxy: cores x clock. Falls back to clock
                        // alone if the core count is unavailable.
                        let perf_score = if num_cores > 0 {
                            num_cores as u64 * sm_clock as u64
                        } else {
                            sm_clock as u64
                        };
                        info!("✓ GPU {}: {} - {:.1} GB free / {:.1} GB total, stable {:.1} GB, {} cores @ {} MHz (perf {})",
                            i, name,
                            mem.free as f64 / 1e9,
                            mem.total as f64 / 1e9,
                            stable_free as f64 / 1e9,
                            num_cores, sm_clock, perf_score);
                        gpu_info.push(CudaGpuInfo {
                            index: i,
                            stable_free,
                            available,
                            sm_clock,
                            num_cores,
                            perf_score,
                            total: mem.total,
                            mem_bw_gbs,
                            host_link_gbs,
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

/// Every card the runtime lists, read through the runtime itself, for a runtime NVML does not
/// see: name and memory from the driver, the throughput proxy from compute units times clock,
/// the host link from the PCI device's own link in sysfs. Sorted fastest first like
/// `probe_cuda_gpus`.
#[cfg(feature = "gpu")]
fn probe_runtime_gpus() -> Vec<CudaGpuInfo> {
    use crate::tensor::gpu_runtime::driver::result as cu;
    use crate::tensor::gpu_runtime::driver::sys::CUdevice_attribute as A;
    if crate::tensor::cuda::runtime::require().is_err() || cu::init().is_err() {
        return Vec::new();
    }
    let count = cu::device::get_count().unwrap_or(0).max(0) as usize;
    let mut gpu_info = Vec::new();
    for i in 0..count {
        let Ok(d) = cu::device::get(i as i32) else {
            continue;
        };
        let attr = |a| {
            unsafe { cu::device::get_attribute(d, a) }
                .unwrap_or(0)
                .max(0) as u32
        };
        let name = cu::device::get_name(d).unwrap_or_else(|_| format!("GPU {i}"));
        let Ok(card) = crate::tensor::cuda::CudaDevice::get(i) else {
            continue;
        };
        // PLACEMENT-EXEMPT: the probe's own reading of the card it lists.
        let Ok((free, total)) = card.context().mem_get_info() else {
            warn!("GPU {i}: the runtime does not report its memory");
            continue;
        };
        let (free, total) = (free as u64, total as u64);
        let sm_clock = attr(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000;
        let num_cores = attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let perf_score = (num_cores.max(1) as u64) * sm_clock as u64;
        let host_link_gbs = pci_host_link_gbs(
            attr(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID),
            attr(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID),
            attr(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID),
        );
        info!(
            "GPU {i}: {name} - {:.1} GB free / {:.1} GB total, {num_cores} compute units @ {sm_clock} MHz (perf {perf_score})",
            free as f64 / 1e9,
            total as f64 / 1e9
        );
        gpu_info.push(CudaGpuInfo {
            index: i,
            stable_free: free,
            available: free,
            sm_clock,
            num_cores,
            perf_score,
            total,
            mem_bw_gbs: cuda_mem_bw_gbs(i),
            host_link_gbs,
        });
    }
    gpu_info.sort_by(|a, b| b.perf_score.cmp(&a.perf_score).then(a.index.cmp(&b.index)));
    gpu_info
}

/// GB/s the PCI device `domain:bus:device` carries to the host: the generation its link
/// supports at the width it negotiated, from sysfs. 0.0 when sysfs does not say.
#[cfg(feature = "gpu")]
fn pci_host_link_gbs(domain: u32, bus: u32, device: u32) -> f64 {
    let dir = std::path::PathBuf::from(format!(
        "/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{device:02x}.0"
    ));
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap_or_default();
    let width: u32 = read("current_link_width").trim().parse().unwrap_or(0);
    let gts: f64 = read("max_link_speed")
        .split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    pcie_lane_gbs(pcie_generation_of_gts(gts)) * width as f64
}

/// The PCIe generation whose lane signals at `gts` GT/s.
fn pcie_generation_of_gts(gts: f64) -> u32 {
    [2.5, 5.0, 8.0, 16.0, 32.0, 64.0]
        .iter()
        .position(|g| (gts - g).abs() < 0.5)
        .map_or(0, |p| p as u32 + 1)
}

/// ACE-Step placement probe: every visible GPU with its usable free VRAM and a
/// native `Device` handle - `(index, usable_free_bytes, Device)` - for the
/// HeteroPlan placement of the LM / DiT / VAE / encoders.
///
/// Built on the shared `probe_cuda_gpus` (process-aware NVML).
/// `reserve` is subtracted from each card's `stable_free`; cards left with no
/// usable budget are dropped. Cards are offered fastest-first (SM-clock order).
///
/// Reserve accounting: `reserve` is the ONLY headroom subtracted here. ACE-Step's
/// callers already pass a per-stage headroom (plus the degrade boost added by
/// `probe_under_pressure`), so the generic engine's 768 MB loader-overhead is NOT
/// applied on this path - that overhead lives only in `llm_engine`.
///
/// Returns empty (-> CPU) when CUDA is unavailable; never panics.
#[cfg(feature = "gpu")]
pub fn probe_cuda_devices(reserve: u64) -> Vec<(usize, u64, crate::tensor::Device)> {
    use crate::tensor::Device;
    let mut out = Vec::new();
    for g in probe_cuda_gpus() {
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
#[cfg(not(feature = "gpu"))]
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
fn apply_pending_demand(owner: &str, cards: &mut [CudaGpuInfo]) {
    let mut pending = crate::inference::place::vram_manager::pending_demand_excluding(owner);
    if pending == 0 {
        return;
    }
    for g in cards.iter_mut() {
        let take = g.stable_free.min(pending);
        g.stable_free -= take;
        pending -= take;
        g.available = g.stable_free;
        if pending == 0 {
            break;
        }
    }
}

/// [`probe_cuda_gpus`], minus what other subsystems have declared they are about to use.
pub fn probe_cuda_gpus_for(owner: &str) -> Vec<CudaGpuInfo> {
    let mut cards = probe_cuda_gpus();
    apply_pending_demand(owner, &mut cards);
    cards
}

/// [`probe_cuda_devices`], minus what other subsystems have declared they are about to use.
#[cfg(feature = "gpu")]
pub fn probe_cuda_devices_for(
    owner: &str,
    reserve: u64,
) -> Vec<(usize, u64, crate::tensor::Device)> {
    use crate::tensor::Device;
    let mut out = Vec::new();
    for g in probe_cuda_gpus_for(owner) {
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
#[cfg(not(feature = "gpu"))]
pub fn probe_cuda_devices_for(
    _owner: &str,
    _reserve: u64,
) -> Vec<(usize, u64, crate::tensor::Device)> {
    Vec::new()
}

#[cfg(all(test, feature = "gpu"))]
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
        if crate::tensor::gpu_runtime::driver::result::init().is_err() {
            return;
        }
        let count =
            crate::tensor::gpu_runtime::driver::result::device::get_count().unwrap_or(0) as usize;
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

#[cfg(all(test, feature = "gpu"))]
mod visible_tests {
    use super::*;

    /// Each ordinal the probe offers is the driver's card of that ordinal: the name NVML gives the
    /// card behind it is the name the driver gives the ordinal, so a restricted
    /// `CUDA_VISIBLE_DEVICES` offers the cards it names and no others.
    #[test]
    fn the_probe_offers_the_cards_the_driver_numbers() {
        if crate::tensor::gpu_runtime::driver::result::init().is_err() {
            eprintln!("skipping: no usable CUDA driver");
            return;
        }
        let Ok(nvml) = nvml_wrapper::Nvml::init() else {
            eprintln!("skipping: no NVML");
            return;
        };
        let visible =
            crate::tensor::gpu_runtime::driver::result::device::get_count().unwrap_or(0) as usize;
        let cards = visible_cards(&nvml);
        assert_eq!(cards.len(), visible, "every visible card and only those");
        for (ordinal, index) in cards {
            let dev =
                crate::tensor::gpu_runtime::driver::result::device::get(ordinal as i32).unwrap();
            let mut name = [0 as std::os::raw::c_char; 128];
            unsafe {
                crate::tensor::gpu_runtime::driver::sys::cuDeviceGetName(
                    name.as_mut_ptr(),
                    128,
                    dev,
                )
            };
            let bytes: Vec<u8> = name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let driver_name = String::from_utf8_lossy(&bytes).to_string();
            let nvml_name = nvml.device_by_index(index).unwrap().name().unwrap();
            assert_eq!(
                driver_name, nvml_name,
                "ordinal {ordinal} is NVML card {index}"
            );
        }
    }
}
