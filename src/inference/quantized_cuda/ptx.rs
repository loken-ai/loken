//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Threads per block for the AWQ GEMV (one thread = one packed column = 8 out).
pub(super) const AWQ_TPB: usize = 128;

/// Split-K chunk size for the AWQ GEMV. One-thread-per-column alone exposes only
/// `N/8` threads (≈5 column-blocks for a 5120-wide matrix -> most of the GPU
/// idle, ~16 GB/s). `blockIdx.y` partitions K into chunks; we size the chunk so
/// the grid has ≈`TARGET_BLOCKS` blocks regardless of N. Measured sweep
/// (deepcoder q_proj 5120²): KCHUNK 64 -> 483 GB/s (beats Q4_K Marlin 403), 128 ->
/// 412, 512 -> 129 - smaller is better until atomic/redundant-read overhead bites,
/// so we floor at 64. Validated by `awq_check`.
pub(super) fn awq_kchunk(k: usize, n: usize, _group_size: usize) -> usize {
    // Per-shape kchunk, from a measured sweep. The optimum keys on N (column
    // parallelism) rather than on a single block target:
    //   narrow-N (q/o 5120², down 13824x5120 - few col-blocks): kc≈128 -> fine
    //     k-splitting fills the GPU & amortizes the coalesced load (680, 931 GB/s)
    //   wide-N (gate_up 5120x27648 - 27 col-blocks): kc≈512 -> columns already
    //     fill the GPU, so coarser splits cut atomic/launch overhead (780 vs the
    //     block-target heuristic's 674; +16%).
    // The old block-target under-split down (826) and over-split gate_up (674).
    let col_blocks = (n / 8).div_ceil(AWQ_TPB).max(1);
    let mut kc = if col_blocks >= 16 { 512 } else { 128 };
    kc = kc.min(k.max(1));
    // Floor: keep >=~200 blocks so small (KxN) shapes still saturate the GPU
    // (a fixed kc128 under-fills e.g. qwen3 4096² -> only 128 blocks -> regression).
    while col_blocks * k.div_ceil(kc) < 200 && kc > 32 {
        kc = (kc * 3 / 4).max(32);
    }
    kc
}

/// AWQ uniform-4-bit decode GEMV kernel. NVRTC-compiled into PTX, cached.
pub(super) const AWQ_GEMV_CU: &str = include_str!("../cuda/awq_gemv.cu");
pub(super) static AWQ_GEMV_PTX: OnceLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, &'static str>>,
> = OnceLock::new();

/// NVRTC-compile awq_gemv.cu (cached). Returns the PTX source.
pub fn get_awq_gemv_ptx(dev: &CudaDevice) -> Result<&'static str> {
    // One PTX per architecture present: each card loads code compiled for itself, and a
    // mixed machine never JITs one card's PTX onto another.
    let arch = nvrtc_arch_of(dev);
    let map = AWQ_GEMV_PTX.get_or_init(Default::default);
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    let key = arch.unwrap_or("default");
    if let Some(p) = g.get(key) {
        if p.is_empty() {
            return Err(anyhow!("NVRTC compile previously failed for {key}"));
        }
        return Ok(p);
    }
    let compiled: String = (|| {
        let opts = cudarc::nvrtc::safe::CompileOptions {
            include_paths: cuda_include_paths(),
            arch,
            ..Default::default()
        };
        let src = format!("{NVRTC_COMPAT_H}\n{AWQ_GEMV_CU}");
        match cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts) {
            Ok(ptx) => {
                let s = ptx.to_src();
                tracing::info!(
                    "quantized_cuda: NVRTC-compiled awq_gemv.cu OK ({} KB PTX)",
                    s.len() / 1024
                );
                s
            }
            Err(e) => {
                tracing::error!("quantized_cuda: awq_gemv.cu NVRTC compile FAILED: {e}");
                String::new()
            }
        }
    })();
    let leaked: &'static str = Box::leak(compiled.into_boxed_str());
    g.insert(key, leaked);
    let ptx = leaked;
    if ptx.is_empty() {
        return Err(anyhow!("awq_gemv.cu failed to NVRTC-compile"));
    }
    Ok(ptx)
}
