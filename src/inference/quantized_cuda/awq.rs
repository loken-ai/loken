//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Production AWQ decode GEMV: resident weight tensors in, on-device `CudaStorage`
/// `[N]` f32 out (no host round-trip). The wrapper builds the output `Tensor`.
pub fn awq_gemv_storage(
    qweight: &crate::tensor::Tensor,
    qzeros: &crate::tensor::Tensor,
    scales: &crate::tensor::Tensor,
    x: &crate::tensor::Tensor,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<CudaStorage> {
    use crate::tensor::StorageView;
    let dev = x.device().as_cuda_device()?;
    let func = awq_kernel(&dev, "awq_gemv_f32")?;
    let (qw_s, _) = qweight.storage_and_layout();
    let (qz_s, _) = qzeros.storage_and_layout();
    let (sc_s, _) = scales.storage_and_layout();
    let (x_s, _) = x.storage_and_layout();
    let qw_cs = match &*qw_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qweight not cuda"),
    };
    let qz_cs = match &*qz_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qzeros not cuda"),
    };
    let sc_cs = match &*sc_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
        _ => anyhow::bail!("scales not cuda"),
    };
    let x_cs = match &*x_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => anyhow::bail!("x not cuda"),
    };
    let dst = dev
        .alloc_zeros::<f32>(n)
        .map_err(|e| anyhow!("awq out alloc: {e}"))?;
    let pcols = n / 8;
    const TPB: u32 = 128;
    let kchunk = awq_kchunk(k, n, group_size);
    let cfg = LaunchConfig {
        grid_dim: (
            (pcols as u32).div_ceil(TPB),
            (k as u32).div_ceil(kchunk as u32),
            1,
        ),
        block_dim: (TPB, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, k_i, g_i, kc_i) = (n as i32, k as i32, group_size as i32, kchunk as i32);
    let mut b = func.builder();
    b.arg(qw_cs);
    b.arg(qz_cs);
    b.arg(sc_cs);
    b.arg(x_cs);
    b.arg(&dst);
    b.arg(&k_i);
    b.arg(&n_i);
    b.arg(&g_i);
    b.arg(&kc_i); // kernel sig is (K,N,gs,kchunk)
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch awq_gemv_f32: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Decode GEMV via dp4a over the repacked layout: quantises `x` (assumed
/// `[K]` f32 contiguous) to per-32-block q8_1, then runs `awq_gemv_dp4a_f32`.
/// Returns the `[N]` f32 result as `CudaStorage`. mmvq-class (~833 GB/s honest
/// DRAM) - the production AWQ decode path once wired.
pub fn awq_gemv_dp4a_storage(
    qw_t: &crate::tensor::Tensor,
    scales_t: &crate::tensor::Tensor,
    zeros_t: &crate::tensor::Tensor,
    x: &crate::tensor::Tensor,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<CudaStorage> {
    use crate::tensor::StorageView;
    let dev = x.device().as_cuda_device()?;
    let (x_s, _) = x.storage_and_layout();
    let x_cs = match &*x_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => anyhow::bail!("x not cuda"),
    };
    let x_view = x_cs.slice(0..k);
    let q8_bytes = q8_1_staged_bytes(k, 1);
    // Reuse a per-(byte-size) q8_1 scratch buffer instead of allocating one per
    // projection (192/tok) - a fresh alloc each call thrashed the caching
    // allocator against the per-token sampling buffers and stalled the GPU
    // (100% util @150W). Reuse is race-free: the dp4a kernel reads `yq` on the
    // device stream, and the next call's quantize writes it on the SAME stream ->
    // strictly ordered. Buffer goes back to the pool after the launch enqueues.
    let mut yq = YQ_POOL
        .with(|p| p.borrow_mut().remove(&q8_bytes))
        .map(Ok)
        .unwrap_or_else(|| {
            unsafe { dev.alloc::<u8>(q8_bytes) }.map_err(|e| anyhow!("yq alloc: {e}"))
        })?;
    quantize_q8_1(&x_view, &mut yq, k, 1, &dev)?;
    // u4 = read-ahead-unrolled variant: +6-7% on short-row shapes (q/o, qkv) that
    // drag the in-forward aggregate; gate_up flat (already DRAM-maxed).
    let func = awq_kernel(&dev, "awq_gemv_dp4a_u4_f32")?;
    let (w_s, _) = qw_t.storage_and_layout();
    let (s_s, _) = scales_t.storage_and_layout();
    let (z_s, _) = zeros_t.storage_and_layout();
    let w_cs = match &*w_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qw_t not cuda"),
    };
    let s_cs = match &*s_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
        _ => anyhow::bail!("scales_t not cuda"),
    };
    let z_cs = match &*z_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<u8>()?,
        _ => anyhow::bail!("zeros_t not cuda"),
    };
    let dst = unsafe { dev.alloc::<f32>(n) }.map_err(|e| anyhow!("dp4a out alloc: {e}"))?;
    const WPB: u32 = 4;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(WPB), 1, 1),
        block_dim: (32, WPB, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, k_i, g_i) = (n as i32, k as i32, group_size as i32);
    let mut b = func.builder();
    b.arg(w_cs);
    b.arg(s_cs);
    b.arg(z_cs);
    b.arg(&yq);
    b.arg(&dst);
    b.arg(&n_i);
    b.arg(&k_i);
    b.arg(&g_i);
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch dp4a: {e}"))?;
    // Return the scratch to the pool for the next projection (launch above has
    // enqueued the read; same-stream ordering keeps reuse safe).
    YQ_POOL.with(|p| {
        p.borrow_mut().insert(q8_bytes, yq);
    });
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

thread_local! {
    /// Per-thread q8_1 activation scratch buffers, keyed by byte size, reused
    /// across dp4a decode projections to avoid a per-call device allocation.
    static YQ_POOL: std::cell::RefCell<std::collections::HashMap<usize, cudarc::driver::CudaSlice<u8>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// AWQ prefill path: dequantize to a dense f16 `[K, N]` weight (`W^T`), returned
/// as `CudaStorage`. The wrapper does `x[.,K] @ W[K,N]` via cuBLAS (the GEMV
/// kernel only handles seq=1). Weights stay resident; this temporary is built per
/// prefill call and dropped after the matmul.
pub fn awq_dequant_f16_storage(
    qweight: &crate::tensor::Tensor,
    qzeros: &crate::tensor::Tensor,
    scales: &crate::tensor::Tensor,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<CudaStorage> {
    use crate::tensor::StorageView;
    let dev = qweight.device().as_cuda_device()?;
    let func = awq_kernel(&dev, "awq_dequant_f16")?;
    let (qw_s, _) = qweight.storage_and_layout();
    let (qz_s, _) = qzeros.storage_and_layout();
    let (sc_s, _) = scales.storage_and_layout();
    let qw_cs = match &*qw_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qweight not cuda"),
    };
    let qz_cs = match &*qz_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qzeros not cuda"),
    };
    let sc_cs = match &*sc_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
        _ => anyhow::bail!("scales not cuda"),
    };
    let dst =
        unsafe { dev.alloc::<half::f16>(k * n) }.map_err(|e| anyhow!("awq dequant alloc: {e}"))?;
    let pcols = n / 8;
    const TPB: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: ((pcols as u32).div_ceil(TPB), k as u32, 1),
        block_dim: (TPB, 1, 1),
        shared_mem_bytes: 0,
    };
    // awq_dequant_f16 kernel signature is (.., K, N, group_size) - pass K then N.
    // (The GEMV kernel takes N,K; do NOT copy its arg order here - that swap left
    // down_proj's k∈[N,K) rows unwritten -> inf. Invisible for K==N projections.)
    let (k_i, n_i, g_i) = (k as i32, n as i32, group_size as i32);
    let mut b = func.builder();
    b.arg(qw_cs);
    b.arg(qz_cs);
    b.arg(sc_cs);
    b.arg(&dst);
    b.arg(&k_i);
    b.arg(&n_i);
    b.arg(&g_i);
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch awq_dequant_f16: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// AWQ prefill dequant from the REPACKED layout -> dense f16 `[K, N]` (`W^T`).
/// Lets the weight keep ONLY the repacked tensors (decode = dp4a over them) and
/// drop the K-major copy -> no 2x VRAM. Same output as `awq_dequant_f16_storage`.
pub fn awq_dequant_f16_repacked_storage(
    qw_t: &crate::tensor::Tensor,
    scales_t: &crate::tensor::Tensor,
    zeros_t: &crate::tensor::Tensor,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<CudaStorage> {
    use crate::tensor::StorageView;
    let dev = qw_t.device().as_cuda_device()?;
    let func = awq_kernel(&dev, "awq_dequant_f16_repacked")?;
    let (w_s, _) = qw_t.storage_and_layout();
    let (s_s, _) = scales_t.storage_and_layout();
    let (z_s, _) = zeros_t.storage_and_layout();
    let w_cs = match &*w_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => anyhow::bail!("qw_t not cuda"),
    };
    let s_cs = match &*s_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
        _ => anyhow::bail!("scales_t not cuda"),
    };
    let z_cs = match &*z_s {
        StorageView::Cuda(c) => c.as_cuda_slice::<u8>()?,
        _ => anyhow::bail!("zeros_t not cuda"),
    };
    let dst =
        unsafe { dev.alloc::<half::f16>(k * n) }.map_err(|e| anyhow!("awq dequant alloc: {e}"))?;
    const TPB: u32 = 128;
    // thread per (k, n), consecutive threads = consecutive n (coalesced writes);
    // grid.y iterates k.
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(TPB), k as u32, 1),
        block_dim: (TPB, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, k_i, g_i) = (n as i32, k as i32, group_size as i32);
    let mut b = func.builder();
    b.arg(w_cs);
    b.arg(s_cs);
    b.arg(z_cs);
    b.arg(&dst);
    b.arg(&n_i);
    b.arg(&k_i);
    b.arg(&g_i);
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch awq_dequant_f16_repacked: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub(crate) fn cuda_include_paths() -> Vec<String> {
    // ONLY the CUDA include dir (for cuda_fp16.h / cuda_bf16.h). Do NOT add
    // /usr/include: NVRTC has its own built-in <stdint.h> etc., and exposing
    // the glibc headers makes `#include <stdint.h>` pull host-only internals
    // (bits/libc-header-start.h) that NVRTC cannot compile.
    let cuda_root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    [
        format!("{cuda_root}/include"),
        "/usr/local/cuda/include".to_string(),
    ]
    .into_iter()
    .filter(|p| std::path::Path::new(p).is_dir())
    .collect::<Vec<_>>()
    .into_iter()
    .fold(Vec::new(), |mut acc, p| {
        if !acc.contains(&p) {
            acc.push(p);
        }
        acc
    })
}

/// NVRTC-compile the full relocated quantized.cu (cached). Returns the PTX.
// pub(crate) (was private): the native substrate's CudaDevice loads the same
// module for the compat QCudaStorage quantize kernels.
pub(crate) fn get_quantized_ptx(dev: &CudaDevice) -> Result<&'static str> {
    get_quantized_ptx_for_ordinal(dev.ordinal())
}

pub(crate) fn get_quantized_ptx_for_ordinal(ordinal: usize) -> Result<&'static str> {
    // One PTX per architecture present: each card loads code compiled for itself, and a
    // mixed machine never JITs one card's PTX onto another.
    let arch = nvrtc_arch_of_ordinal(ordinal);
    let map = QUANTIZED_PTX.get_or_init(Default::default);
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
        // The compat shim first (stdint + INFINITY/NAN, which NVRTC has no headers for),
        // then the block formats and their dot products. `gguf.cuh` states those once for
        // both NVRTC units; this one used to carry its own copy of all of it.
        let src = format!(
            "{NVRTC_COMPAT_H}\n{GGUF_BLOCKS_CUH}\n{QUANTIZED_HELPERS_CU}\n\
             {QUANTIZED_KV_CU}\n{QUANTIZED_ATTENTION_CU}\n\
             {QUANTIZED_FLASH_CU}\n{QUANTIZED_SAMPLING_CU}"
        );
        match cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts) {
            Ok(ptx) => {
                let s = ptx.to_src();
                tracing::info!(
                    "quantized_cuda: NVRTC-compiled quantized.cu OK ({} KB PTX)",
                    s.len() / 1024
                );
                s
            }
            Err(e) => {
                tracing::error!("quantized_cuda: quantized.cu NVRTC compile FAILED: {e}");
                String::new()
            }
        }
    })();
    let leaked: &'static str = Box::leak(compiled.into_boxed_str());
    g.insert(key, leaked);
    let ptx = leaked;
    if ptx.is_empty() {
        return Err(anyhow!("quantized.cu failed to NVRTC-compile"));
    }
    Ok(ptx)
}

/// Prewarm the NVRTC module at model load so the first decode token does not
/// pay the in-band compile. Non-fatal: logs and returns Err on failure.
pub fn prewarm(dev: &CudaDevice) -> Result<()> {
    let _ = get_quantized_ptx(dev)?;
    let _ = get_mmvq_ptx(dev)?;
    Ok(())
}

/// Values one block-quantisation thread block covers.
pub use crate::tensor::quantized::{CUDA_QUANTIZE_BLOCK_SIZE, MATRIX_ROW_PADDING};
pub(super) const Q8_0_BLOCK_SIZE: usize = crate::tensor::quantized::GgmlDType::Q8_0.block_size();
pub(super) const Q8_0_TYPE_SIZE: usize = crate::tensor::quantized::GgmlDType::Q8_0.type_size();

pub fn ceil_div(p: usize, q: usize) -> usize {
    (p + q - 1) / q
}
/// `p` rounded up to a whole number of `q`s.
pub fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// One entry point of the quantised NVRTC module, by name.
///
/// Every launcher over that module opens the same way - compile-or-fetch the module for this
/// card, then resolve the name in it - and a name that does not resolve is worth saying so
/// with the name in hand. The module is compiled once per architecture; this is a lookup
/// after that.
pub(super) fn quantized_kernel(
    dev: &CudaDevice,
    name: &str,
) -> Result<crate::tensor::kernel_ffi::CudaFunc> {
    let ptx = get_quantized_ptx(dev)?;
    dev.get_or_load_custom_func(name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load {name}: {e}"))
}

/// One entry point of the AWQ NVRTC module, by name - [`quantized_kernel`] for the other
/// module these launchers compile.
fn awq_kernel(dev: &CudaDevice, name: &str) -> Result<crate::tensor::kernel_ffi::CudaFunc> {
    let ptx = get_awq_gemv_ptx(dev)?;
    dev.get_or_load_custom_func(name, "loken_awq", ptx)
        .map_err(|e| anyhow!("load {name}: {e}"))
}

/// A thread block of `warps` warps, and `shared_bytes` of dynamic shared memory.
///
/// The attention and mat-vec kernels here all give one warp one unit of work, so the block
/// width is the warp and the block height is how many units a block takes. That count is
/// compile-time inside each kernel, and a launcher that disagrees with it reads shared slots
/// no warp wrote - so the shape is stated once and each launcher supplies only its grid and
/// its warp count.
pub(super) fn warp_block_launch(
    grid: (u32, u32, u32),
    warps: u32,
    shared_bytes: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: (WARP_SIZE as u32, warps, 1),
        shared_mem_bytes: shared_bytes,
    }
}

/// The launch shape of the block-quantisation kernels: one thread block covers
/// [`CUDA_QUANTIZE_BLOCK_SIZE`] values of a row, `blocks_per_row` of them span the row, and
/// the grid's second axis walks `rows`. Callers that also adapt the block width to the SM
/// count go through [`adaptive_block_grid_x`] first and build their own.
pub fn quantize_launch(blocks_per_row: usize, rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks_per_row as u32, rows, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The query-head count a GQA group implies, once `head_dim` is checked against the block
/// width the attention kernels reduce along.
///
/// Every Q8 and KIVI attention launcher opens with this same check and this same product;
/// [`Q8_0_BLOCK_SIZE`] is the width of the 32-value formats they all read (q8_0, q4_0, and
/// the q8_1 the query is staged as).
///
/// The caller names itself by wrapping the error, not by passing its name as a string: a
/// bare lowercase literal in this module is taken for a kernel name, both by anyone reading
/// and by the test that asks the module for every one of them.
pub(super) fn gqa_query_heads(
    head_dim: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
) -> Result<usize> {
    if !head_dim.is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(anyhow!(
            "head_dim {head_dim} not a multiple of {Q8_0_BLOCK_SIZE}"
        ));
    }
    Ok(n_kv_heads * n_q_per_kv)
}

/// How a Q8_0 KV blob laid out `[position, kv-head, head_dim]` is strided, in blocks.
pub(super) struct KvBlockStrides {
    /// Blocks along one head's head_dim - the direction the block format runs in.
    pub hd_blocks: usize,
    /// Blocks from one position to the next: every kv-head of it.
    pub per_position: usize,
    /// Blocks from one kv-head to the next within a position.
    pub per_kv_head: usize,
}

/// The three strides above, from the two numbers that decide them. Six launchers ask.
pub(super) fn kv_block_strides(head_dim: usize, n_kv_heads: usize) -> KvBlockStrides {
    let hd_blocks = head_dim / Q8_0_BLOCK_SIZE;
    KvBlockStrides {
        hd_blocks,
        per_position: n_kv_heads * hd_blocks,
        per_kv_head: hd_blocks,
    }
}

/// The geometry one row of Q8_0 staging is launched with.
pub(super) struct Q8Staging {
    /// Grid and block, the block width adapted to how much of the card the grid fills.
    pub cfg: LaunchConfig,
    /// The padded row width in values, which the kernels take as an argument.
    pub kx_padded: usize,
    /// Bytes one staged row occupies.
    pub row_bytes: usize,
}

/// Check `k` against the Q8_0 block layout and derive the launch for staging one row of it.
///
/// `k` must divide into whole Q8_0 blocks AND already be a multiple of
/// [`MATRIX_ROW_PADDING`]: otherwise the kernel writes padding zeros past the row's logical
/// end, which is the next token's slot. Callers needing an arbitrary `k` stage into a
/// scratch buffer and copy from there instead; each names itself by wrapping the error.
/// `grid_z` is 1 for a single destination, 2 for the paired K/V form whose z axis picks the
/// side.
pub(super) fn q8_0_row_staging(k: usize, grid_z: u32, dev: &CudaDevice) -> Result<Q8Staging> {
    if !k.is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(anyhow!(
            "k {k} not a multiple of block size {Q8_0_BLOCK_SIZE}"
        ));
    }
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    if kx_padded != k {
        return Err(anyhow!(
            "k {k} must be a multiple of {MATRIX_ROW_PADDING} to skip the padding \
             write; callers needing an arbitrary k must use the staging-then-copy path"
        ));
    }
    let (block_x, grid_x) = adaptive_block_grid_x(
        CUDA_QUANTIZE_BLOCK_SIZE as u32,
        ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE) as u32,
        1,
        dev.multiprocessor_count() as u32,
    );
    Ok(Q8Staging {
        cfg: LaunchConfig {
            grid_dim: (grid_x, 1, grid_z),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        },
        kx_padded,
        row_bytes: (kx_padded / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE,
    })
}

/// Bytes that `rows` rows of `k` F32 values occupy once staged as q8_1.
///
/// The row is padded to [`MATRIX_ROW_PADDING`] before the blocks are counted, because the
/// mat-vec kernels read whole blocks past a row's logical end. How many bytes a block takes
/// is the format's to say, so it is asked rather than written out.
pub(super) fn q8_1_staged_bytes(k: usize, rows: usize) -> usize {
    let q8_1 = GgmlDType::Q8_1;
    rows * (pad(k, MATRIX_ROW_PADDING) / q8_1.block_size()) * q8_1.type_size()
}

/// Mirror of the reference grid/block adaptation: expand the grid (and shrink the
/// block) up to 4x while the grid underfills the SMs, keeping >= 64 threads.
pub(super) fn adaptive_block_grid_x(
    block_x: u32,
    grid_x: u32,
    grid_y: u32,
    sm_count: u32,
) -> (u32, u32) {
    const MIN_BLOCK_X: u32 = 64;
    const MAX_EXPANSION: u32 = 4;
    let mut block_x = block_x;
    let mut grid_x = grid_x;
    let mut expanded = 1u32;
    while grid_x * grid_y < sm_count && block_x > MIN_BLOCK_X && expanded < MAX_EXPANSION {
        block_x /= 2;
        grid_x *= 2;
        expanded *= 2;
    }
    (block_x, grid_x)
}

/// Quantize a single K row and V row to Q8_0, writing into `dst_k`/`dst_v` at
/// `dst_byte_offset`, in one fused launch (gridDim.z=2). Returns the number of
/// bytes written per row. `k` must be a multiple of `MATRIX_ROW_PADDING`.
///
///.
/// the kernel from loken's NVRTC-compiled module instead of the tensor-op
/// PTX blob.
#[allow(clippy::too_many_arguments)]
pub fn quantize_q8_0_kv_paired_into_offset(
    src_k: &CudaView<f32>,
    src_v: &CudaView<f32>,
    dst_k: &mut CudaSlice<u8>,
    dst_v: &mut CudaSlice<u8>,
    dst_byte_offset: usize,
    k: usize,
    dev: &CudaDevice,
) -> Result<usize> {
    // z=2 -> one block group for K, one for V.
    let staging = q8_0_row_staging(k, 2, dev)
        .map_err(|e| anyhow!("quantize_q8_0_kv_paired_into_offset: {e}"))?;
    let bytes_to_write = staging.row_bytes;

    if dst_byte_offset + bytes_to_write > dst_k.len()
        || dst_byte_offset + bytes_to_write > dst_v.len()
    {
        return Err(anyhow!(
            "quantize_q8_0_kv_paired_into_offset: write {bytes_to_write} bytes at offset {dst_byte_offset} but K dst is {} V dst is {}",
            dst_k.len(), dst_v.len()
        ));
    }
    if src_k.len() < k || src_v.len() < k {
        return Err(anyhow!(
            "quantize_q8_0_kv_paired_into_offset: src has K={} V={} elems, expected {k}",
            src_k.len(),
            src_v.len()
        ));
    }

    let func = quantized_kernel(dev, "quantize_q8_0_kv_paired")?;
    let src_k_chunk = src_k.slice(0..k);
    let src_v_chunk = src_v.slice(0..k);

    let mut builder = func.builder();
    builder.arg(&src_k_chunk);
    builder.arg(&src_v_chunk);
    builder.arg(&*dst_k);
    builder.arg(&*dst_v);
    barg!(
        builder,
        k as i32,
        staging.kx_padded as i32,
        dst_byte_offset as i32
    );
    unsafe { builder.launch(staging.cfg) }
        .map_err(|e| anyhow!("launch quantize_q8_0_kv_paired: {e}"))?;
    Ok(bytes_to_write)
}

/// Quantize `ky` rows of `k`-element F32 to Q8_1 into `dst` (row-padded to
/// [`MATRIX_ROW_PADDING`]). Internal helper for the attention launchers.
///
/// One row per step of the grid's second axis, which is bounded - so a taller activation
/// takes as many launches as its rows divided by that bound, each over its own window of
/// the source and the destination.
pub(super) fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let blocks_per_row = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    let dst_row_bytes = q8_1_staged_bytes(k, 1);
    const ROWS_PER_LAUNCH: usize = 65535; // gridDim.y limit
    let func = quantized_kernel(dev, "quantize_q8_1")?;

    for first in (0..ky).step_by(ROWS_PER_LAUNCH) {
        let rows = ROWS_PER_LAUNCH.min(ky - first);
        let src_window = src.slice(first * k..(first + rows) * k);
        let dst_window = dst.slice(first * dst_row_bytes..(first + rows) * dst_row_bytes);

        let (block_x, grid_x) = adaptive_block_grid_x(
            CUDA_QUANTIZE_BLOCK_SIZE as u32,
            blocks_per_row as u32,
            rows as u32,
            dev.multiprocessor_count() as u32,
        );
        let cfg = LaunchConfig {
            grid_dim: (grid_x, rows as u32, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_window);
        builder.arg(&dst_window);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch quantize_q8_1: {e}"))?;
    }
    Ok(())
}

/// GQA Q8-attention scores: `scores[h_q, kv] = scale * dot(Q[h_q], K[kv])` over
/// a Q8_0 K cache, with Q quantized to Q8_1 on the fly. Returns the scores as a
/// A `CudaStorage` (shape `[n_q_heads, n_kv]`), driven from
/// cuda.rs; loads from the loken_quantized module.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_q8_0_q8_1_gqa_scaled(
    k_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_score_q8_0_q8_1_gqa: {e}"))?;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_score_q8_0_q8_1_gqa: q_f32 has {} elements, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let kernel_name = match head_dim {
        64 => "attn_score_q8_0_q8_1_gqa_hd64",
        128 => "attn_score_q8_0_q8_1_gqa_hd128",
        256 => "attn_score_q8_0_q8_1_gqa_hd256",
        512 => "attn_score_q8_0_q8_1_gqa_hd512",
        _ => anyhow::bail!(
            "attn_score_q8_0_q8_1_gqa: unsupported head_dim {head_dim} (have 64/128/256/512)"
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    // Quantize Q to Q8_1 once for all kv-heads.
    let k_padded = pad(head_dim, MATRIX_ROW_PADDING);
    let mut q_q8_1 = unsafe { dev.alloc::<u8>(q8_1_staged_bytes(head_dim, n_q_heads))? };
    quantize_q8_1(q_f32, &mut q_q8_1, head_dim, n_q_heads, dev)?;

    let strides = kv_block_strides(head_dim, n_kv_heads);
    let q_stride_blocks = k_padded / GgmlDType::Q8_1.block_size();

    const WARPS_PER_BLOCK: u32 = 4;
    let cfg = warp_block_launch(
        (
            (n_kv as u32).div_ceil(WARPS_PER_BLOCK),
            n_kv_heads as u32,
            1,
        ),
        WARPS_PER_BLOCK,
        0,
    );

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * n_kv)? };
    let mut builder = func.builder();
    builder.arg(k_blob);
    builder.arg(&q_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        n_kv as i32,
        strides.per_position as i32,
        strides.per_kv_head as i32,
        q_stride_blocks as i32,
        n_q_per_kv as i32,
        scale
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// GQA Q8-attention output: softmax(scores) . V over a Q8_0 V cache, fused into
/// one launch. Returns the per-head outputs as a facade `CudaStorage` (shape
/// `[n_q_heads, head_dim]`).
#[allow(clippy::too_many_arguments)]
pub fn attn_softmax_output_q8_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    scores_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_softmax_output_q8_0_f32_gqa: {e}"))?;
    if scores_f32.len() != n_q_heads * seq_kv {
        anyhow::bail!(
            "attn_softmax_output_q8_0_f32_gqa: scores has {} elements, expected {}",
            scores_f32.len(),
            n_q_heads * seq_kv
        );
    }
    let nq_bucket = if n_q_per_kv <= 1 {
        1
    } else if n_q_per_kv <= 4 {
        4
    } else if n_q_per_kv <= 8 {
        8
    } else {
        anyhow::bail!(
            "attn_softmax_output_q8_0_f32_gqa: n_q_per_kv={n_q_per_kv} > 8 not supported"
        );
    };
    let kernel_name = match (head_dim, nq_bucket) {
        (64, 1) => "attn_softmax_output_q8_0_f32_hd64_nq1",
        (64, 4) => "attn_softmax_output_q8_0_f32_hd64_nq4",
        (64, 8) => "attn_softmax_output_q8_0_f32_hd64_nq8",
        (128, 1) => "attn_softmax_output_q8_0_f32_hd128_nq1",
        (128, 4) => "attn_softmax_output_q8_0_f32_hd128_nq4",
        (128, 8) => "attn_softmax_output_q8_0_f32_hd128_nq8",
        _ => anyhow::bail!("attn_softmax_output_q8_0_f32_gqa: unsupported head_dim={head_dim}"),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    let strides = kv_block_strides(head_dim, n_kv_heads);
    let hd_blocks = strides.hd_blocks;

    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = warp_block_launch((hd_blocks as u32, n_kv_heads as u32, 1), WARPS_PER_BLOCK, 0);

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(scores_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        strides.per_position as i32,
        strides.per_kv_head as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
pub fn attn_score_q4_0_f32_kivi(
    k_blocks: &CudaSlice<u8>,
    k_residual: &CudaSlice<f16>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_score_q4_0_f32_kivi: {e}"))?;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_score_q4_0_f32_kivi: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let full_blocks = seq_kv / 32;
    let total_blocks = seq_kv.div_ceil(32);
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_score_q4_0_f32_kivi_hd64_nq1",
        (64, 2) => "attn_score_q4_0_f32_kivi_hd64_nq2",
        (64, 4) => "attn_score_q4_0_f32_kivi_hd64_nq4",
        (64, 5) => "attn_score_q4_0_f32_kivi_hd64_nq5",
        (64, 8) => "attn_score_q4_0_f32_kivi_hd64_nq8",
        (128, 1) => "attn_score_q4_0_f32_kivi_hd128_nq1",
        (128, 2) => "attn_score_q4_0_f32_kivi_hd128_nq2",
        (128, 4) => "attn_score_q4_0_f32_kivi_hd128_nq4",
        (128, 5) => "attn_score_q4_0_f32_kivi_hd128_nq5",
        (128, 8) => "attn_score_q4_0_f32_kivi_hd128_nq8",
        (256, 1) => "attn_score_q4_0_f32_kivi_hd256_nq1",
        (256, 2) => "attn_score_q4_0_f32_kivi_hd256_nq2",
        (256, 4) => "attn_score_q4_0_f32_kivi_hd256_nq4",
        (256, 5) => "attn_score_q4_0_f32_kivi_hd256_nq5",
        (256, 8) => "attn_score_q4_0_f32_kivi_hd256_nq8",
        (512, 1) => "attn_score_q4_0_f32_kivi_hd512_nq1",
        (512, 2) => "attn_score_q4_0_f32_kivi_hd512_nq2",
        (512, 4) => "attn_score_q4_0_f32_kivi_hd512_nq4",
        (512, 5) => "attn_score_q4_0_f32_kivi_hd512_nq5",
        (512, 8) => "attn_score_q4_0_f32_kivi_hd512_nq8",
        (512, 16) => "attn_score_q4_0_f32_kivi_hd512_nq16",
        _ => anyhow::bail!(
            "attn_score_q4_0_f32_kivi: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv})",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    // Grid = (total_blocks, n_kv_heads). Last block of the x axis is the
    // partial residual block when seq_kv is not a multiple of 32; the
    // kernel handles it via the `sb == full_blocks` path.
    // N_WARPS=16: tested 4/8/16 on qwen3-coder and 16 won.
    let n_warps: u32 = 16;
    let cfg = warp_block_launch(
        (total_blocks as u32, n_kv_heads as u32, 1),
        n_warps,
        (n_warps * WARP_SIZE as u32 * 4) as u32,
    );

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * seq_kv)? };
    let mut builder = func.builder();
    builder.arg(k_blocks);
    builder.arg(k_residual);
    builder.arg(q_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        full_blocks as i32,
        n_kv_heads as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
pub fn attn_score_q4_0_f32_kivi_dev_pos(
    k_blocks: &CudaSlice<u8>,
    k_residual: &CudaSlice<f16>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_score_q4_0_f32_kivi_dev_pos: {e}"))?;
    if !max_seq_padded.is_multiple_of(Q8_0_BLOCK_SIZE) {
        anyhow::bail!(
            "attn_score_q4_0_f32_kivi_dev_pos: max_seq_padded {max_seq_padded} not a multiple of {Q8_0_BLOCK_SIZE}"
        );
    }
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_score_q4_0_f32_kivi_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    // Grid covers max_seq_padded blocks. Kernel writes -INFINITY at
    // positions >= seq_kv_dev[0] so softmax_last_dim masks them out.
    let max_blocks = max_seq_padded / 32;
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64,  1) => "attn_score_q4_0_f32_kivi_dev_pos_hd64_nq1",
        (64,  2) => "attn_score_q4_0_f32_kivi_dev_pos_hd64_nq2",
        (64,  4) => "attn_score_q4_0_f32_kivi_dev_pos_hd64_nq4",
        (64,  5) => "attn_score_q4_0_f32_kivi_dev_pos_hd64_nq5",
        (64,  8) => "attn_score_q4_0_f32_kivi_dev_pos_hd64_nq8",
        (128, 1) => "attn_score_q4_0_f32_kivi_dev_pos_hd128_nq1",
        (128, 2) => "attn_score_q4_0_f32_kivi_dev_pos_hd128_nq2",
        (128, 4) => "attn_score_q4_0_f32_kivi_dev_pos_hd128_nq4",
        (128, 5) => "attn_score_q4_0_f32_kivi_dev_pos_hd128_nq5",
        (128, 8) => "attn_score_q4_0_f32_kivi_dev_pos_hd128_nq8",
        (256, 1) => "attn_score_q4_0_f32_kivi_dev_pos_hd256_nq1",
        (256, 2) => "attn_score_q4_0_f32_kivi_dev_pos_hd256_nq2",
        (256, 4) => "attn_score_q4_0_f32_kivi_dev_pos_hd256_nq4",
        (256, 5) => "attn_score_q4_0_f32_kivi_dev_pos_hd256_nq5",
        (256, 8) => "attn_score_q4_0_f32_kivi_dev_pos_hd256_nq8",
        (512, 1) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq1",
        (512, 2) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq2",
        (512, 4) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq4",
        (512, 5) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq5",
        (512, 8) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq8",
        (512, 16) => "attn_score_q4_0_f32_kivi_dev_pos_hd512_nq16",
        _ => anyhow::bail!(
            "attn_score_q4_0_f32_kivi_dev_pos: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv})",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;
    let n_warps: u32 = 16;
    let cfg = warp_block_launch(
        (max_blocks as u32, n_kv_heads as u32, 1),
        n_warps,
        (n_warps * WARP_SIZE as u32 * 4) as u32,
    );
    // Fresh allocation each call - under CUDA graph capture mode the
    // async pool returns the same address for identically-sized allocs,
    // so the captured kernel argument stays valid across replays.
    let dst = unsafe { dev.alloc::<f32>(n_q_heads * max_seq_padded)? };
    let mut builder = func.builder();
    builder.arg(k_blocks);
    builder.arg(k_residual);
    builder.arg(q_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(
        builder,
        max_seq_padded as i32,
        n_kv_heads as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}
