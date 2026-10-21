//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn attn_output_q4_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        anyhow::bail!("attn_output_q4_0_f32_gqa: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if probs_f32.len() != n_q_heads * seq_kv {
        anyhow::bail!(
            "attn_output_q4_0_f32_gqa: probs has {} elements, expected {}",
            probs_f32.len(),
            n_q_heads * seq_kv
        );
    }

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks;
    let kv_head_stride_blocks = hd_blocks;

    // Monolithic 32-warp path (short ctx)
    let kernel_name = "attn_output_q4_0_f32";
    let func = shape_kernel(dev, kernel_name, head_dim, n_q_per_kv)?;

    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = LaunchConfig {
        grid_dim: (hd_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub fn attn_output_q4_0_f32_dev_pos(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        anyhow::bail!("attn_output_q4_0_f32_dev_pos: head_dim {head_dim} not a multiple of 32");
    }
    if !max_seq_padded.is_multiple_of(32) {
        anyhow::bail!(
            "attn_output_q4_0_f32_dev_pos: max_seq_padded {max_seq_padded} not a multiple of 32"
        );
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if probs_f32.len() != n_q_heads * max_seq_padded {
        anyhow::bail!(
            "attn_output_q4_0_f32_dev_pos: probs has {} elements, expected {} (n_q_heads * max_seq_padded)",
            probs_f32.len(),
            n_q_heads * max_seq_padded
        );
    }

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks;
    let kv_head_stride_blocks = hd_blocks;

    let kernel_name = "attn_output_q4_0_f32_dev_pos";
    let func = shape_kernel(dev, kernel_name, head_dim, n_q_per_kv)?;

    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = LaunchConfig {
        grid_dim: (hd_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };
    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(
        builder,
        max_seq_padded as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub fn attn_softmax_output_q4_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    scores_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        anyhow::bail!("attn_softmax_output_q4_0_f32_gqa: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if scores_f32.len() != n_q_heads * seq_kv {
        anyhow::bail!(
            "attn_softmax_output_q4_0_f32_gqa: scores has {} elements, expected {}",
            scores_f32.len(),
            n_q_heads * seq_kv
        );
    }
    // Pick the smallest compile-time NQ instantiation that fits the
    // runtime n_q_per_kv. The kernel's inner loop has a
    // `if (q < n_q_per_kv)` runtime guard, so dispatching to a larger
    // NQ template instantiation just leaves unused lanes/accumulator
    // slots idle - still correct, no extra memory traffic.
    // Relaxed from exact-match {1,4,8} so models with
    // n_q_per_kv ∈ {2,3,5,6,7} (e.g. qwen2 = 40/8 = 5,
    // deepseek-r1:32b same) actually hit the fused path. Previously
    // those silently fell through to the unfused 2-launch chain.
    let kernel_name = "attn_softmax_output_q4_0_f32";
    let func = shape_kernel(dev, kernel_name, head_dim, n_q_per_kv)?;

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks;
    let kv_head_stride_blocks = hd_blocks;

    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = LaunchConfig {
        grid_dim: (hd_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(scores_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub fn attn_fused_q8_decode_dev_pos(
    k_blob: &CudaSlice<u8>,
    v_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !(head_dim == 64 || head_dim == 128) {
        anyhow::bail!("attn_fused_q8_decode_dev_pos: head_dim {head_dim} not in {{64, 128}}");
    }
    let n_q_heads = n_kv_heads; // n_q_per_kv = 1
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_fused_q8_decode_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    // v3: 16-warp design - each warp owns 1/16 of seq_kv with local
    // (m, s, VKQ); cross-warp reduction at block end. Better GPU
    // saturation than v2 (single-warp) at the cost of one __syncthreads
    // at the end.
    let kernel_name = match head_dim {
        64 => "attn_fused_q8_decode_dev_pos_v3_hd64",
        128 => "attn_fused_q8_decode_dev_pos_v3_hd128",
        _ => unreachable!(),
    };
    let func = dev
        .get_or_load_custom_func(kernel_name, "loken_quantized", get_quantized_ptx(dev)?)
        .map_err(|e| anyhow!("load kernel: {e}"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, n_q_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, 16, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(k_blob);
    builder.arg(v_blob);
    builder.arg(q_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(builder, max_seq_padded as i32, n_kv_heads as i32, scale);
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// SPLIT-K flash-decode for Q8_0 KV (dev_pos), loaded from loken_quantized.
/// Phase-1: `(NSPLIT, n_heads)` blocks x 1 warp each compute online-softmax
/// partials over a `ceil(seq_kv/NSPLIT)`-token KV chunk; phase-2 flash-merges
/// the splits per head. Same output as `attn_fused_q8_decode_dev_pos`
/// (`[n_q_heads, head_dim]` F32) but far more parallel - built to beat the
/// 2-kernel score+softmax_output chain on long-KV decode. `NSPLIT` MUST equal
/// `FLASH_SPLITK_NSPLIT` in cuda/quantized.cu (16).
pub fn attn_flash_splitk_q8_decode_dev_pos(
    k_blob: &CudaSlice<u8>,
    v_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    n_kv_heads: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    const NSPLIT: usize = 16;
    if !(head_dim == 64 || head_dim == 128) {
        anyhow::bail!("attn_flash_splitk_q8_decode_dev_pos: head_dim {head_dim} not in {{64,128}}");
    }
    let n_q_heads = n_kv_heads; // n_q_per_kv = 1
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_flash_splitk_q8_decode_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let (partial_name, combine_name) = match head_dim {
        64 => (
            "flash_splitk_q8_partial_hd64",
            "flash_splitk_q8_combine_hd64",
        ),
        128 => (
            "flash_splitk_q8_partial_hd128",
            "flash_splitk_q8_combine_hd128",
        ),
        _ => unreachable!(),
    };
    let ptx = get_quantized_ptx(dev)?;
    let partial_func = dev
        .get_or_load_custom_func(partial_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {partial_name}: {e}"))?;
    let combine_func = dev
        .get_or_load_custom_func(combine_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {combine_name}: {e}"))?;

    let partials = unsafe { dev.alloc::<f32>(n_q_heads * NSPLIT * (head_dim + 2))? };

    let cfg1 = LaunchConfig {
        grid_dim: (NSPLIT as u32, n_q_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b1 = partial_func.builder();
    b1.arg(k_blob);
    b1.arg(v_blob);
    b1.arg(q_f32);
    b1.arg(&partials);
    b1.arg(seq_kv_dev);
    barg!(b1, n_kv_heads as i32, scale);
    unsafe { b1.launch(cfg1) }.map_err(|e| anyhow!("launch partial: {e}"))?;

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let cfg2 = LaunchConfig {
        grid_dim: (n_q_heads as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = combine_func.builder();
    b2.arg(&partials);
    b2.arg(&dst);
    unsafe { b2.launch(cfg2) }.map_err(|e| anyhow!("launch combine: {e}"))?;

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Split-count schedule of the GQA split-K decode. GQA has few KV heads (4-8), so the partial
/// grid is `nsplit x n_kv_heads` single warps, memory-bound on the KV read: `base` keeps
/// about `TARGET_WARPS` in flight at moderate context, and past it the count grows with the
/// sequence so each split covers about `CHUNK` keys (more warps in flight, more bandwidth).
/// The wide combine keeps up with the larger counts.
pub struct GqaSplits {
    pub base: usize,
}

impl GqaSplits {
    /// Warps the partial kernel keeps in flight at moderate context.
    const TARGET_WARPS: usize = 384;
    /// Keys per split once the sequence outgrows `base`.
    pub const CHUNK: usize = 8;
    pub const MIN: usize = 8;
    /// Largest `base`: the moderate-context count never exceeds it.
    const BASE_MAX: usize = 96;
    /// Largest count, and the partials buffer's per-head capacity.
    pub const MAX: usize = 256;

    pub fn for_kv_heads(n_kv_heads: usize) -> Self {
        Self {
            base: (Self::TARGET_WARPS / n_kv_heads.max(1)).clamp(Self::MIN, Self::BASE_MAX),
        }
    }
}

/// Splits of a `seq_kv`-key decode over `n_kv_heads`; `flash_splitk_nsplit` in
/// `flash_splitk.cu` computes the same from the device length.
pub fn gqa_nsplit(seq_kv: usize, n_kv_heads: usize) -> usize {
    GqaSplits::for_kv_heads(n_kv_heads)
        .base
        .max(seq_kv.div_ceil(GqaSplits::CHUNK))
        .clamp(GqaSplits::MIN, GqaSplits::MAX)
        .min(seq_kv.max(1))
}

/// GQA sibling of `attn_flash_splitk_q8_decode_dev_pos`. Fused split-K
/// flash-decode where each (split, h_kv) warp reads its KV head's chunk ONCE
/// and feeds all `n_q_per_kv` query heads - so KV bandwidth (the long-context
/// bottleneck) is read n_kv_headsx instead of n_q_headsx, unlike the 2-kernel
/// `attn_score`+`attn_softmax_output_gqa` chain it replaces (no HBM scores
/// round-trip, NSPLITx more blocks than the single-warp-per-head fused path).
/// The combine kernel is the same per-query-head merge as the MHA variant.
/// `q_f32` is `[n_q_heads, head_dim]` (n_q_heads = n_kv_heads x n_q_per_kv).
///
/// The kernels take their split count from the length they read on the device
/// (`gqa_nsplit`), so an eager step and a graph replay at the same length split alike.
/// `seq_kv_hint` is the longest length the launch may serve: it sizes the grid.
///
/// `rows` queries, `[rows, n_q_heads, head_dim]`, attend in one launch pair: row `r` is the
/// token at the device position plus `r`, reads `r` more keys, and gets exactly the output a
/// one-row launch at its position gives.
#[allow(clippy::too_many_arguments)]
pub fn attn_flash_splitk_q8_gqa_decode_dev_pos(
    k_blob: &CudaSlice<u8>,
    v_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    seq_kv_hint: usize,
    rows: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    // The widest query-head group the devlen partial kernels are built for (their MAXNQ):
    // up to 8 by one build, up to 16 by the `nq16` one.
    const MAX_NQ: usize = 16;
    const NARROW_NQ: usize = 8;
    if !(head_dim == 64 || head_dim == 128 || head_dim == 256) {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: head_dim {head_dim} not in {{64,128,256}}"
        );
    }
    if n_q_per_kv == 0 || n_q_per_kv > MAX_NQ {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: n_q_per_kv {n_q_per_kv} not in 1..={MAX_NQ}"
        );
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if rows == 0 || q_f32.len() != rows * n_q_heads * head_dim {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: Q has {} elems, expected {rows} x {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let nsplit = gqa_nsplit(seq_kv_hint, n_kv_heads);
    let wide = n_q_per_kv > NARROW_NQ;
    let (partial_name, combine_name) = match (head_dim, wide) {
        (64, false) => (
            "flash_splitk_q8_gqa_partial_devlen_hd64",
            "flash_splitk_gqa_combine_wide_devlen_hd64",
        ),
        (64, true) => (
            "flash_splitk_q8_gqa_partial_devlen_nq16_hd64",
            "flash_splitk_gqa_combine_wide_devlen_hd64",
        ),
        (128, false) => (
            "flash_splitk_q8_gqa_partial_devlen_hd128",
            "flash_splitk_gqa_combine_wide_devlen_hd128",
        ),
        (128, true) => (
            "flash_splitk_q8_gqa_partial_devlen_nq16_hd128",
            "flash_splitk_gqa_combine_wide_devlen_hd128",
        ),
        (256, false) => (
            "flash_splitk_q8_gqa_partial_devlen_hd256",
            "flash_splitk_gqa_combine_wide_devlen_hd256",
        ),
        (256, true) => anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: head_dim 256 takes up to {NARROW_NQ} query heads per KV head"
        ),
        _ => unreachable!(),
    };
    let ptx = get_quantized_ptx(dev)?;
    let partial_func = dev
        .get_or_load_custom_func(partial_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {partial_name}: {e}"))?;
    let combine_func = dev
        .get_or_load_custom_func(combine_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {combine_name}: {e}"))?;

    // Allocate partials at the CONSTANT upper bound (not the runtime nsplit) so the
    // per-decode-token alloc size never changes as seq_kv (hence nsplit) grows  -
    // the async mempool then reuses one block instead of fragmenting. The kernels
    // index by the device length's split count (<= GqaSplits::MAX), using only the
    // live prefix.
    debug_assert!(nsplit <= GqaSplits::MAX);
    let partials = unsafe { dev.alloc::<f32>(rows * n_q_heads * GqaSplits::MAX * (head_dim + 2))? };

    // Partial grid is (splits of the longest length served, n_kv_heads, rows): each warp
    // serves n_q_per_kv query heads, and the blocks past the device length's count leave.
    let cfg1 = LaunchConfig {
        grid_dim: (nsplit as u32, n_kv_heads as u32, rows as u32),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b1 = partial_func.builder();
    b1.arg(k_blob);
    b1.arg(v_blob);
    b1.arg(q_f32);
    b1.arg(&partials);
    b1.arg(seq_kv_dev);
    let schedule = GqaSplits::for_kv_heads(n_kv_heads);
    barg!(
        b1,
        n_kv_heads as i32,
        n_q_per_kv as i32,
        schedule.base as i32,
        GqaSplits::CHUNK as i32,
        GqaSplits::MIN as i32,
        GqaSplits::MAX as i32,
        scale
    );
    unsafe { b1.launch(cfg1) }.map_err(|e| anyhow!("launch gqa partial: {e}"))?;

    // WIDE combine: one block per (query head, 32-dim slice) x 8 warps (must
    // match FLASH_SPLITK_COMBINE_WARPS). The per-head grid of the old combine
    // couldn't keep up with the chunk-8 nsplit (nsys: 5.8 µs vs
    // 2.5 µs); HD_BLOCKSx the blocks restores the balance.
    let dst = unsafe { dev.alloc::<f32>(rows * n_q_heads * head_dim)? };
    let cfg2 = LaunchConfig {
        grid_dim: (n_q_heads as u32, (head_dim / 32) as u32, rows as u32),
        block_dim: (WARP_SIZE as u32, 8, 1),
        shared_mem_bytes: 0,
    };
    // One row's output also leaves quantised, for the projection that reads it.
    let q81 =
        if rows == 1 && crate::tensor::cuda::prequant_row(dev.native(), 1, n_q_heads * head_dim) {
            Some(unsafe { dev.alloc::<u8>(n_q_heads * head_dim / 32 * 36)? })
        } else {
            None
        };
    let mut b2 = combine_func.builder();
    b2.arg(&partials);
    b2.arg(&dst);
    b2.arg(seq_kv_dev);
    barg!(
        b2,
        schedule.base as i32,
        GqaSplits::CHUNK as i32,
        GqaSplits::MIN as i32,
        GqaSplits::MAX as i32
    );
    match q81.as_ref() {
        Some(q) => b2.arg(q),
        None => b2.arg(&0u64),
    };
    unsafe { b2.launch(cfg2) }.map_err(|e| anyhow!("launch gqa combine: {e}"))?;
    if let Some(q) = q81 {
        crate::tensor::cuda::stash_prequant(&dev.cuda_stream(), &dst, n_q_heads * head_dim, q);
    }

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Q4_0 KIVI sibling of `attn_flash_splitk_q8_gqa_decode_dev_pos`. One fused
/// pass over the qwen3moe Q4 KV cache (per-channel KIVI K + F16 residual,
/// per-token Q4_0 V) replacing the 2-kernel attn_score(->HBM scores)+
/// attn_softmax_output chain - no HBM scores round-trip, adaptive nsplit for
/// occupancy. seq_kv is a host int (Q4 path is not graph-captured). Reuses the
/// runtime-nsplit gqa combine. `q_f32` is `[n_q_heads, head_dim]`.
pub fn attn_flash_splitk_q4_gqa_decode(
    k_blocks: &CudaSlice<u8>,
    k_residual: &CudaSlice<f16>,
    v_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    seq_kv: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    const MAX_NQ: usize = 8;
    if !(head_dim == 64 || head_dim == 128) {
        anyhow::bail!("attn_flash_splitk_q4_gqa_decode: head_dim {head_dim} not in {{64,128}}");
    }
    if n_q_per_kv == 0 || n_q_per_kv > MAX_NQ {
        anyhow::bail!(
            "attn_flash_splitk_q4_gqa_decode: n_q_per_kv {n_q_per_kv} not in 1..={MAX_NQ}"
        );
    }
    if seq_kv == 0 {
        anyhow::bail!("attn_flash_splitk_q4_gqa_decode: empty cache");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_flash_splitk_q4_gqa_decode: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let full_blocks = seq_kv / 32;
    // Enough splits to fill the device without cutting the KV so fine that the combine pass
    // costs more than the partials save; 384 is the working set one SM holds.
    let nsplit: usize = (384 / n_kv_heads.max(1)).clamp(8, 96).min(seq_kv.max(1));
    let (partial_name, combine_name) = match head_dim {
        64 => (
            "flash_splitk_q4_gqa_partial_hd64",
            "flash_splitk_gqa_combine_hd64",
        ),
        128 => (
            "flash_splitk_q4_gqa_partial_hd128",
            "flash_splitk_gqa_combine_hd128",
        ),
        _ => unreachable!(),
    };
    let ptx = get_quantized_ptx(dev)?;
    let partial_func = dev
        .get_or_load_custom_func(partial_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {partial_name}: {e}"))?;
    let combine_func = dev
        .get_or_load_custom_func(combine_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {combine_name}: {e}"))?;

    let partials = unsafe { dev.alloc::<f32>(n_q_heads * nsplit * (head_dim + 2))? };

    let cfg1 = LaunchConfig {
        grid_dim: (nsplit as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b1 = partial_func.builder();
    b1.arg(k_blocks);
    b1.arg(k_residual);
    b1.arg(v_blob);
    b1.arg(q_f32);
    b1.arg(&partials);
    barg!(
        b1,
        seq_kv as i32,
        full_blocks as i32,
        n_kv_heads as i32,
        n_q_per_kv as i32,
        nsplit as i32,
        scale
    );
    unsafe { b1.launch(cfg1) }.map_err(|e| anyhow!("launch q4 gqa partial: {e}"))?;

    // 8 warps (must match FLASH_SPLITK_COMBINE_WARPS) - shares the same combine
    // kernel as the Q8 GQA path, which now parallelizes the nsplit reduction.
    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let cfg2 = LaunchConfig {
        grid_dim: (n_q_heads as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, 8, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = combine_func.builder();
    b2.arg(&partials);
    b2.arg(&dst);
    barg!(b2, nsplit as i32);
    unsafe { b2.launch(cfg2) }.map_err(|e| anyhow!("launch q4 gqa combine: {e}"))?;

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Device-position (graph-capturable) sibling of `attn_flash_splitk_q4_gqa_decode`.
/// Reads seq_kv from `seq_kv_dev` (= cur_pos_dev, value cur_seq_len-1) so it can
/// run inside the captured decode graph, replacing the 2-kernel
/// attn_score(->HBM scores)+attn_softmax_output chain on the qwen3 Q4 path.
/// Mirrors `attn_flash_splitk_q8_gqa_decode_dev_pos` (adaptive nsplit, constant
/// MAX_NSPLIT partials alloc, shared runtime-nsplit combine).
pub fn attn_flash_splitk_q4_gqa_decode_dev_pos(
    k_blocks: &CudaSlice<u8>,
    k_residual: &CudaSlice<f16>,
    v_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    seq_kv_hint: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    const MAX_NQ: usize = 8;
    if !(head_dim == 64 || head_dim == 128) {
        anyhow::bail!(
            "attn_flash_splitk_q4_gqa_decode_dev_pos: head_dim {head_dim} not in {{64,128}}"
        );
    }
    if n_q_per_kv == 0 || n_q_per_kv > MAX_NQ {
        anyhow::bail!(
            "attn_flash_splitk_q4_gqa_decode_dev_pos: n_q_per_kv {n_q_per_kv} not in 1..={MAX_NQ}"
        );
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_flash_splitk_q4_gqa_decode_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    // Adaptive split count (same rationale as the Q8 dev-pos path): grow nsplit
    // with seq_kv so each warp scans ~24 tokens -> more memory-level parallelism.
    let base = (384 / n_kv_heads.max(1)).clamp(8, 96);
    let by_len = seq_kv_hint.div_ceil(24);
    // nsplit is perf-invariant for qwen3 @2.5K (swept 48..256 -> all 125.7 tok/s):
    // the kernel is per-warp dequant+MAC bound, not occupancy/BW
    // bound, so more splits don't help. Keep the adaptive value (it still helps
    // the very-long-context occupancy floor on other shapes).
    let nsplit: usize = base.max(by_len).clamp(8, 256).min(seq_kv_hint.max(1));
    let (partial_name, combine_name) = match head_dim {
        64 => (
            "flash_splitk_q4_gqa_partial_devpos_hd64",
            "flash_splitk_gqa_combine_hd64",
        ),
        128 => (
            "flash_splitk_q4_gqa_partial_devpos_hd128",
            "flash_splitk_gqa_combine_hd128",
        ),
        _ => unreachable!(),
    };
    let ptx = get_quantized_ptx(dev)?;
    let partial_func = dev
        .get_or_load_custom_func(partial_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {partial_name}: {e}"))?;
    let combine_func = dev
        .get_or_load_custom_func(combine_name, "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel {combine_name}: {e}"))?;

    debug_assert!(nsplit <= GqaSplits::MAX);
    let partials = unsafe { dev.alloc::<f32>(n_q_heads * GqaSplits::MAX * (head_dim + 2))? };

    let cfg1 = LaunchConfig {
        grid_dim: (nsplit as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b1 = partial_func.builder();
    b1.arg(k_blocks);
    b1.arg(k_residual);
    b1.arg(v_blob);
    b1.arg(q_f32);
    b1.arg(&partials);
    b1.arg(seq_kv_dev);
    barg!(
        b1,
        n_kv_heads as i32,
        n_q_per_kv as i32,
        nsplit as i32,
        scale
    );
    unsafe { b1.launch(cfg1) }.map_err(|e| anyhow!("launch q4 gqa partial dev_pos: {e}"))?;

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let cfg2 = LaunchConfig {
        grid_dim: (n_q_heads as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, 8, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = combine_func.builder();
    b2.arg(&partials);
    b2.arg(&dst);
    barg!(b2, nsplit as i32);
    unsafe { b2.launch(cfg2) }.map_err(|e| anyhow!("launch q4 gqa combine dev_pos: {e}"))?;

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub fn dequantize_q8_0_blob_f16(
    data: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    super::dequantize_block32_blob("q8_0", data, elem_count, dev, false)
}

/// Fused quantized GEMV: `dst[nrows] = W(quantized) . y(q8_1)`, driving the
/// `mmvq_gguf_<tag>_f32_plain_cuda1` kernel directly (batch size 1). Relocated
/// from the reference cuda.rs; reads the weight buffer via the public
/// `QCudaStorage::weight_cuda_slice` accessor and loads the kernel from the
/// loken_mmvq module. Default-stream wrapper around the `_on_stream` form.
pub fn mvq_plain_any_via_shared_q8_1(
    qstor: &crate::tensor::quantized::QCudaStorage,
    q8_1_buf: &CudaSlice<u8>,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    mvq_plain_any_via_shared_q8_1_on_stream(qstor, q8_1_buf, ncols, nrows, dev, &dev.cuda_stream())
}

/// Stream-explicit form of [`mvq_plain_any_via_shared_q8_1`].
pub fn mvq_plain_any_via_shared_q8_1_on_stream(
    qstor: &crate::tensor::quantized::QCudaStorage,
    q8_1_buf: &CudaSlice<u8>,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
    stream: &std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
) -> Result<CudaStorage> {
    let tag = match qstor.dtype() {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8_0 => "q8_0",
        other => anyhow::bail!("mvq_plain_any_via_shared_q8_1: unsupported dtype {other:?}"),
    };
    let kernel_name = format!("mmvq_gguf_{tag}_f32_plain_cuda1");
    let ptx = get_mmvq_ptx(dev)?;
    let func = dev
        .get_or_load_custom_func(&kernel_name, "loken_mmvq", ptx)
        .map_err(|e| anyhow!("load {kernel_name}: {e}"))?;

    const FAST_MMVQ_Q8_1_BLOCK_SIZE: usize = 32;
    let k_padded = pad(ncols, MATRIX_ROW_PADDING);
    let stride_col_y = (k_padded / FAST_MMVQ_Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;

    // Allocate dst on the SAME stream as the kernel to avoid a cross-stream dep.
    // SAFETY: dst is freshly allocated and fully written by the kernel below.
    let dst = unsafe { stream.alloc::<f32>(nrows) }.map_err(|e| anyhow!("mvq dst alloc: {e}"))?;

    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };
    let ncols_i = ncols as i32;
    let nrows_i = nrows as i32;
    let weight = qstor.weight_cuda_slice();
    let mut builder = stream.launch_builder(&func);
    builder.arg(weight);
    builder.arg(q8_1_buf);
    builder.arg(&dst);
    builder.arg(&ncols_i);
    builder.arg(&nrows_i);
    builder.arg(&stride_col_y);
    builder.arg(&stride_col_dst);
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Quantize `k` F32 elements to the Q8_1 layout the fused mvq kernels read
/// (row-padded to MATRIX_ROW_PADDING). Drives
/// `mmvq_gguf_quantize_q8_1_f32` from the loken_mmvq module. The `_on_stream`
/// form gains a `dev` param (needed to load the kernel; the substrate reaches it via
/// the FFI launcher).
pub fn quantize_q8_1_fast_mmvq_f32(
    src: &CudaView<f32>,
    out_q8_1: &mut CudaSlice<u8>,
    k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    quantize_q8_1_fast_mmvq_f32_on_stream(src, out_q8_1, k, dev, &dev.cuda_stream())
}

/// Stream-explicit form of [`quantize_q8_1_fast_mmvq_f32`].
pub fn quantize_q8_1_fast_mmvq_f32_on_stream(
    src: &CudaView<f32>,
    out_q8_1: &mut CudaSlice<u8>,
    k: usize,
    dev: &CudaDevice,
    stream: &std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
) -> Result<()> {
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_x = ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    let ptx = get_mmvq_ptx(dev)?;
    let func = dev
        .get_or_load_custom_func("mmvq_gguf_quantize_q8_1_f32", "loken_mmvq", ptx)
        .map_err(|e| anyhow!("load mmvq_gguf_quantize_q8_1_f32: {e}"))?;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks_x as u32, 1, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_i = k as i32;
    let kp_i = k_padded as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(src);
    builder.arg(&*out_q8_1);
    builder.arg(&k_i);
    builder.arg(&kp_i);
    unsafe { builder.launch(cfg) }
        .map_err(|e| anyhow!("launch mmvq_gguf_quantize_q8_1_f32: {e}"))?;
    Ok(())
}

/// Multi-row F32 -> Q8_1 quantize for the tensor-core (IMMA) MoE/dense path.
///
/// Unlike [`quantize_q8_1_fast_mmvq_f32`] this does NOT pad the row to
/// `MATRIX_ROW_PADDING`: the IMMA kernels require `k % 256 == 0` and consume a
/// contiguous `num_rows x (k/32)` array of `block_q8_1`, so `kx_padded == k`.
/// Each `block_q8_1` is 36 bytes (32 x i8 + 2 x f16 scale/sum), giving an
/// output of `num_rows * (k / 32) * 36` bytes.
pub fn quantize_q8_1_mmvq_multirow_f32(
    src: &CudaView<f32>,
    out_q8_1: &mut CudaSlice<u8>,
    k: usize,
    num_rows: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let num_blocks_x = ceil_div(k, CUDA_QUANTIZE_BLOCK_SIZE);
    let ptx = get_mmvq_ptx(dev)?;
    let func = dev
        .get_or_load_custom_func("mmvq_gguf_quantize_q8_1_f32", "loken_mmvq", ptx)
        .map_err(|e| anyhow!("load mmvq_gguf_quantize_q8_1_f32: {e}"))?;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks_x as u32, num_rows as u32, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_i = k as i32;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(src);
    builder.arg(&*out_q8_1);
    builder.arg(&k_i);
    builder.arg(&k_i); // kx_padded == k (IMMA requires k % 256 == 0)
    unsafe { builder.launch(cfg) }
        .map_err(|e| anyhow!("launch mmvq_gguf_quantize_q8_1_f32 (multirow): {e}"))?;
    Ok(())
}

/// The softmax of one band of attention scores, in one pass.
///
/// `scores` is the band as the matmul left it, half precision and contiguous, read as
/// `rows x cols`. `run_max` is each row's largest score from the bands before this one, so
/// that the weights come out on the scale the caller accumulates at; the first band passes
/// negative infinity. Returns the weights, again half precision and the same shape, with
/// each row's new maximum and the sum of its weights.
///
/// The causal mask is an index comparison here rather than a tensor: `c0 + j` is the key a
/// column stands for, and `past + row % seq` is the last key that row may see.
///
/// `Err` for anything not on this path; the caller keeps the operations it would otherwise
/// have chained.
pub fn band_softmax_f16(
    scores: &crate::tensor::Tensor,
    run_max: &crate::tensor::Tensor,
    rows: usize,
    cols: usize,
    seq: usize,
    past: usize,
    c0: usize,
) -> Result<(
    crate::tensor::Tensor,
    crate::tensor::Tensor,
    crate::tensor::Tensor,
)> {
    use crate::tensor::cuda_ext::tensor_from_cuda_storage;
    use crate::tensor::{DType, StorageView};
    if scores.dtype() != DType::F16 {
        anyhow::bail!("band_softmax_f16: scores must be F16");
    }
    if run_max.dtype() != DType::F32 || run_max.elem_count() != rows {
        anyhow::bail!("band_softmax_f16: run_max must be {rows} F32 values");
    }
    if scores.elem_count() != rows * cols || seq == 0 {
        anyhow::bail!("band_softmax_f16: {rows} x {cols} does not match the scores");
    }
    let dev = scores.device().as_cuda_device()?;
    let whole = |t: &crate::tensor::Tensor| {
        let (_, layout) = t.storage_and_layout();
        let (from, to) = layout.contiguous_offsets();
        layout.start_offset() == 0 && from == 0 && to == t.elem_count()
    };
    if !whole(scores) || !whole(run_max) {
        anyhow::bail!("band_softmax_f16: scores and run_max must be contiguous from their start");
    }
    let (scores_storage, _) = scores.storage_and_layout();
    let (run_max_storage, _) = run_max.storage_and_layout();
    let scores_slice = match &*scores_storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?.slice(..),
        _ => anyhow::bail!("band_softmax_f16: scores must be on CUDA"),
    };
    let run_max_slice = match &*run_max_storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?.slice(..),
        _ => anyhow::bail!("band_softmax_f16: run_max must be on CUDA"),
    };

    let ptx = get_quantized_ptx(&dev)?;
    let func = dev
        .get_or_load_custom_func("band_softmax_f16", "loken_quantized", ptx)
        .map_err(|e| anyhow!("load kernel band_softmax_f16: {e}"))?;
    let weights = unsafe { dev.alloc::<half::f16>(rows * cols)? };
    let new_max = unsafe { dev.alloc::<f32>(rows)? };
    let band_sum = unsafe { dev.alloc::<f32>(rows)? };

    // One block per row, so a row's two reductions stay inside one block.
    const THREADS: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = func.builder();
    b.arg(&scores_slice);
    b.arg(&run_max_slice);
    b.arg(&weights);
    b.arg(&new_max);
    b.arg(&band_sum);
    barg!(
        b,
        rows as i32,
        cols as i32,
        seq as i32,
        past as i32,
        c0 as i32
    );
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch band_softmax_f16: {e}"))?;

    let shape = scores.dims().to_vec();
    let weights =
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(weights, dev.clone()), shape)
            .map_err(|e| anyhow!("band_softmax_f16 weights: {e}"))?;
    let new_max =
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(new_max, dev.clone()), rows)
            .map_err(|e| anyhow!("band_softmax_f16 max: {e}"))?;
    let band_sum =
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(band_sum, dev.clone()), rows)
            .map_err(|e| anyhow!("band_softmax_f16 sum: {e}"))?;
    Ok((weights, new_max, band_sum))
}
