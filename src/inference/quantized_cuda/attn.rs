//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

#[allow(clippy::too_many_arguments)]
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

    // Split-K path: distributing the seq reduction
    // across more blocks does NOT speed up this kernel because it is
    // memory-bandwidth bound (reading the V cache once dominates),
    // not parallelism-limited. Same wall-time at seq_kv=16k. Kept
    // as code for future tuning; permanently off in production.
    let splitk_threshold = usize::MAX;

    if seq_kv >= splitk_threshold {
        // Split-K path
        let kernel_name = match (head_dim, n_q_per_kv) {
            (64, 1) => "attn_output_q4_0_f32_splitk_hd64_nq1",
            (64, 4) => "attn_output_q4_0_f32_splitk_hd64_nq4",
            (64, 8) => "attn_output_q4_0_f32_splitk_hd64_nq8",
            (128, 1) => "attn_output_q4_0_f32_splitk_hd128_nq1",
            (128, 4) => "attn_output_q4_0_f32_splitk_hd128_nq4",
            (128, 8) => "attn_output_q4_0_f32_splitk_hd128_nq8",
            (256, 1) => "attn_output_q4_0_f32_splitk_hd256_nq1",
            (256, 4) => "attn_output_q4_0_f32_splitk_hd256_nq4",
            (256, 8) => "attn_output_q4_0_f32_splitk_hd256_nq8",
            _ => anyhow::bail!(
                "attn_output_q4_0_f32_gqa: unsupported splitk combo head_dim={head_dim} n_q_per_kv={n_q_per_kv}",
            ),
        };
        let func = dev
            .get_or_load_custom_func(kernel_name, "loken_quantized", get_quantized_ptx(dev)?)
            .map_err(|e| anyhow!("load kernel: {e}"))?;

        let seq_per_split: usize = 256;
        let s = (seq_kv + seq_per_split - 1) / seq_per_split;

        const SPLITK_WARPS_PER_BLOCK: u32 = 8;
        let cfg = LaunchConfig {
            grid_dim: (hd_blocks as u32, n_kv_heads as u32, s as u32),
            block_dim: (WARP_SIZE as u32, SPLITK_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };

        // CRITICAL: split-K kernel atomicAdd's its partial; output must be zeroed.
        let dst = dev.alloc_zeros::<f32>(n_q_heads * head_dim)?;
        let mut builder = func.builder();
        builder.arg(v_blob);
        builder.arg(probs_f32);
        builder.arg(&dst);
        barg!(
            builder,
            seq_kv as i32,
            n_kv_stride_blocks as i32,
            kv_head_stride_blocks as i32,
            seq_per_split as i32
        );
        unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }

    // Monolithic 32-warp path (short ctx)
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_output_q4_0_f32_hd64_nq1",
        (64, 2) => "attn_output_q4_0_f32_hd64_nq2",
        (64, 4) => "attn_output_q4_0_f32_hd64_nq4",
        (64, 5) => "attn_output_q4_0_f32_hd64_nq5",
        (64, 8) => "attn_output_q4_0_f32_hd64_nq8",
        (128, 1) => "attn_output_q4_0_f32_hd128_nq1",
        (128, 2) => "attn_output_q4_0_f32_hd128_nq2",
        (128, 4) => "attn_output_q4_0_f32_hd128_nq4",
        (128, 5) => "attn_output_q4_0_f32_hd128_nq5",
        (128, 8) => "attn_output_q4_0_f32_hd128_nq8",
        (256, 1) => "attn_output_q4_0_f32_hd256_nq1",
        (256, 2) => "attn_output_q4_0_f32_hd256_nq2",
        (256, 4) => "attn_output_q4_0_f32_hd256_nq4",
        (256, 5) => "attn_output_q4_0_f32_hd256_nq5",
        (256, 8) => "attn_output_q4_0_f32_hd256_nq8",
        (512, 1) => "attn_output_q4_0_f32_hd512_nq1",
        (512, 2) => "attn_output_q4_0_f32_hd512_nq2",
        (512, 4) => "attn_output_q4_0_f32_hd512_nq4",
        (512, 5) => "attn_output_q4_0_f32_hd512_nq5",
        (512, 8) => "attn_output_q4_0_f32_hd512_nq8",
        (512, 16) => "attn_output_q4_0_f32_hd512_nq16",
        _ => anyhow::bail!(
            "attn_output_q4_0_f32_gqa: unsupported combo head_dim={head_dim} n_q_per_kv={n_q_per_kv}",
        ),
    };
    let func = dev
        .get_or_load_custom_func(kernel_name, "loken_quantized", get_quantized_ptx(dev)?)
        .map_err(|e| anyhow!("load kernel: {e}"))?;

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

#[allow(clippy::too_many_arguments)]
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

    let kernel_name = match (head_dim, n_q_per_kv) {
        (64,  1) => "attn_output_q4_0_f32_dev_pos_hd64_nq1",
        (64,  2) => "attn_output_q4_0_f32_dev_pos_hd64_nq2",
        (64,  4) => "attn_output_q4_0_f32_dev_pos_hd64_nq4",
        (64,  5) => "attn_output_q4_0_f32_dev_pos_hd64_nq5",
        (64,  8) => "attn_output_q4_0_f32_dev_pos_hd64_nq8",
        (128, 1) => "attn_output_q4_0_f32_dev_pos_hd128_nq1",
        (128, 2) => "attn_output_q4_0_f32_dev_pos_hd128_nq2",
        (128, 4) => "attn_output_q4_0_f32_dev_pos_hd128_nq4",
        (128, 5) => "attn_output_q4_0_f32_dev_pos_hd128_nq5",
        (128, 8) => "attn_output_q4_0_f32_dev_pos_hd128_nq8",
        (256, 1) => "attn_output_q4_0_f32_dev_pos_hd256_nq1",
        (256, 2) => "attn_output_q4_0_f32_dev_pos_hd256_nq2",
        (256, 4) => "attn_output_q4_0_f32_dev_pos_hd256_nq4",
        (256, 5) => "attn_output_q4_0_f32_dev_pos_hd256_nq5",
        (256, 8) => "attn_output_q4_0_f32_dev_pos_hd256_nq8",
        (512, 1) => "attn_output_q4_0_f32_dev_pos_hd512_nq1",
        (512, 2) => "attn_output_q4_0_f32_dev_pos_hd512_nq2",
        (512, 4) => "attn_output_q4_0_f32_dev_pos_hd512_nq4",
        (512, 5) => "attn_output_q4_0_f32_dev_pos_hd512_nq5",
        (512, 8) => "attn_output_q4_0_f32_dev_pos_hd512_nq8",
        (512, 16) => "attn_output_q4_0_f32_dev_pos_hd512_nq16",
        _ => anyhow::bail!(
            "attn_output_q4_0_f32_dev_pos: unsupported combo head_dim={head_dim} n_q_per_kv={n_q_per_kv}",
        ),
    };
    let func = dev
        .get_or_load_custom_func(kernel_name, "loken_quantized", get_quantized_ptx(dev)?)
        .map_err(|e| anyhow!("load kernel: {e}"))?;

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

#[allow(clippy::too_many_arguments)]
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
    let nq_bucket = if n_q_per_kv <= 1 {
        1
    } else if n_q_per_kv <= 4 {
        4
    } else if n_q_per_kv <= 8 {
        8
    } else if n_q_per_kv <= 16 {
        16
    } else {
        anyhow::bail!(
            "attn_softmax_output_q4_0_f32_gqa: n_q_per_kv={n_q_per_kv} > 8 not supported"
        );
    };
    let kernel_name = match (head_dim, nq_bucket) {
        (64, 1) => "attn_softmax_output_q4_0_f32_hd64_nq1",
        (64, 4) => "attn_softmax_output_q4_0_f32_hd64_nq4",
        (64, 8) => "attn_softmax_output_q4_0_f32_hd64_nq8",
        (128, 1) => "attn_softmax_output_q4_0_f32_hd128_nq1",
        (128, 4) => "attn_softmax_output_q4_0_f32_hd128_nq4",
        (128, 8) => "attn_softmax_output_q4_0_f32_hd128_nq8",
        (256, 1) => "attn_softmax_output_q4_0_f32_hd256_nq1",
        (256, 4) => "attn_softmax_output_q4_0_f32_hd256_nq4",
        (256, 8) => "attn_softmax_output_q4_0_f32_hd256_nq8",
        _ => anyhow::bail!("attn_softmax_output_q4_0_f32_gqa: unsupported head_dim={head_dim}",),
    };
    let func = dev
        .get_or_load_custom_func(kernel_name, "loken_quantized", get_quantized_ptx(dev)?)
        .map_err(|e| anyhow!("load kernel: {e}"))?;

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

#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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

/// GQA sibling of `attn_flash_splitk_q8_decode_dev_pos`. Fused split-K
/// flash-decode where each (split, h_kv) warp reads its KV head's chunk ONCE
/// and feeds all `n_q_per_kv` query heads - so KV bandwidth (the long-context
/// bottleneck) is read n_kv_headsx instead of n_q_headsx, unlike the 2-kernel
/// `attn_score`+`attn_softmax_output_gqa` chain it replaces (no HBM scores
/// round-trip, NSPLITx more blocks than the single-warp-per-head fused path).
/// The combine kernel is the same per-query-head merge as the MHA variant.
/// `q_f32` is `[n_q_heads, head_dim]` (n_q_heads = n_kv_heads x n_q_per_kv).
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
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    const MAX_NQ: usize = 8; // must match FLASH_SPLITK_GQA_PARTIAL_KERNEL MAXNQ
    if !(head_dim == 64 || head_dim == 128) {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: head_dim {head_dim} not in {{64,128}}"
        );
    }
    if n_q_per_kv == 0 || n_q_per_kv > MAX_NQ {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: n_q_per_kv {n_q_per_kv} not in 1..={MAX_NQ}"
        );
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_flash_splitk_q8_gqa_decode_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    // Adaptive split count: GQA has few KV heads (4-8), so blocks = nsplit x
    // n_kv_heads. The partial kernel is one warp per (split, KV-head) and is
    // memory-bound on the KV read - at long context a seq_kv-independent nsplit
    // (the old `384/n_kv` ≈ 48 for qwen3) leaves only ~384 warps in flight (~11%
    // occupancy -> measured ~19% of KV bandwidth) while each warp serially scans
    // 80+ tokens. So GROW nsplit with seq_kv to keep each split's chunk <= ~24
    // tokens (more warps in flight -> more memory-level parallelism -> higher
    // bandwidth), while never dropping below the original moderate-context value
    // (so the short/medium winning cells are unchanged). Cap by seq_kv and a sane
    // upper band; the combine loops nsplit but its cost stays small vs the saving.
    // Chunk target 8 (was 24): nsys on qwen3:0.6b long-ctx decode // measured the partial at 16.3 µs/layer with ~24-token chunks (48 splits x
    // 8 kv-heads = 384 warps ≈ 11% occupancy - latency-bound, not BW) vs
    // 12.5 µs at chunk 8 (nsplit ~3x). The larger nsplit only pays off with
    // the WIDE combine below (grid x HD_BLOCKS); the old single-block-per-head
    // combine grew 2.5 -> 5.8 µs and ate the gain.
    let base = (384 / n_kv_heads.max(1)).clamp(8, 96);
    let by_len = seq_kv_hint.div_ceil(8);
    let nsplit: usize = base.max(by_len).clamp(8, 256).min(seq_kv_hint.max(1));
    // Plain partial (NOT `_pf_`): nsys A/B (qwen3:0.6b, kv≈1.1-1.4K)
    // measured the software-pipelined variant SLOWER (17.1 vs 16.3 µs/layer)  -
    // the +2xHD_BLOCKS registers cost more occupancy than the hidden load
    // latency buys. WIDE combine: one block per (head, 32-dim slice), so the
    // combine keeps up with the chunk-8 nsplit (see kernel comment).
    let (partial_name, combine_name) = match head_dim {
        64 => (
            "flash_splitk_q8_gqa_partial_hd64",
            "flash_splitk_gqa_combine_wide_hd64",
        ),
        128 => (
            "flash_splitk_q8_gqa_partial_hd128",
            "flash_splitk_gqa_combine_wide_hd128",
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
    // index by the runtime `nsplit` (<= MAX_NSPLIT), using only the live prefix.
    const MAX_NSPLIT: usize = 256;
    debug_assert!(nsplit <= MAX_NSPLIT);
    let partials = unsafe { dev.alloc::<f32>(n_q_heads * MAX_NSPLIT * (head_dim + 2))? };

    // Partial grid is (nsplit, n_kv_heads): each warp serves n_q_per_kv query heads.
    let cfg1 = LaunchConfig {
        grid_dim: (nsplit as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b1 = partial_func.builder();
    b1.arg(k_blob);
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
    unsafe { b1.launch(cfg1) }.map_err(|e| anyhow!("launch gqa partial: {e}"))?;

    // WIDE combine: one block per (query head, 32-dim slice) x 8 warps (must
    // match FLASH_SPLITK_COMBINE_WARPS). The per-head grid of the old combine
    // couldn't keep up with the chunk-8 nsplit (nsys: 5.8 µs vs
    // 2.5 µs); HD_BLOCKSx the blocks restores the balance.
    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let cfg2 = LaunchConfig {
        grid_dim: (n_q_heads as u32, (head_dim / 32) as u32, 1),
        block_dim: (WARP_SIZE as u32, 8, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = combine_func.builder();
    b2.arg(&partials);
    b2.arg(&dst);
    barg!(b2, nsplit as i32);
    unsafe { b2.launch(cfg2) }.map_err(|e| anyhow!("launch gqa combine: {e}"))?;

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Q4_0 KIVI sibling of `attn_flash_splitk_q8_gqa_decode_dev_pos`. One fused
/// pass over the qwen3moe Q4 KV cache (per-channel KIVI K + F16 residual,
/// per-token Q4_0 V) replacing the 2-kernel attn_score(->HBM scores)+
/// attn_softmax_output chain - no HBM scores round-trip, adaptive nsplit for
/// occupancy. seq_kv is a host int (Q4 path is not graph-captured). Reuses the
/// runtime-nsplit gqa combine. `q_f32` is `[n_q_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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

    const MAX_NSPLIT: usize = 256;
    debug_assert!(nsplit <= MAX_NSPLIT);
    let partials = unsafe { dev.alloc::<f32>(n_q_heads * MAX_NSPLIT * (head_dim + 2))? };

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

#[allow(clippy::too_many_arguments)]
pub fn dequantize_q8_0_blob_f16(
    data: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !elem_count.is_multiple_of(32) {
        anyhow::bail!("dequantize_q8_0_blob_f16: elem_count {elem_count} must be multiple of 32");
    }
    let nb = elem_count.div_ceil(256);
    let func = dev
        .get_or_load_custom_func(
            "dequantize_block_q8_0_f16",
            "loken_quantized",
            get_quantized_ptx(dev)?,
        )
        .map_err(|e| anyhow!("load kernel: {e}"))?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    let cfg = LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb32 = (elem_count / 32) as i32;
    let mut builder = func.builder();
    builder.arg(data);
    builder.arg(&dst);
    barg!(builder, nb32);
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
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
#[allow(clippy::too_many_arguments)]
pub fn mvq_plain_any_via_shared_q8_1_on_stream(
    qstor: &crate::tensor::quantized::QCudaStorage,
    q8_1_buf: &CudaSlice<u8>,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
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
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
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
