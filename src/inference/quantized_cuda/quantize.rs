//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Public wrapper over the `quantize_q8_1` helper.
pub fn quantize_q8_1_pub(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    quantize_q8_1(src, dst, k, ky, dev)
}

/// Fused quantized GEMV from a pre-quantized Q8_1 activation:
/// `dst[nrows x b_size] = W(quantized) . y(q8_1)`. For Q4_K, b_size=1,
/// ncols%256==0 it uses the tensor-core IMMA kernel; otherwise the generic
/// `mul_mat_vec_<tag>_q8_1_cuda<b_size>` kernel.
/// (reads the weight via the `weight_cuda_slice` accessor).
pub fn mvq_via_pre_quantized_q8_1(
    qstor: &crate::tensor::quantized::QCudaStorage,
    y_q8_1: &CudaSlice<u8>,
    ncols: usize,
    nrows: usize,
    b_size: usize,
) -> Result<CudaStorage> {
    let dev = qstor.device();
    if b_size == 0 || b_size > 8 {
        anyhow::bail!("mvq_via_pre_quantized_q8_1: only bsize 1..=8, got {b_size}");
    }
    // The tensor-core IMMA kernel for Q4_K decode used to be taken here and is
    // NOT: it is slower than the plain MMVQ path it displaced. It only ever
    // received 2 of the 8 Q4_K matmuls per layer - the ones whose callers reach
    // this module rather than the generic one, a split decided by code path and
    // not by any measurement - and taking it cost 15% of end-to-end decode
    // (granite3.1-dense:2b 256.6 tok/s with, 294.8 without, three runs each at
    // 300 tokens). Removing it also moves that model from 12.7% behind the
    // reference to 2.6% ahead. Checked on deepcoder:14b and qwen3:0.6b too:
    // both faster, output unchanged.
    if false && b_size == 1 && qstor.dtype() == GgmlDType::Q4K && ncols.is_multiple_of(256) {
        let stream = dev.cuda_stream();
        let dst = unsafe { dev.alloc::<f32>(nrows)? };
        let weight = qstor.weight_cuda_slice();
        let weights_ptr = weight.device_ptr(&stream).0 as *const core::ffi::c_void;
        let input_ptr = y_q8_1.device_ptr(&stream).0 as *const core::ffi::c_void;
        let dst_ptr = dst.device_ptr(&stream).0 as *mut core::ffi::c_void;
        let stream_h = stream.cu_stream() as i64;
        unsafe {
            q4k_mmvq_imma(
                weights_ptr,
                input_ptr,
                dst_ptr,
                ncols as i32,
                nrows as i32,
                stream_h,
            );
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    // The list is the one thing this site decides: which formats a mat-vec kernel was written
    // for. How each is spelled belongs to the format and comes from its declaration, so a new
    // format arrives here as a missing arm rather than as a name someone has to spell again.
    let dtype = qstor.dtype();
    let served = matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    );
    if !served {
        anyhow::bail!("mvq_via_pre_quantized_q8_1: unsupported dtype {dtype:?}");
    }
    let kernel_name = format!("mul_mat_vec_{}_q8_1_cuda{b_size}", dtype.gguf_name());
    let func = quantized_kernel(dev, &kernel_name)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    // MUST mirror the kernel's compile-time warp count (quantized.cu
    // mul_mat_vec_q): `nwarps = (qk == 32 && ncols_y == 1) ? 8 : 2`. The
    // launcher hardcoding 2 for b_size=1 while the kernel expects 8 for the
    // qk=32 dtypes (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) made the shared-memory
    // reduction read 6 never-written warp slots (uninitialized -> scattered
    // NaN) AND skip 3/4 of the k-blocks (blocks_per_iter is compile-time)  - 
    // root cause of the MIDI-LLM (Q8_0 lm_head) GPU collapse.
    // K-quants (qk=256 -> 2 warps) always matched, which is why
    // every served Q6_K/Q4_K head was fine.
    let qk32 = matches!(
        qstor.dtype(),
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0
    );
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, if qk32 { 8 } else { 2 }),
        2..=4 => ((nrows as u32).div_ceil(2), 2),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => anyhow::bail!("unexpected bsize {b_size}"),
    };
    let cfg = warp_block_launch((nblocks, 1, 1), nwarps, 0);
    let mut builder = func.builder();
    builder.arg(qstor.weight_cuda_slice());
    builder.arg(y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        ncols as i32,
        nrows as i32,
        ncols_padded as i32,
        nrows as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

// ------------------------------------------------------------
// native migration: Q8 KV-cache quantize/attention + Q4_0 dequant
// launchers over the NVRTC-compiled `loken_quantized` module (cuda/quantized.cu) rather
// than the tensor-op PTX blob. Signatures are unchanged so `inference/cache/q8_kv.rs` /
// `inference/cache/q4_kv.rs` only
// retarget their imports.
// ------------------------------------------------------------

/// Quantize `src` (f32) into a pre-allocated destination buffer at a
/// REPLAY-TIME slot index read from a device i32 pointer (`slot_dev[0]`).
/// Required for CUDA graph capture of the Q8 KV append: the captured
/// kernel reads the slot at replay, so the destination offset
/// (`slot_dev[0] * num_blocks_per_token * 34`) advances correctly per
/// token instead of being baked at capture time.
///
/// Same row-width / k-alignment constraint as `quantize_q8_0_f32_into_offset`
/// (k must be a multiple of MATRIX_ROW_PADDING). Single-row decode only.
#[allow(clippy::too_many_arguments)]
pub fn quantize_q8_0_f32_dev_slot(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    slot_dev: &CudaSlice<i32>,
    k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let staging =
        q8_0_row_staging(k, 1, dev).map_err(|e| anyhow!("quantize_q8_0_f32_dev_slot: {e}"))?;
    if src.len() < k {
        anyhow::bail!(
            "quantize_q8_0_f32_dev_slot: src has {} elems, expected {} (k)",
            src.len(),
            k
        );
    }

    let func = quantized_kernel(dev, "quantize_q8_0_dev_slot")?;
    let src_chunk = src.slice(0..k);

    let mut builder = func.builder();
    builder.arg(&src_chunk);
    builder.arg(dst);
    builder.arg(slot_dev);
    barg!(
        builder,
        k as i32,
        staging.kx_padded as i32,
        (staging.kx_padded / Q8_0_BLOCK_SIZE) as i32
    );
    unsafe { builder.launch(staging.cfg) }
        .map_err(|e| anyhow!("launch quantize_q8_0_dev_slot: {e}"))?;
    Ok(())
}

/// Paired K+V twin of `quantize_q8_0_f32_dev_slot`: one launch quantizes both
/// the K and V rows into their buffers at the replay-time slot (gridDim.z
/// selects the side). Saves one graph node + inter-node gap per layer.
pub fn quantize_q8_0_kv_paired_f32_dev_slot(
    src_k: &CudaView<f32>,
    src_v: &CudaView<f32>,
    dst_k: &mut CudaSlice<u8>,
    dst_v: &mut CudaSlice<u8>,
    slot_dev: &CudaSlice<i32>,
    k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // z=2 -> one block group for K, one for V.
    let staging = q8_0_row_staging(k, 2, dev)
        .map_err(|e| anyhow!("quantize_q8_0_kv_paired_f32_dev_slot: {e}"))?;
    if src_k.len() < k || src_v.len() < k {
        anyhow::bail!(
            "quantize_q8_0_kv_paired_f32_dev_slot: src has {}/{} elems, expected {} (k)",
            src_k.len(),
            src_v.len(),
            k
        );
    }

    let func = quantized_kernel(dev, "quantize_q8_0_kv_paired_dev_slot")?;
    let src_k_chunk = src_k.slice(0..k);
    let src_v_chunk = src_v.slice(0..k);

    let mut builder = func.builder();
    builder.arg(&src_k_chunk);
    builder.arg(&src_v_chunk);
    builder.arg(dst_k);
    builder.arg(dst_v);
    builder.arg(slot_dev);
    barg!(
        builder,
        k as i32,
        staging.kx_padded as i32,
        (staging.kx_padded / Q8_0_BLOCK_SIZE) as i32
    );
    unsafe { builder.launch(staging.cfg) }
        .map_err(|e| anyhow!("launch quantize_q8_0_kv_paired_dev_slot: {e}"))?;
    Ok(())
}

/// Quantize `src` (f32) into a pre-allocated destination buffer at the
/// caller-supplied byte offset. Skips the temporary CudaSlice + memcpy
/// pattern used by `quantize_with_layout` + memcpy_dtod by writing the
/// q8_0 blocks directly into the persistent destination at
/// `dst[dst_byte_offset..]`. Used by the KV cache append to fold two
/// launches (quantize-into-staging, memcpy-staging-to-buffer) into one.
///
/// `k` is elements per row (assumed n_kv_heads x head_dim for the
/// single-row decode case). `k` MUST be a multiple of MATRIX_ROW_PADDING
/// (512) - otherwise the kernel writes padding zeros into the slot
/// after the current write, corrupting the next token's KV. This holds
/// for common decode shapes (e.g. n_kv_headsxhead_dim of 32x64, 2x512,
/// 8x128); the caller should fall back to the staging-then-memcpy path
/// when it doesn't.
///
/// Returns the number of bytes written: `(k / 32) x 34`.
pub fn quantize_q8_0_f32_into_offset(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    dst_byte_offset: usize,
    k: usize,
    dev: &CudaDevice,
) -> Result<usize> {
    let staging =
        q8_0_row_staging(k, 1, dev).map_err(|e| anyhow!("quantize_q8_0_f32_into_offset: {e}"))?;
    let bytes_to_write = staging.row_bytes;

    if dst_byte_offset + bytes_to_write > dst.len() {
        anyhow::bail!(
            "quantize_q8_0_f32_into_offset: write {} bytes at offset {} but dst is {} bytes",
            bytes_to_write,
            dst_byte_offset,
            dst.len()
        );
    }
    if src.len() < k {
        anyhow::bail!(
            "quantize_q8_0_f32_into_offset: src has {} elems, expected {} (k)",
            src.len(),
            k
        );
    }

    let func = quantized_kernel(dev, "quantize_q8_0")?;
    let src_chunk = src.slice(0..k);
    let dst_chunk = dst.slice(dst_byte_offset..dst_byte_offset + bytes_to_write);

    let mut builder = func.builder();
    builder.arg(&src_chunk);
    builder.arg(&dst_chunk);
    barg!(builder, k as i32, staging.kx_padded as i32);
    unsafe { builder.launch(staging.cfg) }.map_err(|e| anyhow!("launch quantize_q8_0: {e}"))?;
    Ok(bytes_to_write)
}

/// GQA variant: compute attention scores for **all** kv-heads in a single
/// kernel launch.
///
/// K layout: `[seq_kv, n_kv_heads, head_dim]` as Q8_0 blocks - the packing
/// used by `Q8KvCache`. Q layout: `[n_q_heads, head_dim]` f32 on-device.
/// `n_q_heads` must equal `n_kv_heads * n_q_per_kv`. Q is quantized to Q8_1
/// once, re-used across all kv-heads.
///
/// Output: `float[n_q_heads, n_kv]` scores. Query head `h` corresponds to
/// kv-head `h / n_q_per_kv`.
///
/// Backwards-compat unscaled form: callers should prefer
/// [`attn_score_q8_0_q8_1_gqa_scaled`] to fold the 1/sqrt(d) scale into the
/// kernel and skip the post-matmul affine launch.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_q8_0_q8_1_gqa(
    k_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    attn_score_q8_0_q8_1_gqa_scaled(
        k_blob, q_f32, head_dim, n_kv, n_kv_heads, n_q_per_kv, 1.0f32, dev,
    )
}

/// Device-position variant of `attn_score_q8_0_q8_1_gqa` for CUDA graph
/// capture mode.
///
/// Differences from the non-dev_pos version:
///   - Reads `seq_kv` from a device i32 pointer (`seq_kv_dev`) instead of
///     a host parameter. Host updates `seq_kv_dev` OUTSIDE the captured
///     region each token.
///   - Q is consumed as F32 directly - skips the `quantize_q8_1` step and
///     the associated host-side alloc that breaks graph capture.
///   - Writes scores into a `[n_q_heads, max_seq_padded]` buffer (stable
///     stride across replays). Positions >= seq_kv get -INFINITY so
///     `softmax_last_dim` masks them out.
///
/// `window > 0` enables sliding-window attention (gemma4 SWA): positions
/// older than the last `window` keys are -INF'd and their Q.K dot skipped.
///
/// Returns scores `[n_q_heads, max_seq_padded]` F32. Callers MUST keep
/// `max_seq_padded` constant across all calls in one captured graph.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_q8_0_f32_dev_pos(
    k_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    window: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_score_q8_0_f32_dev_pos: {e}"))?;
    if q_f32.len() != n_q_heads * head_dim {
        anyhow::bail!(
            "attn_score_q8_0_f32_dev_pos: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64,  1) => "attn_score_q8_0_f32_dev_pos_hd64_nq1",
        (64,  2) => "attn_score_q8_0_f32_dev_pos_hd64_nq2",
        (64,  4) => "attn_score_q8_0_f32_dev_pos_hd64_nq4",
        (64,  5) => "attn_score_q8_0_f32_dev_pos_hd64_nq5",
        (64,  8) => "attn_score_q8_0_f32_dev_pos_hd64_nq8",
        (128, 1) => "attn_score_q8_0_f32_dev_pos_hd128_nq1",
        (128, 2) => "attn_score_q8_0_f32_dev_pos_hd128_nq2",
        (128, 4) => "attn_score_q8_0_f32_dev_pos_hd128_nq4",
        (128, 5) => "attn_score_q8_0_f32_dev_pos_hd128_nq5",
        (128, 8) => "attn_score_q8_0_f32_dev_pos_hd128_nq8",
        (256, 1) => "attn_score_q8_0_f32_dev_pos_hd256_nq1",
        (256, 2) => "attn_score_q8_0_f32_dev_pos_hd256_nq2",
        (256, 4) => "attn_score_q8_0_f32_dev_pos_hd256_nq4",
        (256, 5) => "attn_score_q8_0_f32_dev_pos_hd256_nq5",
        (256, 8) => "attn_score_q8_0_f32_dev_pos_hd256_nq8",
        _ => anyhow::bail!(
            "attn_score_q8_0_f32_dev_pos: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv}) - supported {{64,128,256}} x {{1,2,4,5,8}}",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    // Each warp computes one (token, h_kv) and emits scores for all
    // n_q_per_kv query heads in that kv-group.
    const WARPS_PER_BLOCK: u32 = 4;
    let cfg = warp_block_launch(
        (
            (max_seq_padded as u32).div_ceil(WARPS_PER_BLOCK),
            n_kv_heads as u32,
            1,
        ),
        WARPS_PER_BLOCK,
        0,
    );

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * max_seq_padded)? };
    let mut builder = func.builder();
    builder.arg(k_blob);
    builder.arg(q_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(
        builder,
        max_seq_padded as i32,
        n_kv_heads as i32,
        n_q_per_kv as i32,
        window as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// V-path attention output: `out = probs @ V` where V is a Q8_0 cache laid
/// out `[seq_kv, n_kv_heads, head_dim]` (same packing as `Q8KvCache`).
///
/// `probs` is f32 `[n_q_heads, seq_kv]` (softmax output). `n_q_heads` must
/// equal `n_kv_heads * n_q_per_kv`. Returns f32 `[n_q_heads, head_dim]`.
///
/// The reduction is along `seq_kv`, orthogonal to Q8_0's head_dim block
/// direction - so we can't reuse the standard `mul_mat_vec_q` kernels
/// (which reduce along the block axis). Each lane owns one head_dim output
/// and sweeps seq_kv internally.
#[allow(clippy::too_many_arguments)]
pub fn attn_output_q8_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_output_q8_0_f32_gqa: {e}"))?;
    if probs_f32.len() != n_q_heads * seq_kv {
        anyhow::bail!(
            "attn_output_q8_0_f32_gqa: probs has {} elements, expected {}",
            probs_f32.len(),
            n_q_heads * seq_kv
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_output_q8_0_f32_hd64_nq1",
        (64, 2) => "attn_output_q8_0_f32_hd64_nq2",
        (64, 4) => "attn_output_q8_0_f32_hd64_nq4",
        (64, 5) => "attn_output_q8_0_f32_hd64_nq5",
        (64, 8) => "attn_output_q8_0_f32_hd64_nq8",
        (128, 1) => "attn_output_q8_0_f32_hd128_nq1",
        (128, 2) => "attn_output_q8_0_f32_hd128_nq2",
        (128, 4) => "attn_output_q8_0_f32_hd128_nq4",
        (128, 5) => "attn_output_q8_0_f32_hd128_nq5",
        (128, 8) => "attn_output_q8_0_f32_hd128_nq8",
        (256, 1) => "attn_output_q8_0_f32_hd256_nq1",
        (256, 2) => "attn_output_q8_0_f32_hd256_nq2",
        (256, 4) => "attn_output_q8_0_f32_hd256_nq4",
        (256, 5) => "attn_output_q8_0_f32_hd256_nq5",
        (256, 8) => "attn_output_q8_0_f32_hd256_nq8",
        (512, 1) => "attn_output_q8_0_f32_hd512_nq1",
        (512, 2) => "attn_output_q8_0_f32_hd512_nq2",
        (512, 4) => "attn_output_q8_0_f32_hd512_nq4",
        (512, 5) => "attn_output_q8_0_f32_hd512_nq5",
        (512, 8) => "attn_output_q8_0_f32_hd512_nq8",
        (512, 16) => "attn_output_q8_0_f32_hd512_nq16",
        _ => anyhow::bail!(
            "attn_output_q8_0_f32_gqa: unsupported combo head_dim={head_dim} n_q_per_kv={n_q_per_kv} (want {{64,128,256,512}} x {{1,2,4,5,8}})",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    let strides = kv_block_strides(head_dim, n_kv_heads);
    let hd_blocks = strides.hd_blocks;

    // Keep this in lock-step with ATTN_OUTPUT_WARPS in the .cu file.
    // qh is iterated inside the kernel so V is loaded once per (kv, block_idx)
    // tile and reused across n_q_per_kv query heads - critical for GQA
    // groups with n_q_per_kv > 1 (qwen3 = 4, qwen3-coder = 8, Llama-3 70B = 8).
    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = warp_block_launch((hd_blocks as u32, n_kv_heads as u32, 1), WARPS_PER_BLOCK, 0);

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
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

/// Device-position variant of `attn_output_q8_0_f32_gqa` for CUDA graph
/// capture mode.
///
/// Differences from the non-dev_pos version:
///   - Reads `seq_kv` from a device i32 pointer (`seq_kv_dev`).
///   - `probs` is laid out at the fixed `max_seq_padded` stride
///     `[n_q_heads, max_seq_padded]`. Positions >= seq_kv MUST be zero
///     (the softmax of attn_score_q8_0_f32_dev_pos guarantees this since
///     those positions get -INFINITY scores).
///   - V buffer and the output buffer pointers stay valid across replays
///     when graph capture mode allocates them with stable addresses.
///
/// Returns `[n_q_heads, head_dim]` F32.
#[allow(clippy::too_many_arguments)]
pub fn attn_output_q8_0_f32_dev_pos(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_output_q8_0_f32_dev_pos: {e}"))?;
    if probs_f32.len() != n_q_heads * max_seq_padded {
        anyhow::bail!(
            "attn_output_q8_0_f32_dev_pos: probs has {} elements, expected {} ({} x {})",
            probs_f32.len(),
            n_q_heads * max_seq_padded,
            n_q_heads,
            max_seq_padded
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64,  1) => "attn_output_q8_0_f32_dev_pos_hd64_nq1",
        (64,  2) => "attn_output_q8_0_f32_dev_pos_hd64_nq2",
        (64,  4) => "attn_output_q8_0_f32_dev_pos_hd64_nq4",
        (64,  5) => "attn_output_q8_0_f32_dev_pos_hd64_nq5",
        (64,  8) => "attn_output_q8_0_f32_dev_pos_hd64_nq8",
        (128, 1) => "attn_output_q8_0_f32_dev_pos_hd128_nq1",
        (128, 2) => "attn_output_q8_0_f32_dev_pos_hd128_nq2",
        (128, 4) => "attn_output_q8_0_f32_dev_pos_hd128_nq4",
        (128, 5) => "attn_output_q8_0_f32_dev_pos_hd128_nq5",
        (128, 8) => "attn_output_q8_0_f32_dev_pos_hd128_nq8",
        (256, 1) => "attn_output_q8_0_f32_dev_pos_hd256_nq1",
        (256, 2) => "attn_output_q8_0_f32_dev_pos_hd256_nq2",
        (256, 4) => "attn_output_q8_0_f32_dev_pos_hd256_nq4",
        (256, 5) => "attn_output_q8_0_f32_dev_pos_hd256_nq5",
        (256, 8) => "attn_output_q8_0_f32_dev_pos_hd256_nq8",
        _ => anyhow::bail!(
            "attn_output_q8_0_f32_dev_pos: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv}) - supported {{64,128,256}} x {{1,2,4,5,8}}",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    let strides = kv_block_strides(head_dim, n_kv_heads);
    let hd_blocks = strides.hd_blocks;

    // Must match ATTN_OUTPUT_DEV_POS_WARPS in quantized.cu. Lower than
    // the non-dev_pos `ATTN_OUTPUT_WARPS=32` because the dev_pos variant
    // adds a device-memory seq_kv read + max_seq_padded indexing that
    // tip per-thread register count past the 64-reg threshold needed for
    // 1024-thread blocks on smaller compute caps.
    const WARPS_PER_BLOCK: u32 = 16;
    let cfg = warp_block_launch((hd_blocks as u32, n_kv_heads as u32, 1), WARPS_PER_BLOCK, 0);

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(
        builder,
        max_seq_padded as i32,
        strides.per_position as i32,
        strides.per_kv_head as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Fused scale + softmax + V.attn for Q8 KV with device-side seq_kv.
///
/// Replaces the 3-launch chain of `affine(scale)` + `softmax_last_dim` +
/// `attn_output_q8_0_f32_dev_pos` with one kernel that:
///   - applies the 1/√head_dim scale in-register,
///   - computes the row-wise softmax max + denom in warp 0,
///   - re-reads raw scores and folds (scale + softmax) into the V multiply
///     accumulate - no intermediate `probs` buffer allocated or written.
///
/// Inputs match the existing dev_pos contract: `scores` is shape
/// `[n_q_heads, max_seq_padded]` with -INFINITY at positions >= seq_kv,
/// and `seq_kv_dev[0]+1` gives the valid range. V blob layout is the
/// same packed `[max_seq_padded, n_kv, head_dim]` Q8_0 the unfused
/// `attn_output_q8_0_f32_dev_pos` reads. `window > 0` bounds the scan to
/// the last `window` keys (gemma4 SWA; the score kernel -INF'd the rest).
///
/// Returns `[n_q_heads, head_dim]` F32.
#[allow(clippy::too_many_arguments)]
pub fn attn_softmax_output_q8_0_f32_dev_pos(
    v_blob: &CudaSlice<u8>,
    scores_f32: &CudaView<f32>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    max_seq_padded: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    scale: f32,
    window: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let n_q_heads = gqa_query_heads(head_dim, n_kv_heads, n_q_per_kv)
        .map_err(|e| anyhow!("attn_softmax_output_q8_0_f32_dev_pos: {e}"))?;
    if scores_f32.len() != n_q_heads * max_seq_padded {
        anyhow::bail!(
            "attn_softmax_output_q8_0_f32_dev_pos: scores has {} elements, expected {} ({} x {})",
            scores_f32.len(),
            n_q_heads * max_seq_padded,
            n_q_heads,
            max_seq_padded
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64,  1) => "attn_softmax_output_q8_0_f32_dev_pos_hd64_nq1",
        (64,  2) => "attn_softmax_output_q8_0_f32_dev_pos_hd64_nq2",
        (64,  4) => "attn_softmax_output_q8_0_f32_dev_pos_hd64_nq4",
        (64,  5) => "attn_softmax_output_q8_0_f32_dev_pos_hd64_nq5",
        (64,  8) => "attn_softmax_output_q8_0_f32_dev_pos_hd64_nq8",
        (128, 1) => "attn_softmax_output_q8_0_f32_dev_pos_hd128_nq1",
        (128, 2) => "attn_softmax_output_q8_0_f32_dev_pos_hd128_nq2",
        (128, 4) => "attn_softmax_output_q8_0_f32_dev_pos_hd128_nq4",
        (128, 5) => "attn_softmax_output_q8_0_f32_dev_pos_hd128_nq5",
        (128, 8) => "attn_softmax_output_q8_0_f32_dev_pos_hd128_nq8",
        (256, 1) => "attn_softmax_output_q8_0_f32_dev_pos_hd256_nq1",
        (256, 2) => "attn_softmax_output_q8_0_f32_dev_pos_hd256_nq2",
        (256, 4) => "attn_softmax_output_q8_0_f32_dev_pos_hd256_nq4",
        (256, 5) => "attn_softmax_output_q8_0_f32_dev_pos_hd256_nq5",
        (256, 8) => "attn_softmax_output_q8_0_f32_dev_pos_hd256_nq8",
        _ => anyhow::bail!(
            "attn_softmax_output_q8_0_f32_dev_pos: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv}) - supported {{64,128,256}} x {{1,2,4,5,8}}",
        ),
    };
    let func = quantized_kernel(dev, kernel_name)?;

    let strides = kv_block_strides(head_dim, n_kv_heads);
    let hd_blocks = strides.hd_blocks;

    // Must match ATTN_SOFTMAX_OUTPUT_DEV_POS_WARPS in quantized.cu.
    const WARPS_PER_BLOCK: u32 = 16;
    let cfg = warp_block_launch((hd_blocks as u32, n_kv_heads as u32, 1), WARPS_PER_BLOCK, 0);

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(scores_f32);
    builder.arg(&dst);
    builder.arg(seq_kv_dev);
    barg!(
        builder,
        max_seq_padded as i32,
        strides.per_position as i32,
        strides.per_kv_head as i32,
        scale,
        window as i32
    );
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch {kernel_name}: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Dequantize a packed Q4_0 blob (`elem_count / 32` blocks of 18 bytes) to
/// F16. Used by the Q4 KV cache (KIVI) trim/dequant paths.
pub fn dequantize_q4_0_blob_f16(
    data: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !elem_count.is_multiple_of(32) {
        anyhow::bail!("dequantize_q4_0_blob_f16: elem_count {elem_count} must be multiple of 32");
    }
    let nb = elem_count.div_ceil(256);
    let func = quantized_kernel(dev, "dequantize_block_q4_0_f16")?;
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
    unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch dequantize_block_q4_0_f16: {e}"))?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}
