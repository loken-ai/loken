//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Fused LayerNorm on the CUDA F32 path (one launch instead of the ~8-10
/// composed ops). Caller dispatches the fallback for non-CUDA/non-F32.
pub fn fused_layernorm_f32(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let bias_t = match bias {
        Some(b) => b.contiguous()?,
        // dummy 1-elem buffer; kernel ignores it when has_bias = 0
        None => Tensor::zeros_on(1, DType::F32, &x.device())?,
    };
    let cols = *x.dims().last().unwrap_or(&1);
    let rows = x.elem_count() / cols;
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_layernorm_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (w_store, _) = weight.storage_and_layout();
    let (b_store, _) = bias_t.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();
    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(ws),
        crate::tensor::StorageView::Cuda(bs),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*x_store, &*w_store, &*b_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let w_slice = ws.as_cuda_slice::<f32>()?;
        let b_slice = bs.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let block = 256u32.min(cols as u32).max(1).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 8,
        };
        let cols_i32 = cols as i32;
        let has_bias_i32: i32 = if bias.is_some() { 1 } else { 0 };
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(w_slice);
        builder.arg(b_slice);
        builder.arg(o_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        builder.arg(&has_bias_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_layernorm_f32: {e}")))?;
    }
    drop(x_store);
    drop(w_store);
    drop(b_store);
    drop(o_store);
    Ok(out)
}

/// Last-dim softmax on the CUDA F32 path: `out = exp(x-rowmax)/sum(exp(x-rowmax))`
/// over the last dim. Native replacement for
/// `crate::tensor::ops::softmax_last_dim`. Caller dispatches the
/// fallback for non-CUDA/non-F32.
pub fn fused_softmax_lastdim_f32(x: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let cols = *x.dims().last().unwrap_or(&1);
    let rows = x.elem_count() / cols;
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_softmax_lastdim_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();
    if let (crate::tensor::StorageView::Cuda(xs), crate::tensor::StorageView::Cuda(os)) =
        (&*x_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let block = 256u32.min(cols as u32).max(1).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(o_slice);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_softmax_lastdim_f32: {e}")))?;
    }
    drop(x_store);
    drop(o_store);
    Ok(out)
}

/// Variant of `fused_rmsnorm_then_add` that also applies a column-broadcast
/// `scale` to the output: `out = (rms_norm(x, weight, eps) + residual) * scale`.
///
/// Gemma4 lays a per-layer output scale on top of the PLE block; folding
/// it into the same kernel pass saves another launch per layer.
///
/// Strict preconditions (caller falls back if any fail):
/// - CUDA F32 contiguous for x, weight, residual, scale, output
/// - x.shape() == residual.shape()
/// - weight.dims() == [cols] AND scale.dims() == [cols] where
///   cols = x.dims().last()
///
/// Numerically equivalent (within ~1 ULP) to the unfused 3-launch chain
/// `(rms_norm(x, w, eps) + residual).broadcast_mul(scale)`.
pub fn fused_rmsnorm_add_scale(
    x: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
    scale: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let fallback = || -> Result<Tensor> {
        let norm = crate::tensor::ops::rms_norm(x, weight, eps)?;
        (norm + residual)?.broadcast_mul(scale)
    };

    if !x.device().is_cuda() {
        return fallback();
    }
    if x.dtype() != DType::F32
        || residual.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || scale.dtype() != DType::F32
    {
        return fallback();
    }
    if x.shape() != residual.shape() {
        return fallback();
    }
    let cols = match x.dims().last().copied() {
        Some(c) if c > 0 => c,
        _ => return fallback(),
    };
    // Hard shape gate: weight and scale must each be the 1-D column vector.
    // The prior fused_add_then_scale attempt failed live with
    // empty output despite passing numerics tests - a shape mismatch in
    // scale could silently produce nonsense without tripping a CUDA error.
    if weight.dims() != [cols] || scale.dims() != [cols] {
        return fallback();
    }

    let x = x.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.contiguous()?;

    let total_elems = x.elem_count();
    let rows = total_elems / cols;

    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_rmsnorm_add_scale_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (r_store, r_layout) = residual.storage_and_layout();
    let (w_store, _) = weight.storage_and_layout();
    let (s_store, _) = scale.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(rs),
        crate::tensor::StorageView::Cuda(ws),
        crate::tensor::StorageView::Cuda(ss),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*x_store, &*r_store, &*w_store, &*s_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let r_slice = rs.as_cuda_slice::<f32>()?;
        let w_slice = ws.as_cuda_slice::<f32>()?;
        let s_slice = ss.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let r_view = r_slice.slice(r_layout.start_offset()..);

        let block = 256u32.min(cols as u32).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };

        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(w_slice);
        builder.arg(&r_view);
        builder.arg(s_slice);
        builder.arg(o_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_rmsnorm_add_scale: {e}")))?;
    }

    drop(x_store);
    drop(r_store);
    drop(w_store);
    drop(s_store);
    drop(o_store);
    Ok(out)
}

/// Fused last-dim softmax with a per-head attention sink (gpt-oss).
/// `scores`: F32 `[b, n_head, q, kv]` (already masked: masked entries = -inf).
/// `sinks`: F32 `[n_head]`. Returns F32 weights of the same shape as `scores`.
/// Replaces the ~9-op max/exp/sum/div composition with one block-per-row launch.
/// Returns `None` (caller falls back) on any non-CUDA / non-F32 / shape input.
/// CPU fused sink-softmax over the last dim: one pass per `[b,h,q]` row instead of
/// the ~5 materialized passes (`scale` affine, `max_keepdim`, `broadcast_maximum`
/// sink, `broadcast_sub`+`exp`, `sum_keepdim`+sink, `broadcast_div`) that
/// profiling put at ~21% of gpt-oss prefill. Bit-identical to that chain: the max
/// and sum reduce in the SAME ascending order the tensor reductions use, `exp` is
/// the same `f32::exp`, and `scale` is applied as `score * (scale as f32)` - so
/// greedy tokens are unchanged. `scores` is `[b,n_head,q,kv]` f32, `sinks` `[n_head]`.
pub(super) fn cpu_softmax_sinks_fused(
    scores: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    scale: f32,
    n_head: usize,
    q: usize,
    kv: usize,
) -> Result<Option<Tensor>> {
    use rayon::prelude::*;
    let dims = scores.dims().to_vec();
    let sc = scores.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let sk = sinks.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    // Additive causal/window mask `[1,1,q,kv]` folded in per row (index qi*kv+j),
    // matching attend's `scores.broadcast_add(mask)` before the scale.
    let mk = match mask {
        Some(m) => Some(m.contiguous()?.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    let mut out = vec![0f32; sc.len()];
    // Each [b,h,q] row (kv contiguous) is independent. Row r maps to head
    // (r / q) % n_head under the [b, n_head, q, kv] layout; the mask row is qi=r%q.
    out.par_chunks_mut(kv).enumerate().for_each(|(r, orow)| {
        let sink = sk[(r / q) % n_head];
        let srow = &sc[r * kv..r * kv + kv];
        let mrow = mk.as_ref().map(|m| &m[(r % q) * kv..(r % q) * kv + kv]);
        // (raw + mask) * scale, exactly as the materialized broadcast_add + affine.
        let scaled = |j: usize| match mrow {
            Some(mr) => (srow[j] + mr[j]) * scale,
            None => srow[j] * scale,
        };
        // Numerically-stable max over kv of the scaled score, folding in the sink.
        let mut rmax = f32::NEG_INFINITY;
        for j in 0..kv {
            rmax = f32::max(rmax, scaled(j));
        }
        let m = f32::max(rmax, sink);
        // exp(scaled - m), summed ascending, then the sink's own term.
        let mut denom = 0f32;
        for (j, o) in orow.iter_mut().enumerate() {
            let e = (scaled(j) - m).exp();
            *o = e;
            denom += e;
        }
        denom += (sink - m).exp();
        for o in orow.iter_mut() {
            *o /= denom;
        }
    });
    Tensor::from_vec(out, dims, &scores.device()).map(Some)
}

pub fn fused_softmax_sinks(
    scores: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    if scores.dtype() != DType::F32 || sinks.dtype() != DType::F32 {
        return Ok(None);
    }
    let dims = scores.dims();
    if dims.len() != 4 {
        return Ok(None);
    }
    let n_head = dims[1];
    let q = dims[2];
    let kv = dims[3];
    if kv == 0 || sinks.dims() != [n_head] {
        return Ok(None);
    }
    if !scores.device().is_cuda() {
        return cpu_softmax_sinks_fused(scores, mask, sinks, scale, n_head, q, kv);
    }
    // The CUDA kernel takes no mask input - pre-add it (exactly as attend did),
    // then run the maskless device softmax.
    let scores_owned = match mask {
        Some(m) => Some(scores.broadcast_add(m)?),
        None => None,
    };
    let scores = scores_owned.as_ref().unwrap_or(scores);
    let scores = scores.contiguous()?;
    let sinks = sinks.contiguous()?;
    let rows = scores.elem_count() / kv;
    let out = unsafe { Tensor::empty(scores.shape(), DType::F32, &scores.device())? };

    let cuda_dev = scores.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_softmax_sinks_f32", "loken_fused", ptx)?;

    let (sc_store, sc_layout) = scores.storage_and_layout();
    let (sk_store, _) = sinks.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(scs),
        crate::tensor::StorageView::Cuda(sks),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*sc_store, &*sk_store, &*o_store)
    {
        let sc_slice = scs.as_cuda_slice::<f32>()?;
        let sk_slice = sks.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let sc_view = sc_slice.slice(sc_layout.start_offset()..);

        let block = 256u32.min((kv as u32).max(1)).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let kv_i32 = kv as i32;
        let q_i32 = q as i32;
        let nh_i32 = n_head as i32;
        let mut builder = func.builder();
        builder.arg(&sc_view);
        builder.arg(sk_slice);
        builder.arg(o_slice);
        builder.arg(&kv_i32);
        builder.arg(&q_i32);
        builder.arg(&nh_i32);
        builder.arg(&scale);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_softmax_sinks: {e}")))?;
    }
    drop(sc_store);
    drop(sk_store);
    drop(o_store);
    Ok(Some(out))
}

/// On-device additive attention mask `[1,1,seq,kv]` (gpt-oss): 0 visible, -inf
/// masked (causal + optional sliding window). Generated by a kernel - no host
/// Vec + H2D upload - so it leaves no transient-host memcpy in a captured CUDA
/// graph. `window` 0 = full causal. Returns `None` (caller uses the host path)
/// on non-CUDA. The output tensor is substrate-allocated (arena-backed at capture).
pub fn fused_gptoss_mask(
    seq: usize,
    kv: usize,
    input_pos: usize,
    window: usize,
    device: &crate::tensor::Device,
) -> Result<Option<Tensor>> {
    if !device.is_cuda() {
        return Ok(None);
    }
    let out = unsafe { Tensor::empty((1, 1, seq, kv), DType::F32, device)? };
    let cuda_dev = device.as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_gptoss_mask_f32", "loken_fused", ptx)?;
    let (o_store, _) = out.storage_and_layout();
    if let crate::tensor::StorageView::Cuda(os) = &*o_store {
        let o_slice = os.as_cuda_slice::<f32>()?;
        let n = (seq * kv) as u32;
        let block = 256u32;
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let kv_i = kv as i32;
        let pos_i = input_pos as i32;
        let win_i = window as i32;
        let mut builder = func.builder();
        builder.arg(o_slice);
        builder.arg(&seq_i);
        builder.arg(&kv_i);
        builder.arg(&pos_i);
        builder.arg(&win_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_gptoss_mask: {e}")))?;
    }
    drop(o_store);
    Ok(Some(out))
}

/// Fused dual RmsNorm + add residual:
///   out = RmsNorm(a, w1, eps) + RmsNorm(b, w2, eps) + c
/// Replaces 4 launches (norm_a + norm_b + add_a_b + add_c) with 1
/// for the gemma4-MoE post-FFN combine path. All tensors must be F32
/// and contiguous on CUDA.
pub fn fused_dual_rmsnorm_add(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    w1: &Tensor,
    w2: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if !a.device().is_cuda()
        || a.dtype() != DType::F32
        || b.dtype() != DType::F32
        || c.dtype() != DType::F32
        || w1.dtype() != DType::F32
        || w2.dtype() != DType::F32
    {
        let na = crate::tensor::ops::rms_norm(a, w1, eps)?;
        let nb = crate::tensor::ops::rms_norm(b, w2, eps)?;
        return (na + nb)? + c;
    }

    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let c_c = c.contiguous()?;
    let w1_c = w1.contiguous()?;
    let w2_c = w2.contiguous()?;

    let total_elems = a_c.elem_count();
    let cols = *a_c.dims().last().unwrap_or(&1);
    let rows = total_elems / cols;
    let out = unsafe { Tensor::empty(a_c.shape(), DType::F32, &a_c.device())? };

    let cuda_dev = a_c.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_dual_rmsnorm_add_f32", "loken_fused", ptx)?;

    let (a_store, a_layout) = a_c.storage_and_layout();
    let (b_store, b_layout) = b_c.storage_and_layout();
    let (c_store, c_layout) = c_c.storage_and_layout();
    let (w1_store, _) = w1_c.storage_and_layout();
    let (w2_store, _) = w2_c.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(asg),
        crate::tensor::StorageView::Cuda(bs),
        crate::tensor::StorageView::Cuda(cs),
        crate::tensor::StorageView::Cuda(w1s),
        crate::tensor::StorageView::Cuda(w2s),
        crate::tensor::StorageView::Cuda(os),
    ) = (
        &*a_store, &*b_store, &*c_store, &*w1_store, &*w2_store, &*o_store,
    ) {
        let a_slice = asg.as_cuda_slice::<f32>()?;
        let b_slice = bs.as_cuda_slice::<f32>()?;
        let c_slice = cs.as_cuda_slice::<f32>()?;
        let w1_slice = w1s.as_cuda_slice::<f32>()?;
        let w2_slice = w2s.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let a_view = a_slice.slice(a_layout.start_offset()..);
        let b_view = b_slice.slice(b_layout.start_offset()..);
        let c_view = c_slice.slice(c_layout.start_offset()..);

        let block = 256u32.min(cols as u32).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4 * 2,
        };

        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&a_view);
        builder.arg(&b_view);
        builder.arg(&c_view);
        builder.arg(w1_slice);
        builder.arg(w2_slice);
        builder.arg(o_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_dual_rmsnorm_add: {e}")))?;
    }

    drop(a_store);
    drop(b_store);
    drop(c_store);
    drop(w1_store);
    drop(w2_store);
    drop(o_store);
    Ok(out)
}

/// Fused repeat penalty + argmax - replaces 6+ kernel launches with 1.
/// Applies penalty to specified token indices, then finds argmax.
/// Returns the sampled token ID (with host-blocking sync).
pub fn fused_penalty_argmax(
    logits: &Tensor,
    penalty_token_ids: &[u32],
    repeat_penalty: f32,
) -> Result<u32> {
    let (u, _slice) = fused_penalty_argmax_with_device(logits, penalty_token_ids, repeat_penalty)?;
    Ok(u)
}

/// Same kernel as `fused_penalty_argmax` but also returns the device-side
/// CudaSlice<i32> holding the sampled token. The host u32 is synced for
/// stop-check/EOS logic; the device buffer can be used to launch the next
/// forward's embedding lookup WITHOUT waiting for the sync - /// Path B's pipelining hook.
///
/// The slice lifetime ends when the caller drops the returned tuple, so
/// callers must extract whatever they need (wrap as Tensor, store as
/// session field) before dropping.
pub fn fused_penalty_argmax_with_device(
    logits: &Tensor,
    penalty_token_ids: &[u32],
    repeat_penalty: f32,
) -> Result<(u32, crate::tensor::cuda_ext::CudaSlice<i32>)> {
    if !logits.device().is_cuda() {
        crate::tensor::bail!("fused_penalty_argmax requires CUDA");
    }

    let logits = logits.contiguous()?;
    let vocab_size = logits.elem_count();

    let cuda_dev = logits.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;

    // Two-kernel chain: block_argmax over many blocks (one per SM) +
    // final_reduce over the block candidates.
    // Replaces the single-block scan that nsys showed at 417 µs/call
    // (5.6% of gemma4 decode time at vocab=262144).
    let block_func =
        cuda_dev.get_or_load_custom_func("fused_penalty_argmax_block_f32", "loken_fused", ptx)?;
    let final_func =
        cuda_dev.get_or_load_custom_func("fused_penalty_argmax_final_f32", "loken_fused", ptx)?;

    let penalty_ids: Vec<i32> = penalty_token_ids.iter().map(|&t| t as i32).collect();
    let n_penalty = penalty_ids.len() as i32;
    let penalty_gpu = cuda_dev
        .cuda_stream()
        .clone_htod(&penalty_ids)
        .map_err(|e| crate::tensor::Error::msg(format!("htod: {e}")))?;

    // Pick block count to give each block ~1024 elements (one thread per
    // element after stride). Cap at 256 so the final-reduce stays trivial.
    let block_threads: u32 = 1024;
    let n_blocks: u32 = ((vocab_size as u32).div_ceil(block_threads))
        .min(256)
        .max(1);

    let block_vals = cuda_dev
        .cuda_stream()
        .alloc_zeros::<f32>(n_blocks as usize)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc block_vals: {e}")))?;
    let block_idxs = cuda_dev
        .cuda_stream()
        .alloc_zeros::<i32>(n_blocks as usize)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc block_idxs: {e}")))?;
    let out_token = cuda_dev
        .cuda_stream()
        .alloc_zeros::<i32>(1)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc out: {e}")))?;

    let (logits_store, logits_layout) = logits.storage_and_layout();
    if let crate::tensor::StorageView::Cuda(ls) = &*logits_store {
        let logits_slice = ls.as_cuda_slice::<f32>()?;
        let logits_view = logits_slice.slice(logits_layout.start_offset()..);

        let block_cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: block_threads * 8,
        };
        let vocab_i32 = vocab_size as i32;
        let mut builder = block_func.builder();
        builder.arg(&logits_view);
        builder.arg(&penalty_gpu);
        builder.arg(&n_penalty);
        builder.arg(&repeat_penalty);
        builder.arg(&vocab_i32);
        builder.arg(&block_vals);
        builder.arg(&block_idxs);
        unsafe { builder.launch(block_cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_penalty_argmax_block: {e}")))?;

        // Final reduce: 256-thread block matches n_blocks cap.
        let final_threads: u32 = n_blocks.next_power_of_two().max(32);
        let final_cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (final_threads, 1, 1),
            shared_mem_bytes: final_threads * 8,
        };
        let n_blocks_i32 = n_blocks as i32;
        let mut b2 = final_func.builder();
        b2.arg(&block_vals);
        b2.arg(&block_idxs);
        b2.arg(&n_blocks_i32);
        b2.arg(&out_token);
        unsafe { b2.launch(final_cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_penalty_argmax_final: {e}")))?;
    }
    drop(logits_store);

    let mut result = vec![0i32];
    cuda_dev
        .cuda_stream()
        .memcpy_dtoh(&out_token, &mut result)
        .map_err(|e| crate::tensor::Error::msg(format!("dtoh: {e}")))?;

    Ok((result[0] as u32, out_token))
}

/// Path B step 2: u32-output sister of
/// `fused_penalty_argmax_with_device`. The kernel writes the argmax index
/// directly into an owned `CudaSlice<u32>` - same memory layout as the
/// i32 variant (bit-identical for non-negative vocab indices), but the
/// Rust type matches the reference `DType::U32` so we can wrap as a Tensor via
/// `CudaStorage::wrap_cuda_slice` without an i32->u32 copy kernel and use
/// it as the next forward's embedding-lookup input - no host sync.
pub fn fused_penalty_argmax_u32_with_device(
    logits: &Tensor,
    penalty_token_ids: &[u32],
    repeat_penalty: f32,
) -> Result<(u32, crate::tensor::cuda_ext::CudaSlice<u32>)> {
    if !logits.device().is_cuda() {
        crate::tensor::bail!("fused_penalty_argmax_u32 requires CUDA");
    }

    let logits = logits.contiguous()?;
    let vocab_size = logits.elem_count();

    let cuda_dev = logits.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;

    // Multi-block argmax (mirror of the i32 path). Saves ~5% of decode for
    // gemma4 by using 256 blocks instead of 1.
    let block_func =
        cuda_dev.get_or_load_custom_func("fused_penalty_argmax_block_f32", "loken_fused", ptx)?;
    let final_func = cuda_dev.get_or_load_custom_func(
        "fused_penalty_argmax_final_f32_u32_out",
        "loken_fused",
        ptx,
    )?;

    let penalty_ids: Vec<i32> = penalty_token_ids.iter().map(|&t| t as i32).collect();
    let n_penalty = penalty_ids.len() as i32;
    let penalty_gpu = cuda_dev
        .cuda_stream()
        .clone_htod(&penalty_ids)
        .map_err(|e| crate::tensor::Error::msg(format!("htod: {e}")))?;

    let block_threads: u32 = 1024;
    let n_blocks: u32 = ((vocab_size as u32).div_ceil(block_threads))
        .min(256)
        .max(1);
    let block_vals = cuda_dev
        .cuda_stream()
        .alloc_zeros::<f32>(n_blocks as usize)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc block_vals: {e}")))?;
    let block_idxs = cuda_dev
        .cuda_stream()
        .alloc_zeros::<i32>(n_blocks as usize)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc block_idxs: {e}")))?;
    let out_token = cuda_dev
        .cuda_stream()
        .alloc_zeros::<u32>(1)
        .map_err(|e| crate::tensor::Error::msg(format!("alloc out: {e}")))?;

    let (logits_store, logits_layout) = logits.storage_and_layout();
    if let crate::tensor::StorageView::Cuda(ls) = &*logits_store {
        let logits_slice = ls.as_cuda_slice::<f32>()?;
        let logits_view = logits_slice.slice(logits_layout.start_offset()..);

        let block_cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: block_threads * 8,
        };
        let vocab_i32 = vocab_size as i32;
        let mut b = block_func.builder();
        b.arg(&logits_view);
        b.arg(&penalty_gpu);
        b.arg(&n_penalty);
        b.arg(&repeat_penalty);
        b.arg(&vocab_i32);
        b.arg(&block_vals);
        b.arg(&block_idxs);
        unsafe { b.launch(block_cfg) }.map_err(|e| {
            crate::tensor::Error::msg(format!("fused_penalty_argmax_block (u32): {e}"))
        })?;

        let final_threads: u32 = n_blocks.next_power_of_two().max(32);
        let final_cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (final_threads, 1, 1),
            shared_mem_bytes: final_threads * 8,
        };
        let n_blocks_i32 = n_blocks as i32;
        let mut b2 = final_func.builder();
        b2.arg(&block_vals);
        b2.arg(&block_idxs);
        b2.arg(&n_blocks_i32);
        b2.arg(&out_token);
        unsafe { b2.launch(final_cfg) }.map_err(|e| {
            crate::tensor::Error::msg(format!("fused_penalty_argmax_final (u32): {e}"))
        })?;
    }
    drop(logits_store);

    let mut result = vec![0u32];
    cuda_dev
        .cuda_stream()
        .memcpy_dtoh(&out_token, &mut result)
        .map_err(|e| crate::tensor::Error::msg(format!("dtoh: {e}")))?;

    Ok((result[0], out_token))
}

/// F32 single-query attention for HD=512 graph-mode bypass.
///
/// Replaces the cuBLAS-backed Q@K^T + softmax + P@V chain used by
/// `padded_standard_attention` for gemma4 Global layers. The cuBLAS
/// path crashes in captured graph because cuBLAS workspace state isn't
/// tracked by the Tensor lifetime. The external flash-attention kernel
/// fallback is also broken at HD=512 on sm_120. This custom kernel
/// bypasses both.
///
/// Shapes:
/// - `q`: `[1, n_q_heads, 1, 512]` F32 contiguous
/// - `k`: `[1, n_kv_heads, max_kv_padded, 512]` F32 contiguous (padded buffer)
/// - `v`: `[1, n_kv_heads, max_kv_padded, 512]` F32 contiguous (padded buffer)
/// - `mask`: `[max_kv_padded]` F32 (`0.0` for valid positions, `-INFINITY` for padding)
/// - `seq_kv_dev`: device `i32` storing `current_seq_len - 1` (matches Q8 convention)
/// - `scale`: typically `1.0 / sqrt(head_dim)`
///
/// Returns `[1, n_q_heads, 1, 512]` F32. NOT thread-safe across streams  -
/// caller must ensure exclusive access during graph capture.
pub fn fused_attn_decode_f32_hd512(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    seq_kv_dev: &crate::tensor::cuda_ext::CudaSlice<i32>,
    n_q_per_kv: usize,
    scale: f32,
) -> Result<Tensor> {
    use crate::tensor::cuda_ext::PushKernelArg;
    if !q.device().is_cuda() {
        crate::tensor::bail!("fused_attn_decode_f32_hd512 requires CUDA");
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        crate::tensor::bail!("fused_attn_decode_f32_hd512 requires F32 Q/K/V");
    }
    let (qb, n_q_heads, qs, qhd) = q.dims4()?;
    let (kb, n_kv_heads, max_kv_padded, khd) = k.dims4()?;
    let (vb, n_kv_v, max_kv_v, vhd) = v.dims4()?;
    if qb != 1 || kb != 1 || vb != 1 {
        crate::tensor::bail!("fused_attn_decode_f32_hd512 requires batch=1");
    }
    if qs != 1 {
        crate::tensor::bail!("fused_attn_decode_f32_hd512 requires seq_q=1 (decode)");
    }
    if qhd != 512 || khd != 512 || vhd != 512 {
        crate::tensor::bail!("fused_attn_decode_f32_hd512 requires HD=512");
    }
    if n_kv_heads != n_kv_v {
        crate::tensor::bail!("K and V kv_head mismatch");
    }
    if max_kv_padded != max_kv_v {
        crate::tensor::bail!("K and V max_kv mismatch");
    }
    if n_q_heads != n_kv_heads * n_q_per_kv {
        crate::tensor::bail!(
            "n_q_heads {n_q_heads} != n_kv_heads {n_kv_heads} * n_q_per_kv {n_q_per_kv}"
        );
    }

    let cuda_dev = q.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_attn_decode_f32_hd512", "loken_fused", ptx)?;

    let out = unsafe { Tensor::empty(&[1, n_q_heads, 1, 512], DType::F32, &q.device())? };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (m_s, m_l) = mask.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(qc),
        crate::tensor::StorageView::Cuda(kc),
        crate::tensor::StorageView::Cuda(vc),
        crate::tensor::StorageView::Cuda(mc),
        crate::tensor::StorageView::Cuda(oc),
    ) = (&*q_s, &*k_s, &*v_s, &*m_s, &*o_s)
    {
        let q_view = qc.as_cuda_slice::<f32>()?.slice(q_l.start_offset()..);
        let k_view = kc.as_cuda_slice::<f32>()?.slice(k_l.start_offset()..);
        let v_view = vc.as_cuda_slice::<f32>()?.slice(v_l.start_offset()..);
        let m_view = mc.as_cuda_slice::<f32>()?.slice(m_l.start_offset()..);
        let o_slice = oc.as_cuda_slice::<f32>()?;

        const HD: u32 = 512;
        const THREADS: u32 = 256;
        const TILE_KV: u32 = 256;
        // Online-softmax kernel: smem footprint is O(1) in max_kv_padded.
        // = HD * 4 (Q) + TILE_KV * 4 (per-tile scores) = 2048 + 1024 = 3 KB.
        let smem_bytes = (HD + TILE_KV) as usize * 4;

        // One-shot diagnostic on first invocation.
        use std::sync::atomic::{AtomicBool, Ordering};
        static FIRST_CALL: AtomicBool = AtomicBool::new(true);
        if FIRST_CALL.swap(false, Ordering::Relaxed) {
            tracing::info!(
                "fused_attn_decode_f32_hd512 first call: n_q_heads={n_q_heads} \
                 n_kv_heads={n_kv_heads} n_q_per_kv={n_q_per_kv} max_kv_padded={max_kv_padded} \
                 smem_bytes={smem_bytes} (tiled online-softmax, O(1) in max_kv_padded)"
            );
        }

        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_q_heads as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };

        let max_kv_i32 = max_kv_padded as i32;
        let n_kv_i32 = n_kv_heads as i32;
        let n_q_per_kv_i32 = n_q_per_kv as i32;
        let mut builder = func.builder();
        builder.arg(&q_view);
        builder.arg(&k_view);
        builder.arg(&v_view);
        builder.arg(&m_view);
        builder.arg(o_slice);
        builder.arg(seq_kv_dev);
        builder.arg(&max_kv_i32);
        builder.arg(&n_kv_i32);
        builder.arg(&n_q_per_kv_i32);
        builder.arg(&scale);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_attn_decode_f32_hd512: {e}")))?;
    }
    drop(q_s);
    drop(k_s);
    drop(v_s);
    drop(m_s);
    drop(o_s);
    Ok(out)
}

/// Split-K variant of `fused_attn_decode_f32_hd512`: grid=(n_q_heads, nsplit)
/// for the partial pass (good occupancy at long KV, unlike the 8-block single-
/// pass kernel) + a per-head combine. Same args/contract. Closes the gemma4
/// global-hd512 long-ctx loss where the single-block kernel loses to cuBLAS.
pub fn fused_attn_decode_f32_hd512_splitk(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    seq_kv_dev: &crate::tensor::cuda_ext::CudaSlice<i32>,
    n_q_per_kv: usize,
    scale: f32,
) -> Result<Tensor> {
    use crate::tensor::cuda_ext::PushKernelArg;
    if !q.device().is_cuda() {
        crate::tensor::bail!("hd512_splitk requires CUDA");
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        crate::tensor::bail!("hd512_splitk requires F32 Q/K/V");
    }
    let (qb, n_q_heads, qs, qhd) = q.dims4()?;
    let (kb, n_kv_heads, max_kv_padded, khd) = k.dims4()?;
    let (vb, _n_kv_v, max_kv_v, vhd) = v.dims4()?;
    if qb != 1 || kb != 1 || vb != 1 || qs != 1 {
        crate::tensor::bail!("hd512_splitk requires batch=1, seq_q=1");
    }
    if qhd != 512 || khd != 512 || vhd != 512 || max_kv_padded != max_kv_v {
        crate::tensor::bail!("hd512_splitk shape mismatch");
    }
    if n_q_heads != n_kv_heads * n_q_per_kv {
        crate::tensor::bail!("hd512_splitk head mismatch");
    }
    const HD: usize = 512;
    const THREADS: u32 = 256;
    // Adaptive split count. Each block runs 256 threads, one per KV position
    // in its TILE_KV=256 window - so a split should own ≈256 positions to keep
    // the threads busy (nsplit=ceil(seq_kv/256) ⇒ chunk≈256, ~full occupancy).
    // The old `256/n_heads` formula ignored seq_kv: at 1453 KV it made 32
    // splits of ~45 positions each, idling 82 % of every block's threads.
    // Floor of 4 keeps enough blocks (4.n_heads) to fill the SMs at short KV;
    // cap of 64 bounds the partials buffer + combine cost at very long KV.
    let nsplit: usize = max_kv_padded.div_ceil(256).clamp(4, 64);

    let cuda_dev = q.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let partial_func = cuda_dev.get_or_load_custom_func(
        "fused_attn_decode_f32_hd512_splitk_partial",
        "loken_fused",
        ptx,
    )?;
    let combine_func = cuda_dev.get_or_load_custom_func(
        "fused_attn_decode_f32_hd512_splitk_combine",
        "loken_fused",
        ptx,
    )?;

    let partials =
        unsafe { Tensor::empty(&[n_q_heads * nsplit * (HD + 2)], DType::F32, &q.device())? };
    let out = unsafe { Tensor::empty(&[1, n_q_heads, 1, 512], DType::F32, &q.device())? };

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (m_s, m_l) = mask.storage_and_layout();
    let (p_s, _) = partials.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();
    if let (
        crate::tensor::StorageView::Cuda(qc),
        crate::tensor::StorageView::Cuda(kc),
        crate::tensor::StorageView::Cuda(vc),
        crate::tensor::StorageView::Cuda(mc),
        crate::tensor::StorageView::Cuda(pc),
        crate::tensor::StorageView::Cuda(oc),
    ) = (&*q_s, &*k_s, &*v_s, &*m_s, &*p_s, &*o_s)
    {
        let q_view = qc.as_cuda_slice::<f32>()?.slice(q_l.start_offset()..);
        let k_view = kc.as_cuda_slice::<f32>()?.slice(k_l.start_offset()..);
        let v_view = vc.as_cuda_slice::<f32>()?.slice(v_l.start_offset()..);
        let m_view = mc.as_cuda_slice::<f32>()?.slice(m_l.start_offset()..);
        let p_slice = pc.as_cuda_slice::<f32>()?;
        let o_slice = oc.as_cuda_slice::<f32>()?;

        let smem_bytes = (HD + 256) * 4;
        let max_kv_i32 = max_kv_padded as i32;
        let n_kv_i32 = n_kv_heads as i32;
        let n_q_per_kv_i32 = n_q_per_kv as i32;
        let nsplit_i32 = nsplit as i32;

        let cfg1 = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_q_heads as u32, nsplit as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };
        let mut b1 = partial_func.builder();
        b1.arg(&q_view);
        b1.arg(&k_view);
        b1.arg(&v_view);
        b1.arg(&m_view);
        b1.arg(p_slice);
        b1.arg(seq_kv_dev);
        b1.arg(&max_kv_i32);
        b1.arg(&n_kv_i32);
        b1.arg(&n_q_per_kv_i32);
        b1.arg(&nsplit_i32);
        b1.arg(&scale);
        unsafe { b1.launch(cfg1) }
            .map_err(|e| crate::tensor::Error::msg(format!("hd512_splitk partial: {e}")))?;

        let cfg2 = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_q_heads as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b2 = combine_func.builder();
        b2.arg(p_slice);
        b2.arg(o_slice);
        b2.arg(&nsplit_i32);
        unsafe { b2.launch(cfg2) }
            .map_err(|e| crate::tensor::Error::msg(format!("hd512_splitk combine: {e}")))?;
    }
    drop(q_s);
    drop(k_s);
    drop(v_s);
    drop(m_s);
    drop(p_s);
    drop(o_s);
    Ok(out)
}

#[cfg(test)]
mod adaptive_grid_tests {
    use super::*;

    /// `adaptive_grid_1d` is shared by every fused-kernel launch site
    /// (silu_mul, gelu_mul, split_gelu_mul, add_rmsnorm, ...). Its job
    /// is to shrink the default 256-thread block down to MIN_BLOCK=64
    /// when the natural grid count would leave SMs idle on a large
    /// GPU - without it, small inputs on a 128-SM card would launch
    /// only a few blocks and waste 90 % of the SMs. Pin the contract
    /// so a refactor can't silently degrade kernel occupancy.

    #[test]
    pub(super) fn default_block_size_is_256_when_grid_already_covers_sms() {
        // Large n / small SM count: default block of 256 is fine because
        // grid (n/256) already exceeds sm_count.
        let (block, grid) = adaptive_grid_1d(1_000_000, 32);
        assert_eq!(block, 256, "block should stay at default 256");
        assert!(grid >= 32, "grid {grid} must cover all 32 SMs");
    }

    #[test]
    pub(super) fn block_shrinks_to_fill_idle_sms_on_small_inputs() {
        // Small n on a wide GPU: default block=256, grid=ceil(1024/256)=4.
        // With 32 SMs we'd waste 28 of them. The helper should shrink
        // block until grid >= sm_count OR the floors trip.
        let (block, grid) = adaptive_grid_1d(1024, 32);
        // Block must shrink below 256 to grow grid past the SM count.
        assert!(block < 256, "block should shrink, got {block}");
        // Grid must cover (or come close to) the SM count.
        assert!(grid >= 16, "grid {grid} too small to keep SMs busy");
    }

    #[test]
    pub(super) fn block_floors_at_min_block_64() {
        // Very small n on a very wide GPU: even shrinking to MIN_BLOCK
        // can't generate sm_count blocks. The helper must stop at the
        // floor (64) rather than going to 32 / 16 / etc., which would
        // hurt per-block performance more than the SM-occupancy gain.
        let (block, _grid) = adaptive_grid_1d(128, 128);
        assert!(
            block >= 64,
            "block must stay at or above MIN_BLOCK=64, got {block}"
        );
    }

    #[test]
    pub(super) fn block_size_caps_expansion_at_4x_shrink() {
        // MAX_EXPANSION = 4. Starting from 256, the smallest reachable
        // block is 256 / 4 = 64 - which happens to equal MIN_BLOCK.
        // Pin: the helper never goes below 64 regardless of inputs.
        for n in [1, 32, 64, 128, 256, 511] {
            let (block, _) = adaptive_grid_1d(n, 512);
            assert!(block >= 64, "n={n}: block {block} dropped below 64");
        }
    }

    #[test]
    pub(super) fn grid_x_block_always_covers_n() {
        // Critical invariant: every work item must have at least one
        // thread. grid * block >= n for any input the helper sees.
        // Otherwise the kernel skips items past the last block boundary.
        for (n, sm) in [(1u32, 1), (255, 8), (1000, 32), (1_000_000, 128)] {
            let (block, grid) = adaptive_grid_1d(n, sm);
            assert!(
                (block as u64) * (grid as u64) >= n as u64,
                "n={n} sm={sm}: grid {grid} x block {block} = {} < n",
                (block as u64) * (grid as u64),
            );
        }
    }

    #[test]
    pub(super) fn returns_at_least_one_block_for_nonzero_n() {
        // Even tiny n must yield grid >= 1 so the kernel actually runs.
        let (_block, grid) = adaptive_grid_1d(1, 32);
        assert!(grid >= 1, "grid {grid} must be >= 1");
    }

    #[test]
    pub(super) fn handles_n_equal_to_zero_without_panicking() {
        // n=0 shouldn't crash the kernel-dispatch path; the launch sites
        // check elem_count themselves before calling, but the helper
        // should still be safe.
        let (block, grid) = adaptive_grid_1d(0, 32);
        assert!(block >= 64, "block {block}");
        assert_eq!(grid, 0, "zero work -> zero blocks");
    }
}
