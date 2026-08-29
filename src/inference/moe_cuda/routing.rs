//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// cfg-parity twin of `moe_cuda_cpu::clear_expert_cache` - the CUDA module
/// keeps no per-expert CPU QMatMul cache, so there is nothing to drop.
pub fn clear_expert_cache() {}

/// Top-k softmax over expert logits -> (weights, expert ids).
pub fn topk_softmax(
    logits: &Tensor,
    n_expert_used: usize,
    with_norm: bool,
) -> Result<(Tensor, Tensor)> {
    let dims = logits.dims();
    if dims.len() != 2 {
        crate::tensor::bail!("topk_softmax requires a 2D logits tensor, got {:?}", dims);
    }
    let n_rows = dims[0];
    let n_experts = dims[1];
    if !matches!(n_experts, 32 | 64 | 128 | 256 | 512) {
        crate::tensor::bail!(
            "topk_softmax only supports n_experts ∈ {{32, 64, 128, 256}}; got {}",
            n_experts
        );
    }
    if n_expert_used == 0 || n_expert_used > n_experts {
        crate::tensor::bail!(
            "topk_softmax: n_expert_used={} out of range (n_experts={})",
            n_expert_used,
            n_experts
        );
    }
    if logits.dtype() != DType::F32 {
        crate::tensor::bail!(
            "topk_softmax: expected f32 logits, got {:?}",
            logits.dtype()
        );
    }
    let dev = logits.device().as_cuda_device()?;
    let logits = logits.contiguous()?;
    let (logits_storage, _) = logits.storage_and_layout();
    let logits_slice = match &*logits_storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("topk_softmax: logits must be a cuda tensor"),
    };

    let weights = unsafe { dev.alloc::<f32>(n_rows * n_expert_used) }?;
    let ids = unsafe { dev.alloc::<u32>(n_rows * n_expert_used) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        loken_topk_softmax(
            logits_slice.device_ptr(logits_slice.stream()).0 as *const f32,
            weights.device_ptr(weights.stream()).0 as *mut f32,
            ids.device_ptr(ids.stream()).0 as *mut u32,
            n_rows as i32,
            n_experts as i32,
            n_expert_used as i32,
            if with_norm { 1 } else { 0 },
            stream,
        );
    }

    let weights_storage = CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let weights = tensor_from_cuda_storage(weights_storage, (n_rows, n_expert_used))?;
    let ids_storage = CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids = tensor_from_cuda_storage(ids_storage, (n_rows, n_expert_used))?;
    Ok((weights, ids))
}

/// Batched argmax over `[n_rows, vocab]` logits -> `n_rows` token ids in ONE custom CUDA
/// launch (not a slow per-op argmax + per-row host scan). For the continuous-batching
/// deterministic decode sampler. Ties break to the lowest index (host-argmax parity).
pub fn batched_argmax(logits: &Tensor) -> Result<Vec<u32>> {
    let (n_rows, vocab) = logits.dims2()?;
    let dev = logits.device().as_cuda_device()?;
    let logits = logits.contiguous()?;
    let (storage, _) = logits.storage_and_layout();
    let out = unsafe { dev.alloc::<u32>(n_rows) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let out_ptr = out.device_ptr(out.stream()).0 as *mut u32;
    match logits.dtype() {
        DType::F16 => {
            let s = match &*storage {
                StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => crate::tensor::bail!("batched_argmax: cuda only"),
            };
            unsafe {
                loken_batched_argmax_f16(
                    s.device_ptr(s.stream()).0 as *const std::ffi::c_void,
                    out_ptr,
                    n_rows as i32,
                    vocab as i32,
                    stream,
                )
            };
        }
        DType::F32 => {
            let s = match &*storage {
                StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => crate::tensor::bail!("batched_argmax: cuda only"),
            };
            unsafe {
                loken_batched_argmax_f32(
                    s.device_ptr(s.stream()).0 as *const std::ffi::c_void,
                    out_ptr,
                    n_rows as i32,
                    vocab as i32,
                    stream,
                )
            };
        }
        d => crate::tensor::bail!("batched_argmax: unsupported dtype {d:?}"),
    }
    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    let t = tensor_from_cuda_storage(out_storage, (n_rows,))?;
    Ok(t.to_vec1::<u32>()?)
}

/// Batched top-k + softmax-denominator front-end for the CB sampled path. Per row returns
/// the top-`k` logits (descending) + their vocab indices + (rowmax, denom) - one custom
/// CUDA launch, transferring only [n_rows.k]+[n_rows.2] instead of the full [n_rows,vocab].
/// The host reconstructs full-vocab probs `exp((l-rowmax)/T)/denom` and finishes top-p +
/// multinomial over just k entries. WARNING: masks `logits` in place (safe on the CB decode
/// buffer, overwritten by the next graph replay). Returns (top_logits, top_idx, stats).
pub fn batched_topk_denom(
    logits: &Tensor,
    k: usize,
    temperature: f64,
    pen_rows: &[i32],
    pen_toks: &[u32],
    rp: f32,
) -> Result<(Vec<f32>, Vec<u32>, Vec<f32>)> {
    let (n_rows, vocab) = logits.dims2()?;
    let dev = logits.device().as_cuda_device()?;
    let logits = logits.contiguous()?;
    let (storage, _) = logits.storage_and_layout();
    let out_logit = unsafe { dev.alloc::<f32>(n_rows * k) }?;
    let out_idx = unsafe { dev.alloc::<u32>(n_rows * k) }?;
    let out_stats = unsafe { dev.alloc::<f32>(n_rows * 2) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (lp, ip, sp) = (
        out_logit.device_ptr(out_logit.stream()).0 as *mut f32,
        out_idx.device_ptr(out_idx.stream()).0 as *mut u32,
        out_stats.device_ptr(out_stats.stream()).0 as *mut f32,
    );
    let inv_temp = (1.0 / temperature) as f32;
    // Upload the flattened (row, recent-token) pairs for the in-place repeat-penalty scatter.
    let n_pen = pen_rows.len();
    let (rp_ptr, tk_ptr, rows_dev, toks_dev) = if n_pen > 0 && rp > 1.0 {
        let rows_dev = Tensor::from_vec(pen_rows.to_vec(), (n_pen,), &crate::tensor::Device::Cpu)?
            .to_device(&logits.device())?;
        let toks_dev = Tensor::from_vec(pen_toks.to_vec(), (n_pen,), &crate::tensor::Device::Cpu)?
            .to_device(&logits.device())?;
        let (rs, _) = rows_dev.storage_and_layout();
        let (ts, _) = toks_dev.storage_and_layout();
        let rptr = match &*rs {
            StorageView::Cuda(c) => {
                let sl = c.as_cuda_slice::<i32>()?;
                sl.device_ptr(sl.stream()).0 as *const i32
            }
            _ => std::ptr::null(),
        };
        let tptr = match &*ts {
            StorageView::Cuda(c) => {
                let sl = c.as_cuda_slice::<u32>()?;
                sl.device_ptr(sl.stream()).0 as *const u32
            }
            _ => std::ptr::null(),
        };
        (rptr, tptr, Some(rows_dev), Some(toks_dev))
    } else {
        (std::ptr::null(), std::ptr::null(), None, None)
    };
    let _keep = (rows_dev, toks_dev);
    match logits.dtype() {
        DType::F16 => {
            let s = match &*storage {
                StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => crate::tensor::bail!("cuda only"),
            };
            let lg = s.device_ptr(s.stream()).0 as *mut std::ffi::c_void;
            if !rp_ptr.is_null() {
                unsafe {
                    loken_repeat_penalty_f16(
                        lg,
                        rp_ptr,
                        tk_ptr,
                        rp,
                        n_pen as i32,
                        vocab as i32,
                        stream,
                    )
                };
            }
            unsafe {
                loken_batched_topk_denom_f16(
                    lg,
                    lp,
                    ip,
                    sp,
                    n_rows as i32,
                    vocab as i32,
                    k as i32,
                    inv_temp,
                    stream,
                )
            };
        }
        DType::F32 => {
            let s = match &*storage {
                StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => crate::tensor::bail!("cuda only"),
            };
            let lg = s.device_ptr(s.stream()).0 as *mut std::ffi::c_void;
            if !rp_ptr.is_null() {
                unsafe {
                    loken_repeat_penalty_f32(
                        lg,
                        rp_ptr,
                        tk_ptr,
                        rp,
                        n_pen as i32,
                        vocab as i32,
                        stream,
                    )
                };
            }
            unsafe {
                loken_batched_topk_denom_f32(
                    lg,
                    lp,
                    ip,
                    sp,
                    n_rows as i32,
                    vocab as i32,
                    k as i32,
                    inv_temp,
                    stream,
                )
            };
        }
        d => crate::tensor::bail!("batched_topk_denom: unsupported dtype {d:?}"),
    }
    let tl = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out_logit, dev.clone()),
        (n_rows * k,),
    )?
    .to_vec1::<f32>()?;
    let ti = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out_idx, dev.clone()),
        (n_rows * k,),
    )?
    .to_vec1::<u32>()?;
    let ts = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out_stats, dev.clone()),
        (n_rows * 2,),
    )?
    .to_vec1::<f32>()?;
    Ok((tl, ti, ts))
}
