//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Fused router GEMV + top-k softmax for the MoE gate.
/// Returns `Ok(None)` if the shape isn't supported.
pub fn gate_topk_softmax(
    xs: &Tensor,
    gate_w: &Tensor,
    n_expert_used: usize,
    with_norm: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if xs.dtype() != DType::F32 || gate_w.dtype() != DType::F32 || !xs.device().is_cuda() {
        return Ok(None);
    }
    let (xd, wd) = (xs.dims(), gate_w.dims());
    if xd.len() != 2 || wd.len() != 2 {
        return Ok(None);
    }
    let (n_rows, hidden) = (xd[0], xd[1]);
    let (n_experts, w_hidden) = (wd[0], wd[1]);
    if hidden != w_hidden || hidden > 8192 || (hidden & 3) != 0 {
        return Ok(None);
    }
    if !matches!(n_experts, 32 | 64 | 128 | 256) {
        return Ok(None);
    }
    if n_expert_used == 0 || n_expert_used > n_experts {
        return Ok(None);
    }
    let dev = xs.device().as_cuda_device()?;
    let xs = xs.contiguous()?;
    let gate_w = gate_w.contiguous()?;
    let (xs_st, _) = xs.storage_and_layout();
    let (gw_st, _) = gate_w.storage_and_layout();
    let xs_s = match &*xs_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let gw_s = match &*gw_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let weights = unsafe { dev.alloc::<f32>(n_rows * n_expert_used) }?;
    let ids = unsafe { dev.alloc::<u32>(n_rows * n_expert_used) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_gate_topk_softmax(
            xs_s.device_ptr(xs_s.stream()).0 as *const c_void,
            gw_s.device_ptr(gw_s.stream()).0 as *const c_void,
            weights.device_ptr(weights.stream()).0 as *mut c_void,
            ids.device_ptr(ids.stream()).0 as *mut c_void,
            hidden as i32,
            n_rows as i32,
            n_experts as i32,
            n_expert_used as i32,
            if with_norm { 1 } else { 0 },
            stream,
        );
    }
    let weights = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(weights, dev.clone()),
        (n_rows, n_expert_used),
    )?;
    let ids = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(ids, dev.clone()),
        (n_rows, n_expert_used),
    )?;
    Ok(Some((weights, ids)))
}

/// POST-matmul sigmoid router: takes pre-computed logits [n_rows, n_experts]
/// (from the fast cuBLAS gemv) and fuses sigmoid + bias-select top-k +
/// gather(unbiased) + renorm + scale into ONE launch - keeps the gemv, erases
/// the ~7 tiny post-ops (incl. the asort) that dominate launch-bound MoE decode.
/// Returns `Ok(None)` if the shape/expert-count isn't supported.
pub fn topk_sigmoid_post(
    logits: &Tensor,
    bias: Option<&Tensor>,
    n_expert_used: usize,
    with_norm: bool,
    scale: f64,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if logits.dtype() != DType::F32 || !logits.device().is_cuda() {
        return Ok(None);
    }
    let d = logits.dims();
    if d.len() != 2 {
        return Ok(None);
    }
    let (n_rows, n_experts) = (d[0], d[1]);
    if !matches!(n_experts, 32 | 64 | 128 | 256) {
        return Ok(None);
    }
    if n_expert_used == 0 || n_expert_used > n_experts {
        return Ok(None);
    }
    let bias_c = match bias {
        Some(b) => {
            if b.dtype() != DType::F32 || b.dims() != [n_experts] {
                return Ok(None);
            }
            Some(b.contiguous()?)
        }
        None => None,
    };
    let dev = logits.device().as_cuda_device()?;
    let logits = logits.contiguous()?;
    let (lg_st, _) = logits.storage_and_layout();
    let lg_s = match &*lg_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let bias_st = bias_c.as_ref().map(|b| b.storage_and_layout());
    let bias_ptr = match &bias_st {
        Some((bs, _)) => match &**bs {
            StorageView::Cuda(c) => {
                let s = c.as_cuda_slice::<f32>()?;
                s.device_ptr(s.stream()).0 as *const c_void
            }
            _ => return Ok(None),
        },
        None => core::ptr::null(),
    };
    let weights = unsafe { dev.alloc::<f32>(n_rows * n_expert_used) }?;
    let ids = unsafe { dev.alloc::<u32>(n_rows * n_expert_used) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_topk_sigmoid_post(
            lg_s.device_ptr(lg_s.stream()).0 as *const c_void,
            bias_ptr,
            weights.device_ptr(weights.stream()).0 as *mut c_void,
            ids.device_ptr(ids.stream()).0 as *mut c_void,
            n_rows as i32,
            n_experts as i32,
            n_expert_used as i32,
            if with_norm { 1 } else { 0 },
            scale as f32,
            stream,
        );
    }
    let weights = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(weights, dev.clone()),
        (n_rows, n_expert_used),
    )?;
    let ids = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(ids, dev.clone()),
        (n_rows, n_expert_used),
    )?;
    Ok(Some((weights, ids)))
}

/// Decode-path argsort of a flat [m] u32 expert-id tensor (m <= 32). Drop-in for
/// `topk_flat.sort_last_dim(true)` - returns (sorted_ascending, stable_argsort) with
/// identical semantics, in one warp launch (avoids the reference pipeline-stalling sort).
/// Returns `Ok(None)` if not applicable (m>32, non-u32, non-CUDA).
pub fn argsort_small_u32(flat: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if flat.dtype() != DType::U32 || !flat.device().is_cuda() {
        return Ok(None);
    }
    let d = flat.dims();
    if d.len() != 1 {
        return Ok(None);
    }
    let m = d[0];
    if m == 0 || m > 32 {
        return Ok(None);
    }
    let dev = flat.device().as_cuda_device()?;
    let flat = flat.contiguous()?;
    let (in_st, _) = flat.storage_and_layout();
    let in_s = match &*in_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => return Ok(None),
    };
    let sorted = unsafe { dev.alloc::<u32>(m) }?;
    let index = unsafe { dev.alloc::<u32>(m) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_argsort_small_u32(
            in_s.device_ptr(in_s.stream()).0 as *const c_void,
            sorted.device_ptr(sorted.stream()).0 as *mut c_void,
            index.device_ptr(index.stream()).0 as *mut c_void,
            m as i32,
            stream,
        );
    }
    let sorted = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(sorted, dev.clone()), (m,))?;
    let index = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(index, dev.clone()), (m,))?;
    Ok(Some((sorted, index)))
}

/// F16-I/O LFM2 gated short-conv: bcx [b,3D] F16, state [b,D,L-1] F32, conv_w
/// [D,L] F32 -> (y [b,D] F16, new_state [b,D,L-1] F32). Bit-identical to the F32
/// kernel (same internal F32 math + F16↔F32 conversions) but reads/writes F16 so
/// the surrounding in_proj->F32 / y->F16 cast launches vanish. `Ok(None)` if shapes
/// unsupported (non-CUDA, wrong dtypes).
pub fn lfm2_shortconv_f16io(
    bcx: &Tensor,
    state: &Tensor,
    conv_w: &Tensor,
    d_model: usize,
    l_cache: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if !bcx.device().is_cuda() || bcx.dtype() != DType::F16 {
        return Ok(None);
    }
    if state.dtype() != DType::F32 || conv_w.dtype() != DType::F32 {
        return Ok(None);
    }
    let bd = bcx.dims();
    if bd.len() != 2 || bd[1] != 3 * d_model {
        return Ok(None);
    }
    let b = bd[0];
    let dev = bcx.device().as_cuda_device()?;
    let in_place = super::in_place_state_on();
    // An in-place state is written where it lies, so it is used as it is, never a copy.
    if in_place && !state.is_contiguous() {
        crate::tensor::bail!("lfm2_shortconv_f16io: an in-place state must be contiguous");
    }
    let bcx = bcx.contiguous()?;
    let state = state.contiguous()?;
    let conv_w = conv_w.contiguous()?;
    let (bs, _) = bcx.storage_and_layout();
    let (ss, _) = state.storage_and_layout();
    let (ws, _) = conv_w.storage_and_layout();
    let bcx_s = match &*bs {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
        _ => return Ok(None),
    };
    let st_s = match &*ss {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let cw_s = match &*ws {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let y = unsafe { dev.alloc::<half::f16>(b * d_model) }?;
    let new_state = if in_place {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(b * d_model * (l_cache - 1)) }?)
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_lfm2_shortconv_f16io(
            bcx_s.device_ptr(bcx_s.stream()).0 as *const c_void,
            st_s.device_ptr(st_s.stream()).0 as *const c_void,
            cw_s.device_ptr(cw_s.stream()).0 as *const c_void,
            y.device_ptr(y.stream()).0 as *mut c_void,
            match &new_state {
                Some(n) => n.device_ptr(n.stream()).0 as *mut c_void,
                None => st_s.device_ptr(st_s.stream()).0 as *mut c_void,
            },
            b as i32,
            d_model as i32,
            l_cache as i32,
            stream,
        );
    }
    let y = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(y, dev.clone()), (b, d_model))?;
    let new_state = match new_state {
        Some(n) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(n, dev.clone()),
            (b, d_model, l_cache - 1),
        )?,
        None => state.clone(),
    };
    Ok(Some((y, new_state)))
}

/// The LFM2 gated short convolution over every row of a chunk in one launch: `bcx`
/// `[b, seq, 3D]` F16 or F32, `state` `[b, D, L-1]` F32, `conv_w` `[D, L]` F32.
/// Returns `y` `[b, seq, D]` in the dtype of `bcx` and the state after the last row.
/// The rows and the state are those of `seq` calls of [`lfm2_shortconv_f16io`], bit for bit.
/// `Ok(None)` when the kernel does not take the inputs: not on a card, another dtype, or a
/// window longer than the kernel carries.
pub fn lfm2_shortconv_rows(
    bcx: &Tensor,
    state: &Tensor,
    conv_w: &Tensor,
    d_model: usize,
    l_cache: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if !bcx.device().is_cuda() || !matches!(bcx.dtype(), DType::F16 | DType::F32) {
        return Ok(None);
    }
    if state.dtype() != DType::F32 || conv_w.dtype() != DType::F32 {
        return Ok(None);
    }
    let (b, seq, width) = bcx.dims3()?;
    if width != 3 * d_model || l_cache < 2 || seq == 0 {
        return Ok(None);
    }
    if l_cache - 1 > unsafe { loken_lfm2_shortconv_rows_window() } as usize {
        return Ok(None);
    }
    let dev = bcx.device().as_cuda_device()?;
    let in_place = super::in_place_state_on();
    // An in-place state is written where it lies, so it is used as it is, never a copy.
    if in_place && !state.is_contiguous() {
        crate::tensor::bail!("lfm2_shortconv_rows: an in-place state must be contiguous");
    }
    let in_f16 = bcx.dtype() == DType::F16;
    let bcx = bcx.contiguous()?;
    let state = state.contiguous()?;
    let conv_w = conv_w.contiguous()?;
    let (bs, bl) = bcx.storage_and_layout();
    let (ss, sl) = state.storage_and_layout();
    let (ws, wl) = conv_w.storage_and_layout();
    let StorageView::Cuda(bc) = &*bs else {
        return Ok(None);
    };
    let StorageView::Cuda(sc) = &*ss else {
        return Ok(None);
    };
    let StorageView::Cuda(wc) = &*ws else {
        return Ok(None);
    };
    let elem = bcx.dtype().size_in_bytes();
    let bcx_p = if in_f16 {
        let s = bc.as_cuda_slice::<half::f16>()?;
        s.device_ptr(s.stream()).0
    } else {
        let s = bc.as_cuda_slice::<f32>()?;
        s.device_ptr(s.stream()).0
    } + (bl.start_offset() * elem) as u64;
    let st_s = sc.as_cuda_slice::<f32>()?;
    let st_p = (st_s.device_ptr(st_s.stream()).0 as *mut f32).wrapping_add(sl.start_offset());
    let cw_s = wc.as_cuda_slice::<f32>()?;
    let cw_p = (cw_s.device_ptr(cw_s.stream()).0 as *const f32).wrapping_add(wl.start_offset());
    let n = b * seq * d_model;
    let y_f16 = if in_f16 {
        Some(unsafe { dev.alloc::<half::f16>(n) }?)
    } else {
        None
    };
    let y_f32 = if in_f16 {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(n) }?)
    };
    let y_p = match (&y_f16, &y_f32) {
        (Some(y), _) => y.device_ptr(y.stream()).0,
        (_, Some(y)) => y.device_ptr(y.stream()).0,
        _ => unreachable!("one output is allocated"),
    };
    let new_state = if in_place {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(b * d_model * (l_cache - 1)) }?)
    };
    let out_p = match &new_state {
        Some(n) => n.device_ptr(n.stream()).0 as *mut f32,
        None => st_p,
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_lfm2_shortconv_rows(
            bcx_p as *const c_void,
            st_p,
            cw_p,
            y_p as *mut c_void,
            out_p,
            b as i32,
            seq as i32,
            d_model as i32,
            l_cache as i32,
            in_f16 as i32,
            stream,
        );
    }
    drop(bs);
    drop(ss);
    drop(ws);
    let y = match (y_f16, y_f32) {
        (Some(y), _) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(y, dev.clone()),
            (b, seq, d_model),
        )?,
        (_, Some(y)) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(y, dev.clone()),
            (b, seq, d_model),
        )?,
        _ => unreachable!("one output is allocated"),
    };
    let new_state = match new_state {
        Some(n) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(n, dev.clone()),
            (b, d_model, l_cache - 1),
        )?,
        None => state.clone(),
    };
    Ok(Some((y, new_state)))
}
