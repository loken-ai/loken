//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// In-place variant for the CUDA-graph decode path: updates `state` IN PLACE
/// (state_out == state_in - the shift kernel is in-place-safe: it reads all of
/// `state` into the accumulator before writing, and the shift `so[k]=st[k+1]`
/// only overwrites lower indices it has already consumed). Returns y only. No
/// per-token allocation -> no MEM_ALLOC node to block graph replay. Bit-identical
/// to `lfm2_shortconv_f16io` (same kernel, same math). `state` must be a fixed,
/// contiguous, persistent buffer.
pub fn lfm2_shortconv_f16io_inplace(
    bcx: &Tensor,
    state: &Tensor,
    conv_w: &Tensor,
    d_model: usize,
    l_cache: usize,
) -> Result<Option<Tensor>> {
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
    let bcx = bcx.contiguous()?;
    let conv_w = conv_w.contiguous()?;
    let (bs, _) = bcx.storage_and_layout();
    let (ss, _) = state.storage_and_layout(); // persistent buffer is already contiguous
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
    let st_ptr = st_s.device_ptr(st_s.stream()).0;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_lfm2_shortconv_f16io(
            bcx_s.device_ptr(bcx_s.stream()).0 as *const c_void,
            st_ptr as *const c_void,
            cw_s.device_ptr(cw_s.stream()).0 as *const c_void,
            y.device_ptr(y.stream()).0 as *mut c_void,
            st_ptr as *mut c_void, // state_out == state_in: in-place shift
            b as i32,
            d_model as i32,
            l_cache as i32,
            stream,
        );
    }
    let y = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(y, dev.clone()), (b, d_model))?;
    Ok(Some(y))
}

/// Scatter a per-token F16 K/V residual into the ring slot.
///
/// # Safety
/// `src` and `dst` must be valid CUDA device pointers on the device `stream`
/// belongs to, `src` readable for `n_kv * head_dim` F16 elements and `dst`
/// writable for the whole ring, `slot` in `[0, 32)`. The kernel is enqueued on
/// `stream`, so both must stay alive until that stream has been synchronised.
pub unsafe fn kv_residual_scatter_f16_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    slot: i32,
    stream: i64,
) {
    loken_kv_residual_scatter_f16(src, dst, n_kv, head_dim, slot, stream)
}

/// Device-slot variant of [`kv_residual_scatter_f16_raw`] - reads the slot from
/// `slot_dev[0] % 32` on the GPU (needed under CUDA-graph capture).
///
/// # Safety
/// As [`kv_residual_scatter_f16_raw`], plus `slot_dev` must be a valid device
/// pointer readable for one 32-bit slot index. The slot is read on the GPU, so
/// it is not validated host-side - a wild value indexes the ring out of bounds.
pub unsafe fn kv_residual_scatter_f16_dev_slot_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    slot_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    stream: i64,
) {
    loken_kv_residual_scatter_f16_dev_slot(src, dst, slot_dev, n_kv, head_dim, stream)
}

/// Byte-copy scatter for the Q4_0 V append path, with a device-side position.
///
/// # Safety
/// `src`, `dst` and `pos_dev` must be valid device pointers on `stream`'s device:
/// `src` readable for `token_bytes`, `dst` writable at the position `pos_dev`
/// names, `pos_dev` readable for one index. `token_bytes` must be a multiple of 4
/// (checked only in debug builds). The position is read on the GPU and therefore
/// unvalidated here.
pub unsafe fn q4_v_scatter_bytes_dev_pos_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    token_bytes: i32,
    stream: i64,
) {
    debug_assert!(token_bytes % 4 == 0, "token_bytes must be multiple of 4");
    loken_q4_v_scatter_bytes_dev_pos(src, dst, pos_dev, token_bytes, stream)
}

/// Conditional Q4_0 quantize+flush of the K residual under graph capture.
///
/// # Safety
/// All four pointers must be valid device pointers on `stream`'s device:
/// `residual` readable for `n_kv * head_dim` elements, `k_blocks` writable for
/// `max_seq_blocks` Q4_0 blocks, `pos_dev` readable for one index. The flush is
/// conditional on the device-side position, which is not validated host-side.
pub unsafe fn flush_k_residual_q4_dev_pos_raw(
    residual: *const core::ffi::c_void,
    k_blocks: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    max_seq_blocks: i32,
    stream: i64,
) {
    loken_flush_k_residual_q4_dev_pos(
        residual,
        k_blocks,
        pos_dev,
        n_kv,
        head_dim,
        max_seq_blocks,
        stream,
    )
}

/// Gated DeltaNet linear-attention recurrence (qwen3.5). Returns `(output,
/// new_state)`.rs`.
pub fn gated_delta_net(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device().as_cuda_device()?;
    let (n_tokens, h, s_v) = q.dims3()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(c) => {
                let c = c.as_cuda_slice::<f32>()?;
                let base = c.device_ptr(c.stream()).0 as *const f32;
                // `device_ptr` is the storage base; a contiguous tensor can still
                // carry a non-zero layout start_offset (e.g. `v` is a contiguous
                // narrow of the conv output at seq==1, offset 2*kd*kg). Without
                // this the kernel reads the wrong slice and the recurrence drifts.
                Ok(unsafe { base.add(layout.start_offset()) })
            }
            _ => crate::tensor::bail!("gated_delta_net: inputs must be CUDA f32"),
        }
    };
    let (qp, kp, vp) = (cuptr(q)?, cuptr(k)?, cuptr(v)?);
    let (gp, bp, sp) = (cuptr(g)?, cuptr(beta)?, cuptr(state)?);
    let o_slice = unsafe { dev.alloc::<f32>(n_tokens * h * s_v) }?;
    let ns_slice = unsafe { dev.alloc::<f32>(h * s_v * s_v) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (sq1, sq2) = (s_v as i64, (h * s_v) as i64);
    let (sv1, sv2) = (s_v as i64, (h * s_v) as i64);
    let (sg1, sg2) = (1i64, h as i64);
    unsafe {
        loken_gated_delta_net(
            qp,
            kp,
            vp,
            gp,
            bp,
            sp,
            o_slice.device_ptr(o_slice.stream()).0 as *mut f32,
            ns_slice.device_ptr(ns_slice.stream()).0 as *mut f32,
            h as i32,
            n_tokens as i32,
            s_v as i32,
            sq1,
            sq2,
            sv1,
            sv2,
            sg1,
            sg2,
            scale,
            stream,
        );
    }
    let o = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(o_slice, dev.clone()),
        (n_tokens, h, s_v),
    )?;
    let ns = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(ns_slice, dev.clone()),
        (h, s_v, s_v),
    )?;
    Ok((o, ns))
}

/// Fused causal depthwise conv1d + SiLU for the DeltaNet input projection.
/// `qkv` [seq, C], `conv_state` [C, K-1], `w` [C, K] (all f32, CUDA). Returns
/// `(out [seq, C], new_conv_state [C, K-1])`. Replaces ~16 tensor ops/layer.
pub fn fused_conv_silu(
    qkv: &Tensor,
    conv_state: &Tensor,
    w: &Tensor,
    conv_kernel: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = qkv.device().as_cuda_device()?;
    let (seq, c) = qkv.dims2()?;
    let k = conv_kernel;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("fused_conv_silu: inputs must be CUDA f32"),
        }
    };
    let (qp, csp, wp) = (cuptr(qkv)?, cuptr(conv_state)?, cuptr(w)?);
    let out_slice = unsafe { dev.alloc::<f32>(seq * c) }?;
    let ncs_slice = unsafe { dev.alloc::<f32>(c * (k - 1)) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_fused_conv_silu(
            qp,
            csp,
            wp,
            out_slice.device_ptr(out_slice.stream()).0 as *mut f32,
            ncs_slice.device_ptr(ncs_slice.stream()).0 as *mut f32,
            c as i32,
            seq as i32,
            k as i32,
            stream,
        );
    }
    let out = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out_slice, dev.clone()),
        (seq, c),
    )?;
    let ncs = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(ncs_slice, dev.clone()),
        (c, k - 1),
    )?;
    Ok((out, ncs))
}

/// Fused z-gated RMSNorm for the DeltaNet output. `o`,`z` [N, D], `norm_w` [D]
/// (f32, CUDA). Returns `y` [N, D] = rmsnorm(o).norm_w.silu(z). Replaces ~9
/// tensor ops/layer.
pub fn zgate_rmsnorm(o: &Tensor, z: &Tensor, norm_w: &Tensor, eps: f32) -> Result<Tensor> {
    let dev = o.device().as_cuda_device()?;
    let (n, d) = o.dims2()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("zgate_rmsnorm: inputs must be CUDA f32"),
        }
    };
    let (op, zp, wp) = (cuptr(o)?, cuptr(z)?, cuptr(norm_w)?);
    let y_slice = unsafe { dev.alloc::<f32>(n * d) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_zgate_rmsnorm(
            op,
            zp,
            wp,
            y_slice.device_ptr(y_slice.stream()).0 as *mut f32,
            n as i32,
            d as i32,
            eps,
            stream,
        );
    }
    Ok(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(y_slice, dev.clone()),
        (n, d),
    )?)
}

/// Fused DeltaNet gating. `alpha`,`beta_in` [N, H]; `a_log`,`dt_bias` [H]
/// (f32, CUDA). Returns `(g_pre [N,H], beta [N,H])` where
/// `g_pre = a_log.softplus(alpha+dt_bias)` (PRE-exp) and `beta = sigmoid(beta_in)`.
pub fn deltanet_gate(
    alpha: &Tensor,
    beta_in: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = alpha.device().as_cuda_device()?;
    let (n, h) = alpha.dims2()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("deltanet_gate: inputs must be CUDA f32"),
        }
    };
    let (ap, bp, alp, dtp) = (
        cuptr(alpha)?,
        cuptr(beta_in)?,
        cuptr(a_log)?,
        cuptr(dt_bias)?,
    );
    let g_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let b_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_deltanet_gate(
            ap,
            bp,
            alp,
            dtp,
            g_slice.device_ptr(g_slice.stream()).0 as *mut f32,
            b_slice.device_ptr(b_slice.stream()).0 as *mut f32,
            n as i32,
            h as i32,
            stream,
        );
    }
    let g = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(g_slice, dev.clone()), (n, h))?;
    let beta =
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(b_slice, dev.clone()), (n, h))?;
    Ok((g, beta))
}
