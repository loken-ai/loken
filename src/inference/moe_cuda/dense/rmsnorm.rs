//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// F16-I/O zgate_rmsnorm: `o` [N,D] F32, `z` [N,D] F16 (z_proj output, no ->F32
/// cast), `norm_w` [D] F32 -> `y` [N,D] F16 (no y->F16 cast). Bit-identical.
/// `Ok(None)` if not F16/CUDA.
pub fn zgate_rmsnorm_f16io(
    o: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    use core::ffi::c_void;
    if z.dtype() != DType::F16 || !o.device().is_cuda() {
        return Ok(None);
    }
    if o.dtype() != DType::F32 || norm_w.dtype() != DType::F32 {
        return Ok(None);
    }
    let dev = o.device().as_cuda_device()?;
    let (n, d) = o.dims2()?;
    let o = o.contiguous()?;
    let z = z.contiguous()?;
    let norm_w = norm_w.contiguous()?;
    // start_offset-aware: z may be a contiguous VIEW into the fused DeltaNet
    // projection output - never read the storage base pointer blindly.
    let (os, ol) = o.storage_and_layout();
    let (zs, zl) = z.storage_and_layout();
    let (ws, wl) = norm_w.storage_and_layout();
    let op = match &*os {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(ol.start_offset()) }
        }
        _ => return Ok(None),
    };
    let zp = match &*zs {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<half::f16>()?;
            unsafe {
                (s.device_ptr(s.stream()).0 as *const half::f16).add(zl.start_offset())
                    as *const c_void
            }
        }
        _ => return Ok(None),
    };
    let wp = match &*ws {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(wl.start_offset()) }
        }
        _ => return Ok(None),
    };
    let y = unsafe { dev.alloc::<half::f16>(n * d) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_zgate_rmsnorm_f16io(
            op,
            zp,
            wp,
            y.device_ptr(y.stream()).0 as *mut c_void,
            n as i32,
            d as i32,
            eps,
            stream,
        );
    }
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(y, dev.clone()),
        (n, d),
    )?))
}

/// F16-INPUT fused_conv_silu: `qkv` [seq,C] F16 (in_qkv output, no ->F32 cast);
/// `conv_state`,`w` F32; returns F32 (out, new_conv_state). Bit-identical (exact
/// F16->F32). `Ok(None)` if not F16/CUDA.
pub fn fused_conv_silu_f16in(
    qkv: &Tensor,
    conv_state: &Tensor,
    w: &Tensor,
    conv_kernel: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if qkv.dtype() != DType::F16 || !qkv.device().is_cuda() {
        return Ok(None);
    }
    if conv_state.dtype() != DType::F32 || w.dtype() != DType::F32 {
        return Ok(None);
    }
    let dev = qkv.device().as_cuda_device()?;
    let (seq, c) = qkv.dims2()?;
    let k = conv_kernel;
    let qkv = qkv.contiguous()?;
    let conv_state = conv_state.contiguous()?;
    let w = w.contiguous()?;
    // start_offset-aware (qkv can be a view into the fused projection output).
    let (qs, ql) = qkv.storage_and_layout();
    let (css, csl) = conv_state.storage_and_layout();
    let (ws, wl) = w.storage_and_layout();
    let qp = match &*qs {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<half::f16>()?;
            unsafe {
                (s.device_ptr(s.stream()).0 as *const half::f16).add(ql.start_offset())
                    as *const c_void
            }
        }
        _ => return Ok(None),
    };
    let csp = match &*css {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(csl.start_offset()) }
        }
        _ => return Ok(None),
    };
    let wp = match &*ws {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(wl.start_offset()) }
        }
        _ => return Ok(None),
    };
    let out_slice = unsafe { dev.alloc::<f32>(seq * c) }?;
    let ncs_slice = unsafe { dev.alloc::<f32>(c * (k - 1)) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_fused_conv_silu_f16in(
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
    Ok(Some((out, ncs)))
}

/// F16-INPUT deltanet_gate: `alpha`,`beta_in` [N,H] F16 (a_proj/b_proj outputs, no
/// ->F32 cast); `a_log`,`dt_bias` F32; returns F32 (g_pre, beta). Bit-identical.
/// `Ok(None)` if not F16/CUDA.
pub fn deltanet_gate_f16in(
    alpha: &Tensor,
    beta_in: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    use core::ffi::c_void;
    if alpha.dtype() != DType::F16 || beta_in.dtype() != DType::F16 || !alpha.device().is_cuda() {
        return Ok(None);
    }
    if a_log.dtype() != DType::F32 || dt_bias.dtype() != DType::F32 {
        return Ok(None);
    }
    let dev = alpha.device().as_cuda_device()?;
    let (n, h) = alpha.dims2()?;
    let alpha = alpha.contiguous()?;
    let beta_in = beta_in.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    // start_offset-aware: alpha/beta are contiguous VIEWS into the fused
    // DeltaNet projection output at decode.
    let (as_, al) = alpha.storage_and_layout();
    let (bs, bl) = beta_in.storage_and_layout();
    let (als, all_) = a_log.storage_and_layout();
    let (dts, dtl) = dt_bias.storage_and_layout();
    let ap = match &*as_ {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<half::f16>()?;
            unsafe {
                (s.device_ptr(s.stream()).0 as *const half::f16).add(al.start_offset())
                    as *const c_void
            }
        }
        _ => return Ok(None),
    };
    let bp = match &*bs {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<half::f16>()?;
            unsafe {
                (s.device_ptr(s.stream()).0 as *const half::f16).add(bl.start_offset())
                    as *const c_void
            }
        }
        _ => return Ok(None),
    };
    let alp = match &*als {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(all_.start_offset()) }
        }
        _ => return Ok(None),
    };
    let dtp = match &*dts {
        StorageView::Cuda(s) => {
            let s = s.as_cuda_slice::<f32>()?;
            unsafe { (s.device_ptr(s.stream()).0 as *const f32).add(dtl.start_offset()) }
        }
        _ => return Ok(None),
    };
    let g_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let b_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_deltanet_gate_f16in(
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
    Ok(Some((g, beta)))
}

/// Fused per-head RMSNorm. `x` [N, D] f32 (CUDA), `w` [D]. Returns `x.rsqrt(mean(x²)+eps).w`.
/// One warp per row. Replaces ~6 tensor ops.
pub fn head_rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let dev = x.device().as_cuda_device()?;
    let (n, d) = x.dims2()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("head_rmsnorm: inputs must be CUDA f32"),
        }
    };
    let (xp, wp) = (cuptr(x)?, cuptr(w)?);
    let y_slice = unsafe { dev.alloc::<f32>(n * d) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_head_rmsnorm(
            xp,
            wp,
            y_slice.device_ptr(y_slice.stream()).0 as *mut f32,
            n as i32,
            d as i32,
            eps,
            stream,
        );
    }
    tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(y_slice, dev.clone()), (n, d))
}

/// Fused L2-norm + GQA tile for DeltaNet q/k. `x` [seq, kg, kd] (f32, CUDA,
/// contiguous). Returns `y` [seq, vg, kd] with vg=rep.kg, where each v-head hv
/// gets `l2norm(x[:, hv%kg, :])` (TILED GQA). Replaces ~5 tensor ops per q/k.
pub fn l2norm_gqa(x: &Tensor, kg: usize, kd: usize, rep: usize, eps: f32) -> Result<Tensor> {
    let dev = x.device().as_cuda_device()?;
    let (seq, _kg, _kd) = x.dims3()?;
    let vg = rep * kg;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("l2norm_gqa: input must be CUDA f32"),
        }
    };
    let xp = cuptr(x)?;
    let y_slice = unsafe { dev.alloc::<f32>(seq * vg * kd) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_l2norm_gqa(
            xp,
            y_slice.device_ptr(y_slice.stream()).0 as *mut f32,
            seq as i32,
            kg as i32,
            kd as i32,
            rep as i32,
            eps,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(y_slice, dev.clone()),
        (seq, vg, kd),
    )
}

/// Fused RMS-norm (BF16 input) + quantized matmul, single-token decode.
pub fn rms_norm_then_qmatmul_bf16(
    x: &Tensor,
    w_norm: &Tensor,
    w_mm: &crate::tensor::quantized::QTensor,
    rms_eps: f32,
) -> Result<Tensor> {
    use core::ffi::c_void;
    if x.dtype() != DType::BF16 {
        crate::tensor::bail!("rms_norm_then_qmatmul_bf16: x must be BF16");
    }
    if w_norm.dtype() != DType::F32 {
        crate::tensor::bail!("rms_norm_then_qmatmul_bf16: w_norm must be F32");
    }
    let hidden = x.dim(crate::tensor::D::Minus1)?;
    if w_norm.shape().elem_count() != hidden {
        crate::tensor::bail!("rms_norm_then_qmatmul_bf16: w_norm len mismatch");
    }
    let (out_rows, k) = w_mm.shape().dims2()?;
    if k != hidden {
        crate::tensor::bail!(
            "rms_norm_then_qmatmul_bf16: w_mm K={} != hidden={}",
            k,
            hidden
        );
    }
    if hidden % 32 != 0 {
        crate::tensor::bail!("rms_norm_then_qmatmul_bf16: hidden must be divisible by 32");
    }
    if hidden > 16384 {
        crate::tensor::bail!(
            "rms_norm_then_qmatmul_bf16: hidden {} exceeds single-block limit 16384",
            hidden
        );
    }
    let dev = x.device().as_cuda_device()?;
    let x_c = x.contiguous()?;
    let wn_c = w_norm.contiguous()?;
    let kx_padded = hidden.div_ceil(512) * 512;
    let y_q8_1_bytes = (kx_padded / 32) * 36;
    // plain (uninit) alloc: the quantize grid covers every padded block and writes all
    // 36 bytes of each, so zeroing here is a memset the kernel immediately overwrites -
    // 42 of them per token on this path, and the API round trip is what leaves the device
    // idle between launches. Same treatment as the two quantize_q8_1 sites in cuda.rs.
    let y_q8_1 = unsafe { dev.cuda_stream().alloc::<u8>(y_q8_1_bytes) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let x_off = {
        let (_s, l) = x_c.storage_and_layout();
        l.start_offset()
    };
    let (xs_st, _) = x_c.storage_and_layout();
    let xs_slice = match &*xs_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
        _ => crate::tensor::bail!("x must be cuda"),
    };
    let (ws_st, _) = wn_c.storage_and_layout();
    let ws_slice = match &*ws_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("w_norm must be cuda"),
    };
    let x_base = xs_slice.device_ptr(xs_slice.stream()).0 as usize;
    let x_ptr = (x_base + x_off * 2) as *const c_void; // BF16 = 2 bytes
    let w_ptr = ws_slice.device_ptr(ws_slice.stream()).0 as *const f32;
    let y_ptr = y_q8_1.device_ptr(y_q8_1.stream()).0 as *mut c_void;
    unsafe {
        loken_rms_quantize_q8_1_bf16(x_ptr, w_ptr, y_ptr, hidden as i32, rms_eps, stream);
    }
    let qstor = match &w_mm.storage() {
        crate::tensor::quantized::QStorage::Cuda(s) => s,
        _ => crate::tensor::bail!("w_mm must be on CUDA"),
    };
    let out_storage = crate::inference::quantized_cuda::mvq_via_pre_quantized_q8_1(
        qstor, &y_q8_1, hidden, out_rows, 1,
    )
    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    tensor_from_cuda_storage(out_storage, (1, 1, out_rows))
}
