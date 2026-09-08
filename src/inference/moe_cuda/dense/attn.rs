//! The fused post-QKV decode launchers: RMS-norm, RoPE and the split into Q/K/V in one kernel.
//!
//! Four variants exist because four things can be absent - the norms, the second half of the
//! head, or nothing at all - and each is a different kernel. What they share is the plumbing:
//! read three or five addresses out of tensors, allocate three outputs, launch, wrap the results
//! back into tensors. That plumbing is written once here and the variants only choose a kernel.
//!
//! The outputs' carrier is a runtime choice. The kernel writes through `void*` and learns the
//! dtype from its own argument, so the only thing a carrier decides on this side is how many
//! bytes an element takes - which is what makes one generic body serve all three.

use super::*;
use crate::tensor::kernel_ffi::{CudaDType, CudaDevice};
use core::ffi::c_void;
use half::{bf16, f16};

/// The device address a tensor's storage starts at, read as `T`.
fn address_of<T: CudaDType>(storage: &StorageView, what: &str) -> Result<u64> {
    let slice = match storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => crate::tensor::bail!("post-QKV decode: {what} must be on CUDA"),
    };
    Ok(slice.device_ptr(slice.stream()).0)
}

/// The code the kernel reads its output carrier from.
fn carrier_code(out_dtype: DType) -> Result<i32> {
    Ok(match out_dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        d => crate::tensor::bail!("post-QKV decode: unsupported output dtype {d:?}"),
    })
}

/// The RoPE table pair's addresses.
///
/// Both tables are carried in the output dtype, so the dtype decides nothing but how the slice
/// is read - the kernel takes them as `void*` either way.
fn rope_addresses(
    out_dtype: DType,
    cos: &StorageView,
    sin: &StorageView,
) -> Result<(*const c_void, *const c_void)> {
    let (c, s) = match out_dtype {
        DType::F16 => (
            address_of::<f16>(cos, "rope_cos")?,
            address_of::<f16>(sin, "rope_sin")?,
        ),
        DType::BF16 => (
            address_of::<bf16>(cos, "rope_cos")?,
            address_of::<bf16>(sin, "rope_sin")?,
        ),
        DType::F32 => (
            address_of::<f32>(cos, "rope_cos")?,
            address_of::<f32>(sin, "rope_sin")?,
        ),
        d => crate::tensor::bail!("post-QKV decode: unsupported output dtype {d:?}"),
    };
    Ok((c as *const c_void, s as *const c_void))
}

/// Allocate the three outputs, hand the kernel their addresses, and hand back tensors.
///
/// `Qv` carries Q and V, `K` carries K: the variants that feed the Q4 KV-cache quantiser keep Q
/// and V in F32 so no cast launch follows, and only K takes the requested carrier.
fn outputs<Qv: CudaDType, K: CudaDType>(
    dev: &CudaDevice,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    launch: impl FnOnce(*mut c_void, *mut c_void, *mut c_void),
) -> Result<(Tensor, Tensor, Tensor)> {
    let q = unsafe { dev.alloc::<Qv>(n_q * hd) }?;
    let k = unsafe { dev.alloc::<K>(n_kv * hd) }?;
    let v = unsafe { dev.alloc::<Qv>(n_kv * hd) }?;
    launch(
        q.device_ptr(q.stream()).0 as *mut c_void,
        k.device_ptr(k.stream()).0 as *mut c_void,
        v.device_ptr(v.stream()).0 as *mut c_void,
    );
    Ok((
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(q, dev.clone()), (n_q, hd))?,
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(k, dev.clone()), (n_kv, hd))?,
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(v, dev.clone()), (n_kv, hd))?,
    ))
}

/// All three outputs in the requested carrier.
fn outputs_in_carrier(
    out_dtype: DType,
    dev: &CudaDevice,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    launch: impl FnOnce(*mut c_void, *mut c_void, *mut c_void),
) -> Result<(Tensor, Tensor, Tensor)> {
    match out_dtype {
        DType::F16 => outputs::<f16, f16>(dev, n_q, n_kv, hd, launch),
        DType::BF16 => outputs::<bf16, bf16>(dev, n_q, n_kv, hd, launch),
        DType::F32 => outputs::<f32, f32>(dev, n_q, n_kv, hd, launch),
        d => crate::tensor::bail!("post-QKV decode: unsupported output dtype {d:?}"),
    }
}

/// Q and V in F32 for the Q4 KV-cache quantiser; K in the requested carrier.
fn outputs_q_in_f32(
    out_dtype: DType,
    dev: &CudaDevice,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    launch: impl FnOnce(*mut c_void, *mut c_void, *mut c_void),
) -> Result<(Tensor, Tensor, Tensor)> {
    match out_dtype {
        DType::F16 => outputs::<f32, f16>(dev, n_q, n_kv, hd, launch),
        DType::BF16 => outputs::<f32, bf16>(dev, n_q, n_kv, hd, launch),
        DType::F32 => outputs::<f32, f32>(dev, n_q, n_kv, hd, launch),
        d => crate::tensor::bail!("post-QKV decode: unsupported output dtype {d:?}"),
    }
}

/// Fused post-QKV decode: RMS-norm, RoPE and the reshape into separate Q/K/V tensors.
///
/// `q_scale` is multiplied into Q only - the caller passes `1/sqrt(head_dim)` to fold the
/// attention scale in here and spare a downstream affine. `rope_cos` / `rope_sin` are the full
/// `[max_seq, hd/2]` tables: the kernel picks its row from `rope_pos`, so no offset is needed.
pub fn attn_post_qkv_decode(
    qkv: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    rms_eps: f32,
    q_scale: f32,
    out_dtype: DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    if qkv.dtype() != DType::F32 {
        crate::tensor::bail!(
            "attn_post_qkv_decode: qkv must be F32, got {:?}",
            qkv.dtype()
        );
    }
    if q_norm_w.dtype() != DType::F32 || k_norm_w.dtype() != DType::F32 {
        crate::tensor::bail!("attn_post_qkv_decode: norm weights must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        crate::tensor::bail!(
            "attn_post_qkv_decode: rope_cos/sin dtype {:?}/{:?} must match out_dtype {:?}",
            rope_cos.dtype(),
            rope_sin.dtype(),
            out_dtype
        );
    }

    let dev = qkv.device().as_cuda_device()?;
    let dtype_int = carrier_code(out_dtype)?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let qkv_c = qkv.contiguous()?;
    let qn_c = q_norm_w.contiguous()?;
    let kn_c = k_norm_w.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let (qn_storage, _) = qn_c.storage_and_layout();
    let (kn_storage, _) = kn_c.storage_and_layout();
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();

    let qkv_ptr = address_of::<f32>(&qkv_storage, "qkv")? as *const f32;
    let qn_ptr = address_of::<f32>(&qn_storage, "q_norm_w")? as *const f32;
    let kn_ptr = address_of::<f32>(&kn_storage, "k_norm_w")? as *const f32;
    let (rc_ptr, rs_ptr) = rope_addresses(out_dtype, &rc_storage, &rs_storage)?;

    outputs_in_carrier(
        out_dtype,
        &dev,
        n_q,
        n_kv,
        hd,
        |q_ptr, k_ptr, v_ptr| unsafe {
            loken_attn_post_qkv_decode(
                qkv_ptr,
                qn_ptr,
                kn_ptr,
                rc_ptr,
                rs_ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                rope_pos as i32,
                rms_eps,
                q_scale,
                dtype_int,
                rope_style,
                stream_i64,
            );
        },
    )
}

/// [`attn_post_qkv_decode`] with Q and V returned in F32, to feed the Q4 KV-cache quantise
/// without an extra cast launch.
pub fn attn_post_qkv_decode_qf32(
    qkv: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    rms_eps: f32,
    q_scale: f32,
    out_dtype: DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    if qkv.dtype() != DType::F32 {
        crate::tensor::bail!("attn_post_qkv_decode_qf32: qkv must be F32");
    }
    if q_norm_w.dtype() != DType::F32 || k_norm_w.dtype() != DType::F32 {
        crate::tensor::bail!("attn_post_qkv_decode_qf32: norm weights must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        crate::tensor::bail!("attn_post_qkv_decode_qf32: rope dtype mismatch");
    }

    let dev = qkv.device().as_cuda_device()?;
    let dtype_int = carrier_code(out_dtype)?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let qkv_c = qkv.contiguous()?;
    let qn_c = q_norm_w.contiguous()?;
    let kn_c = k_norm_w.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let (qn_storage, _) = qn_c.storage_and_layout();
    let (kn_storage, _) = kn_c.storage_and_layout();
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();

    let qkv_ptr = address_of::<f32>(&qkv_storage, "qkv")? as *const f32;
    let qn_ptr = address_of::<f32>(&qn_storage, "q_norm_w")? as *const f32;
    let kn_ptr = address_of::<f32>(&kn_storage, "k_norm_w")? as *const f32;
    let (rc_ptr, rs_ptr) = rope_addresses(out_dtype, &rc_storage, &rs_storage)?;

    outputs_q_in_f32(
        out_dtype,
        &dev,
        n_q,
        n_kv,
        hd,
        |q_ptr, k_ptr, v_ptr| unsafe {
            loken_attn_post_qkv_decode_qf32(
                qkv_ptr,
                qn_ptr,
                kn_ptr,
                rc_ptr,
                rs_ptr,
                q_ptr as *mut f32,
                k_ptr,
                v_ptr as *mut f32,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                rope_pos as i32,
                rms_eps,
                q_scale,
                dtype_int,
                rope_style,
                stream_i64,
            );
        },
    )
}

/// [`attn_post_qkv_decode_qf32`] for the architectures that carry no Q/K norm.
pub fn attn_post_qkv_decode_qf32_no_norm(
    qkv: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    q_scale: f32,
    out_dtype: DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    if qkv.dtype() != DType::F32 {
        crate::tensor::bail!("attn_post_qkv_decode_qf32_no_norm: qkv must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        crate::tensor::bail!("attn_post_qkv_decode_qf32_no_norm: rope dtype mismatch");
    }

    let dev = qkv.device().as_cuda_device()?;
    let dtype_int = carrier_code(out_dtype)?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let qkv_c = qkv.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();

    let qkv_ptr = address_of::<f32>(&qkv_storage, "qkv")? as *const f32;
    let (rc_ptr, rs_ptr) = rope_addresses(out_dtype, &rc_storage, &rs_storage)?;

    outputs_q_in_f32(
        out_dtype,
        &dev,
        n_q,
        n_kv,
        hd,
        |q_ptr, k_ptr, v_ptr| unsafe {
            loken_attn_post_qkv_decode_qf32_no_norm(
                qkv_ptr,
                rc_ptr,
                rs_ptr,
                q_ptr as *mut f32,
                k_ptr,
                v_ptr as *mut f32,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                rope_pos as i32,
                q_scale,
                dtype_int,
                rope_style,
                stream_i64,
            );
        },
    )
}

/// [`attn_post_qkv_decode_qf32_no_norm`] where RoPE turns only the first `rope_dim` of each
/// head and the rest of the head passes through.
pub fn attn_post_qkv_decode_qf32_no_norm_partial_rope(
    qkv: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_dim: usize,
    rope_pos: usize,
    q_scale: f32,
    out_dtype: DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    if qkv.dtype() != DType::F32 {
        crate::tensor::bail!("attn_post_qkv_decode_qf32_no_norm_partial_rope: qkv must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        crate::tensor::bail!("attn_post_qkv_decode_qf32_no_norm_partial_rope: rope dtype mismatch");
    }
    if rope_dim == 0 || rope_dim > hd {
        crate::tensor::bail!(
            "attn_post_qkv_decode_qf32_no_norm_partial_rope: rope_dim {rope_dim} must be in (0, {hd}]"
        );
    }
    if !rope_dim.is_multiple_of(2) {
        crate::tensor::bail!(
            "attn_post_qkv_decode_qf32_no_norm_partial_rope: rope_dim {rope_dim} must be even"
        );
    }

    let dev = qkv.device().as_cuda_device()?;
    let dtype_int = carrier_code(out_dtype)?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let qkv_c = qkv.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();

    let qkv_ptr = address_of::<f32>(&qkv_storage, "qkv")? as *const f32;
    let (rc_ptr, rs_ptr) = rope_addresses(out_dtype, &rc_storage, &rs_storage)?;

    outputs_q_in_f32(
        out_dtype,
        &dev,
        n_q,
        n_kv,
        hd,
        |q_ptr, k_ptr, v_ptr| unsafe {
            loken_attn_post_qkv_decode_qf32_no_norm_partial_rope(
                qkv_ptr,
                rc_ptr,
                rs_ptr,
                q_ptr as *mut f32,
                k_ptr,
                v_ptr as *mut f32,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                rope_dim as i32,
                rope_pos as i32,
                q_scale,
                dtype_int,
                rope_style,
                stream_i64,
            );
        },
    )
}

/// Multi-block F32 GEMV for the MoE router gate.
/// Returns `Ok(None)` if the shape isn't supported (caller falls back to gate.forward).
pub fn gate_gemv_f32(xs: &Tensor, gate_w: &Tensor) -> Result<Option<Tensor>> {
    if xs.dtype() != DType::F32 || gate_w.dtype() != DType::F32 || !xs.device().is_cuda() {
        return Ok(None);
    }
    let (xd, wd) = (xs.dims(), gate_w.dims());
    if xd.len() != 2 || wd.len() != 2 {
        return Ok(None);
    }
    let (n_rows, hidden) = (xd[0], xd[1]);
    let (n_experts, w_hidden) = (wd[0], wd[1]);
    if hidden != w_hidden || (hidden & 31) != 0 {
        return Ok(None);
    }
    let dev = xs.device().as_cuda_device()?;
    let xs = xs.contiguous()?;
    let gate_w = gate_w.contiguous()?;
    let (xs_st, _) = xs.storage_and_layout();
    let (gw_st, _) = gate_w.storage_and_layout();
    let xs_ptr = address_of::<f32>(&xs_st, "gate input")? as *const c_void;
    let gw_ptr = address_of::<f32>(&gw_st, "gate weight")? as *const c_void;
    let logits = unsafe { dev.alloc::<f32>(n_rows * n_experts) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_gate_gemv_f32(
            xs_ptr,
            gw_ptr,
            logits.device_ptr(logits.stream()).0 as *mut c_void,
            hidden as i32,
            n_rows as i32,
            n_experts as i32,
            stream,
        );
    }
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(logits, dev.clone()),
        (n_rows, n_experts),
    )?))
}
