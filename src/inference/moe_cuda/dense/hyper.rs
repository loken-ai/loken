//! The wide residual's elementwise stages as launches: the grouped norm, the low-rank
//! nonlinearity, the gated mean over the streams, and the write-back with the next norm.
//!
//! The residual is F32 `[rows, streams * dim]`; the sublayer reads and writes F32 or F16.

use super::*;

/// Device address of an F32 or F16 tensor's first element.
pub(super) fn addr<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor, what: &str) -> Result<u64> {
    let (st, layout) = t.storage_and_layout();
    match &*st {
        StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            let (base, _) = slice.device_ptr(slice.stream());
            Ok(base + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("{what} must be on a CUDA device"),
    }
}

fn f32_t(t: &Tensor, what: &str) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        crate::tensor::bail!("{what}: expected F32, got {:?}", t.dtype());
    }
    t.contiguous()
}

/// `r` `[rows, streams * dim]` normalised stream by stream and scaled by `gamma`
/// `[streams * dim]`.
pub fn hc_norm(r: &Tensor, gamma: &Tensor, streams: usize, dim: usize, eps: f32) -> Result<Tensor> {
    let dev = r.device().as_cuda_device()?;
    let (r, gamma) = (f32_t(r, "hc_norm input")?, f32_t(gamma, "hc_norm gamma")?);
    let rows = r.elem_count() / (streams * dim);
    let out = unsafe { dev.alloc::<f32>(r.elem_count()) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_norm(
            addr::<f32>(&r, "r")? as *const f32,
            addr::<f32>(&gamma, "gamma")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            (rows * streams) as i32,
            streams as i32,
            dim as i32,
            eps,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        r.shape().clone(),
    )
}

/// `silu(down / streams)` elementwise over an F32 tensor.
pub fn hc_low(down: &Tensor, streams: usize) -> Result<Tensor> {
    let dev = down.device().as_cuda_device()?;
    let down = f32_t(down, "hc_low input")?;
    let n = down.elem_count();
    let out = unsafe { dev.alloc::<f32>(n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_low(
            addr::<f32>(&down, "down")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            n as i32,
            streams as i32,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        down.shape().clone(),
    )
}

/// The sublayer input: the mean over the streams of `xn * sigmoid(up)`, `[rows, dim]`, F16 when
/// `f16_out` else F32.
pub fn hc_mean_gate(
    xn: &Tensor,
    up: &Tensor,
    streams: usize,
    dim: usize,
    f16_out: bool,
) -> Result<Tensor> {
    let dev = xn.device().as_cuda_device()?;
    let (xn, up) = (f32_t(xn, "hc_mean_gate xn")?, f32_t(up, "hc_mean_gate up")?);
    let rows = xn.elem_count() / (streams * dim);
    let stream = dev.cuda_stream().cu_stream() as i64;
    let shape = crate::tensor::Shape::from((1, rows, dim));
    if f16_out {
        let out = unsafe { dev.alloc::<half::f16>(rows * dim) }?;
        unsafe {
            loken_hc_mean_gate(
                addr::<f32>(&xn, "xn")? as *const f32,
                addr::<f32>(&up, "up")? as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                1,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)
    } else {
        let out = unsafe { dev.alloc::<f32>(rows * dim) }?;
        unsafe {
            loken_hc_mean_gate(
                addr::<f32>(&xn, "xn")? as *const f32,
                addr::<f32>(&up, "up")? as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                0,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)
    }
}

/// Most rows a step may carry for [`hc_mix_fused`]: a decode step or a draft verify, whose rows
/// then see the arithmetic of the decode steps they stand for. A longer chunk reads each weight
/// once for all its rows in a matrix product instead.
pub const FUSED_MIX_MAX_ROWS: usize = crate::inference::engine::decode_step::VERIFY_ROWS;

/// A dense F32 weight `[n, k]` that is contiguous on a card, or `None`.
fn dense_rows(w: &Tensor, n: Option<usize>, k: usize) -> Option<(usize, bool)> {
    let [rows, cols] = w.dims() else { return None };
    ((w.dtype() == DType::F32 || w.dtype() == DType::F16)
        && w.device().is_cuda()
        && w.is_contiguous()
        && *cols == k
        && n.is_none_or(|n| n == *rows))
    .then_some((*rows, w.dtype() == DType::F16))
}

/// The mixer of a decode step in two launches: `down` and `inject` together, then `up` with the
/// gated mean over the streams. Returns the sublayer input `[1, rows, dim]` (F16 when `f16_out`)
/// and the raw write-back weights `[1, rows, streams]`; `None` when the weights or the step are
/// not of the shapes the launches take, and the caller runs the stage-by-stage chain.
pub fn hc_mix_fused(
    xn: &Tensor,
    down: &Tensor,
    up: &Tensor,
    inject: Option<&Tensor>,
    streams: usize,
    dim: usize,
    f16_out: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let k = streams * dim;
    let rows = xn.elem_count() / k.max(1);
    if rows == 0 || rows > FUSED_MIX_MAX_ROWS || xn.dtype() != DType::F32 || !xn.is_contiguous() {
        return Ok(None);
    }
    let Some((rank, half_w)) = dense_rows(down, None, k) else {
        return Ok(None);
    };
    match dense_rows(up, Some(k), rank) {
        Some((_, h)) if h == half_w => {}
        _ => return Ok(None),
    }
    let heads = match inject {
        Some(w) => match dense_rows(w, None, k) {
            Some((n, h)) if h == half_w => n,
            _ => return Ok(None),
        },
        None => 0,
    };
    let dev = xn.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let low = unsafe { dev.alloc::<f32>(rows * rank) }?;
    let inj = unsafe { dev.alloc::<f32>((rows * heads).max(1)) }?;
    let xn_p = addr::<f32>(xn, "xn")? as *const f32;
    let weight = |w: &Tensor, name: &str| -> Result<*const c_void> {
        Ok(if half_w {
            addr::<half::f16>(w, name)? as *const c_void
        } else {
            addr::<f32>(w, name)? as *const c_void
        })
    };
    let down_p = weight(down, "down")?;
    let up_p = weight(up, "up")?;
    let inj_w = match inject {
        Some(w) => weight(w, "inject")?,
        None => std::ptr::null(),
    };
    let low_p = low.device_ptr(low.stream()).0;
    unsafe {
        loken_hc_down_inject(
            xn_p,
            down_p,
            inj_w,
            low_p as *mut f32,
            inj.device_ptr(inj.stream()).0 as *mut f32,
            rows as i32,
            k as i32,
            rank as i32,
            heads as i32,
            streams as i32,
            i32::from(half_w),
            stream,
        );
    }
    let shape = crate::tensor::Shape::from((1, rows, dim));
    let mixed = if f16_out {
        let out = unsafe { dev.alloc::<half::f16>(rows * dim) }?;
        unsafe {
            loken_hc_up_mean(
                xn_p,
                up_p,
                low_p as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                rank as i32,
                1,
                i32::from(half_w),
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)?
    } else {
        let out = unsafe { dev.alloc::<f32>(rows * dim) }?;
        unsafe {
            loken_hc_up_mean(
                xn_p,
                up_p,
                low_p as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                rank as i32,
                0,
                i32::from(half_w),
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)?
    };
    let injected = if heads == 0 {
        None
    } else {
        Some(tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(inj, dev.clone()),
            crate::tensor::Shape::from((1, rows, heads)),
        )?)
    };
    Ok(Some((mixed, injected)))
}

/// `r + out (x) 2 sigmoid(inject / streams)` over `r` `[rows, streams * dim]`, with `inject`
/// `[rows, streams]` F32 and `out` `[rows, dim]` F32 or F16. When `next_gamma` is given the next
/// mixer's grouped norm of the result comes back too.
pub fn hc_combine(
    r: &Tensor,
    out: &Tensor,
    inject: &Tensor,
    next_gamma: Option<(&Tensor, f32)>,
    streams: usize,
    dim: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    let dev = r.device().as_cuda_device()?;
    let r = f32_t(r, "hc_combine residual")?;
    let inject = f32_t(inject, "hc_combine inject")?;
    let out = out.contiguous()?;
    let out_f16 = match out.dtype() {
        DType::F16 => true,
        DType::F32 => false,
        other => crate::tensor::bail!("hc_combine: the sublayer output is {other:?}"),
    };
    let rows = r.elem_count() / (streams * dim);
    let out_addr = if out_f16 {
        addr::<half::f16>(&out, "out")?
    } else {
        addr::<f32>(&out, "out")?
    };
    let new_r = unsafe { dev.alloc::<f32>(r.elem_count()) }?;
    let normed = match next_gamma {
        Some(_) => Some(unsafe { dev.alloc::<f32>(r.elem_count()) }?),
        None => None,
    };
    let gamma_c = next_gamma
        .map(|(g, _)| f32_t(g, "hc_combine gamma"))
        .transpose()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_combine(
            addr::<f32>(&r, "r")? as *const f32,
            out_addr as *const c_void,
            addr::<f32>(&inject, "inject")? as *const f32,
            new_r.device_ptr(new_r.stream()).0 as *mut f32,
            normed.as_ref().map_or(std::ptr::null_mut(), |n| {
                n.device_ptr(n.stream()).0 as *mut f32
            }),
            match &gamma_c {
                Some(g) => addr::<f32>(g, "gamma")? as *const f32,
                None => std::ptr::null(),
            },
            (rows * streams) as i32,
            streams as i32,
            dim as i32,
            next_gamma.map_or(0.0, |(_, e)| e),
            out_f16 as i32,
            stream,
        );
    }
    let shape = r.shape().clone();
    let new_r = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(new_r, dev.clone()),
        shape.clone(),
    )?;
    let normed = normed
        .map(|n| tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(n, dev.clone()), shape))
        .transpose()?;
    Ok((new_r, normed))
}
