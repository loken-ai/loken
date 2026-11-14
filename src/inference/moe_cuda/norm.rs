//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Fused (a+b) then RMSNorm(.)*gamma in one launch -> (sum, normed). Relocated
/// from candle-nn `moe::add_rms_norm`.
pub fn add_rms_norm(a: &Tensor, b: &Tensor, gamma: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 || gamma.dtype() != DType::F32 {
        crate::tensor::bail!(
            "add_rms_norm: all inputs must be F32 (got {:?} {:?} {:?})",
            a.dtype(),
            b.dtype(),
            gamma.dtype()
        );
    }
    let hidden = a.shape().elem_count();
    if b.shape().elem_count() != hidden || gamma.shape().elem_count() != hidden {
        crate::tensor::bail!(
            "add_rms_norm: shape mismatch (a={}, b={}, gamma={})",
            hidden,
            b.shape().elem_count(),
            gamma.shape().elem_count()
        );
    }
    if hidden > 16384 {
        crate::tensor::bail!(
            "add_rms_norm: hidden {} exceeds single-block limit 16384",
            hidden
        );
    }
    let dev = a.device().as_cuda_device()?;
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let g_c = gamma.contiguous()?;
    let stream_ptr = dev.cuda_stream().cu_stream() as i64;
    let xs_alloc = unsafe { dev.alloc::<f32>(hidden) }?;
    let nm_alloc = unsafe { dev.alloc::<f32>(hidden) }?;
    let a_off = a_c.layout().start_offset();
    let b_off = b_c.layout().start_offset();
    let g_off = g_c.layout().start_offset();
    let (a_st, _) = a_c.storage_and_layout();
    let (b_st, _) = b_c.storage_and_layout();
    let (g_st, _) = g_c.storage_and_layout();
    let a_slice = match &*a_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("a must be cuda"),
    };
    let b_slice = match &*b_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("b must be cuda"),
    };
    let g_slice = match &*g_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("gamma must be cuda"),
    };
    let a_base = a_slice.device_ptr(a_slice.stream()).0 as usize;
    let b_base = b_slice.device_ptr(b_slice.stream()).0 as usize;
    let g_base = g_slice.device_ptr(g_slice.stream()).0 as usize;
    let a_ptr = (a_base + a_off * 4) as *const f32;
    let b_ptr = (b_base + b_off * 4) as *const f32;
    let g_ptr = (g_base + g_off * 4) as *const f32;
    let xs_ptr = xs_alloc.device_ptr(xs_alloc.stream()).0 as *mut f32;
    let nm_ptr = nm_alloc.device_ptr(nm_alloc.stream()).0 as *mut f32;
    unsafe {
        loken_add_rms_norm(
            a_ptr,
            b_ptr,
            g_ptr,
            xs_ptr,
            nm_ptr,
            hidden as i32,
            eps,
            stream_ptr,
        );
    }
    let xs_storage = CudaStorage::wrap_cuda_slice(xs_alloc, dev.clone());
    let nm_storage = CudaStorage::wrap_cuda_slice(nm_alloc, dev.clone());
    let xs_t = tensor_from_cuda_storage(xs_storage, a.shape().clone())?;
    let nm_t = tensor_from_cuda_storage(nm_storage, a.shape().clone())?;
    Ok((xs_t, nm_t))
}

/// Fused RMSNorm(x)*w_norm -> Q8_1, then quantized matmul with `w_mm`. Relocated
/// from candle-nn `moe::rms_norm_then_qmatmul`; reuses the relocated
/// `quantized_cuda::mvq_via_pre_quantized_q8_1` for the matmul.
pub fn rms_norm_then_qmatmul(
    x: &Tensor,
    w_norm: &Tensor,
    w_mm: &crate::tensor::quantized::QTensor,
    rms_eps: f32,
) -> Result<Tensor> {
    use crate::tensor::quantized::QStorage;
    if x.dtype() != DType::F32 || w_norm.dtype() != DType::F32 {
        crate::tensor::bail!("rms_norm_then_qmatmul: x and w_norm must be F32");
    }
    let hidden = x.dim(crate::tensor::D::Minus1)?;
    if w_norm.elem_count() != hidden {
        crate::tensor::bail!("rms_norm_then_qmatmul: w_norm len mismatch");
    }
    let (out_rows, k) = w_mm.shape().dims2()?;
    if k != hidden {
        crate::tensor::bail!("rms_norm_then_qmatmul: w_mm K={} != hidden={}", k, hidden);
    }
    if hidden % 32 != 0 {
        crate::tensor::bail!("rms_norm_then_qmatmul: hidden must be divisible by 32");
    }
    if hidden > 16384 {
        crate::tensor::bail!(
            "rms_norm_then_qmatmul: hidden {} exceeds single-block limit 16384",
            hidden
        );
    }
    if !crate::inference::kernel::fused::fused_norm_fits(
        x,
        crate::inference::kernel::fused::rms_quantize_shared_cols(hidden),
    ) {
        crate::tensor::bail!(
            "rms_norm_then_qmatmul: hidden {} does not fit the shared memory of one block",
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
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("x must be cuda"),
    };
    let (ws_st, _) = wn_c.storage_and_layout();
    let ws_slice = match &*ws_st {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("w_norm must be cuda"),
    };
    let x_base = xs_slice.device_ptr(xs_slice.stream()).0 as usize;
    let x_ptr = (x_base + x_off * 4) as *const f32;
    let w_ptr = ws_slice.device_ptr(ws_slice.stream()).0 as *const f32;
    let y_ptr = y_q8_1.device_ptr(y_q8_1.stream()).0 as *mut core::ffi::c_void;
    unsafe {
        loken_rms_quantize_q8_1(x_ptr, w_ptr, y_ptr, hidden as i32, rms_eps, stream);
    }
    let qstor = match &w_mm.storage() {
        QStorage::Cuda(s) => s,
        _ => crate::tensor::bail!("w_mm must be on CUDA"),
    };
    let out_storage = crate::inference::quantized_cuda::mvq_via_pre_quantized_q8_1(
        qstor, &y_q8_1, hidden, out_rows, 1,
    )
    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    tensor_from_cuda_storage(out_storage, (1, 1, out_rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The q8_1 row the fused norm writes carries the codes and pairs the quantise kernel
    /// writes from the same normed row, byte for byte.
    ///
    /// Each input value is 1 or -1 and the epsilon is 0, so the row's mean square is exactly 1.
    /// The kernel's reciprocal square root of 1 may land a few units of the last place off 1.
    /// The normed row is the input times that factor times the norm weight, on the host for
    /// each factor near 1.
    /// One of them gives the fused kernel's bytes.
    #[test]
    fn the_fused_norm_quantises_its_row_as_the_quantise_kernel() {
        let Ok(device) = crate::tensor::Device::new_cuda(0) else {
            eprintln!("no card: the kernels run on a card only");
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let dev = cuda.native().clone();
        let stream = dev.stream().clone();
        let (rows, hidden) = (256usize, 4096usize);
        let x: Vec<f32> = (0..hidden)
            .map(|i| if (i * 7919) % 13 < 6 { -1.0 } else { 1.0 })
            .collect();
        let w: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.618_034).sin() * 3.0 + ((i / hidden) as f32 * 0.37).cos())
            .collect();
        let x_dev = stream.clone_htod(&x).unwrap();
        let w_dev = stream.clone_htod(&w).unwrap();
        let block_bytes = hidden / 32 * 36;
        let fused = unsafe { stream.alloc::<u8>(rows * block_bytes) }.unwrap();
        let x_ptr = x_dev.device_ptr(&stream).0 as *const f32;
        let w_base = w_dev.device_ptr(&stream).0 as usize;
        let y_base = fused.device_ptr(&stream).0 as usize;
        let stream_ptr = cuda.cuda_stream().cu_stream() as i64;
        for row in 0..rows {
            let w_ptr = (w_base + row * hidden * 4) as *const f32;
            let y_ptr = (y_base + row * block_bytes) as *mut c_void;
            unsafe { loken_rms_quantize_q8_1(x_ptr, w_ptr, y_ptr, hidden as i32, 0.0, stream_ptr) };
        }
        let fused = stream.clone_dtoh(&fused).unwrap();
        // The bytes that differ from the quantise kernel's for each factor near 1.
        let differing: Vec<(f32, Vec<String>)> = (-4i32..=4)
            .map(|ulps| {
                let factor = f32::from_bits((1.0f32.to_bits() as i32 + ulps) as u32);
                let normed: Vec<f32> = w
                    .iter()
                    .enumerate()
                    .map(|(i, v)| x[i % hidden] * factor * v)
                    .collect();
                let normed_dev = stream.clone_htod(&normed).unwrap();
                let plain =
                    crate::tensor::cuda::quantize_q8_1_rows(&dev, &normed_dev, hidden, rows)
                        .unwrap();
                let plain = stream.clone_dtoh(&plain).unwrap();
                // A block is 36 bytes: the pair of the scale and the sum, then the 32 codes.
                let bytes = fused
                    .iter()
                    .zip(&plain)
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(at, (a, b))| format!("block {} byte {}: {a} for {b}", at / 36, at % 36))
                    .collect();
                (factor, bytes)
            })
            .collect();
        let closest = differing
            .iter()
            .min_by_key(|(_, bytes)| bytes.len())
            .unwrap();
        assert!(
            closest.1.is_empty(),
            "at the closest factor {}, {} of {} bytes differ from the quantise kernel's, first: {:?}",
            closest.0,
            closest.1.len(),
            fused.len(),
            closest.1.first()
        );
    }
}
