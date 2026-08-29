//! Normalisation as an operation, for callers holding a raw weight tensor rather than an
//! [`RmsNorm`].
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// RMS normalization: `x / sqrt(mean(x², last_dim) + eps) * weight`.
///
/// Drop-in replacement for `rms_norm`. `weight` is a 1-D
/// `[last(x)]` tensor broadcast over the last dim. The variance reduction runs
/// in F32 (F16/BF16 inputs are promoted), matching the reference numerics; the
/// result is cast back to the input dtype before the weight multiply.
/// CPU F32 fused `residual-add + RMSNorm` in ONE pooled pass: returns
/// `(sum, rmsnorm(sum))` where `sum = a + b`. Bit-identical to a separate
/// `broadcast_add(a,b)` followed by [`rms_norm`] (same f32 ops, same order:
/// `x = a[i]+b[i]`, `ss += x*x`, `denom = sqrt(ss/hidden+eps)`, `x/denom*w`).
/// It saves one pool.run barrier AND one full read+write pass over the
/// `[rows, hidden]` intermediate per call (twice per transformer layer on the
/// prefill critical path). Returns `None` (caller falls back) unless both inputs
/// are CPU-F32, equal-shape, and FULL contiguous storage (`f32_data` length ==
/// element count ⇒ no view offset/padding, so the borrowed slices are correct).
pub fn fused_add_rmsnorm_f32(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    if !matches!(a.device(), Device::Cpu)
        || a.dtype() != DType::F32
        || b.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || a.dims() != b.dims()
        || !a.is_contiguous()
        || !b.is_contiguous()
    {
        return Ok(None);
    }
    let hidden = match a.dims().last() {
        Some(&h) if h > 0 => h,
        _ => return Ok(None),
    };
    if weight.dims() != [hidden] {
        return Ok(None);
    }
    let n = a.elem_count();
    let (ad, bd) = (a.f32_data()?, b.f32_data()?);
    // Strict guard: storage must be exactly the tensor's elements (offset 0, no
    // padding) so the raw f32_data borrow is the logical row-major data.
    if ad.len() != n || bd.len() != n {
        return Ok(None);
    }
    let wd = weight.flatten_all()?.to_vec1::<f32>()?;
    let rows = n / hidden;
    let mut sum = vec![0f32; n];
    let mut norm = vec![0f32; n];
    struct P(*mut f32);
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    let sum_p = P(sum.as_mut_ptr());
    crate::tensor::quant_cpu::pool_par_chunks_mut(&mut norm, hidden, &|r, norm_row| {
        let sum_p = &sum_p;
        let base = r * hidden;
        if base + hidden > n {
            return;
        }
        // SAFETY: chunk `r` owns the disjoint `sum[base..base+hidden]` range.
        let s_row = unsafe { std::slice::from_raw_parts_mut(sum_p.0.add(base), hidden) };
        let a_row = &ad[base..base + hidden];
        let b_row = &bd[base..base + hidden];
        let mut ss = 0f32;
        for i in 0..hidden {
            let x = a_row[i] + b_row[i];
            s_row[i] = x;
            ss += x * x;
        }
        let denom = (ss / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            norm_row[i] = (s_row[i] / denom) * wd[i];
        }
    });
    let _ = rows;
    let dims = a.dims().to_vec();
    Ok(Some((
        Tensor::from_vec(sum, dims.clone(), &a.device())?,
        Tensor::from_vec(norm, dims, &a.device())?,
    )))
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // CUDA F32 fast path -> our own fused kernel (no `candle_nn`).
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 && weight.dtype() == DType::F32 {
            let cols = *x.dims().last().unwrap_or(&1);
            if weight.dims() == [cols] {
                return crate::inference::kernel::fused::fused_rmsnorm_f32(x, weight, eps);
            }
        }
    }
    // CPU F32 fused fast path: one pass over each row, single output
    // allocation. The generic chain below issues ~9 separate allocating
    // tensor ops per call; with rms_norm invoked ~4x per layer (attn/ffn
    // norm + q/k norm) that dispatch+alloc churn dominates CPU decode.
    if matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 && weight.dtype() == DType::F32
    {
        let hidden = x.dim(D::Minus1)?;
        if weight.dims() == [hidden] {
            let w = weight.flatten_all()?.to_vec1::<f32>()?;
            let mut out = x.flatten_all()?.to_vec1::<f32>()?;
            let rows = out.len() / hidden;
            let norm_row = |row: &mut [f32]| {
                let mut ss = 0f32;
                for &v in row.iter() {
                    ss += v * v;
                }
                let denom = (ss / hidden as f32 + eps).sqrt();
                for (o, &wv) in row.iter_mut().zip(w.iter()) {
                    *o = (*o / denom) * wv;
                }
            };
            if rows == 1 {
                norm_row(&mut out);
            } else {
                // Spin-pool (not rayon): keeps one thread pool active across the
                // layer - mixing rayon here with the matmul spin-pool makes small
                // models spend most of prefill in rayon's work-stealing bridge.
                crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, hidden, &|_, row| {
                    norm_row(row)
                });
            }
            return Tensor::from_vec(out, x.dims().to_vec(), &x.device());
        }
    }
    // Fallback (CPU, F16/BF16, or odd shapes): portable primitives only,
    // matching the reference `rms_norm_slow` exactly.
    let x_dtype = x.dtype();
    let internal = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden = x.dim(D::Minus1)?;
    let xf = x.to_dtype(internal)?;
    let norm_x = (xf.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let x_normed = xf.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)
}
