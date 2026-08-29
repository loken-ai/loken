//! Softmax, plain and in the masked/scaled shapes attention needs.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Softmax over the last dim: `exp(x - rowmax) / sum(exp(x - rowmax))`.
///
/// Drop-in replacement for `softmax_last_dim`. Numerically
/// stable (row-max subtraction); handles `-inf` mask entries. CUDA F32 uses
/// our own kernel; the fallback promotes F16/BF16 to F32 for the reduction
/// (matching the reference internal F32 accumulation) and casts back.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 {
            return crate::inference::kernel::fused::fused_softmax_lastdim_f32(x);
        }
    }
    // CPU F32 fused: per-row max-subtract -> exp -> normalize in one pass
    // (vs to_dtype/max/sub/exp/sum/div = ~6 allocating ops). Handles -inf mask
    // entries (exp(-inf)=0). One softmax per attention layer per token.
    if matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 {
        let cols = x.dim(D::Minus1)?;
        if cols > 0 {
            let mut out = x.flatten_all()?.to_vec1::<f32>()?;
            let rows = out.len() / cols;
            let sm_row = |row: &mut [f32]| {
                let mut m = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for o in row.iter_mut() {
                    let e = (*o - m).exp();
                    *o = e;
                    sum += e;
                }
                for o in row.iter_mut() {
                    *o /= sum;
                }
            };
            if rows == 1 {
                sm_row(&mut out);
            } else {
                crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, cols, &|_, row| {
                    sm_row(row)
                });
            }
            return Tensor::from_vec(out, x.dims().to_vec(), &x.device());
        }
    }
    let x_dtype = x.dtype();
    let internal = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let xf = x.to_dtype(internal)?;
    let max = xf.max_keepdim(D::Minus1)?;
    let num = xf.broadcast_sub(&max)?.exp()?;
    let den = num.sum_keepdim(D::Minus1)?;
    num.broadcast_div(&den)?.to_dtype(x_dtype)
}

/// Scale, causal mask and softmax over the last dim, in one pass.
///
/// The scores leave the q.kᵀ matmul in half precision. Widening them, applying
/// the scale, and adding the mask each wrote the whole score tensor before the
/// softmax read it back, so the tensor was walked several times before any
/// reduction happened. Here the widen, the scale and the mask ride along with
/// the pass that already has to find each row's maximum, and only the result is
/// written. Arithmetic and order are unchanged, so results are bit-identical to
/// the separate steps; anything not on this path falls back to them.
pub fn softmax_scaled_masked(att: &Tensor, scale: f64, mask: &Tensor) -> Result<Tensor> {
    if matches!(att.device(), Device::Cpu)
        && att.dtype() == DType::F16
        && mask.dtype() == DType::F32
    {
        let cols = att.dim(D::Minus1)?;
        let pq = att.dim(D::Minus2).unwrap_or(0);
        let mvec = mask.flatten_all()?.to_vec1::<f32>()?;
        // Only the [Pq,Pk] broadcast layout is safe to index as r%Pq; any other
        // mask shape falls back, as in the unfused pair.
        if cols > 0 && pq > 0 && mvec.len() == pq * cols {
            let src = att.f16_data()?;
            let rows = src.len() / cols;
            let s = scale as f32;
            let mut out: Vec<f32> = Vec::with_capacity(rows * cols);
            // Fully overwritten below, so skip the zero-fill.
            #[allow(clippy::uninit_vec)]
            unsafe {
                out.set_len(rows * cols)
            };
            let sm_row = |r: usize, row: &mut [f32]| {
                let base = r * cols;
                let mb = (r % pq) * cols; // this query position's mask row
                let mut m = f32::NEG_INFINITY;
                for (j, o) in row.iter_mut().enumerate() {
                    // Widen, scale and mask in the pass that finds the maximum.
                    let v = src[base + j].to_f32() * s + mvec[mb + j];
                    *o = v;
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for o in row.iter_mut() {
                    let e = (*o - m).exp();
                    *o = e;
                    sum += e;
                }
                for o in row.iter_mut() {
                    *o /= sum; // match the unfused softmax (division)
                }
            };
            if rows == 1 {
                sm_row(0, &mut out);
            } else {
                crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, cols, &|r, row| {
                    sm_row(r, row)
                });
            }
            return Tensor::from_vec(out, att.dims().to_vec(), &att.device());
        }
    }
    // Fallback: the separate steps, unchanged.
    let scaled = (att.to_dtype(DType::F32)? * scale)?;
    softmax_last_dim_masked(&scaled, mask)
}

pub fn softmax_last_dim_masked(att: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if matches!(att.device(), Device::Cpu)
        && att.dtype() == DType::F32
        && mask.dtype() == DType::F32
    {
        let cols = att.dim(D::Minus1)?;
        let pq = att.dim(D::Minus2).unwrap_or(0);
        let mvec = mask.flatten_all()?.to_vec1::<f32>()?;
        // only the [Pq,Pk] broadcast layout is safe to index as r%Pq (att row r =
        // ..*Pq + i is row-major, so i = r % Pq). Any other mask shape -> fallback.
        if cols > 0 && pq > 0 && mvec.len() == pq * cols {
            let mrows = pq; // mask rows == att query positions
            let mut out = att.flatten_all()?.to_vec1::<f32>()?;
            let rows = out.len() / cols;
            let sm_row = |r: usize, row: &mut [f32]| {
                let mb = (r % mrows) * cols; // this query position's mask row
                let mut m = f32::NEG_INFINITY;
                for (j, o) in row.iter().enumerate() {
                    let s = *o + mvec[mb + j];
                    if s > m {
                        m = s;
                    }
                }
                let mut sum = 0f32;
                for (j, o) in row.iter_mut().enumerate() {
                    let e = (*o + mvec[mb + j] - m).exp();
                    *o = e;
                    sum += e;
                }
                for o in row.iter_mut() {
                    *o /= sum;
                } // match unfused softmax (division)
            };
            if rows == 1 {
                sm_row(0, &mut out);
            } else {
                crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, cols, &|r, row| {
                    sm_row(r, row)
                });
            }
            return Tensor::from_vec(out, att.dims().to_vec(), &att.device());
        }
    }
    // fallback (CUDA / non-F32): the unfused pair - GPU path unchanged.
    softmax_last_dim(&att.broadcast_add(mask)?)
}

/// Softmax over an arbitrary dim. Drop-in for `softmax(x, dim)`
/// (keeps input dtype; numerically stable via max-subtraction).
pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let num = x.broadcast_sub(&max)?.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

/// Log-softmax over an arbitrary dim: `x - logsumexp(x, dim)`. Drop-in for
/// `log_softmax(x, dim)`.
pub fn log_softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let diff = x.broadcast_sub(&max)?;
    let log_sum = diff.exp()?.sum_keepdim(dim)?.log()?;
    diff.broadcast_sub(&log_sum)
}
