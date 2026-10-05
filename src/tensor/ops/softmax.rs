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

/// Softmax over the last (key) dim with a per-head sink logit added to each row's
/// denominator: `exp(s - m) / (sum(exp(s - m)) + exp(sink - m))` with `m` the row maximum
/// taken over the scores and the sink.
///
/// - `scores`: attention logits `[b, n_head, q, kv]`, before `scale`.
/// - `mask`: additive `[.., q, kv]` mask (0 visible, -inf hidden), or `None`.
/// - `sinks`: learned per-head sink logits `[n_head]`, not scaled.
///
/// Returns weights `[b, n_head, q, kv]` whose rows sum below 1: the missing mass is what
/// the sink, which has no value, absorbed.
pub fn softmax_last_dim_with_sinks(
    scores: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    // One pass where a fused kernel serves the shape: the mask, the scale on the scores
    // and the whole sink softmax.
    if let Some(out) =
        crate::inference::kernel::fused::fused_softmax_sinks(scores, mask, sinks, scale as f32)?
    {
        return Ok(out);
    }
    let masked = match mask {
        Some(m) => Some(scores.broadcast_add(m)?),
        None => None,
    };
    let scores = masked.as_ref().unwrap_or(scores);
    let (_b, n_head, _q, _kv) = scores.dims4()?;
    let scores = (scores * scale)?;
    let sink = sinks.reshape((1, n_head, 1, 1))?;
    // The stabilising maximum covers the sink logit as well as the scores.
    let m = scores.max_keepdim(D::Minus1)?.broadcast_maximum(&sink)?;
    let exp_scores = scores.broadcast_sub(&m)?.exp()?;
    let exp_sink = sink.broadcast_sub(&m)?.exp()?;
    let denom = (exp_scores.sum_keepdim(D::Minus1)? + exp_sink)?;
    exp_scores.broadcast_div(&denom)
}

#[cfg(test)]
mod sink_softmax_tests {
    use super::softmax_last_dim_with_sinks;
    use crate::tensor::{Device, Tensor};

    #[test]
    fn a_sink_absorbs_part_of_the_mass() {
        // scores [1, 2], sink 0: m = 2, exp = [e^-1, 1], sink e^-2,
        // denominator 1.5032, weights [0.2448, 0.6652], sum 0.9100.
        let dev = Device::Cpu;
        let scores = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 1, 2), &dev).unwrap();
        let sinks = Tensor::new(&[0.0f32], &dev).unwrap();
        let w = softmax_last_dim_with_sinks(&scores, None, &sinks, 1.0).unwrap();
        let v: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - 0.2448).abs() < 1e-3, "w0={}", v[0]);
        assert!((v[1] - 0.6652).abs() < 1e-3, "w1={}", v[1]);
        let sum: f32 = v.iter().sum();
        assert!((sum - 0.9100).abs() < 1e-3, "sum={sum}");
    }

    #[test]
    fn a_very_negative_sink_leaves_the_plain_softmax() {
        let dev = Device::Cpu;
        let scores = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 1, 3), &dev).unwrap();
        let sinks = Tensor::new(&[-1e30f32], &dev).unwrap();
        let w = softmax_last_dim_with_sinks(&scores, None, &sinks, 1.0).unwrap();
        let sum: f32 = w
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }
}
