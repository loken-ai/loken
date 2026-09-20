//! DeepSeek V4.1 hyper-connections (bet phase 4).
//!
//! The residual stream is carried as `hc_mult` parallel copies. Around each sublayer, `hc_pre`
//! collapses the copies into one input and `hc_post` expands the output back out, mixing the
//! residual in through a doubly-stochastic `comb` matrix. All three coefficient sets - `pre`,
//! `post`, `comb` - come from the stream itself via `hc_mixes`, `comb` made doubly stochastic by
//! Sinkhorn iteration. The reference computes this path in f32.
//!
//! The reference is `notes/deepseek-oracle`; the coefficients are judged against a dump in the
//! block test.

use crate::tensor::{Device, Result, Tensor};

/// The pre/post/comb coefficients for one sublayer, per token.
pub struct HcMixes {
    /// [n, hc]
    pub pre: Vec<Vec<f32>>,
    /// [n, hc]
    pub post: Vec<Vec<f32>>,
    /// [n, hc, hc], row i column j.
    pub comb: Vec<Vec<Vec<f32>>>,
}

/// Derive the mixing coefficients from the stream. `x` is [b, s, hc, d]; `hc_fn` [mix_hc, hc*d],
/// `scale` [3], `base` [mix_hc], with `mix_hc = (2 + hc) * hc`. `norm_eps` is the stream RMS eps,
/// `hc_eps` the Sinkhorn floor.
pub fn hc_mixes(
    x: &Tensor,
    hc_fn: &Tensor,
    scale: &[f32],
    base: &[f32],
    hc: usize,
    sinkhorn_iters: usize,
    norm_eps: f32,
    hc_eps: f32,
) -> Result<HcMixes> {
    use rayon::prelude::*;
    let (b, s, hcx, d) = x.dims4()?;
    debug_assert_eq!(hcx, hc);
    let n = b * s;
    let flat = d * hc;
    let xf = x.reshape((n, flat))?;
    // One statistic per token over the whole flattened hc*d stream, then the projection scaled by
    // it - the reference's `F.linear(x, hc_fn) * rsqrt`.
    let rows = xf.flatten_all()?.to_vec1::<f32>()?;
    let mixes = crate::inference::offload::linear(&xf, hc_fn)?
        .flatten_all()?
        .to_vec1::<f32>()?; // [n, mix_hc]
    let mix_hc = mixes.len() / n.max(1);

    // Tokens are independent: each is computed in the order a single token would be.
    let per_token: Vec<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> = (0..n)
        .into_par_iter()
        .map(|t| {
            let ss: f32 = rows[t * flat..(t + 1) * flat].iter().map(|v| v * v).sum();
            let rsqrt = 1.0 / (ss / flat as f32 + norm_eps).sqrt();
            let m: Vec<f32> = mixes[t * mix_hc..(t + 1) * mix_hc]
                .iter()
                .map(|v| v * rsqrt)
                .collect();
            let pre_t: Vec<f32> = (0..hc)
                .map(|c| sigmoid(m[c] * scale[0] + base[c]) + hc_eps)
                .collect();
            let post_t: Vec<f32> = (0..hc)
                .map(|c| 2.0 * sigmoid(m[hc + c] * scale[1] + base[hc + c]))
                .collect();
            let mut cm = vec![vec![0f32; hc]; hc];
            for i in 0..hc {
                for j in 0..hc {
                    cm[i][j] = m[2 * hc + i * hc + j] * scale[2] + base[2 * hc + i * hc + j];
                }
            }
            sinkhorn(&mut cm, sinkhorn_iters, hc_eps);
            (pre_t, post_t, cm)
        })
        .collect();
    let mut pre = Vec::with_capacity(n);
    let mut post = Vec::with_capacity(n);
    let mut comb = Vec::with_capacity(n);
    for (p, q, c) in per_token {
        pre.push(p);
        post.push(q);
        comb.push(c);
    }
    Ok(HcMixes { pre, post, comb })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Make `cm` doubly stochastic: a row softmax with a floor, one column normalisation, then
/// `iters - 1` row/column normalisation passes. Matches the reference kernel's order exactly.
fn sinkhorn(cm: &mut [Vec<f32>], iters: usize, eps: f32) {
    let hc = cm.len();
    // comb = softmax(comb, dim=-1) + eps
    for row in cm.iter_mut() {
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            den += *v;
        }
        for v in row.iter_mut() {
            *v = *v / den + eps;
        }
    }
    // comb = comb / (comb.sum(-2) + eps)  (column sums)
    normalise_cols(cm, hc, eps);
    for _ in 0..iters.saturating_sub(1) {
        normalise_rows(cm, eps);
        normalise_cols(cm, hc, eps);
    }
}

fn normalise_rows(cm: &mut [Vec<f32>], eps: f32) {
    for row in cm.iter_mut() {
        let sum: f32 = row.iter().sum::<f32>() + eps;
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

fn normalise_cols(cm: &mut [Vec<f32>], hc: usize, eps: f32) {
    for j in 0..hc {
        let sum: f32 = (0..hc).map(|i| cm[i][j]).sum::<f32>() + eps;
        for row in cm.iter_mut() {
            row[j] /= sum;
        }
    }
}

/// Collapse the hc copies of `x` [b, s, hc, d] into one [b, s, d], weighted by `pre_mix` [n, hc].
pub fn hc_pre(x: &Tensor, pre_mix: &[Vec<f32>]) -> Result<Tensor> {
    let (b, s, hc, d) = x.dims4()?;
    let n = b * s;
    use rayon::prelude::*;
    let xv = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(t, o)| {
        for c in 0..hc {
            let w = pre_mix[t][c];
            let base = (t * hc + c) * d;
            for k in 0..d {
                o[k] += w * xv[base + k];
            }
        }
    });
    Tensor::from_vec(out, (b, s, d), &Device::Cpu)
}

/// Expand `x` [b, s, d] back to hc copies and mix in `residual` [b, s, hc, d] through `comb`.
/// `y[.., j, ..] = post[.., j] * x + sum_i comb[.., i, j] * residual[.., i, ..]`.
pub fn hc_post(
    x: &Tensor,
    residual: &Tensor,
    post: &[Vec<f32>],
    comb: &[Vec<Vec<f32>>],
    hc: usize,
) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    let n = b * s;
    let xv = x.flatten_all()?.to_vec1::<f32>()?;
    let rv = residual.flatten_all()?.to_vec1::<f32>()?;
    use rayon::prelude::*;
    let mut out = vec![0f32; n * hc * d];
    out.par_chunks_mut(hc * d)
        .enumerate()
        .for_each(|(t, token)| {
            for j in 0..hc {
                let o = j * d;
                let pj = post[t][j];
                for k in 0..d {
                    token[o + k] = pj * xv[t * d + k];
                }
                for (i, crow) in comb[t].iter().enumerate() {
                    let c = crow[j];
                    let r = (t * hc + i) * d;
                    for k in 0..d {
                        token[o + k] += c * rv[r + k];
                    }
                }
            }
        });
    Tensor::from_vec(out, (b, s, hc, d), &Device::Cpu)
}
