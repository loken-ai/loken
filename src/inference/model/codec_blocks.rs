//! Blocks shared by the decoders of neural speech codecs.
//!
//! The dequantisation of a residual vector quantizer, and the causal convolutions a
//! streaming codec runs.

use crate::tensor::VarBuilder;
use crate::tensor::{Error, Result, Tensor};

/// `y [B, out, L]` plus `bias [out]` on every position.
pub(crate) fn add_bias(y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    match bias {
        None => Ok(y),
        Some(b) => {
            let out = b.dims()[0];
            y.broadcast_add(&b.reshape((1, out, 1))?)
        }
    }
}

/// Causal Conv1d: left-pad `(k-1)*dilation+1-stride`, then conv with no pad.
pub(crate) fn causal_conv1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let k = w.shape().dims3()?.2;
    let pad = (k - 1) * dilation + 1 - stride;
    let xp = x.pad_with_zeros(2, pad, 0)?;
    add_bias(xp.conv1d(w, 0, stride, dilation, groups)?, bias)
}

/// Causal ConvTranspose1d: transpose then trim `(k-stride)` from the right.
pub(crate) fn causal_convtr1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    groups: usize,
) -> Result<Tensor> {
    let k = w.shape().dims3()?.2;
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, groups)?;
    let l = y.shape().dims3()?.2;
    add_bias(y.narrow(2, 0, l.saturating_sub(k - stride))?, bias)
}

/// max(x, c) = relu(x - c) + c, built from relu and affine.
pub(crate) fn max_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    x.affine(1.0, -c)?.relu()?.affine(1.0, c)
}

/// One residual vector quantizer's decode: the sum over codebooks of the code's entry, then
/// the output projection.
pub(crate) struct Rvq {
    /// Each `[entries, vq_dim]`, the EMA sums divided by the usage.
    codebooks: Vec<Tensor>,
    /// `[dim, vq_dim, 1]`, a conv of kernel 1.
    output_proj: Tensor,
}

impl Rvq {
    /// The codebooks under `vq.layers` and the output projection of `vb`.
    ///
    /// A codebook is `embedding_sum / max(cluster_usage, eps)`.
    pub(crate) fn load(vb: &VarBuilder, eps: f32) -> Result<Self> {
        let n_cb = vb.count_children("vq.layers");
        let mut codebooks = Vec::with_capacity(n_cb);
        for k in 0..n_cb {
            let sum = format!("vq.layers.{k}._codebook.embedding_sum");
            let es = vb.get(vb.shape_of(&sum)?, &sum)?;
            let (entries, _) = es.shape().dims2()?;
            let cu = vb.get(entries, &format!("vq.layers.{k}._codebook.cluster_usage"))?;
            let denom = max_scalar(&cu, eps)?.reshape((entries, 1))?;
            codebooks.push(es.broadcast_div(&denom)?);
        }
        Ok(Self {
            codebooks,
            output_proj: vb.get(vb.shape_of("output_proj.weight")?, "output_proj.weight")?,
        })
    }

    /// Codebooks of this quantizer.
    pub(crate) fn len(&self) -> usize {
        self.codebooks.len()
    }

    /// `codes`: `[n_cb, T]` (u32 indices) -> `[dim, T]`.
    pub(crate) fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        let mut acc: Option<Tensor> = None;
        for (k, cb) in self.codebooks.iter().enumerate() {
            let idx = codes.narrow(0, k, 1)?.reshape(t)?;
            let e = cb.index_select(&idx, 0)?;
            acc = Some(match acc {
                None => e,
                Some(a) => a.add(&e)?,
            });
        }
        let acc = acc.ok_or_else(|| Error("a quantizer with no codebooks".into()))?;
        let vq_dim = acc.shape().dims2()?.1;
        let q = acc.transpose(0, 1)?.reshape((1, vq_dim, t))?;
        let dim = self.output_proj.shape().dims3()?.0;
        causal_conv1d(&q, &self.output_proj, None, 1, 1, 1)?.reshape((dim, t))
    }
}
