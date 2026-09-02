//! The vision transformer block, once, for every tower in the tree that is one.
//!
//! A ViT block is two sublayers over residuals: normalise, attend, add; normalise, expand,
//! activate, contract, add. Nothing in that depends on how the checkpoint that supplies the
//! weights was written - yet the two packagings of the same model here differ in exactly two
//! such ways. One stores Q, K and V fused in a single projection; the other stores them apart.
//! One cuts the image with a flat linear layer over already-flattened patches; the other with a
//! convolution whose kernel equals its stride, which is the same arithmetic spelled as a layer.
//!
//! So the block takes the projections it is handed and asks nothing about where they came from,
//! and the two loaders are left doing the only thing that genuinely differs between them:
//! reading names out of a file.

use crate::tensor::layer::qlinear::{QLinear, QMlp};
use crate::tensor::layer::LayerNorm;
use crate::tensor::ops::Activation;
use crate::tensor::Module;
use crate::tensor::{Result, Tensor};

/// Q, K and V, however the checkpoint stores them.
#[derive(Debug)]
pub enum Qkv {
    /// One projection producing all three in a row, split after the fact.
    Fused(QLinear),
    /// Three projections, each producing one operand.
    Split { q: QLinear, k: QLinear, v: QLinear },
}

impl Qkv {
    /// The three operands, each `[batch, heads, positions, head_dim]`.
    ///
    /// The fused form carries the three consecutively along the feature axis, so the reshape
    /// below grows a leading three and the permutation brings it to the front; the split form
    /// reshapes each projection on its own. Both end in the same layout, which is what lets the
    /// attention below be written once.
    fn project(
        &self,
        xs: &Tensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, n, _) = xs.dims3()?;
        match self {
            Self::Fused(qkv) => {
                let all = qkv
                    .forward(xs)?
                    .reshape((b, n, 3, heads, head_dim))?
                    .permute((2, 0, 3, 1, 4))?;
                Ok((
                    all.i(0)?.contiguous()?,
                    all.i(1)?.contiguous()?,
                    all.i(2)?.contiguous()?,
                ))
            }
            Self::Split { q, k, v } => {
                let one = |p: &QLinear| -> Result<Tensor> {
                    p.forward(xs)?
                        .reshape((b, n, heads, head_dim))?
                        .transpose(1, 2)?
                        .contiguous()
                };
                Ok((one(q)?, one(k)?, one(v)?))
            }
        }
    }
}

/// One block: attention over a normalised copy of the input, then an MLP over a normalised copy
/// of that, each added back.
#[derive(Debug)]
pub struct VitBlock {
    norm1: LayerNorm,
    qkv: Qkv,
    proj: QLinear,
    norm2: LayerNorm,
    mlp: QMlp,
    heads: usize,
    head_dim: usize,
}

impl VitBlock {
    /// `heads` divides the width; `head_dim` follows from it and is kept rather than recomputed
    /// per forward.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        norm1: LayerNorm,
        qkv: Qkv,
        proj: QLinear,
        norm2: LayerNorm,
        mlp: QMlp,
        width: usize,
        heads: usize,
    ) -> Result<Self> {
        if heads == 0 || width % heads != 0 {
            crate::tensor::bail!("a ViT block of width {width} cannot have {heads} heads");
        }
        Ok(Self {
            norm1,
            qkv,
            proj,
            norm2,
            mlp,
            heads,
            head_dim: width / heads,
        })
    }

    /// `[batch, positions, width]` in, the same out.
    ///
    /// The scores go through the shared attention, which tiles them: a patch grid is short
    /// enough that the bound is never reached, and a tower of this many blocks each holding a
    /// full `positions x positions` score matrix at once is how one of these ran out of memory.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, width) = xs.dims3()?;
        let normed = self.norm1.forward(xs)?;
        let (q, k, v) = self.qkv.project(&normed, self.heads, self.head_dim)?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let attended =
            crate::inference::model::acestep::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, n, width))?;
        let xs = (xs + self.proj.forward(&attended)?)?;

        let hidden = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        &xs + &hidden
    }
}

use crate::tensor::IndexOp;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::layer::qlinear::Weight;
    use crate::tensor::layer::Linear;
    use crate::tensor::{DType, Device};

    /// The largest element-wise disagreement between two tensors of the same shape.
    fn worst(a: &Tensor, b: &Tensor) -> f32 {
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        av.iter()
            .zip(&bv)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    fn deterministic(shape: (usize, usize), seed: usize, dev: &Device) -> Tensor {
        let (r, c) = shape;
        let v: Vec<f32> = (0..r * c)
            .map(|i| (((i * 37 + seed * 11) % 97) as f32) * 0.021 - 1.0)
            .collect();
        Tensor::from_vec(v, (r, c), dev).unwrap()
    }

    fn dense(weight: Tensor, bias: Option<Tensor>) -> QLinear {
        let (out_dim, in_dim) = (weight.dim(0).unwrap(), weight.dim(1).unwrap());
        QLinear::new(
            Weight::Dense(Linear::new(weight, None).unwrap(), DType::F32),
            bias,
            in_dim,
            out_dim,
        )
    }

    /// The two ways a checkpoint stores the projections are the same three projections.
    ///
    /// The split form is cut out of the fused weight, so any disagreement is the block's
    /// handling of the two layouts and not the numbers it was given. This is the whole claim
    /// that lets one block serve both towers, and it is the one a reshape gets wrong quietly:
    /// the fused weight's rows are Q, then K, then V, and reading them as heads-then-operand
    /// instead of operand-then-heads produces a plausible tensor of the right shape.
    #[test]
    fn the_fused_and_split_projections_are_the_same_projections() {
        let dev = Device::Cpu;
        let (width, heads, n) = (32usize, 4usize, 6usize);
        let fused_w = deterministic((3 * width, width), 1, &dev);
        let fused_b = Tensor::from_vec(
            (0..3 * width)
                .map(|i| (i as f32) * 0.003 - 0.1)
                .collect::<Vec<_>>(),
            3 * width,
            &dev,
        )
        .unwrap();
        let xs = deterministic((n, width), 5, &dev)
            .reshape((1, n, width))
            .unwrap();

        let fused = Qkv::Fused(dense(fused_w.clone(), Some(fused_b.clone())));
        let split = Qkv::Split {
            q: dense(
                fused_w.narrow(0, 0, width).unwrap(),
                Some(fused_b.narrow(0, 0, width).unwrap()),
            ),
            k: dense(
                fused_w.narrow(0, width, width).unwrap(),
                Some(fused_b.narrow(0, width, width).unwrap()),
            ),
            v: dense(
                fused_w.narrow(0, 2 * width, width).unwrap(),
                Some(fused_b.narrow(0, 2 * width, width).unwrap()),
            ),
        };

        let a = fused.project(&xs, heads, width / heads).unwrap();
        let b = split.project(&xs, heads, width / heads).unwrap();
        for (name, x, y) in [("q", a.0, b.0), ("k", a.1, b.1), ("v", a.2, b.2)] {
            let gap = worst(&x, &y);
            assert!(gap < 1e-5, "{name}: the two layouts disagree by {gap}");
        }
    }

    /// The block's attention is the attention it replaces.
    ///
    /// The tower this came from scaled the score matrix; the shared attention folds the scale
    /// into the query instead and tiles the softmax. Those are the same function and this says
    /// so - against a reference written the long way, on the same weights, so nothing but the
    /// attention itself is under test.
    #[test]
    fn the_shared_attention_answers_what_the_hand_written_one_did() {
        let dev = Device::Cpu;
        let (width, heads, n) = (32usize, 4usize, 6usize);
        let head_dim = width / heads;
        let xs = deterministic((n, width), 7, &dev)
            .reshape((1, n, width))
            .unwrap();

        let one = |seed: usize| dense(deterministic((width, width), seed, &dev), None);
        let qkv = Qkv::Split {
            q: one(2),
            k: one(3),
            v: one(4),
        };
        let (q, k, v) = qkv.project(&xs, heads, head_dim).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul_t(&k).unwrap() * scale).unwrap();
        let want = crate::tensor::ops::softmax_last_dim(&scores)
            .unwrap()
            .matmul(&v)
            .unwrap();
        let got =
            crate::inference::model::acestep::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)
                .unwrap();

        let gap = worst(&want, &got);
        let size = want
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .fold(1e-6f32, |m, x| m.max(x.abs()));
        assert!(
            gap / size < 1e-5,
            "the two attentions disagree by {gap} of {size}"
        );
    }
}
