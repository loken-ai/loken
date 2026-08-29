//! Low-rank adapters, as tensor algebra.
//!
//! A LoRA is two thin matrices and a strength; applying it is `x.down.up.scale` added to
//! whatever the base weight produced. That is arithmetic on tensors, not a layer - which is
//! why it lives here rather than in `nn`: the quantised tensor types attach adapters too, and
//! a substrate that had to reach up into the layer library to do it would have the dependency
//! backwards.
//!
//! Several adapters on one projection fuse into a single pair, so the forward cost does not
//! grow with how many the user attached.

use super::{Result, Tensor};

/// One low-rank correction: `x -> ((x @ down) @ up) * scale`.
///
/// `down` is `[in, r]` and `up` is `[r, out]`, i.e. both already in this layer's
/// transposed convention, so the forward needs no reshaping beyond the matmuls.
#[derive(Clone, Debug)]
pub struct LoraDelta {
    pub down: Tensor,
    pub up: Tensor,
    pub scale: f32,
}

/// Fuse attached adapters into one `([1, in, R], [1, R, out])` pair, or `None` when
/// there is nothing live to apply.
///
/// Adapters at scale zero are excluded: a caller who asked for an adapter at strength 0
/// wants it disabled, and a zero block would cost exactly as much as a live one.
///
/// A fusion that fails to build yields `None` - the layer then runs on its base weight
/// rather than half-applied, because an adapter that silently applies to some
/// projections and not others is worse than one that does not apply at all.
pub fn fuse_loras(loras: &[LoraDelta], in_dim: usize, out_dim: usize) -> Option<(Tensor, Tensor)> {
    let live: Vec<&LoraDelta> = loras.iter().filter(|d| d.scale != 0.0).collect();
    if live.is_empty() {
        return None;
    }
    let downs: Vec<Tensor> = live.iter().map(|d| d.down.clone()).collect();
    let ups: Result<Vec<Tensor>> = live.iter().map(|d| d.up.affine(d.scale, 0.0)).collect();
    let ups = ups.ok()?;
    let dref: Vec<&Tensor> = downs.iter().collect();
    let uref: Vec<&Tensor> = ups.iter().collect();
    Tensor::cat(&dref, 1)
        .and_then(|d| {
            let r = d.dims()[1];
            let d3 = d.reshape(vec![1, in_dim, r])?.contiguous()?;
            let u3 = Tensor::cat(&uref, 0)?
                .reshape(vec![1, r, out_dim])?
                .contiguous()?;
            Ok((d3, u3))
        })
        .ok()
}
