//! A split layer run whole on every card of its group: each card keeps its own copy of the
//! residual stream and computes the norms and the residual adds itself, and the partial outputs
//! of attention and of the feed-forward are summed onto every card at once. A token crosses the
//! cards twice per layer, and a run of split layers passes each card's copy on to the next.

use super::GenericTransformerLayer;
use crate::tensor::{Result, Tensor};

impl GenericTransformerLayer {
    /// Whether this layer runs through [`Self::forward_split`] for `x`: its shares are on other
    /// cards with communicators between them, `x` is F32, and the layer is a pre-norm attention
    /// and feed-forward block whose every step the split reproduces.
    pub(super) fn runs_split(&self, x: &Tensor) -> bool {
        #[cfg(feature = "cuda")]
        {
            !self.tp_shards.is_empty()
                && self.tp_comms.is_some()
                && x.dtype() == crate::tensor::DType::F32
                && self.moe.is_none()
                && self.ffn_norm.is_some()
                && !self.flags.parallel_attn
                && self.ple_inp_gate.is_none()
                && !self.is_wide()
                && !self.is_hybrid()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = x;
            false
        }
    }

    /// The layer over `x`, every rank computing it whole: the attention input normed on each
    /// card, each share's attention summed onto every card, the residual added and normed on
    /// each, each share's feed-forward summed onto every card, the residual added again. The
    /// copies on the other cards are kept for the next split layer. `index_pos` is `None` in a
    /// recorded step, where each cache gives its own position.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_split(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: Option<usize>,
    ) -> Result<Tensor> {
        let comms = self
            .tp_comms
            .clone()
            .ok_or_else(|| crate::tensor::Error::msg("a split layer without communicators"))?;
        let mut shards = std::mem::take(&mut self.tp_shards);
        let out = {
            let mut ranks: Vec<&mut GenericTransformerLayer> = std::iter::once(&mut *self)
                .chain(shards.iter_mut())
                .collect();
            split_layer(&mut ranks, &comms, x, mask, index_pos)
        };
        let joined = comms.join();
        self.tp_shards = shards;
        let out = out?;
        joined?;
        Ok(out)
    }
}

/// [`GenericTransformerLayer::forward_split`] over the ranks of the group, rank 0 first.
#[cfg(feature = "cuda")]
fn split_layer(
    ranks: &mut [&mut GenericTransformerLayer],
    comms: &super::TpComms,
    x: &Tensor,
    mask: Option<&Tensor>,
    index_pos: Option<usize>,
) -> Result<Tensor> {
    let to: Vec<_> = ranks[1..].iter().map(|l| l.cos.device()).collect();
    let xs = comms.replicas_of(x, &to)?;
    let mut parts = Vec::with_capacity(ranks.len());
    for (layer, x) in ranks.iter_mut().zip(&xs) {
        let xn = layer.attn_norm.forward(x)?;
        let pos = index_pos.unwrap_or_else(|| layer.cached_len());
        let m = mask.map(|m| m.to_device(&layer.cos.device())).transpose()?;
        let mut a = layer.forward_attn_inner(&xn, m.as_ref(), pos, None, None, None)?;
        // The output bias is on rank 0 alone, so the sum carries it once.
        if let Some(b) = &layer.attn_output_bias {
            a = a.broadcast_add(b)?;
        }
        parts.push(a.contiguous()?);
    }
    let attn = comms.all_reduce(&parts)?;
    drop(parts);
    let mut hs = Vec::with_capacity(ranks.len());
    let mut parts = Vec::with_capacity(ranks.len());
    for ((layer, x), mut a) in ranks.iter_mut().zip(&xs).zip(attn) {
        if let Some(s) = layer.residual_scale {
            a = (a * s)?;
        }
        let (h, hn) = residual_and_norm(layer, &a, x)?;
        parts.push(layer.forward_ffn_own(&hn)?.contiguous()?);
        hs.push(h);
    }
    let ffn = comms.all_reduce(&parts)?;
    drop(parts);
    let mut outs = Vec::with_capacity(ranks.len());
    for ((layer, h), mut f) in ranks.iter().zip(&hs).zip(ffn) {
        if let Some(s) = layer.residual_scale {
            f = (f * s)?;
        }
        let mut out = match &layer.post_ffn_norm {
            Some(n) => (n.forward(&f)? + h)?,
            None => (f + h)?,
        };
        if let Some(scale) = &layer.ple_output_scale {
            out = out.broadcast_mul(scale)?;
        }
        outs.push(out);
    }
    let own = outs[0].clone();
    comms.keep(outs)?;
    Ok(own)
}

/// The residual after attention, `a` added to `x` (normed first by the post-attention norm when
/// the layer has one), and its norm for the feed-forward.
#[cfg(feature = "cuda")]
fn residual_and_norm(
    layer: &GenericTransformerLayer,
    a: &Tensor,
    x: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if layer.post_attn_norm.is_none() {
        if let Some(w) = layer.ffn_norm_weight.as_ref() {
            return crate::inference::kernel::fused::fused_add_rmsnorm_dual(
                a,
                x,
                w,
                layer.ffn_norm_eps,
            );
        }
    }
    let h = match &layer.post_attn_norm {
        Some(n) => (n.forward(a)? + x)?,
        None => (a + x)?,
    };
    let norm = layer
        .ffn_norm
        .as_ref()
        .ok_or_else(|| crate::tensor::Error::msg("ffn_norm missing on a split layer"))?;
    let hn = norm.forward(&h)?;
    Ok((h, hn))
}
