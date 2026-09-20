//! DeepSeek V4.1 block for a ratio-0 layer (bet phase 4).
//!
//! One block is attention then FFN, each wrapped in hyper-connections: `hc_pre` collapses the
//! `hc_mult` residual copies into the sublayer input, an RMS norm, the sublayer, then `hc_post`
//! expands back out and folds the residual in. Each sublayer's `hc_mixes` produces the coefficients
//! the *next* sublayer uses - attention uses the pre-mix handed in from the previous layer, the FFN
//! uses the one attention just produced, and the block returns the FFN's pre-mix for the layer
//! below.
//!
//! The reference is `notes/deepseek-oracle`; the whole block is judged against a dump in the test.

use super::attention::Ratio0Attention;
use super::band::{BandAttention, SharedAttn};
use super::cache::AttnCache;
use super::hyper_connections::{hc_mixes, hc_post, hc_pre, HcMixes};
use super::moe::Moe;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Result, Tensor};

/// A layer's attention, in whichever mode its band uses: a window-only ratio-0 layer, or a
/// compressed-KV band layer. Both take the same collapsed input and rope table, so a block holds
/// one of these and the rest of the block (MoE, norms, hyper-connections) is identical.
pub enum LayerAttn {
    Ratio0(Ratio0Attention),
    Band(BandAttention),
}

impl LayerAttn {
    /// The bytes of the attention projections as stored.
    pub fn bytes(&self) -> usize {
        match self {
            LayerAttn::Ratio0(a) => {
                a.wq_a.bytes() + a.wq_b.bytes() + a.wkv.bytes() + a.wo_a.bytes() + a.wo_b.bytes()
            }
            LayerAttn::Band(a) => {
                // Every weight a token's attention reads whole, the dense ones as f32: what a
                // card keeps for it, so a reserve sized from this leaves none of them out.
                let dense = |t: &Tensor| t.elem_count() * std::mem::size_of::<f32>();
                a.wq_a.bytes()
                    + a.wq_b.bytes()
                    + a.wkv.bytes()
                    + a.wo_a.bytes()
                    + a.wo_b.bytes()
                    + a.indexer
                        .as_ref()
                        .map(|i| {
                            i.wq_b.bytes()
                                + dense(&i.weights_proj)
                                + i.wk.as_ref().map(dense).unwrap_or(0)
                        })
                        .unwrap_or(0)
                    + a.compressor
                        .as_ref()
                        .map(|c| dense(&c.wkv) + c.wgate.as_ref().map(dense).unwrap_or(0))
                        .unwrap_or(0)
            }
        }
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
    ) -> Result<Tensor> {
        self.forward_prefill_cached(x, cos, sin, shared, None)
    }

    /// `forward_prefill`, leaving in `cache` what a decode continuing from here reads.
    pub fn forward_prefill_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
        cache: Option<&mut AttnCache>,
    ) -> Result<Tensor> {
        match self {
            LayerAttn::Ratio0(a) => a.forward_prefill_cached(x, cos, sin, cache),
            LayerAttn::Band(a) => a.forward_prefill_cached(x, cos, sin, shared, cache),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        cache: &mut AttnCache,
        shared: &mut SharedAttn,
    ) -> Result<Tensor> {
        match self {
            LayerAttn::Ratio0(a) => a.forward_decode(x, cos, sin, pos, cache),
            LayerAttn::Band(a) => a.forward_decode(x, cos, sin, pos, cache, shared),
        }
    }
}

/// The hyper-connection weights of one sublayer: the projection, its per-set scales, and biases.
pub struct HcWeights {
    pub func: Tensor,  // [mix_hc, hc*d]
    pub scale: Tensor, // [3]
    pub base: Tensor,  // [mix_hc]
}

pub struct Block {
    pub attn: LayerAttn,
    pub moe: Moe,
    pub attn_norm: Tensor, // [d]
    pub ffn_norm: Tensor,  // [d]
    pub hc_attn: HcWeights,
    pub hc_ffn: HcWeights,

    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
}

impl Block {
    fn mixes(&self, x: &Tensor, w: &HcWeights) -> Result<HcMixes> {
        hc_mixes(
            x,
            &w.func,
            &w.scale.flatten_all()?.to_vec1::<f32>()?,
            &w.base.flatten_all()?.to_vec1::<f32>()?,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
    }

    /// Compute this block's attention hyper-connection mixes for `x` [b, s, hc, d].
    pub fn attn_mixes(&self, x: &Tensor) -> Result<HcMixes> {
        self.mixes(x, &self.hc_attn)
    }

    /// Prefill forward. `x` is [b, s, hc, d], `pre_mix` [n, hc] the collapse weights from the layer
    /// above, `cos`/`sin` the rope table. Returns the next stream [b, s, hc, d] and the FFN pre-mix
    /// this block hands to the layer below.
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        pre_mix: &[Vec<f32>],
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        let (x, pre) = self.forward_prefill_attn(x, pre_mix, cos, sin, shared, None)?;
        self.forward_prefill_ffn(&x, &pre)
    }

    /// The attention half of `forward_prefill`: the stream after attention, and the pre-mix the
    /// FFN half collapses it with. With `cache`, attention leaves there what a decode continuing
    /// from here reads.
    pub fn forward_prefill_attn(
        &self,
        x: &Tensor,
        pre_mix: &[Vec<f32>],
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
        cache: Option<&mut AttnCache>,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        use crate::inference::offload::stage;
        let residual = x;
        let am = stage("hc mixes", || self.attn_mixes(x))?;
        let xin = stage("hc pre and norm", || {
            rms_norm(&hc_pre(x, pre_mix)?, &self.attn_norm, self.norm_eps)
        })?;
        let xattn = stage("attention", || {
            self.attn
                .forward_prefill_cached(&xin, cos, sin, shared, cache)
        })?;
        let x = stage("hc post", || {
            hc_post(&xattn, residual, &am.post, &am.comb, self.hc_mult)
        })?;
        Ok((x, am.pre))
    }

    /// The FFN half of `forward_prefill`, over `x` [b, s, hc, d] and the pre-mix attention produced.
    /// Every step is per token, so the streams of several prefills run as one when concatenated
    /// along the sequence.
    pub fn forward_prefill_ffn(
        &self,
        x: &Tensor,
        pre_mix: &[Vec<f32>],
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        use crate::inference::offload::stage;
        let residual = x;
        let fm = stage("hc mixes", || self.mixes(x, &self.hc_ffn))?;
        let xin = stage("hc pre and norm", || {
            rms_norm(&hc_pre(x, pre_mix)?, &self.ffn_norm, self.norm_eps)
        })?;
        let xffn = stage("moe", || self.moe.forward(&xin))?;
        let out = stage("hc post", || {
            hc_post(&xffn, residual, &fm.post, &fm.comb, self.hc_mult)
        })?;
        Ok((out, fm.pre))
    }

    /// Decode one token. Same structure as `forward_prefill` at s == 1 - the hyper-connections,
    /// norms and MoE are per-token - but attention reads from and appends to `cache`. `x` is
    /// [1, 1, hc, d], `pre_mix` [1, hc].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode(
        &self,
        x: &Tensor,
        pre_mix: &[Vec<f32>],
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        cache: &mut AttnCache,
        shared: &mut SharedAttn,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        use crate::inference::offload::stage;
        let residual = x;
        let am = stage("hc mixes", || self.attn_mixes(x))?;
        let xin = stage("hc pre and norm", || {
            rms_norm(&hc_pre(x, pre_mix)?, &self.attn_norm, self.norm_eps)
        })?;
        let xattn = stage("attention", || {
            self.attn.forward_decode(&xin, cos, sin, pos, cache, shared)
        })?;
        let x = stage("hc post", || {
            hc_post(&xattn, residual, &am.post, &am.comb, self.hc_mult)
        })?;

        let residual = &x;
        let fm = stage("hc mixes", || self.mixes(&x, &self.hc_ffn))?;
        let xin = stage("hc pre and norm", || {
            rms_norm(&hc_pre(&x, &am.pre)?, &self.ffn_norm, self.norm_eps)
        })?;
        let xffn = stage("moe", || self.moe.forward(&xin))?;
        let out = stage("hc post", || {
            hc_post(&xffn, residual, &fm.post, &fm.comb, self.hc_mult)
        })?;
        Ok((out, fm.pre))
    }
}
