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
    /// The mapped storage (address, length) of the quantised projections a prompt's attention
    /// reads, in the order it reads them.
    pub fn storage_spans(&self) -> Vec<(usize, usize)> {
        let ps: Vec<&crate::inference::offload::projection::Projection> = match self {
            LayerAttn::Ratio0(a) => vec![&a.wq_a, &a.wkv, &a.wq_b, &a.wo_a, &a.wo_b],
            LayerAttn::Band(a) => {
                let mut v = vec![&a.wq_a, &a.wkv];
                if let Some(i) = &a.indexer {
                    v.push(&i.wq_b);
                }
                v.extend([&a.wq_b, &a.wo_a, &a.wo_b]);
                v
            }
        };
        ps.into_iter().filter_map(|p| p.storage_span()).collect()
    }

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

/// Where a prefill's stream stands after a layer run against a card that holds it.
pub enum DevLayer {
    /// Still on the card; the collapse weights for the next layer.
    Device(Vec<Vec<f32>>),
    /// Handed back to the host: the stream and the collapse weights.
    Host(Tensor, Vec<Vec<f32>>),
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
    /// Start reading this layer's weights for a prompt: its attention's, then its experts'.
    pub fn prefetch_layer(&self) {
        self.moe.prefetch_layer(&self.attn.storage_spans());
    }

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

    /// One layer of a prefill whose residual stream is held by `ds` (a card): both halves, with the
    /// stream's mixes, collapse and update done there and only each sublayer's input and output
    /// crossing. `pre_mix` is the collapse weights from the layer above. A step the card cannot
    /// do hands the stream back to the host, which finishes the layer and carries on without it.
    pub fn forward_prefill_dev(
        &self,
        ds: &mut dyn crate::inference::offload::HcStream,
        pre_mix: &[Vec<f32>],
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
        cache: Option<&mut AttnCache>,
    ) -> Result<DevLayer> {
        use crate::inference::offload::stage;
        let hc = self.hc_mult;
        let n = pre_mix.len();
        let d = self.attn_norm.elem_count();
        let flat = |rows: &[Vec<f32>]| -> Vec<f32> { rows.iter().flatten().copied().collect() };
        let comb_flat =
            |c: &[Vec<Vec<f32>>]| -> Vec<f32> { c.iter().flatten().flatten().copied().collect() };
        let mixes =
            |ds: &mut dyn crate::inference::offload::HcStream, w: &HcWeights| -> Result<HcMixes> {
                let func = super::hyper_connections::dense_f32(&w.func)?;
                let (pre, post, comb) = ds.mixes(
                    &func,
                    &w.scale.flatten_all()?.to_vec1::<f32>()?,
                    &w.base.flatten_all()?.to_vec1::<f32>()?,
                    self.hc_sinkhorn_iters,
                    self.norm_eps,
                    self.hc_eps,
                )?;
                Ok(HcMixes {
                    pre: pre.chunks(hc).map(<[f32]>::to_vec).collect(),
                    post: post.chunks(hc).map(<[f32]>::to_vec).collect(),
                    comb: comb
                        .chunks(hc * hc)
                        .map(|t| t.chunks(hc).map(<[f32]>::to_vec).collect())
                        .collect(),
                })
            };
        let input = |ds: &mut dyn crate::inference::offload::HcStream,
                     pre: &[Vec<f32>],
                     norm: &Tensor|
         -> Result<Tensor> {
            let w = norm.flatten_all()?.to_vec1::<f32>()?;
            let xin = ds.pre_norm(&flat(pre), &w, self.norm_eps)?;
            Tensor::from_vec(xin, (1, n, d), &crate::tensor::Device::Cpu)
        };
        // The stream as the host holds it from here on.
        let back = |ds: &mut dyn crate::inference::offload::HcStream,
                    why: &crate::tensor::Error|
         -> Result<Tensor> {
            tracing::warn!(
                "the card cannot hold the prompt's stream ({why}); the host finishes it"
            );
            Tensor::from_vec(ds.rows(0, n)?, (1, n, hc, d), &crate::tensor::Device::Cpu)
        };
        let attended = (|| -> Result<(HcMixes, Tensor)> {
            let am = stage("hc mixes", || mixes(ds, &self.hc_attn))?;
            let xin = stage("hc pre and norm", || input(ds, pre_mix, &self.attn_norm))?;
            Ok((am, xin))
        })();
        let (am, xin) = match attended {
            Ok(v) => v,
            Err(e) => {
                let x = back(ds, &e)?;
                let (mid, mix) = self.forward_prefill_attn(&x, pre_mix, cos, sin, shared, cache)?;
                let (nx, npm) = self.forward_prefill_ffn(&mid, &mix)?;
                return Ok(DevLayer::Host(nx, npm));
            }
        };
        let xattn = stage("attention", || {
            self.attn
                .forward_prefill_cached(&xin, cos, sin, shared, cache)
        })?;
        let posted = stage("hc post", || {
            let xo = super::hyper_connections::dense_f32(&xattn)?;
            ds.post(&xo, &flat(&am.post), &comb_flat(&am.comb))
        });
        if let Err(e) = posted {
            let x = back(ds, &e)?;
            let mid = hc_post(&xattn, &x, &am.post, &am.comb, hc)?;
            let (nx, npm) = self.forward_prefill_ffn(&mid, &am.pre)?;
            return Ok(DevLayer::Host(nx, npm));
        }
        let ffn = (|| -> Result<(HcMixes, Tensor)> {
            let fm = stage("hc mixes", || mixes(ds, &self.hc_ffn))?;
            let xin = stage("hc pre and norm", || input(ds, &am.pre, &self.ffn_norm))?;
            Ok((fm, xin))
        })();
        let (fm, xin) = match ffn {
            Ok(v) => v,
            Err(e) => {
                let x = back(ds, &e)?;
                let (nx, npm) = self.forward_prefill_ffn(&x, &am.pre)?;
                return Ok(DevLayer::Host(nx, npm));
            }
        };
        let xffn = stage("moe", || self.moe.forward(&xin))?;
        let posted = stage("hc post", || {
            let xo = super::hyper_connections::dense_f32(&xffn)?;
            ds.post(&xo, &flat(&fm.post), &comb_flat(&fm.comb))
        });
        if let Err(e) = posted {
            let x = back(ds, &e)?;
            let out = hc_post(&xffn, &x, &fm.post, &fm.comb, hc)?;
            return Ok(DevLayer::Host(out, fm.pre));
        }
        Ok(DevLayer::Device(fm.pre))
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

    /// The FFN half of a decode step that may leave up to `defer` of its host experts running:
    /// the stream returned lacks their share, and the job comes back with this layer's `post`
    /// weights, which `add_deferred` needs to fold the share into the stream later.
    #[allow(clippy::type_complexity)]
    pub fn forward_decode_ffn_deferring(
        &self,
        x: &Tensor,
        pre_mix: &[Vec<f32>],
        defer: usize,
    ) -> Result<(
        Tensor,
        Vec<Vec<f32>>,
        Option<(super::moe::DeferredExperts, Vec<f32>)>,
    )> {
        use crate::inference::offload::stage;
        let residual = x;
        let fm = stage("hc mixes", || self.mixes(x, &self.hc_ffn))?;
        let xin = stage("hc pre and norm", || {
            rms_norm(&hc_pre(x, pre_mix)?, &self.ffn_norm, self.norm_eps)
        })?;
        let (xffn, deferred) = stage("moe", || self.moe.forward_with(&xin, defer))?;
        let out = stage("hc post", || {
            hc_post(&xffn, residual, &fm.post, &fm.comb, self.hc_mult)
        })?;
        let post = fm.post.first().cloned().unwrap_or_default();
        Ok((out, fm.pre, deferred.map(|d| (d, post))))
    }

    /// Wait for the experts a deferring FFN half left running and add their share to `x`.
    pub fn add_deferred(
        &self,
        x: &Tensor,
        deferred: super::moe::DeferredExperts,
        post: &[f32],
    ) -> Result<Tensor> {
        let delta = crate::inference::offload::stage("moe deferred wait", || {
            self.moe.finish_deferred(deferred)
        })?;
        super::hyper_connections::hc_add(x, post, &delta)
    }

    /// Read ahead the routed experts this block is expected to use for the token whose stream
    /// reaches the layer below as `x` [1, 1, hc, d]: the FFN input is approximated by the
    /// stream collapsed with this block's own mixes, which skips the attention in between.
    pub fn advise_experts(&self, x: &Tensor) -> Result<()> {
        let am = self.attn_mixes(x)?;
        let xin = rms_norm(&hc_pre(x, &am.pre)?, &self.ffn_norm, self.norm_eps)?;
        let dim = xin.elem_count();
        self.moe.predict_and_advise(&xin.reshape((1, dim))?)?;
        Ok(())
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
        let (x, am_pre) = self.forward_decode_attn(x, pre_mix, cos, sin, pos, cache, shared)?;
        self.forward_prefill_ffn(&x, &am_pre)
    }

    /// The attention half of `forward_decode`: the stream after ring-aware attention, and the
    /// pre-mix the FFN half collapses it with. Split out so a batched verify can run this per
    /// token, each attending over the ring, and then the FFN half once over the whole block -
    /// the MoE reads each expert once for the block, which is where the speed of a verify is.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_attn(
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
        Ok((x, am.pre))
    }
}
