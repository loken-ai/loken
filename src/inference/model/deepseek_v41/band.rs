//! DeepSeek V4.1 CSA2 band attention for a kv-source, single-level index-source layer (phase 5).
//!
//! Above ratio 0 a layer attends over two KV sources at once, concatenated into one sparse_attn
//! call: the sliding window of raw KV, plus `index_topk` compressed positions reaching further
//! back. A compressor pools `compress_ratio` tokens into one latent with a learned softmax gate; an
//! indexer scores the compressed positions with a small fp4 side-attention and keeps the best ones.
//! This is the single-level indexer - a layer that neither sources nor consumes candidate blocks;
//! the two-level case is phase 6.
//!
//! The reference is `notes/deepseek-oracle`; the whole band is judged against a dump in the test.

use super::attention::{act_quant_fp8_e4m3, pow2_ceil, push_window, rope_partial};
use super::cache::AttnCache;
use crate::inference::offload::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Device, Result, Tensor};

/// The e2m1 float4 grid; rounding a magnitude to it is what "fp4" means, the sign carried apart.
const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_MAX: f32 = 6.0;

/// Block-wise fp4 (e2m1) quant/dequant round trip on the last dimension. With `pow2_scale` the
/// block scale is rounded up to a power of two (the e8m0 format the indexer q/k use); without it
/// the raw f32 scale is kept (the e4m3-scale path the compressed latent uses).
fn fp4_act_quant(x: &Tensor, block: usize, pow2_scale: bool) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let n = *dims.last().unwrap();
    debug_assert_eq!(n % block, 0);
    let mut v = x.flatten_all()?.to_vec1::<f32>()?;
    let mut i = 0;
    while i < v.len() {
        let seg = &mut v[i..i + block];
        let amax = seg.iter().fold(0f32, |m, &e| m.max(e.abs()));
        let mut scale = (amax / FP4_MAX).max(1e-30);
        if pow2_scale {
            scale = pow2_ceil(scale);
        }
        for e in seg.iter_mut() {
            let scaled = (*e / scale).clamp(-FP4_MAX, FP4_MAX);
            *e = e2m1_round(scaled.abs()).copysign(scaled) * scale;
        }
        i += block;
    }
    Tensor::from_vec(v, dims, &Device::Cpu)
}

/// Nearest e2m1 grid magnitude to `m` (>= 0), first grid point winning a tie (torch argmin).
fn e2m1_round(m: f32) -> f32 {
    let mut best = FP4_GRID[0];
    let mut bestd = (m - FP4_GRID[0]).abs();
    for &g in &FP4_GRID[1..] {
        let d = (m - g).abs();
        if d < bestd {
            bestd = d;
            best = g;
        }
    }
    best
}

/// Gather rows `start, start+step, ...` (count of them) of a 2-D table [rows, cols].
fn stride_rows(t: &Tensor, start: usize, step: usize, count: usize) -> Result<Tensor> {
    let cols = t.dim(1)?;
    let all = t.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(count * cols);
    for j in 0..count {
        out.extend_from_slice(&all[start + j * step]);
    }
    Tensor::from_vec(out, (count, cols), &Device::Cpu)
}

/// Pools `ratio` consecutive tokens into one KV latent with a learned per-channel softmax gate.
/// `wkv`/`wgate` are [head_dim, dim], `norm` [head_dim]. Returns the pre-RoPE latent
/// [b, s/ratio, head_dim] for the prefill case (ratio > 1).
pub struct Compressor {
    pub norm: Tensor,
    pub wkv: Tensor,
    /// The pooling gate; a ratio-1 layer pools nothing and carries none.
    pub wgate: Option<Tensor>,
    pub ratio: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl Compressor {
    fn forward_prefill(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _dim) = x.dims3()?;
        let groups = s / self.ratio;
        let hd = self.head_dim;
        let x2 = x.reshape((b * s, x.dim(2)?))?;
        let kv = crate::inference::offload::linear(&x2, &self.wkv)?.to_vec2::<f32>()?; // [b*s, hd]
                                                                                       // A ratio-1 layer pools nothing - one token per group - and carries no gate: its latent
                                                                                       // is the projected token itself.
        let Some(wgate) = &self.wgate else {
            if self.ratio != 1 {
                return Err(crate::tensor::Error::msg(
                    "compressor without a gate at ratio > 1",
                ));
            }
            let flat: Vec<f32> = kv.into_iter().flatten().collect();
            let pooled = Tensor::from_vec(flat, (b * s, hd), &Device::Cpu)?;
            return rms_norm(&pooled, &self.norm, self.eps)?.reshape((b, groups, hd));
        };
        let score = crate::inference::offload::linear(&x2, wgate)?.to_vec2::<f32>()?;
        // Per (group, channel) softmax over the ratio tokens, then the weighted sum.
        let mut pooled = vec![0f32; b * groups * hd];
        for bi in 0..b {
            for g in 0..groups {
                for c in 0..hd {
                    let mut mx = f32::NEG_INFINITY;
                    for r in 0..self.ratio {
                        let row = bi * s + g * self.ratio + r;
                        mx = mx.max(score[row][c]);
                    }
                    let mut den = 0f32;
                    let mut acc = 0f32;
                    for r in 0..self.ratio {
                        let row = bi * s + g * self.ratio + r;
                        let w = (score[row][c] - mx).exp();
                        den += w;
                        acc += w * kv[row][c];
                    }
                    pooled[(bi * groups + g) * hd + c] = acc / den;
                }
            }
        }
        let pooled = Tensor::from_vec(pooled, (b * groups, hd), &Device::Cpu)?;
        rms_norm(&pooled, &self.norm, self.eps)?.reshape((b, groups, hd))
    }
}

/// State a band layer publishes for the layers below it. A kv-source writes `compress_kv`; an
/// index-key owner writes `index_k`; an index-source writes `topk_idxs`; a candidate source writes
/// `candidates`. Layers run in order, so a source always writes before its readers read. Reset (a
/// fresh default) at the start of each forward.
#[derive(Default)]
pub struct SharedAttn {
    pub compress_kv: Option<Tensor>, // [b, compress_len, head_dim], rope'd + fp4
    pub index_k: Option<Tensor>,     // [b, compress_len, index_head_dim], rope'd + fp4
    pub topk_idxs: Option<Vec<i32>>, // [b*s*index_topk], already offset into the concatenated KV
    pub candidates: Option<Vec<bool>>, // [b*s*compress_len]
}

/// The indexer of an index-source layer. It owns its index keys only when it is also a kv-source
/// (`owns_k`); otherwise it reads them from the kv-source below. A candidate source publishes the
/// block mask; a candidate consumer scores only inside that mask.
pub struct Indexer {
    pub wq_b: Projection,       // [index_n_heads*index_head_dim, q_lora]
    pub weights_proj: Tensor,   // [index_n_heads, dim]
    pub wk: Option<Tensor>,     // [index_head_dim, head_dim], only when it owns its keys
    pub k_norm: Option<Tensor>, // [index_head_dim], likewise
    pub n_heads: usize,
    pub index_head_dim: usize,
    pub rope_head_dim: usize,
    pub index_topk: usize,
    pub eps: f32,
    pub owns_k: bool,
    pub is_candidate_source: bool,
    pub uses_candidates: bool,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
}

impl Indexer {
    /// Returns `[b, s, index_topk]` int32 indices into the compressed positions, shifted by
    /// `offset`, or -1 for an unreachable slot, and publishes them plus (if owned) the index keys
    /// and (if a candidate source) the block mask. `latent` is this layer's pre-RoPE compressor
    /// latent, `Some` only when the layer owns its keys.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        x: &Tensor,
        qr: &Tensor,
        latent: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        ratio: usize,
        offset: usize,
        groups: usize,
        shared: &mut SharedAttn,
    ) -> Result<Vec<i32>> {
        let (b, s, _dim) = x.dims3()?;
        let ihd = self.index_head_dim;
        let rd = self.rope_head_dim;
        let nh = self.n_heads;

        // Own the index keys (derive from the pre-RoPE latent), or read the kv-source's.
        if self.owns_k {
            let latent = latent.expect("owns_k needs the compressor latent");
            let cos_c = stride_rows(cos, 0, ratio, groups)?;
            let sin_c = stride_rows(sin, 0, ratio, groups)?;
            let hd_latent = latent.dim(2)?;
            let wk = self.wk.as_ref().expect("owns_k needs index keys");
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("owns_k needs an index key norm");
            let k = rms_norm(
                &crate::inference::offload::linear(&latent.reshape((b * groups, hd_latent))?, wk)?,
                k_norm,
                self.eps,
            )?;
            let k = rope_partial(&k.reshape((b, groups, 1, ihd))?, &cos_c, &sin_c, rd)?
                .reshape((b, groups, ihd))?;
            shared.index_k = Some(fp4_act_quant(&k, 32, true)?);
        }
        let index_k = shared
            .index_k
            .clone()
            .expect("index_source needs an index_k source");
        let gtot = index_k.dim(1)?;

        // Index queries from the shared query latent.
        let q = self
            .wq_b
            .apply(&qr.reshape((b * s, qr.dim(2)?))?)?
            .reshape((b, s, nh, ihd))?;
        let q = rope_partial(&q, cos, sin, rd)?;
        let q = fp4_act_quant(&q, 32, true)?;

        let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        let weights =
            crate::inference::offload::linear(&x.reshape((b * s, x.dim(2)?))?, &self.weights_proj)?
                .to_vec2::<f32>()?;

        let qv = q.flatten_all()?.to_vec1::<f32>()?;
        let kv = index_k.flatten_all()?.to_vec1::<f32>()?;
        let clen_of = |i: usize| (i + 1) / ratio;
        let offloaded = match (crate::inference::offload::current(), b) {
            (Some(off), 1) => {
                let flat: Vec<f32> = weights.iter().flatten().copied().collect();
                off.index_scores(&qv, &kv, &flat, (s, nh, ihd, gtot), ratio, scale)
            }
            _ => None,
        };
        // Full score matrix [b*s, gtot], -inf beyond each query's reachable count.
        let computed = offloaded.is_some();
        let mut scores = match offloaded {
            Some(scores) => scores?,
            None => vec![f32::NEG_INFINITY; b * s * gtot],
        };
        for bi in (0..b).filter(|_| !computed) {
            for i in 0..s {
                let clen = clen_of(i).min(gtot);
                for tpos in 0..clen {
                    let mut acc = 0f32;
                    for h in 0..nh {
                        let qo = ((bi * s + i) * nh + h) * ihd;
                        let ko = (bi * gtot + tpos) * ihd;
                        let mut dot = 0f32;
                        for t in 0..ihd {
                            dot += qv[qo + t] * kv[ko + t];
                        }
                        acc += dot.max(0.0) * weights[bi * s + i][h] * scale;
                    }
                    scores[(bi * s + i) * gtot + tpos] = acc;
                }
            }
        }

        // Two-level candidate pre-filter: a source publishes the block mask; a consumer restricts
        // its scores to it before its own top-k.
        let compress_lens: Vec<usize> = (0..b * s).map(|q| clen_of(q % s)).collect();
        if self.is_candidate_source {
            shared.candidates = Some(super::candidates::select_candidate_blocks(
                &scores,
                b * s,
                gtot,
                &compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            ));
        } else if self.uses_candidates {
            let mask = shared
                .candidates
                .as_ref()
                .expect("uses_candidates needs a candidate source");
            scores = super::candidates::apply_candidate_mask(&scores, mask);
        }

        // Top-k by score, kept only where reachable, compacted into slots (order is irrelevant to
        // sparse_attn, which treats each slot independently), the rest -1.
        let topk = self.index_topk.min(gtot);
        let mut out = vec![-1i32; b * s * self.index_topk];
        {
            use rayon::prelude::*;
            // The same set a stable sort by descending score keeps - ties to the lower position -
            // found by a partial selection, each query on its own.
            out.par_chunks_mut(self.index_topk)
                .enumerate()
                .for_each(|(qi, slots)| {
                    if topk == 0 {
                        return;
                    }
                    let clen = clen_of(qi % s);
                    let row = &scores[qi * gtot..qi * gtot + gtot];
                    let mut order: Vec<usize> = (0..gtot).collect();
                    let by = |a: &usize, c: &usize| {
                        row[*c].partial_cmp(&row[*a]).unwrap().then(a.cmp(c))
                    };
                    if topk < gtot {
                        order.select_nth_unstable_by(topk - 1, by);
                    }
                    let kept = &mut order[..topk];
                    kept.sort_by(by);
                    let mut slot = 0;
                    for &t in kept.iter() {
                        if t < clen {
                            slots[slot] = (t + offset) as i32;
                            slot += 1;
                        }
                    }
                });
        }
        shared.topk_idxs = Some(out.clone());
        Ok(out)
    }

    /// Decode: top-k compressed positions for one query token at `pos`. `qr` is the shared query
    /// latent [1, q_lora], `x` this layer's input [1, 1, dim]. Reads its index keys from `cache`
    /// (owns_k) or the shared state, scores the reachable groups, and returns `index_topk` indices
    /// into the concatenated cache (shifted by `w_offset`, the window ring size), -1 for empty. Also
    /// publishes them to `shared.topk_idxs` for a reader below.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode(
        &self,
        x: &Tensor,
        qr: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        ratio: usize,
        w_offset: usize,
        cache: &AttnCache,
        shared: &mut SharedAttn,
    ) -> Result<Vec<i32>> {
        let ihd = self.index_head_dim;
        let rd = self.rope_head_dim;
        let nh = self.n_heads;
        let cos_p = cos.narrow(0, pos, 1)?;
        let sin_p = sin.narrow(0, pos, 1)?;

        // Index keys: this layer's own cache, or the kv-source's published keys.
        let (kv, gtot) = if self.owns_k {
            (cache.index_k.clone(), cache.n_groups)
        } else {
            let ik = shared
                .index_k
                .as_ref()
                .expect("index_source needs an index_k source");
            (ik.flatten_all()?.to_vec1::<f32>()?, ik.dim(1)?)
        };

        let q = self.wq_b.apply(qr)?.reshape((1, 1, nh, ihd))?;
        let q = rope_partial(&q, &cos_p, &sin_p, rd)?;
        let q = fp4_act_quant(&q, 32, true)?;
        let qv = q.flatten_all()?.to_vec1::<f32>()?;

        let weights = x
            .reshape((1, x.dim(2)?))?
            .matmul(&self.weights_proj.t()?)?
            .to_vec2::<f32>()?;
        let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);

        let clen = ((pos + 1) / ratio).min(gtot);
        let mut scores = vec![f32::NEG_INFINITY; gtot];
        for (tpos, sc) in scores.iter_mut().enumerate().take(clen) {
            let mut acc = 0f32;
            for h in 0..nh {
                let qo = h * ihd;
                let ko = tpos * ihd;
                let mut dot = 0f32;
                for t in 0..ihd {
                    dot += qv[qo + t] * kv[ko + t];
                }
                acc += dot.max(0.0) * weights[0][h] * scale;
            }
            *sc = acc;
        }

        // Two-level candidate pre-filter, one query row.
        if self.is_candidate_source {
            shared.candidates = Some(super::candidates::select_candidate_blocks(
                &scores,
                1,
                gtot,
                &[clen],
                self.candidate_topk_blocks,
                self.candidate_block_size,
            ));
        } else if self.uses_candidates {
            let mask = shared
                .candidates
                .as_ref()
                .expect("uses_candidates needs a source");
            scores = super::candidates::apply_candidate_mask(&scores, mask);
        }

        let topk = self.index_topk.min(gtot);
        let mut order: Vec<usize> = (0..gtot).collect();
        order.sort_by(|&a, &c| scores[c].partial_cmp(&scores[a]).unwrap());
        let mut out = vec![-1i32; self.index_topk];
        let mut slot = 0;
        for &t in order.iter().take(topk) {
            if t < clen {
                out[slot] = (t + w_offset) as i32;
                slot += 1;
            }
        }
        shared.topk_idxs = Some(out.clone());
        Ok(out)
    }
}

/// General sparse attention over gathered positions with a learned per-head sink, in F32. `q`
/// [b, s, h, d]; `kv` [b, n, d] one shared head; `idxs` [b, s, topk] into kv, negative = empty.
pub(super) fn sparse_attn(
    q: &Tensor,
    kv: &Tensor,
    sink: &[f32],
    idxs: &[i32],
    topk: usize,
    scale: f32,
) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let n = kv.dim(1)?;
    let qv = q.flatten_all()?.to_vec1::<f32>()?;
    let kvv = kv.flatten_all()?.to_vec1::<f32>()?;
    if let (Some(offload), 1) = (crate::inference::offload::current(), b) {
        if let Some(out) = offload.sparse_attention(&qv, &kvv, sink, idxs, (s, h, d, topk), scale) {
            return Tensor::from_vec(out?, (b, s, h, d), &Device::Cpu);
        }
    }
    use rayon::prelude::*;
    let mut out = vec![0f32; b * s * h * d];
    // Every (query, head) pair on its own: the logits over its slots, their softmax with the sink,
    // and the keys - which are the values too - summed by those weights.
    out.par_chunks_mut(d).enumerate().for_each(|(pair, o)| {
        let (row, hi) = (pair / h, pair % h);
        let bi = row / s;
        let qv = &qv[pair * d..(pair + 1) * d];
        let slots = &idxs[row * topk..(row + 1) * topk];
        let sink_logit = sink[hi];
        let mut m = sink_logit;
        let mut logits = Vec::with_capacity(topk);
        for &idx in slots {
            if idx < 0 {
                continue;
            }
            let key = &kvv[(bi * n + idx as usize) * d..(bi * n + idx as usize + 1) * d];
            let l = qv.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
            if l > m {
                m = l;
            }
            logits.push((idx as usize, l));
        }
        let mut denom = (sink_logit - m).exp();
        for &(idx, l) in &logits {
            let w = (l - m).exp();
            denom += w;
            let key = &kvv[(bi * n + idx) * d..(bi * n + idx + 1) * d];
            for (acc, &v) in o.iter_mut().zip(key) {
                *acc += w * v;
            }
        }
        for v in o.iter_mut() {
            *v /= denom;
        }
    });
    Tensor::from_vec(out, (b, s, h, d), &Device::Cpu)
}

/// Which window cache slots each prefill query attends to, as [b, s, window] int32 into the window
/// KV (positions 0..s-1), -1 for slots the query cannot yet see. Matches get_window_topk_idxs.
pub(super) fn window_idxs(b: usize, s: usize, window: usize) -> Vec<i32> {
    let mut out = vec![-1i32; b * s * window];
    for bi in 0..b {
        for i in 0..s {
            let base = i.saturating_sub(window - 1);
            for k in 0..window.min(s) {
                let pos = base + k;
                if pos <= i {
                    out[(bi * s + i) * window + k] = pos as i32;
                }
            }
        }
    }
    out
}

/// A band layer: sliding window plus compressed KV. Its role in the band decides which parts it
/// owns - a kv-source has a `compressor`, an index-source has an `indexer`; a reader has neither
/// and takes both from the shared state a source below it published.
pub struct BandAttention {
    // Base attention.
    pub wq_a: Projection,
    pub q_norm: Tensor,
    pub wq_b: Projection,
    pub wkv: Projection,
    pub kv_norm: Tensor,
    pub attn_sink: Tensor,
    pub wo_a: Projection,
    pub wo_b: Projection,
    // CSA2, present per role.
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,

    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub ratio: usize,
    pub index_topk: usize,
    pub eps: f32,
}

impl BandAttention {
    fn softmax_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }

    /// Prefill forward for a band layer. `x` is [b, s, dim] (post attn_norm); `cos`/`sin` the
    /// layer's rope table; `shared` the state sources publish and readers consume. Returns
    /// [b, s, dim].
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
    ) -> Result<Tensor> {
        self.forward_prefill_cached(x, cos, sin, shared, None)
    }

    /// `forward_prefill`, leaving in `cache` what a decode continuing from here reads: the window
    /// ring, and for a kv-source its completed groups' compressed KV (and index keys, when it owns
    /// them) with the inputs of the group still filling.
    pub fn forward_prefill_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        shared: &mut SharedAttn,
        mut cache: Option<&mut AttnCache>,
    ) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let (nh, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        let x2 = x.reshape((b * s, dim))?;

        let qr = rms_norm(&self.wq_a.apply(&x2)?, &self.q_norm, self.eps)?;
        let q = self.wq_b.apply(&qr)?.reshape((b, s, nh, hd))?;
        let q = rope_partial(&q, cos, sin, rd)?;

        // Window KV (fp8 round trip), keys 0..s-1 of the concatenated cache.
        let kv = rms_norm(&self.wkv.apply(&x2)?, &self.kv_norm, self.eps)?;
        let kv = rope_partial(&kv.reshape((b, s, 1, hd))?, cos, sin, rd)?.reshape((b, s, hd))?;
        let window_kv = act_quant_fp8_e4m3(&kv, 32)?;
        if let Some(cache) = cache.as_deref_mut() {
            super::attention::fill_window(cache, &window_kv, b, self.window_size)?;
        }
        let offset = s;
        let groups = s / self.ratio;

        // The compressor (kv-source only) yields the pre-RoPE latent, which the indexer reads
        // before it is rotated and quantised for attention.
        let latent = match &self.compressor {
            Some(c) => Some(crate::inference::offload::stage("compressor", || {
                c.forward_prefill(x)
            })?),
            None => None,
        };
        let qr3 = qr.reshape((b, s, qr.dim(1)?))?;
        let indexer_started = std::time::Instant::now();
        let compress_idxs = match &self.indexer {
            Some(idx) => idx.forward_prefill(
                x,
                &qr3,
                latent.as_ref(),
                cos,
                sin,
                self.ratio,
                offset,
                groups,
                shared,
            )?,
            None => shared
                .topk_idxs
                .clone()
                .expect("a reader band layer needs a source's topk_idxs"),
        };
        if let Some(off) = crate::inference::offload::current() {
            off.record("indexer", indexer_started.elapsed().as_nanos() as u64);
        }
        // A kv-source rotates and fp4-quantises its latent, then publishes it; a reader takes the
        // compressed KV a source below it published.
        if let Some(latent) = latent {
            let cos_c = stride_rows(cos, 0, self.ratio, groups)?;
            let sin_c = stride_rows(sin, 0, self.ratio, groups)?;
            let latent = rope_partial(&latent.reshape((b, groups, 1, hd))?, &cos_c, &sin_c, rd)?
                .reshape((b, groups, hd))?;
            shared.compress_kv = Some(fp4_act_quant(&latent, 16, false)?);
        }
        let compress_kv = shared
            .compress_kv
            .clone()
            .expect("a band layer needs a compressed KV source");
        if let (Some(cache), true) = (cache, self.compressor.is_some()) {
            cache.comp_kv = compress_kv.flatten_all()?.to_vec1::<f32>()?;
            cache.n_groups = groups;
            if self.indexer.as_ref().is_some_and(|i| i.owns_k) {
                if let Some(k) = &shared.index_k {
                    cache.index_k = k.flatten_all()?.to_vec1::<f32>()?;
                }
            }
            let rows = x2.flatten_all()?.to_vec1::<f32>()?;
            cache.group_x = (groups * self.ratio..s)
                .map(|t| rows[t * dim..(t + 1) * dim].to_vec())
                .collect();
        }

        // One sparse_attn over [window || compressed] keys and their concatenated indices.
        let kv_all = Tensor::cat(&[&window_kv, &compress_kv], 1)?;
        let win = self.window_size;
        let wi = window_idxs(b, s, win);
        let itopk = self.index_topk;
        let topk = win + itopk;
        let mut idxs = vec![-1i32; b * s * topk];
        for bi in 0..b {
            for i in 0..s {
                let src_w = (bi * s + i) * win;
                let src_c = (bi * s + i) * itopk;
                let dst = (bi * s + i) * topk;
                idxs[dst..dst + win].copy_from_slice(&wi[src_w..src_w + win]);
                idxs[dst + win..dst + topk].copy_from_slice(&compress_idxs[src_c..src_c + itopk]);
            }
        }
        let sink = self.attn_sink.flatten_all()?.to_vec1::<f32>()?;
        let o = crate::inference::offload::stage("sparse attention", || {
            sparse_attn(&q, &kv_all, &sink, &idxs, topk, self.softmax_scale())
        })?;
        let o =
            crate::inference::offload::stage("rope", || rope_partial(&o, cos, &sin.neg()?, rd))?;
        crate::inference::offload::stage("grouped out", || self.grouped_out(&o, b, s, dim))
    }

    /// Grouped low-rank output projection, block-diagonal over `o_groups` (as ratio 0). `o` is
    /// [b, s, n_heads, head_dim]; returns [b, s, dim].
    fn grouped_out(&self, o: &Tensor, b: usize, s: usize, dim: usize) -> Result<Tensor> {
        let p = self.n_heads * self.head_dim / self.o_groups;
        let og = o.reshape((b * s, self.o_groups, p))?;
        let mut parts = Vec::with_capacity(self.o_groups);
        for g in 0..self.o_groups {
            let slice = og.narrow(1, g, 1)?.reshape((b * s, p))?;
            let wa = self.wo_a.rows(g * self.o_lora_rank, self.o_lora_rank)?;
            parts.push(wa.apply(&slice)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let o = Tensor::cat(&refs, 1)?;
        self.wo_b.apply(&o)?.reshape((b, s, dim))
    }

    /// Decode one token at `pos`. `x` is [1, 1, dim], `cos`/`sin` the layer's full rope tables,
    /// `cache` this layer's state, `shared` the per-step cross-layer state. A kv-source pools the
    /// next group when its buffer fills and publishes the compressed KV (and, if it owns them, the
    /// index keys); an index-source scores its query and publishes the top-k; a reader consumes
    /// both. Returns [1, 1, dim].
    pub fn forward_decode(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        cache: &mut AttnCache,
        shared: &mut SharedAttn,
    ) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        debug_assert_eq!((b, s), (1, 1));
        let (nh, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        let cos_p = cos.narrow(0, pos, 1)?;
        let sin_p = sin.narrow(0, pos, 1)?;
        let x2 = x.reshape((b * s, dim))?;

        // Query and this token's window KV, exactly as ratio 0.
        use crate::inference::offload::stage;
        let qr = stage("attn q_a", || {
            rms_norm(&self.wq_a.apply(&x2)?, &self.q_norm, self.eps)
        })?;
        let q = stage("attn q_b", || self.wq_b.apply(&qr))?.reshape((b, s, nh, hd))?;
        let q = stage("attn rope", || rope_partial(&q, &cos_p, &sin_p, rd))?;
        let kv = stage("attn kv", || {
            rms_norm(&self.wkv.apply(&x2)?, &self.kv_norm, self.eps)
        })?;
        let kv = stage("attn rope", || {
            rope_partial(&kv.reshape((b, s, 1, hd))?, &cos_p, &sin_p, rd)
        })?
        .reshape((b, s, hd))?;
        let kv = stage("attn kv fp8", || act_quant_fp8_e4m3(&kv, 32))?;
        push_window(
            cache,
            &kv.flatten_all()?.to_vec1::<f32>()?,
            self.window_size,
        );
        let w = cache.win.len();

        // kv-source: buffer this token, pool a group when it completes, publish the compressed KV
        // (and index keys, if owned).
        let compressor_started = std::time::Instant::now();
        if let Some(comp) = &self.compressor {
            cache.group_x.push(x2.flatten_all()?.to_vec1::<f32>()?);
            if cache.group_x.len() == self.ratio {
                let g = cache.n_groups;
                let flat: Vec<f32> = cache.group_x.iter().flatten().copied().collect();
                let xg = Tensor::from_vec(flat, (b, self.ratio, dim), &Device::Cpu)?;
                let latent = comp.forward_prefill(&xg)?; // [1, 1, hd], pre-RoPE
                let cos_g = cos.narrow(0, g * self.ratio, 1)?;
                let sin_g = sin.narrow(0, g * self.ratio, 1)?;
                let lr = rope_partial(&latent.reshape((b, 1, 1, hd))?, &cos_g, &sin_g, rd)?
                    .reshape((b, 1, hd))?;
                let ck = fp4_act_quant(&lr, 16, false)?;
                cache.comp_kv.extend(ck.flatten_all()?.to_vec1::<f32>()?);
                if let Some(idx) = &self.indexer {
                    if idx.owns_k {
                        let ihd = idx.index_head_dim;
                        let wk = idx.wk.as_ref().expect("owns_k needs index keys");
                        let k_norm = idx.k_norm.as_ref().expect("owns_k needs an index key norm");
                        let k =
                            rms_norm(&latent.reshape((b, hd))?.matmul(&wk.t()?)?, k_norm, idx.eps)?;
                        let k = rope_partial(&k.reshape((b, 1, 1, ihd))?, &cos_g, &sin_g, rd)?
                            .reshape((b, 1, ihd))?;
                        let k = fp4_act_quant(&k, 32, true)?;
                        cache.index_k.extend(k.flatten_all()?.to_vec1::<f32>()?);
                    }
                }
                cache.group_x.clear();
                cache.n_groups += 1;
            }
            shared.compress_kv = Some(Tensor::from_vec(
                cache.comp_kv.clone(),
                (b, cache.n_groups, hd),
                &Device::Cpu,
            )?);
            if let Some(idx) = &self.indexer {
                if idx.owns_k {
                    let ihd = idx.index_head_dim;
                    shared.index_k = Some(Tensor::from_vec(
                        cache.index_k.clone(),
                        (b, cache.n_groups, ihd),
                        &Device::Cpu,
                    )?);
                }
            }
        }

        if let Some(off) = crate::inference::offload::current() {
            off.record("compressor", compressor_started.elapsed().as_nanos() as u64);
        }
        let indexer_started = std::time::Instant::now();
        // Compressed indices: an index-source scores its own query; a reader takes the published
        // top-k. Both are shifted by the window ring size w.
        let compress_idxs = match &self.indexer {
            Some(idx) => {
                let qr1 = qr.reshape((1, qr.dim(1)?))?;
                idx.forward_decode(x, &qr1, cos, sin, pos, self.ratio, w, cache, shared)?
            }
            None => shared
                .topk_idxs
                .clone()
                .expect("a reader band layer needs a source's topk_idxs"),
        };
        if let Some(off) = crate::inference::offload::current() {
            off.record("indexer", indexer_started.elapsed().as_nanos() as u64);
        }
        let compress_kv = shared
            .compress_kv
            .clone()
            .expect("a band layer needs a compressed KV source");

        // One sparse_attn over [window ring || compressed] keys.
        let gather_started = std::time::Instant::now();
        let mut buf = Vec::with_capacity(w * hd);
        for row in &cache.win {
            buf.extend_from_slice(row);
        }
        let ring = Tensor::from_vec(buf, (b, w, hd), &Device::Cpu)?;
        let kv_all = Tensor::cat(&[&ring, &compress_kv], 1)?;
        let itopk = self.index_topk;
        let topk = w + itopk;
        let mut idxs = vec![-1i32; topk];
        for (k, slot) in idxs.iter_mut().enumerate().take(w) {
            *slot = k as i32;
        }
        idxs[w..topk].copy_from_slice(&compress_idxs[..itopk]);
        let sink = self.attn_sink.flatten_all()?.to_vec1::<f32>()?;
        if let Some(off) = crate::inference::offload::current() {
            off.record("attn kv gather", gather_started.elapsed().as_nanos() as u64);
        }
        let o = stage("sparse attention", || {
            sparse_attn(&q, &kv_all, &sink, &idxs, topk, self.softmax_scale())
        })?;
        let o = stage("attn rope", || rope_partial(&o, &cos_p, &sin_p.neg()?, rd))?;
        stage("attn grouped out", || self.grouped_out(&o, b, s, dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ratio-1 compressor carries no gate: one token per group, so its latent is the normed
    /// projection of the token itself, checked against that computation directly.
    #[test]
    fn ratio1_compressor_pools_nothing() {
        let (dim, hd, s) = (16usize, 8usize, 5usize);
        let wkv = Tensor::from_vec(
            (0..hd * dim).map(|i| (i as f32 * 0.37).sin()).collect(),
            (hd, dim),
            &Device::Cpu,
        )
        .unwrap();
        let norm = Tensor::from_vec(
            (0..hd).map(|i| 1.0 + i as f32 * 0.1).collect(),
            (hd,),
            &Device::Cpu,
        )
        .unwrap();
        let x = Tensor::from_vec(
            (0..s * dim).map(|i| (i as f32 * 0.11).cos()).collect(),
            (1, s, dim),
            &Device::Cpu,
        )
        .unwrap();
        let c = Compressor {
            norm: norm.clone(),
            wkv: wkv.clone(),
            wgate: None,
            ratio: 1,
            head_dim: hd,
            eps: 1e-6,
        };
        let got = c.forward_prefill(&x).unwrap();
        let want = rms_norm(
            &x.reshape((s, dim))
                .unwrap()
                .matmul(&wkv.t().unwrap())
                .unwrap(),
            &norm,
            1e-6,
        )
        .unwrap();
        assert_eq!(got.dims(), &[1, s, hd]);
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(g, w);
    }

    /// The card's sparse attention gives the CPU's, on window slots and on gathered slots with
    /// empty ones among them.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_card_attends_as_the_cpu_does() {
        use crate::inference::offload::cuda::Card;
        use crate::inference::offload::{room, with_offload};
        use crate::tensor::cuda::CudaDevice;
        use crate::tensor::{Device, Tensor};
        use std::sync::Arc;
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card sparse attention is NOT covered by this run");
            return;
        };
        let (_, total) = crate::tensor::cuda_ext::mem_get_info(&Device::Cuda(dev.clone())).unwrap();
        let room = room::open(dev.ordinal(), total, 0);
        let (s, h, d, n) = (37usize, 3usize, 16usize, 45usize);
        let wave = |len: usize, k: f32| {
            (0..len)
                .map(|i| ((i as f32) * k).sin())
                .collect::<Vec<f32>>()
        };
        let q = Tensor::from_vec(wave(s * h * d, 0.37), (1, s, h, d), &Device::Cpu).unwrap();
        let kv = Tensor::from_vec(wave(n * d, 0.11), (1, n, d), &Device::Cpu).unwrap();
        let sink = vec![0.3f32, -0.2, 1.1];
        let topk = 9;
        let mut idxs: Vec<i32> = window_idxs(1, s, topk);
        for (i, v) in idxs.iter_mut().enumerate() {
            if i % 5 == 4 {
                *v = ((i * 7) % n) as i32;
            }
        }
        let cpu = sparse_attn(&q, &kv, &sink, &idxs, topk, 0.25).unwrap();
        let card = with_offload(Arc::new(Card::new(dev, room)), || {
            sparse_attn(&q, &kv, &sink, &idxs, topk, 0.25)
        })
        .unwrap();
        let (cpu, card) = (
            cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            card.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        for (c, g) in cpu.iter().zip(&card) {
            assert!((c - g).abs() <= 1e-5 * c.abs().max(1.0), "{g} vs {c}");
        }
    }
}
