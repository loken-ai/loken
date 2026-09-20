//! DeepSeek V4.1 attention for compress_ratio-0 layers (bet phase 2).
//!
//! A ratio-0 layer runs sliding-window attention only: no compressor, no indexer. What remains is
//! the whole of the mechanism this phase gates - latent low-rank Q, a single shared KV head that
//! goes through an fp8 round trip before it is attended, a learned per-head attention sink, and a
//! grouped low-rank output projection. Higher ratios (phases 5-6) add compressed KV on top of this
//! same window path, so this code is the base the later phases extend rather than replace.
//!
//! The reference is `notes/deepseek-oracle` (DeepSeek's own `model.py`, unmodified). This forward
//! reproduces its `Attention.forward` for the prefill case (`start_pos == 0`) of a ratio-0 layer,
//! and is judged tensor-for-tensor against a dump of that path in the module test.

use super::band::sparse_attn;
use super::cache::AttnCache;
use crate::inference::offload::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Device, Result, Tensor};

/// e4m3fn saturates here; the KV round trip clamps to this before rounding to the grid.
const FP8_E4M3_MAX: f32 = 448.0;

/// The weights of one ratio-0 attention block, all held as F32.
///
/// Shapes follow the reference (`o_lora_rank` = output low-rank width, `o_groups` = the block
/// diagonal group count of the output projection):
/// - `wq_a` [q_lora, dim], `q_norm` [q_lora], `wq_b` [n_heads * head_dim, q_lora]
/// - `wkv` [head_dim, dim], `kv_norm` [head_dim]
/// - `attn_sink` [n_heads]
/// - `wo_a` [o_groups * o_lora, n_heads * head_dim / o_groups], `wo_b` [dim, o_groups * o_lora]
pub struct Ratio0Attention {
    pub wq_a: Projection,
    pub q_norm: Tensor,
    pub wq_b: Projection,
    pub wkv: Projection,
    pub kv_norm: Tensor,
    pub attn_sink: Tensor,
    pub wo_a: Projection,
    pub wo_b: Projection,

    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub eps: f32,
}

impl Ratio0Attention {
    fn softmax_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }

    /// Prefill forward for a ratio-0 layer.
    ///
    /// `x` is [b, s, dim]; `cos` and `sin` are the real and imaginary halves of the rope table,
    /// each [>= s, rope_head_dim / 2], so `cos[p, j]` and `sin[p, j]` rotate the j-th pair at
    /// position p. Positions run from 0 (this is `start_pos == 0`). Returns [b, s, dim].
    pub fn forward_prefill(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        self.forward_prefill_cached(x, cos, sin, None)
    }

    /// `forward_prefill`, leaving in `cache` the window a decode continuing from here reads: the
    /// last `window_size` rope'd, fp8 KV rows, as `forward_decode` would have pushed them.
    pub fn forward_prefill_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut AttnCache>,
    ) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let (nh, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        let x2 = x.reshape((b * s, dim))?;

        // Latent Q: down-project, RMS-norm the latent, up-project, split into heads, rope the tail.
        let qr = rms_norm(&self.wq_a.apply(&x2)?, &self.q_norm, self.eps)?;
        let q = self.wq_b.apply(&qr)?.reshape((b, s, nh, hd))?;
        let q = rope_partial(&q, cos, sin, rd)?;

        // Shared KV: one head, RMS-normed, rope on the tail, then the fp8 round trip the reference
        // applies to the cache regardless of weight dtype (it was trained with it).
        let kv = rms_norm(&self.wkv.apply(&x2)?, &self.kv_norm, self.eps)?;
        let kv = rope_partial(&kv.reshape((b, s, 1, hd))?, cos, sin, rd)?.reshape((b, s, hd))?;
        let kv = act_quant_fp8_e4m3(&kv, 32)?;
        if let Some(cache) = cache {
            fill_window(cache, &kv, b, self.window_size)?;
        }

        // Sliding-window attention with the learned per-head sink, one KV head shared by every
        // query head. o is [b, s, nh, hd].
        let o = sparse_window_attn(
            &q,
            &kv,
            &self.attn_sink,
            self.window_size,
            self.softmax_scale(),
        )?;
        // The cache stays in one rotated form, so the query's rotation is removed from the output.
        let o = rope_partial(&o, cos, &sin.neg()?, rd)?;

        // Grouped low-rank output projection.
        self.grouped_out(&o, b, s, dim)
    }

    /// Grouped low-rank output projection. `wo_a` is block-diagonal over groups: group g projects
    /// only its own slice of the flattened heads, so it is an einsum, not a plain matmul. `o` is
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

    /// Decode one token at `pos`. `x` is [b, 1, dim] (b == 1), `cos`/`sin` the layer's full rope
    /// tables (row `pos` is used), `cache` this layer's window ring. Appends this token's rope'd,
    /// fp8 KV to the ring, attends the query over the ring, returns [b, 1, dim].
    pub fn forward_decode(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        cache: &mut AttnCache,
    ) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        debug_assert_eq!((b, s), (1, 1));
        let (nh, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        let cos = &cos.narrow(0, pos, 1)?;
        let sin = &sin.narrow(0, pos, 1)?;
        let x2 = x.reshape((b * s, dim))?;

        use crate::inference::offload::stage;
        let qr = stage("attn q_a", || {
            rms_norm(&self.wq_a.apply(&x2)?, &self.q_norm, self.eps)
        })?;
        let q = stage("attn q_b", || self.wq_b.apply(&qr))?.reshape((b, s, nh, hd))?;
        let q = stage("attn rope", || rope_partial(&q, cos, sin, rd))?;

        let kv = stage("attn kv", || {
            rms_norm(&self.wkv.apply(&x2)?, &self.kv_norm, self.eps)
        })?;
        let kv = stage("attn rope", || {
            rope_partial(&kv.reshape((b, s, 1, hd))?, cos, sin, rd)
        })?
        .reshape((b, s, hd))?;
        let kv = stage("attn kv fp8", || act_quant_fp8_e4m3(&kv, 32))?;

        push_window(
            cache,
            &kv.flatten_all()?.to_vec1::<f32>()?,
            self.window_size,
        );
        let o = stage("attn window", || {
            attend_window(&q, cache, hd, &self.attn_sink, self.softmax_scale())
        })?;
        let o = stage("attn rope", || rope_partial(&o, cos, &sin.neg()?, rd))?;
        stage("attn grouped out", || self.grouped_out(&o, b, s, dim))
    }
}

/// The window ring a decode would hold after a prefill's `kv` [1, s, hd]: its last `window` rows.
pub(super) fn fill_window(
    cache: &mut AttnCache,
    kv: &Tensor,
    b: usize,
    window: usize,
) -> Result<()> {
    if b != 1 {
        return Err(crate::tensor::Error::msg(
            "a prefill that fills a decode cache runs one sequence",
        ));
    }
    let (_, s, hd) = kv.dims3()?;
    let rows = kv.flatten_all()?.to_vec1::<f32>()?;
    for t in s.saturating_sub(window)..s {
        push_window(cache, &rows[t * hd..(t + 1) * hd], window);
    }
    Ok(())
}

/// Append a rope'd, quantised KV row to the window ring, evicting the oldest past `window`.
pub(super) fn push_window(cache: &mut AttnCache, kv_row: &[f32], window: usize) {
    cache.win.push_back(kv_row.to_vec());
    while cache.win.len() > window {
        cache.win.pop_front();
    }
}

/// Attend a single-token query [b, 1, h, hd] over the whole window ring (its last <= window rows),
/// with the learned per-head sink. Returns [b, 1, h, hd].
pub(super) fn attend_window(
    q: &Tensor,
    cache: &AttnCache,
    hd: usize,
    sink: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let w = cache.win.len();
    let mut buf = Vec::with_capacity(w * hd);
    for row in &cache.win {
        buf.extend_from_slice(row);
    }
    let kv = Tensor::from_vec(buf, (1, w.max(1), hd), &Device::Cpu)?;
    let idxs: Vec<i32> = (0..w as i32).collect();
    let sink = sink.flatten_all()?.to_vec1::<f32>()?;
    sparse_attn(q, &kv, &sink, &idxs, w, scale)
}

/// Rope the last `rd` elements of each head with the interleaved (GPT-J) convention the reference
/// uses, leaving the leading `head_dim - rd` (the nope part) untouched. `x` is [b, s, h, hd].
pub(crate) fn rope_partial(x: &Tensor, cos: &Tensor, sin: &Tensor, rd: usize) -> Result<Tensor> {
    use rayon::prelude::*;
    let (b, s, h, hd) = x.dims4()?;
    let half = rd / 2;
    let (rows, chalf) = cos.dims2()?;
    if rows < s || chalf != half {
        return Err(crate::tensor::Error::msg(format!(
            "rope: a table of {rows} x {chalf} for {s} positions of {rd} rotated dims"
        )));
    }
    let cv = cos.flatten_all()?.to_vec1::<f32>()?;
    let sv = sin.flatten_all()?.to_vec1::<f32>()?;
    let mut v = x.flatten_all()?.to_vec1::<f32>()?;
    // One pass over the rotated tail of each head, in place: the nope part is never copied. The
    // same float operations as `rope_i`, pair (2j, 2j + 1) at position p.
    v.par_chunks_mut(h * hd)
        .enumerate()
        .for_each(|(row, token)| {
            let p = row % s;
            let (c, sn) = (&cv[p * half..(p + 1) * half], &sv[p * half..(p + 1) * half]);
            for head in token.chunks_mut(hd) {
                let tail = &mut head[hd - rd..];
                for j in 0..half {
                    let (x0, x1) = (tail[2 * j], tail[2 * j + 1]);
                    tail[2 * j] = x0 * c[j] - x1 * sn[j];
                    tail[2 * j + 1] = x0 * sn[j] + x1 * c[j];
                }
            }
        });
    let _ = b;
    Tensor::from_vec(v, x.dims().to_vec(), &Device::Cpu)
}

/// Block-wise fp8 (e4m3) quant/dequant round trip on the last dimension, with power-of-two scales
/// (the ue8m0 format). This is the reference's `act_quant(..., inplace=True)` on the KV cache: it
/// rounds every value to what the fp8 grid can hold and reads it back, so the attention sees the
/// quantised K and V rather than the exact ones.
pub(crate) fn act_quant_fp8_e4m3(x: &Tensor, block: usize) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let n = *dims.last().unwrap();
    debug_assert_eq!(n % block, 0);
    let mut v = x.flatten_all()?.to_vec1::<f32>()?;
    let mut i = 0;
    while i < v.len() {
        let seg = &mut v[i..i + block];
        let amax = seg.iter().fold(0f32, |m, &e| m.max(e.abs()));
        // A power-of-two scale that puts the block's largest magnitude at or below fp8's max.
        let scale = pow2_ceil((amax / FP8_E4M3_MAX).max(1e-30));
        for e in seg.iter_mut() {
            let clamped = (*e / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
            *e = e4m3_round(clamped) * scale;
        }
        i += block;
    }
    Tensor::from_vec(v, dims, &Device::Cpu)
}

/// The smallest power of two that is >= x (x > 0), matching `exp2(ceil(log2(x)))`.
pub(crate) fn pow2_ceil(x: f32) -> f32 {
    (x.log2().ceil()).exp2()
}

/// Round a value to the nearest e4m3fn magnitude, ties to even. `a` is finite and already clamped
/// to [-448, 448]; e4m3fn has three mantissa bits, min normal 2^-6, subnormal step 2^-9.
fn e4m3_round(a: f32) -> f32 {
    if a == 0.0 {
        return 0.0;
    }
    let sign = a.signum();
    let mag = a.abs();
    const MIN_NORMAL: f32 = 0.015625; // 2^-6
    const SUB_STEP: f32 = 0.001953125; // 2^-9
    let step = if mag < MIN_NORMAL {
        SUB_STEP
    } else {
        // step between representable values in mag's binade: 2^(e-3).
        2f32.powi(mag.log2().floor() as i32 - 3)
    };
    let r = round_half_to_even(mag / step);
    (sign * r * step).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
}

fn round_half_to_even(y: f32) -> f32 {
    let f = y.floor();
    let d = y - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Sliding-window attention with a learned per-head sink, computed in F32.
///
/// `q` is [b, s, h, d]; `kv` [b, n, d] is the single shared KV head serving as both K and V;
/// `sink` [h] is an extra logit per head that takes softmax mass but contributes no value, so a
/// query with an empty window yields zeros. At prefill `n == s` and query i attends keys in
/// [max(0, i - window + 1), i], which is exactly the window index set the reference builds.
fn sparse_window_attn(
    q: &Tensor,
    kv: &Tensor,
    sink: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let n = kv.dim(1)?;
    let qv = q.flatten_all()?.to_vec1::<f32>()?;
    let kvv = kv.flatten_all()?.to_vec1::<f32>()?;
    let sinkv = sink.flatten_all()?.to_vec1::<f32>()?;
    if let (Some(offload), 1) = (crate::inference::offload::current(), b) {
        let idxs = super::band::window_idxs(b, s, window);
        if let Some(out) =
            offload.sparse_attention(&qv, &kvv, &sinkv, &idxs, (s, h, d, window), scale)
        {
            return Tensor::from_vec(out?, (b, s, h, d), &Device::Cpu);
        }
    }
    let mut out = vec![0f32; b * s * h * d];

    for bi in 0..b {
        for i in 0..s {
            let lo = i.saturating_sub(window - 1);
            for hi in 0..h {
                let q_off = ((bi * s + i) * h + hi) * d;
                let sink_logit = sinkv[hi];
                // First pass: logits over the window and their running max (the sink included).
                let mut logits = Vec::with_capacity(i - lo + 1);
                let mut m = sink_logit;
                for j in lo..=i {
                    let kv_off = (bi * n + j) * d;
                    let mut dot = 0f32;
                    for t in 0..d {
                        dot += qv[q_off + t] * kvv[kv_off + t];
                    }
                    let l = dot * scale;
                    if l > m {
                        m = l;
                    }
                    logits.push(l);
                }
                // Second pass: softmax denominator (with the sink) and the value-weighted sum.
                let mut denom = (sink_logit - m).exp();
                let o_off = q_off;
                for (k, j) in (lo..=i).enumerate() {
                    let w = (logits[k] - m).exp();
                    denom += w;
                    let kv_off = (bi * n + j) * d;
                    for t in 0..d {
                        out[o_off + t] += w * kvv[kv_off + t];
                    }
                }
                for t in 0..d {
                    out[o_off + t] /= denom;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, s, h, d), &Device::Cpu)
}
