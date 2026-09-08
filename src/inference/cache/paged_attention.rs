//! Paged GQA decode attention - the core data-plane computation for continuous
//! batching. Each decoding sequence has ONE query token attending over its own
//! KV, which lives in non-contiguous physical blocks ([`crate::inference::cache::paged_kv`]). For
//! each sequence we gather its KV via the block table and run GQA softmax
//! attention.
//!
//! This is the scalar f32 REFERENCE - obviously correct, the spec a fused CUDA
//! paged-attention kernel must match bit-for-bit (greedy-identical). It already
//! runs the real loop (gather + GQA + online-max-stable softmax) so the batched
//! forward can call it for an initial correct continuous-batching path; the
//! kernel then drops in for speed without changing semantics.

/// One decoding sequence's attention inputs/outputs (host f32, GQA layout).
pub struct PagedAttnSeq<'a> {
    /// Query for the new token: `[n_head * head_dim]`.
    pub q: &'a [f32],
    /// Gathered keys over the context: `[context_len * n_kv_head * head_dim]`
    /// (row-major by position, then kv-head, then dim - the [`crate::inference::cache::paged_kv::
    /// PagedKvStore::gather_seq`] layout reshaped to per-kv-head).
    pub k: &'a [f32],
    /// Gathered values, same layout as `k`.
    pub v: &'a [f32],
    /// Number of tokens attended (including the new one).
    pub context_len: usize,
    /// Output buffer for the attended result: `[n_head * head_dim]`.
    pub out: &'a mut [f32],
}

/// Compute GQA decode attention for one sequence. `n_head` query heads share
/// `n_kv_head` KV heads (group = n_head / n_kv_head). `scale` multiplies scores
/// pre-softmax. Causal over the full gathered context (the new token is last).
pub fn paged_attn_decode_seq(
    s: &mut PagedAttnSeq,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f32,
) {
    debug_assert_eq!(s.q.len(), n_head * head_dim);
    debug_assert_eq!(s.out.len(), n_head * head_dim);
    debug_assert_eq!(s.k.len(), s.context_len * n_kv_head * head_dim);
    let group = n_head / n_kv_head;
    let ctx = s.context_len;
    let kv_stride = n_kv_head * head_dim; // per-position stride in k/v
    let mut scores = vec![0f32; ctx];
    for h in 0..n_head {
        let kvh = h / group;
        let qh = &s.q[h * head_dim..(h + 1) * head_dim];
        // scores[t] = scale * (qh . k[t, kvh])
        let mut mx = f32::NEG_INFINITY;
        for t in 0..ctx {
            let kt = &s.k[t * kv_stride + kvh * head_dim..][..head_dim];
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * kt[d];
            }
            let sc = dot * scale;
            scores[t] = sc;
            if sc > mx {
                mx = sc;
            }
        }
        // softmax (max-stable)
        let mut sum = 0f32;
        for t in 0..ctx {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        // out[h] = Σ_t p[t] . v[t, kvh]
        let oh = &mut s.out[h * head_dim..(h + 1) * head_dim];
        for d in 0..head_dim {
            oh[d] = 0.0;
        }
        for t in 0..ctx {
            let p = scores[t] * inv;
            let vt = &s.v[t * kv_stride + kvh * head_dim..][..head_dim];
            for d in 0..head_dim {
                oh[d] += p * vt[d];
            }
        }
    }
}

/// GPU-ready NEOX (split-half) RoPE on `x` `[B, n_heads, head_dim]`, each batch
/// row rotated at its own `positions[b]`. Tensor ops -> runs on the input's device
/// (no host round-trip). Returns the rotated tensor. Matches the scalar rope used
/// by the decode reference (validated <1e-5).
pub fn rope_inplace_tensor(
    x: &crate::tensor::Tensor,
    positions: &[usize],
    base: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    let dims = x.dims();
    let (_bsz, _n_heads, hd) = (dims[0], dims[1], dims[2]);
    let (cos, sin) = rope_cos_sin(positions, hd, base, &x.device(), x.dtype())?;
    rope_apply(x, &cos, &sin, false)
}

/// Build split-half RoPE cos/sin tables for `positions`, shaped `[B,1,half]` to
/// broadcast over heads, in `dt`. Depends ONLY on positions/hd/base - so a
/// batched decode computes this ONCE per step and reuses it across all layers
/// (the per-layer recompute was ~2.(n_layers-1) redundant trig+alloc ops/step).
pub fn rope_cos_sin(
    positions: &[usize],
    hd: usize,
    base: f32,
    dev: &crate::tensor::Device,
    dt: crate::tensor::DType,
) -> crate::tensor::Result<(crate::tensor::Tensor, crate::tensor::Tensor)> {
    let bsz = positions.len();
    let half = hd / 2;
    let mut cos = vec![0f32; bsz * half];
    let mut sin = vec![0f32; bsz * half];
    let inv_freq = crate::inference::model::rope::inverse_frequencies(hd, base);
    for b in 0..bsz {
        for i in 0..half {
            let (s, c) = (positions[b] as f32 * inv_freq[i]).sin_cos();
            cos[b * half + i] = c;
            sin[b * half + i] = s;
        }
    }
    let cos = crate::tensor::Tensor::from_vec(cos, &[bsz, 1, half][..], dev)?.to_dtype(dt)?;
    let sin = crate::tensor::Tensor::from_vec(sin, &[bsz, 1, half][..], dev)?.to_dtype(dt)?;
    Ok((cos, sin))
}

/// Apply precomputed RoPE cos/sin (`[B,1,half]`) to `x` (`[B,heads,hd]`).
/// `interleaved` selects the pairing convention: interleaved/GPT-J (Llama/Mistral,
/// pairs `(2i,2i+1)` - mirrors `ops::rope_i`) vs non-interleaved/NeoX
/// (all others, pairs `(i,i+half)` - mirrors `ops::rope`). This MUST match
/// the serial forward's `flags.use_rope_i`, else the learned Q/K get the wrong
/// dim pairing and attention scrambles - subtly at low positions, catastrophically
/// at high positions (the CB >256-prompt garbage/EOS bug).
pub fn rope_apply(
    x: &crate::tensor::Tensor,
    cos: &crate::tensor::Tensor,
    sin: &crate::tensor::Tensor,
    interleaved: bool,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    // Fused single-kernel path (CUDA F16 - the CB decode hot path): replaces the
    // ~8 tensor ops below (hundreds of graph nodes across the layers) with one
    // launch. Falls back to the tensor path for CPU / non-F16 / odd shapes.
    if x.dims().len() == 3 {
        if let Some(out) =
            crate::inference::kernel::fused::paged_rope_f16(x, cos, sin, interleaved)?
        {
            return Ok(out);
        }
    }
    let half = x.dims()[2] / 2;
    if interleaved {
        let (b, nh, hd) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let xr = x.reshape(&[b, nh, half, 2][..])?;
        let x1 = xr.narrow(3, 0, 1)?.squeeze(3)?; // even indices [B, nh, half]
        let x2 = xr.narrow(3, 1, 1)?.squeeze(3)?; // odd indices
                                                  // o1 = x1*cos - x2*sin ; o2 = x1*sin + x2*cos ; then re-interleave.
        let o1 = x1
            .broadcast_mul(cos)?
            .broadcast_add(&x2.broadcast_mul(sin)?.affine(-1.0, 0.0)?)?;
        let o2 = x1
            .broadcast_mul(sin)?
            .broadcast_add(&x2.broadcast_mul(cos)?)?;
        return crate::tensor::Tensor::stack(&[&o1, &o2], 3)?.reshape(&[b, nh, hd][..]);
    }
    let x1 = x.narrow(2, 0, half)?; // [B, nh, half]
    let x2 = x.narrow(2, half, half)?;
    // o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin
    let o1 = x1
        .broadcast_mul(cos)?
        .broadcast_add(&x2.broadcast_mul(sin)?.affine(-1.0, 0.0)?)?;
    let o2 = x2
        .broadcast_mul(cos)?
        .broadcast_add(&x1.broadcast_mul(sin)?)?;
    crate::tensor::Tensor::cat(&[&o1, &o2], 2)
}

/// GPU-ready paged GQA decode attention via tensor ops (matmul + softmax) - runs
/// on whatever device the inputs live on (CUDA in production, no host round-trip,
/// unlike the scalar reference). `q`: `[nh, hd]`; `k`,`v`: `[ctx, nkv, hd]`.
/// Returns `[nh, hd]`. Per-KV-head GQA: the `nh/nkv` query heads in a group
/// attend over that KV head's keys/values. Validated bit-close to the scalar
/// reference (test below); this is the path the GPU batched forward calls.
pub fn paged_attn_decode_seq_tensor(
    q: &crate::tensor::Tensor,
    k: &crate::tensor::Tensor,
    v: &crate::tensor::Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    use crate::tensor::ops::softmax_last_dim;
    let g = nh / nkv;
    let ctx = k.dims()[0];
    let mut outs = Vec::with_capacity(nkv);
    for kvh in 0..nkv {
        let qg = q.narrow(0, kvh * g, g)?; // [g, hd]
        let kk = k.narrow(1, kvh, 1)?.reshape(&[ctx, hd][..])?.contiguous()?; // [ctx, hd]
        let vv = v.narrow(1, kvh, 1)?.reshape(&[ctx, hd][..])?.contiguous()?; // [ctx, hd]
        let scores = qg
            .matmul(&kk.transpose(0, 1)?.contiguous()?)?
            .affine(scale, 0.0)?; // [g, ctx]
        let p = softmax_last_dim(&scores)?; // [g, ctx]
        outs.push(p.matmul(&vv)?); // [g, hd]
    }
    let refs: Vec<&crate::tensor::Tensor> = outs.iter().collect();
    crate::tensor::Tensor::cat(&refs, 0) // [nh, hd]
}

/// Batched GQA paged decode attention over B sequences that share the SAME
/// context length (the burst/lockstep case). Runs in O(1) batched-matmul launches
/// per layer instead of O(B.nkv) tiny kernels - the GPU throughput lever (the
/// per-seq loop is launch-bound once weight GEMMs amortize across the batch).
///   q: [B, nh, hd]   k,v: [B, ctx, nkv, hd]   ->  [B, nh, hd]
pub fn paged_attn_decode_batched_tensor(
    q: &crate::tensor::Tensor,
    k: &crate::tensor::Tensor,
    v: &crate::tensor::Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    use crate::tensor::ops::softmax_last_dim;
    let b = q.dims()[0];
    let _ctx = k.dims()[1];
    let g = nh / nkv;
    // q [B,nh,hd] -> [B,nkv,g,hd]
    let qg = q.reshape(&[b, nkv, g, hd][..])?;
    // k,v [B,ctx,nkv,hd] -> [B,nkv,ctx,hd]
    let kt = k.transpose(1, 2)?.contiguous()?;
    let vt = v.transpose(1, 2)?.contiguous()?;
    // scores [B,nkv,g,ctx] = qg @ kt^T
    let scores = qg
        .matmul(&kt.transpose(2, 3)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let p = softmax_last_dim(&scores)?; // [B,nkv,g,ctx]
    let out = p.matmul(&vt)?; // [B,nkv,g,hd]
    out.reshape(&[b, nh, hd][..])
}

/// Ragged batched GQA decode attention: B sequences of DIFFERENT context lengths
/// `lens`, K/V padded to `max_ctx` ([B,max_ctx,nkv,hd]). Padding positions are
/// masked to -inf so they contribute nothing. ONE batched matmul/softmax for the
/// whole batch - the real-serving attention (contexts desync during the prefill
/// ramp, so the equal-context fast path rarely applies). q: [B,nh,hd] -> [B,nh,hd].
/// Build the ragged decode mask `[B,1,1,max_ctx]` (0 where j<len[b], -inf beyond).
/// Depends only on lens - compute ONCE per step (not per layer) and reuse.
pub fn ragged_mask(
    lens: &[usize],
    max_ctx: usize,
    dev: &crate::tensor::Device,
    dt: crate::tensor::DType,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    let b = lens.len();
    let mut m = vec![0f32; b * max_ctx];
    for bi in 0..b {
        for j in lens[bi]..max_ctx {
            m[bi * max_ctx + j] = f32::NEG_INFINITY;
        }
    }
    crate::tensor::Tensor::from_vec(m, &[b, 1, 1, max_ctx][..], dev)?.to_dtype(dt)
}

pub fn paged_attn_decode_ragged_tensor(
    q: &crate::tensor::Tensor,
    k: &crate::tensor::Tensor,
    v: &crate::tensor::Tensor,
    mask: &crate::tensor::Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    use crate::tensor::ops::softmax_last_dim;
    let b = q.dims()[0];
    let g = nh / nkv;
    let qg = q.reshape(&[b, nkv, g, hd][..])?; // [B,nkv,g,hd]
    let kt = k.transpose(1, 2)?.contiguous()?; // [B,nkv,max_ctx,hd]
    let vt = v.transpose(1, 2)?.contiguous()?;
    let scores = qg
        .matmul(&kt.transpose(2, 3)?.contiguous()?)?
        .affine(scale, 0.0)?; // [B,nkv,g,max_ctx]
    let scores = scores.broadcast_add(mask)?;
    let p = softmax_last_dim(&scores)?;
    let out = p.matmul(&vt)?; // [B,nkv,g,hd]
    out.reshape(&[b, nh, hd][..])
}

/// Causal multi-token PREFILL attention for ONE sequence (GQA). Processes all T
/// prompt positions in one shot - q,k,v are [T, heads, hd]. Returns [T, nh, hd].
/// This is the prefill counterpart to the decode paths: one forward over the
/// whole prompt instead of T serial single-token forwards (the TTFT lever).
pub fn paged_prefill_attn_tensor(
    q: &crate::tensor::Tensor,
    k: &crate::tensor::Tensor,
    v: &crate::tensor::Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    use crate::tensor::ops::softmax_last_dim;
    let t = q.dims()[0];
    let g = nh / nkv;
    let dt = q.dtype();
    let dev = q.device().clone();
    // causal mask [T,T]: 0 on/below diagonal, -inf above.
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mask = crate::tensor::Tensor::from_vec(m, &[t, t][..], &dev)?.to_dtype(dt)?;
    // q [T,nh,hd]->[nh,T,hd], k/v [T,nkv,hd]->[nkv,T,hd]
    let qh = q.transpose(0, 1)?.contiguous()?;
    let kh = k.transpose(0, 1)?.contiguous()?;
    let vh = v.transpose(0, 1)?.contiguous()?;
    let mut outs = Vec::with_capacity(nkv);
    for kvh in 0..nkv {
        let qg = qh.narrow(0, kvh * g, g)?; // [g,T,hd]
        let kk = kh.narrow(0, kvh, 1)?.reshape(&[t, hd][..])?; // [T,hd]
        let vv = vh.narrow(0, kvh, 1)?.reshape(&[t, hd][..])?; // [T,hd]
                                                               // scores [g,T,T] = qg @ kk^T
        let scores = qg
            .broadcast_matmul(&kk.transpose(0, 1)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(&mask)?;
        let p = softmax_last_dim(&scores)?; // [g,T,T]
        outs.push(p.broadcast_matmul(&vv)?); // [g,T,hd]
    }
    let refs: Vec<&crate::tensor::Tensor> = outs.iter().collect();
    let cat = crate::tensor::Tensor::cat(&refs, 0)?; // [nh,T,hd]
    cat.transpose(0, 1)?.contiguous() // [T,nh,hd]
}

/// Offset-causal attention for a PREFIX-CACHED suffix prefill: `q` are the suffix
/// queries (`tsfx` positions starting at absolute position `cached`), `k`/`v` are
/// the FULL sequence keys/values (`t = cached + tsfx`: the gathered cached prefix
/// KV concatenated with the freshly-computed suffix KV). Suffix query `i` (absolute
/// position `cached+i`) attends causally over keys `0..=cached+i`. Same math as
/// [`paged_prefill_attn_tensor`] but with a rectangular `[tsfx, t]` mask offset by
/// `cached` - this is what lets a multi-turn / shared-prefix request skip
/// recomputing the prefix's KV and attend over the cached blocks instead.
pub fn paged_prefill_attn_suffix(
    q: &crate::tensor::Tensor,
    k: &crate::tensor::Tensor,
    v: &crate::tensor::Tensor,
    cached: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
) -> crate::tensor::Result<crate::tensor::Tensor> {
    use crate::tensor::ops::softmax_last_dim;
    let tsfx = q.dims()[0];
    let t = k.dims()[0];
    debug_assert_eq!(
        t,
        cached + tsfx,
        "k/v must be the full prefix+suffix length"
    );
    let g = nh / nkv;
    let dt = q.dtype();
    let dev = q.device().clone();
    // mask [tsfx, t]: suffix query i attends keys j <= cached+i.
    let mut m = vec![0f32; tsfx * t];
    for i in 0..tsfx {
        for j in (cached + i + 1)..t {
            m[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mask = crate::tensor::Tensor::from_vec(m, &[tsfx, t][..], &dev)?.to_dtype(dt)?;
    let qh = q.transpose(0, 1)?.contiguous()?; // [nh, tsfx, hd]
    let kh = k.transpose(0, 1)?.contiguous()?; // [nkv, t, hd]
    let vh = v.transpose(0, 1)?.contiguous()?;
    let mut outs = Vec::with_capacity(nkv);
    for kvh in 0..nkv {
        let qg = qh.narrow(0, kvh * g, g)?; // [g, tsfx, hd]
        let kk = kh.narrow(0, kvh, 1)?.reshape(&[t, hd][..])?; // [t, hd]
        let vv = vh.narrow(0, kvh, 1)?.reshape(&[t, hd][..])?; // [t, hd]
        let scores = qg
            .broadcast_matmul(&kk.transpose(0, 1)?.contiguous()?)?
            .affine(scale, 0.0)?; // [g,tsfx,t]
        let scores = scores.broadcast_add(&mask)?;
        let p = softmax_last_dim(&scores)?;
        outs.push(p.broadcast_matmul(&vv)?); // [g, tsfx, hd]
    }
    let refs: Vec<&crate::tensor::Tensor> = outs.iter().collect();
    let cat = crate::tensor::Tensor::cat(&refs, 0)?; // [nh, tsfx, hd]
    cat.transpose(0, 1)?.contiguous() // [tsfx, nh, hd]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_attn_matches_full_prefill_for_suffix_positions() {
        // The prefix-cache correctness contract: a suffix prefill attending over
        // (gathered prefix KV ++ suffix KV) must produce the SAME outputs for the
        // suffix positions as a full prefill over the whole sequence.
        use crate::tensor::{Device, Tensor};
        let (t, cached, nh, nkv, hd) = (6usize, 4usize, 2usize, 1usize, 2usize);
        let tsfx = t - cached;
        let scale = 1.0 / (hd as f32).sqrt();
        let dev = Device::Cpu;
        let qv: Vec<f32> = (0..t * nh * hd)
            .map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.5)
            .collect();
        let kv: Vec<f32> = (0..t * nkv * hd)
            .map(|i| ((i * 5) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let vv: Vec<f32> = (0..t * nkv * hd)
            .map(|i| ((i * 3) % 9) as f32 * 0.1 - 0.4)
            .collect();
        let q = Tensor::from_vec(qv, &[t, nh, hd][..], &dev).unwrap();
        let k = Tensor::from_vec(kv, &[t, nkv, hd][..], &dev).unwrap();
        let v = Tensor::from_vec(vv, &[t, nkv, hd][..], &dev).unwrap();
        let full = paged_prefill_attn_tensor(&q, &k, &v, nh, nkv, hd, scale).unwrap();
        let q_sfx = q.narrow(0, cached, tsfx).unwrap();
        let sfx = paged_prefill_attn_suffix(&q_sfx, &k, &v, cached, nh, nkv, hd, scale).unwrap();
        let full_sfx = full.narrow(0, cached, tsfx).unwrap();
        let a: Vec<f32> = sfx.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = full_sfx.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "suffix attn {x} != full prefill {y}");
        }
    }

    // Independent naive reference (different code path: explicit float softmax,
    // no max-subtraction) to cross-check the optimized max-stable version.
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        ctx: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
    ) -> Vec<f32> {
        let group = nh / nkv;
        let kvs = nkv * hd;
        let mut out = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / group;
            let mut e = vec![0f32; ctx];
            let mut sum = 0f32;
            for t in 0..ctx {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += q[h * hd + d] * k[t * kvs + kvh * hd + d];
                }
                e[t] = (dot * scale).exp();
                sum += e[t];
            }
            for t in 0..ctx {
                let p = e[t] / sum;
                for d in 0..hd {
                    out[h * hd + d] += p * v[t * kvs + kvh * hd + d];
                }
            }
        }
        out
    }

    #[test]
    fn single_head_two_keys_hand_check() {
        // 1 head, hd=2, ctx=2, scale=1. q=[1,0]. k0=[1,0], k1=[0,1].
        // scores = [1, 0] -> softmax = [e/(e+1), 1/(e+1)]. v0=[10,0], v1=[0,10].
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [10.0f32, 0.0, 0.0, 10.0];
        let mut out = [0f32; 2];
        let mut s = PagedAttnSeq {
            q: &q,
            k: &k,
            v: &v,
            context_len: 2,
            out: &mut out,
        };
        paged_attn_decode_seq(&mut s, 1, 1, 2, 1.0);
        let e = std::f32::consts::E;
        let p0 = e / (e + 1.0);
        let p1 = 1.0 / (e + 1.0);
        assert!(
            (out[0] - 10.0 * p0).abs() < 1e-5,
            "out0={} want {}",
            out[0],
            10.0 * p0
        );
        assert!(
            (out[1] - 10.0 * p1).abs() < 1e-5,
            "out1={} want {}",
            out[1],
            10.0 * p1
        );
    }

    #[test]
    fn matches_independent_naive_reference_gqa() {
        // GQA: 4 query heads, 2 kv heads, hd=3, ctx=5, scale=1/sqrt(3).
        let (nh, nkv, hd, ctx) = (4usize, 2usize, 3usize, 5usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..nh * hd)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let k: Vec<f32> = (0..ctx * nkv * hd)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2)
            .collect();
        let v: Vec<f32> = (0..ctx * nkv * hd)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.5)
            .collect();
        let mut out = vec![0f32; nh * hd];
        let mut s = PagedAttnSeq {
            q: &q,
            k: &k,
            v: &v,
            context_len: ctx,
            out: &mut out,
        };
        paged_attn_decode_seq(&mut s, nh, nkv, hd, scale);
        let want = naive(&q, &k, &v, ctx, nh, nkv, hd, scale);
        for i in 0..nh * hd {
            assert!(
                (out[i] - want[i]).abs() < 1e-5,
                "i={i} got {} want {}",
                out[i],
                want[i]
            );
        }
    }

    #[test]
    fn tensor_rope_matches_scalar() {
        use crate::tensor::{Device, Tensor};
        let (bsz, nh, hd, base) = (3usize, 2usize, 8usize, 10000f32);
        let positions = [0usize, 5, 17];
        let data: Vec<f32> = (0..bsz * nh * hd)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        // scalar reference (split-half, per the batched_forward rope)
        let half = hd / 2;
        let mut want = data.clone();
        for b in 0..bsz {
            for h in 0..nh {
                let off = (b * nh + h) * hd;
                for i in 0..half {
                    let f = 1.0f32 / base.powf((2 * i) as f32 / hd as f32);
                    let (s, c) = (positions[b] as f32 * f).sin_cos();
                    let (x1, x2) = (want[off + i], want[off + half + i]);
                    want[off + i] = x1 * c - x2 * s;
                    want[off + half + i] = x2 * c + x1 * s;
                }
            }
        }
        let xt = Tensor::from_vec(data, &[bsz, nh, hd][..], &Device::Cpu).unwrap();
        let got = rope_inplace_tensor(&xt, &positions, base)
            .unwrap()
            .reshape(&[bsz * nh * hd][..])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..bsz * nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "i={i} {} != {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn tensor_version_matches_scalar_reference_gqa() {
        use crate::tensor::{Device, Tensor};
        let (nh, nkv, hd, ctx) = (4usize, 2usize, 3usize, 5usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..nh * hd)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let k: Vec<f32> = (0..ctx * nkv * hd)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2)
            .collect();
        let v: Vec<f32> = (0..ctx * nkv * hd)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.5)
            .collect();
        // scalar reference
        let mut want = vec![0f32; nh * hd];
        let mut s = PagedAttnSeq {
            q: &q,
            k: &k,
            v: &v,
            context_len: ctx,
            out: &mut want,
        };
        paged_attn_decode_seq(&mut s, nh, nkv, hd, scale);
        // tensor (GPU-ready) version on the same data
        let qt = Tensor::from_vec(q, &[nh, hd][..], &Device::Cpu).unwrap();
        let kt = Tensor::from_vec(k, &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let vt = Tensor::from_vec(v, &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let got = paged_attn_decode_seq_tensor(&qt, &kt, &vt, nh, nkv, hd, scale)
            .unwrap()
            .reshape(&[nh * hd][..])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "i={i} tensor {} != scalar {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn batched_tensor_matches_per_seq_tensor() {
        // The equal-context batched GQA path must equal the validated per-seq path
        // run independently for each of B sequences.
        use crate::tensor::{Device, Tensor};
        let (b, nh, nkv, hd, ctx) = (3usize, 4usize, 2usize, 3usize, 5usize);
        let scale = 1.0 / (hd as f32).sqrt();
        // distinct data per sequence
        let mut want = Vec::with_capacity(b * nh * hd);
        let mut qb = Vec::with_capacity(b * nh * hd);
        let mut kb = Vec::with_capacity(b * ctx * nkv * hd);
        let mut vb = Vec::with_capacity(b * ctx * nkv * hd);
        for bi in 0..b {
            let q: Vec<f32> = (0..nh * hd)
                .map(|i| (((i + bi * 3) * 7 % 11) as f32 - 5.0) * 0.3)
                .collect();
            let k: Vec<f32> = (0..ctx * nkv * hd)
                .map(|i| (((i + bi) * 5 % 13) as f32 - 6.0) * 0.2)
                .collect();
            let v: Vec<f32> = (0..ctx * nkv * hd)
                .map(|i| (((i + bi * 2) * 3 % 7) as f32 - 3.0) * 0.5)
                .collect();
            let qt = Tensor::from_vec(q.clone(), &[nh, hd][..], &Device::Cpu).unwrap();
            let kt = Tensor::from_vec(k.clone(), &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
            let vt = Tensor::from_vec(v.clone(), &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
            let per = paged_attn_decode_seq_tensor(&qt, &kt, &vt, nh, nkv, hd, scale)
                .unwrap()
                .reshape(&[nh * hd][..])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            want.extend_from_slice(&per);
            qb.extend_from_slice(&q);
            kb.extend_from_slice(&k);
            vb.extend_from_slice(&v);
        }
        let qt = Tensor::from_vec(qb, &[b, nh, hd][..], &Device::Cpu).unwrap();
        let kt = Tensor::from_vec(kb, &[b, ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let vt = Tensor::from_vec(vb, &[b, ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let got = paged_attn_decode_batched_tensor(&qt, &kt, &vt, nh, nkv, hd, scale)
            .unwrap()
            .reshape(&[b * nh * hd][..])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..b * nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "i={i} batched {} != per-seq {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn ragged_tensor_matches_per_seq_unequal_lens() {
        // The ragged batched path (padding + per-seq mask) must equal the per-seq
        // path run independently, for B sequences of DIFFERENT context lengths.
        use crate::tensor::{Device, Tensor};
        let (nh, nkv, hd) = (4usize, 2usize, 3usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let lens = [5usize, 2, 4];
        let b = lens.len();
        let max_ctx = *lens.iter().max().unwrap();
        let mut want = Vec::with_capacity(b * nh * hd);
        let mut qrows = Vec::with_capacity(b * nh * hd);
        let mut kpad = vec![0f32; b * max_ctx * nkv * hd];
        let mut vpad = vec![0f32; b * max_ctx * nkv * hd];
        for bi in 0..b {
            let ctx = lens[bi];
            let q: Vec<f32> = (0..nh * hd)
                .map(|i| (((i + bi * 3) * 7 % 11) as f32 - 5.0) * 0.3)
                .collect();
            let k: Vec<f32> = (0..ctx * nkv * hd)
                .map(|i| (((i + bi) * 5 % 13) as f32 - 6.0) * 0.2)
                .collect();
            let v: Vec<f32> = (0..ctx * nkv * hd)
                .map(|i| (((i + bi * 2) * 3 % 7) as f32 - 3.0) * 0.5)
                .collect();
            let qt = Tensor::from_vec(q.clone(), &[nh, hd][..], &Device::Cpu).unwrap();
            let kt = Tensor::from_vec(k.clone(), &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
            let vt = Tensor::from_vec(v.clone(), &[ctx, nkv, hd][..], &Device::Cpu).unwrap();
            let per = paged_attn_decode_seq_tensor(&qt, &kt, &vt, nh, nkv, hd, scale)
                .unwrap()
                .reshape(&[nh * hd][..])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            want.extend_from_slice(&per);
            qrows.extend_from_slice(&q);
            // scatter k/v into the padded [B,max_ctx,nkv,hd] buffer
            let base = bi * max_ctx * nkv * hd;
            kpad[base..base + ctx * nkv * hd].copy_from_slice(&k);
            vpad[base..base + ctx * nkv * hd].copy_from_slice(&v);
        }
        let qt = Tensor::from_vec(qrows, &[b, nh, hd][..], &Device::Cpu).unwrap();
        let kt = Tensor::from_vec(kpad, &[b, max_ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let vt = Tensor::from_vec(vpad, &[b, max_ctx, nkv, hd][..], &Device::Cpu).unwrap();
        let mask = ragged_mask(&lens, max_ctx, &Device::Cpu, crate::tensor::DType::F32).unwrap();
        let got = paged_attn_decode_ragged_tensor(&qt, &kt, &vt, &mask, nh, nkv, hd, scale)
            .unwrap()
            .reshape(&[b * nh * hd][..])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..b * nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "i={i} ragged {} != per-seq {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn single_context_token_is_just_value() {
        // ctx=1 -> softmax of one score = 1 -> out == v[0].
        let q = [0.5f32, -0.5];
        let k = [3.0f32, 7.0];
        let v = [42.0f32, -9.0];
        let mut out = [0f32; 2];
        let mut s = PagedAttnSeq {
            q: &q,
            k: &k,
            v: &v,
            context_len: 1,
            out: &mut out,
        };
        paged_attn_decode_seq(&mut s, 1, 1, 2, 0.123);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[1] + 9.0).abs() < 1e-6);
    }
}
