//! Attention: the fused prefill kernel, scaled dot-product, rotary embedding, and the
//! key/value repeat a grouped-query model needs.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Fused additive-mask softmax: `softmax_last_dim(att + mask)` in ONE pass, with
/// NO materialization of `att+mask`. `att` is `[.., Pq, Pk]`, `mask` broadcasts
/// over the leading dims as `[Pq, Pk]` (additive causal/window mask, -inf above
/// the diagonal). The prefill attention's `att.broadcast_add(mask)` allocates a
/// full `[B,H,Pq,Pk]` tensor (per layer) that softmax then re-reads - ~20% of
/// CPU prefill at long prompts. ggml/llama.cpp fold the mask into `soft_max`;
/// this matches that. Bit-identical to broadcast_add->softmax on CPU-F32; other
/// devices/dtypes fall back to the unfused pair so the GPU path is unchanged.
/// Causal attention over a whole prefill chunk without building the scores.
///
/// The query-by-key score tensor is never needed as a whole: each query row's
/// output is a weighted sum of value rows, and the weights can be accumulated
/// one key at a time while tracking that row's running maximum and sum,
/// rescaling what is already accumulated whenever the maximum moves. So the
/// scores, their softmax and the product with V collapse into one pass that
/// allocates only the result - where materializing them costs a tensor
/// proportional to chunk length times context length, per layer, per chunk.
///
/// `q` is `[b, n_kv_head, groups*seq, head_dim]` and `k`/`v` are
/// `[b, n_kv_head, kv, head_dim]`, the grouped-query layout the caller already
/// builds. Rows are independent, so the work is spread over them. Returns
/// `None` when the shapes or dtypes are off this path, and the caller keeps its
/// existing chain.
///
/// The result is not bit-identical to that chain: the running rescale sums in a
/// different order, and the chain rounds its weights to half precision before
/// the product with V while this keeps them at full width throughout - so this
/// is the more accurate of the two.
pub fn flash_attn_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    past: usize,
    seq: usize,
) -> Result<Option<Tensor>> {
    if !matches!(q.device(), Device::Cpu)
        || q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
    {
        return Ok(None);
    }
    let (qb, qh, qrows, hd) = match q.dims() {
        [a, b, c, d] => (*a, *b, *c, *d),
        _ => return Ok(None),
    };
    let (kb, kh, kv, khd) = match k.dims() {
        [a, b, c, d] => (*a, *b, *c, *d),
        _ => return Ok(None),
    };
    if qb != kb || qh != kh || hd != khd || k.dims() != v.dims() || seq == 0 || kv == 0 {
        return Ok(None);
    }
    if past + seq != kv || qrows % seq != 0 || hd == 0 || hd > MAX_HEAD_DIM {
        return Ok(None);
    }
    let qd = q.f16_data()?;
    let kd = k.f16_data()?;
    let vd = v.f16_data()?;
    if qd.len() != qb * qh * qrows * hd || kd.len() != kb * kh * kv * hd {
        return Ok(None);
    }

    // Rows are grouped per task so the score scratch is allocated once per task
    // rather than once per row.
    const ROWS_PER_TASK: usize = 8;
    let total_rows = qb * qh * qrows;
    let mut out = vec![0f32; total_rows * hd];
    let s = scale as f32;
    crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, hd * ROWS_PER_TASK, &|t, ochunk| {
        // Holds one row's scores at a time: sized by context length, never by
        // the number of query rows, so it stays small and cache-resident.
        let mut sc = vec![0f32; kv];
        let mut qrow = [0f32; MAX_HEAD_DIM];
        let mut acc = [0f32; MAX_HEAD_DIM];
        for (j, o) in ochunk.chunks_mut(hd).enumerate() {
            let r = t * ROWS_PER_TASK + j;
            // Row r belongs to one kv head; within that head the group index
            // and the query position are folded together, and the position
            // drives the causal limit.
            let head = r / qrows;
            let qpos = (r % qrows) % seq;
            let base = head * kv * hd;
            let qrow = &mut qrow[..hd];
            for (i, dst) in qrow.iter_mut().enumerate() {
                *dst = qd[r * hd + i].to_f32();
            }
            let last = past + qpos;
            let sc = &mut sc[..=last];
            // First pass: this row's scores, and the maximum they reach.
            let mut m = f32::NEG_INFINITY;
            for (c, dst) in sc.iter_mut().enumerate() {
                let x = crate::inference::cache::cpu_f16_kv::f16_dot(
                    qrow,
                    &kd[base + c * hd..base + (c + 1) * hd],
                ) * s;
                *dst = x;
                if x > m {
                    m = x;
                }
            }
            // The maximum is known before any exponential runs, so they go as
            // one vectorized sweep and the accumulator never needs rescaling.
            let sum = crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(sc, m);
            let acc = &mut acc[..hd];
            acc.fill(0.0);
            for (c, &p) in sc.iter().enumerate() {
                crate::inference::cache::cpu_f16_kv::f16_axpy(
                    p,
                    &vd[base + c * hd..base + (c + 1) * hd],
                    acc,
                );
            }
            for (i, dst) in o.iter_mut().enumerate() {
                *dst = acc[i] / sum;
            }
        }
    });
    let t = Tensor::from_vec(out, vec![qb, qh, qrows, hd], &q.device())?;
    Ok(Some(t.to_dtype(v.dtype())?))
}

/// Upper bound on head dimension for the stack buffers above; larger heads take
/// the caller's existing chain.
const MAX_HEAD_DIM: usize = 256;

/// NeoX (non-interleaved) RoPE. `x`:[b,h,s,d], `cos`/`sin`:[>=s, d/2]. Drop-in
/// for `rope`: pairs (j, j+d/2) rotated by cos/sin[seq,j].
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 {
            return crate::inference::kernel::fused::fused_rope_neox_f32(x, cos, sin);
        }
    }
    let (_b, _h, s, d) = x.dims4()?;
    let half = d / 2;
    // CPU F32 fused: rotate each [b,h,s,d] row in one pass (vs narrow/cat +
    // 4 broadcast_muls + 2 adds = ~10 allocating ops). Bit-identical math.
    if matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 && cos.dtype() == DType::F32 {
        let chalf = cos.dim(D::Minus1)?;
        let xv = x.flatten_all()?.to_vec1::<f32>()?;
        let cv = cos.flatten_all()?.to_vec1::<f32>()?;
        let sv = sin.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![0f32; xv.len()];
        // Parallelize the per-(b*h) row rotation on the shared spin-pool instead
        // of running it serially on the caller thread while the GEMM pool's
        // workers idle - one more prefill critical-path op moved onto the one pool
        // (cf. the silu/quantize fixes). Bit-identical (disjoint per-row output).
        crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, s * d, &|r, orow| {
            let roff = r * s * d;
            for si in 0..s {
                let base = si * d;
                let cbase = si * chalf;
                for j in 0..half {
                    let c = cv[cbase + j];
                    let sn = sv[cbase + j];
                    let a = xv[roff + base + j];
                    let b2 = xv[roff + base + half + j];
                    orow[base + j] = a * c - b2 * sn;
                    orow[base + half + j] = b2 * c + a * sn;
                }
            }
        });
        return Tensor::from_vec(out, x.dims().to_vec(), &x.device());
    }
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos = cos.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let sin = sin.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let o2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&o1, &o2], D::Minus1)
}

/// Interleaved (GPT-J / llama-ggml) RoPE. Drop-in for
/// `rope_i`: pairs (2j, 2j+1) rotated by cos/sin[seq,j].
pub fn rope_i(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 {
            return crate::inference::kernel::fused::fused_rope_interleaved_f32(x, cos, sin);
        }
    }
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;
    // CPU F32 fused: rotate each [b,h,s,d] row of interleaved pairs (2j, 2j+1) in
    // ONE pooled pass, instead of reshape + 2 narrows + 4 broadcast_muls + a
    // stack/reshape (~10 allocating tensor ops, several of them separate pooled
    // passes on the prefill critical path - the broadcast_muls here are a chunk
    // of the profile's broadcast time). Bit-identical: same `x_even*cos - x_odd*sin`
    // / `x_even*sin + x_odd*cos` float ops in the same order.
    if matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 && cos.dtype() == DType::F32 {
        let chalf = cos.dim(D::Minus1)?;
        let xv = x.flatten_all()?.to_vec1::<f32>()?;
        let cv = cos.flatten_all()?.to_vec1::<f32>()?;
        let sv = sin.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![0f32; xv.len()];
        crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, s * d, &|r, orow| {
            let roff = r * s * d;
            for si in 0..s {
                let base = si * d;
                let cbase = si * chalf;
                for j in 0..half {
                    let c = cv[cbase + j];
                    let sn = sv[cbase + j];
                    let x0 = xv[roff + base + 2 * j];
                    let x1 = xv[roff + base + 2 * j + 1];
                    orow[base + 2 * j] = x0 * c - x1 * sn;
                    orow[base + 2 * j + 1] = x0 * sn + x1 * c;
                }
            }
        });
        return Tensor::from_vec(out, x.dims().to_vec(), &x.device());
    }
    let xr = x.reshape((b, h, s, half, 2))?;
    let x1 = xr.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // even indices
    let x2 = xr.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?; // odd indices
    let cos = cos.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let sin = sin.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    Tensor::stack(&[&o1, &o2], D::Minus1)?.reshape((b, h, s, d))
}

/// A projection's rows split into heads: `[b, seq, heads . head_dim]` in,
/// `[b, heads, seq, head_dim]` out.
///
/// A projection writes every head of a position side by side inside that position's row, and
/// everything that reads them back - a per-head norm, the rotation, the score matmul - wants
/// the head axis in front of the positions instead. The head count is the only thing that
/// separates a query projection from a key or value one, so it is the only thing the caller
/// passes; `b` and `seq` are the projection's own and are read off it.
///
/// Materialised rather than left as a transposed view, because the matmul that follows takes
/// packed operands.
pub fn heads_first(t: Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    let (b, seq, _) = t.dims3()?;
    let split = t.reshape((b, seq, heads, head_dim))?;
    split.transpose(1, 2)?.contiguous()
}

/// How many keys one band of a banded attention covers.
///
/// The scores of a band are `rows x band` at full width, and that tensor is the whole
/// reason a long context could not be served: held over the entire context it grows
/// without bound, held over a band it does not. So the band is chosen to keep it near a
/// fixed size rather than to be a round number, and never wider than the context itself.
///
/// Wide enough that the matmul it feeds is worth launching: below a few hundred keys the
/// launch costs more than the work it does.
fn score_band(rows: usize, kv_len: usize) -> usize {
    /// What one band's scores may occupy.
    const MOST_BYTES: usize = 64 << 20;
    const NARROWEST: usize = 256;
    const WIDEST: usize = 8192;
    let per_key = rows.max(1) * std::mem::size_of::<f32>();
    (MOST_BYTES / per_key.max(1))
        .clamp(NARROWEST, WIDEST)
        .min(kv_len.max(1))
}

/// The additive causal mask for one band of keys: `[seq, band]`, zero where a query row
/// may look and negative infinity where it may not.
///
/// Built for the band rather than for the context. A whole-context mask is a square in the
/// length of the conversation, assembled on the host and copied to the card on every pass,
/// and at a long context that copy outweighs the weights of a layer - to say something
/// about the shape of the attention that carries no information at all.
fn causal_band_mask(
    seq: usize,
    past: usize,
    c0: usize,
    band: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq)
        .flat_map(|p| {
            let last = past + p;
            (0..band).map(move |j| {
                if c0 + j > last {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (seq, band), device)
}

/// The card's one-pass band softmax, where it can serve this shape.
///
/// `None` when there is no CUDA, the scores are not half precision, or the tensors are not
/// laid out the way it reads them. The caller then chains the separate operations, which
/// answer the same and read the band several times over.
fn fused_band_softmax(
    scores: &Tensor,
    run_max: &Tensor,
    rows: usize,
    width: usize,
    seq: usize,
    past: usize,
    c0: usize,
) -> Option<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let (kv_rows, _) = (scores.dim(1).ok()?, ());
        match crate::inference::quantized_cuda::band_softmax_f16(
            scores, run_max, rows, width, seq, past, c0,
        ) {
            Ok((weights, top, sum)) => Some((
                weights,
                top.reshape((1, kv_rows, seq, 1)).ok()?,
                sum.reshape((1, kv_rows, seq, 1)).ok()?,
            )),
            Err(e) => {
                // Once, and loudly enough to be seen: the chained operations answer the
                // same but read the band several times over, and a prefill that quietly
                // became half as fast because a shape stopped matching is the kind of
                // silence that takes a day to find.
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        "band softmax kernel not used, chaining the operations instead \
                         (prefill will be slower): {e}"
                    );
                });
                None
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (scores, run_max, rows, width, seq, past, c0);
        None
    }
}

/// Causal attention over a prefill chunk, one band of keys at a time.
///
/// `softmax(QK^T + mask)V` written so that no tensor in it grows with the context. That
/// form holds a score for every query against every key at once, and at a long context
/// that tensor is the largest allocation in the layer by far - large enough that the card
/// runs out of memory and the engine answers by shrinking a prefill chunk that was never
/// what made it too big. Accumulating band by band, carrying each row's running maximum
/// and sum and rescaling what is already accumulated whenever the maximum moves, gives the
/// same answer out of memory that grows with the band instead.
///
/// `q` is `[1, kv_heads, groups * seq, head_dim]` and `k`/`v` are `[1, kv_heads, kv,
/// head_dim]` - the grouped-query layout, which lets one matmul answer every query head
/// that shares a key head without copying K and V once per query head. The scale belongs
/// on `q` before the call: it is `seq x head_dim` where the scores are `seq x kv`.
///
/// The result is `[1, kv_heads, groups * seq, head_dim]` in `f32`, and is not bit-identical
/// to the unbanded form: the running rescale sums in a different order, and it keeps the
/// weights at full width where the unbanded chain rounds them to the value dtype before the
/// product with V - so this is the more accurate of the two.
pub fn banded_causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq: usize,
    past: usize,
) -> Result<Tensor> {
    let (_, kv_heads, rows, head_dim) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let band = score_band(kv_heads * rows, kv_len);
    let groups = rows / seq.max(1);

    // Each row's largest score so far and the weight it was accumulated at. Kept as
    // tensors rather than numbers because every row carries its own.
    let rows_total = kv_heads * groups * seq;
    let mut acc: Option<Tensor> = None;
    let mut run_max =
        Tensor::from_vec(vec![f32::NEG_INFINITY; rows_total], rows_total, &q.device())?;
    let mut run_sum: Option<Tensor> = None;
    let mut c0 = 0usize;
    while c0 < kv_len {
        // Every row of this chunk is causally before this band, and so before every band
        // after it.
        if c0 + 1 > past + seq {
            break;
        }
        let width = band.min(kv_len - c0);
        let scores = q
            .matmul(&k.narrow(2, c0, width)?.transpose(2, 3)?)?
            .reshape((1, kv_heads * groups, seq, width))?;

        // One pass over the band where the card has a kernel for it: widen, mask, reduce,
        // shift, exponentiate, sum and narrow are six passes over a tensor the size of the
        // band when written separately, which is three times what the matmuls around them
        // cost.
        let fused = fused_band_softmax(&scores, &run_max, rows_total, width, seq, past, c0);
        let (weights, top, band_sum) = match fused {
            Some(answer) => answer,
            None => {
                let scores = scores.to_dtype(DType::F32)?;
                // Masked only where the band crosses the diagonal. A band every row can
                // see in full needs no mask, which is most of them at a long context, and
                // building one there would be building a tensor of zeros.
                let scores = if c0 + width <= past + 1 {
                    scores
                } else {
                    scores.broadcast_add(&causal_band_mask(
                        seq,
                        past,
                        c0,
                        width,
                        &scores.device(),
                    )?)?
                };
                let top = scores.max_keepdim(D::Minus1)?.maximum(&run_max.reshape((
                    1,
                    kv_heads * groups,
                    seq,
                    1,
                ))?)?;
                let weights = scores.broadcast_sub(&top)?.exp()?;
                let band_sum = weights.sum_keepdim(D::Minus1)?;
                (weights.to_dtype(v.dtype())?, top, band_sum)
            }
        };

        let band_ctx = weights
            .contiguous()?
            .reshape((1, kv_heads, rows, width))?
            .matmul(&v.narrow(2, c0, width)?)?
            .reshape((1, kv_heads * groups, seq, head_dim))?
            .to_dtype(DType::F32)?;
        let previous = run_max.reshape((1, kv_heads * groups, seq, 1))?;
        // The first band leaves nothing to rescale: every row sees key zero, so its
        // maximum is a real number and the ones after it can be compared against it, and
        // the rescale from negative infinity is zero.
        let rescale = previous.broadcast_sub(&top)?.exp()?;
        run_sum = Some(match run_sum.take() {
            None => band_sum,
            Some(sum) => (sum.broadcast_mul(&rescale)? + band_sum)?,
        });
        acc = Some(match acc.take() {
            None => band_ctx,
            Some(a) => (a.broadcast_mul(&rescale)? + band_ctx)?,
        });
        run_max = top.reshape(rows_total)?;
        c0 += width;
    }
    acc.expect("a chunk always sees its own first key")
        .broadcast_div(&run_sum.expect("a sum accompanies every accumulator"))?
        .reshape((1, kv_heads, rows, head_dim))
}

/// Give every query group its own copy of the key or value head it reads.
///
/// `[b, kv_heads, seq, head_dim]` in, `[b, kv_heads * n_rep, seq, head_dim]` out, with query
/// head `h * n_rep + r` seeing kv head `h` - the grouping the checkpoints are trained with, and
/// the one a `cat` along the wrong axis silently gets wrong.
///
/// Broadcast rather than concatenated: the repeats are the same rows, so they are made a view
/// first and materialised once, instead of `n_rep` copies being built and then reshaped.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, kv_heads, seq, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, seq, head_dim))?
        .reshape((b, kv_heads * n_rep, seq, head_dim))
}

/// [`repeat_kv`] for a stack that carries no batch axis.
pub fn repeat_kv_unbatched(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (kv_heads, seq, head_dim) = xs.dims3()?;
    xs.unsqueeze(1)?
        .expand((kv_heads, n_rep, seq, head_dim))?
        .reshape((kv_heads * n_rep, seq, head_dim))
}

/// The additive causal mask, `[seq, seq]` F32 on `device`: `0` on and below the
/// diagonal, `masked` above it, to be added to the scores before the softmax.
///
/// `masked` is the caller's and not this function's: some towers are written against
/// `-inf`, which softmaxes to exactly zero, and others against `f32::MIN`, which is
/// finite and so leaves an entirely masked row a usable distribution rather than NaN.
/// The two disagree in the last bits, so neither stands in for the other. A caller
/// wanting `[1, 1, seq, seq]` reshapes - the result is contiguous.
pub fn causal_mask(seq: usize, masked: f32, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = masked;
        }
    }
    Tensor::from_vec(data, (seq, seq), device)
}

#[cfg(test)]
mod repeat_kv_tests {
    use super::*;
    use crate::tensor::Device;

    /// A banded attention answers what the whole-context form answers.
    ///
    /// The banded form never holds a score for every query against every key, which is
    /// what lets a long context be served at all, but it only earns that by giving the
    /// same answer. The running maximum and the rescale are where it would not: a band
    /// whose maximum exceeds the running one has to correct everything accumulated before
    /// it, and getting that wrong is a plausible-looking answer rather than an error.
    ///
    /// So it is compared against `softmax(QK^T + mask)V` computed in one piece, on a
    /// context wide enough to need several bands and values spread widely enough that the
    /// maximum really does move between them.
    #[test]
    fn a_banded_attention_answers_what_the_whole_context_answers() {
        let (kv_heads, groups, seq, past, hd) = (2usize, 3usize, 6usize, 11usize, 4usize);
        let rows = groups * seq;
        let kv = past + seq;
        let mk = |n: usize, shape: (usize, usize, usize, usize), spread: f32| {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * spread).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
        };
        // Spread wide enough that each band's maximum differs from the one before it,
        // which is the case the rescale exists for.
        let q = mk(kv_heads * rows * hd, (1, kv_heads, rows, hd), 6.0);
        let k = mk(kv_heads * kv * hd, (1, kv_heads, kv, hd), 6.0);
        let v = mk(kv_heads * kv * hd, (1, kv_heads, kv, hd), 2.0);

        let banded = banded_causal_attention(&q, &k, &v, seq, past).unwrap();

        // The whole-context form, on the same grouped layout.
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap())
            .unwrap()
            .reshape((1, kv_heads * groups, seq, kv))
            .unwrap();
        let mask: Vec<f32> = (0..seq)
            .flat_map(|p| {
                (0..kv).map(move |j| {
                    if j > past + p {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (seq, kv), &Device::Cpu).unwrap();
        let whole = softmax_last_dim(&scores.broadcast_add(&mask).unwrap())
            .unwrap()
            .reshape((1, kv_heads, rows, kv))
            .unwrap()
            .matmul(&v)
            .unwrap();

        let a: Vec<f32> = banded.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = whole.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-4, "element {i}: {x} vs {y}");
        }
    }

    /// A band never covers more than the context, and never so little that the launch
    /// costs more than the work.
    #[test]
    fn the_band_stays_between_its_bounds() {
        assert_eq!(score_band(64, 100), 100);
        assert!(score_band(64, 1_000_000) <= 8192);
        // A great many rows still get a band worth launching a matmul for.
        assert!(score_band(1_000_000, 1_000_000) >= 256);
    }

    /// Grouping the queries answers the same attention as copying the keys.
    ///
    /// A grouped-query prefill can be written either way: expand K and V so every query
    /// head has its own copy, or view Q so the heads sharing a kv head sit in one matrix
    /// and matmul against K as stored. The second allocates `n_rep` times less and is what
    /// the prefill path uses, but it is only correct because query head `h * n_rep + r`
    /// reads kv head `h` - a layout a reshape gets silently wrong if it ever changes.
    ///
    /// So the two are compared element by element on distinct values, with the softmax
    /// left out: it is over the key axis, which neither arrangement moves.
    #[test]
    fn grouping_the_queries_answers_what_copying_the_keys_does() {
        let (kv, rep, seq, kv_len, hd) = (3usize, 4usize, 5usize, 7usize, 6usize);
        let heads = kv * rep;
        let mk = |n: usize, shape: (usize, usize, usize, usize)| {
            let v: Vec<f32> = (0..n).map(|i| ((i * 37) % 23) as f32 - 11.0).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
        };
        let q = mk(heads * seq * hd, (1, heads, seq, hd));
        let k = mk(kv * kv_len * hd, (1, kv, kv_len, hd));
        let v = mk(kv * kv_len * hd, (1, kv, kv_len, hd));

        // Copying the keys: one K and V per query head.
        let ke = repeat_kv(k.clone(), rep).unwrap().contiguous().unwrap();
        let ve = repeat_kv(v.clone(), rep).unwrap().contiguous().unwrap();
        let copied = q
            .matmul(&ke.transpose(2, 3).unwrap())
            .unwrap()
            .matmul(&ve)
            .unwrap();

        // Grouping the queries: K and V as stored.
        let qg = q.reshape((1, kv, rep * seq, hd)).unwrap();
        let scores = qg.matmul(&k.transpose(2, 3).unwrap()).unwrap();
        // Back to one row per query head, as the mask and the softmax need it, and
        // grouped again for the value matmul - the round trip the prefill path makes.
        let scores = scores.reshape((1, heads, seq, kv_len)).unwrap();
        let grouped = scores
            .reshape((1, kv, rep * seq, kv_len))
            .unwrap()
            .matmul(&v)
            .unwrap()
            .reshape((1, heads, seq, hd))
            .unwrap();

        let a: Vec<f32> = copied.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = grouped.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-3, "element {i}: {x} vs {y}");
        }
    }

    /// The three ways this was written across the tree are one function.
    ///
    /// Each spelling puts the repeats somewhere different before reshaping - concatenated along
    /// the sequence, selected by index along the heads, broadcast along a new axis - and a
    /// wrong one still produces a tensor of the right shape. So they are compared element by
    /// element, on a tensor whose every element is distinct.
    #[test]
    fn the_three_ways_to_repeat_kv_heads_agree() {
        let (b, kv, seq, hd, rep) = (2usize, 3usize, 4usize, 5usize, 3usize);
        let v: Vec<f32> = (0..b * kv * seq * hd).map(|i| i as f32).collect();
        let xs = Tensor::from_vec(v, (b, kv, seq, hd), &Device::Cpu).unwrap();

        let broadcast = repeat_kv(xs.clone(), rep).unwrap();

        let concatenated = Tensor::cat(&vec![&xs; rep], 2)
            .unwrap()
            .reshape((b, kv * rep, seq, hd))
            .unwrap();

        let ids: Vec<u32> = (0..kv as u32)
            .flat_map(|h| std::iter::repeat_n(h, rep))
            .collect();
        let selected = xs
            .index_select(&Tensor::from_vec_u32(ids, vec![kv * rep]).unwrap(), 1)
            .unwrap();

        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&broadcast), flat(&concatenated), "concatenated");
        assert_eq!(flat(&broadcast), flat(&selected), "selected");

        // And the grouping itself: query head h*rep+r has to see kv head h, which is the claim
        // the shapes above cannot make on their own.
        let got = flat(&broadcast);
        let want = flat(&xs);
        for h in 0..kv {
            for r in 0..rep {
                let out = (h * rep + r) * seq * hd;
                let src = h * seq * hd;
                assert_eq!(
                    got[out..out + seq * hd],
                    want[src..src + seq * hd],
                    "kv head {h}, repeat {r}"
                );
            }
        }
    }
}
