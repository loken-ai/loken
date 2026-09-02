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
                let out = ((0 * kv + h) * rep + r) * seq * hd;
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
