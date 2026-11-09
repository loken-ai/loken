//! Routed MoE experts computed on the CPU cores, model-neutral: one thread-pool region for a
//! whole batch of experts, the quant format read from the experts themselves. A model whose
//! resident experts run on the cards hands its misses here, to run beside the cards' block.

use super::experts::Expert;
use std::sync::Arc;

/// The decode's host experts in one thread-pool region: gate and up over the token's row, the
/// SwiGLU, then the down projection, each a flat (expert x output column) GEMV that spreads across
/// every core. `Expert::forward` per expert fans out on rayon instead, and nested under the block's
/// own threads the decode collapses onto two or three cores; this keeps the pool to one publisher
/// so all of them run. Bit-for-bit the per-expert products (the kernel is the same dot). `None` if
/// a weight is not in the format the kernels read, and the caller runs the experts the old way.
fn batched_one_format(experts: &[Arc<Expert>], x: &[f32], limit: f32) -> Option<Vec<Vec<f32>>> {
    use crate::inference::offload::projection::Projection;
    use crate::tensor::quant_cpu::matmul_bytes_multi;
    use crate::tensor::quantized::GgmlDType;
    if experts.is_empty() {
        return Some(Vec::new());
    }
    fn qbytes(p: &Projection, want: GgmlDType) -> Option<std::borrow::Cow<'_, [u8]>> {
        match p {
            Projection::Quant(q) if q.dtype() == want => q.data().ok(),
            _ => None,
        }
    }
    let dim = x.len();
    let inter = experts[0].w1.dims()[0];
    let ne = experts.len();
    // The quant format the kernels read is taken from the experts themselves: whatever the
    // model quantised its experts to, not a fixed recipe. A batch of one format only.
    let ex_dtype = |p: &Projection| match p {
        Projection::Quant(q) => Some(q.dtype()),
        _ => None,
    };
    let gate_dt = ex_dtype(&experts[0].w1)?;
    let up_dt = ex_dtype(&experts[0].w3)?;
    let down_dt = ex_dtype(&experts[0].w2)?;
    let (mut gate_c, mut up_c, mut down_c) = (
        Vec::with_capacity(ne),
        Vec::with_capacity(ne),
        Vec::with_capacity(ne),
    );
    for e in experts {
        gate_c.push(qbytes(&e.w1, gate_dt)?);
        up_c.push(qbytes(&e.w3, up_dt)?);
        down_c.push(qbytes(&e.w2, down_dt)?);
    }
    let lhs: Vec<&[f32]> = vec![x; ne];
    let gate_r: Vec<&[u8]> = gate_c.iter().map(|c| c.as_ref()).collect();
    let up_r: Vec<&[u8]> = up_c.iter().map(|c| c.as_ref()).collect();
    let mut gate_o = vec![vec![0f32; inter]; ne];
    let mut up_o = vec![vec![0f32; inter]; ne];
    {
        let mut o: Vec<&mut [f32]> = gate_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(gate_dt, dim, inter, &gate_r, &lhs, &mut o).ok()?;
    }
    {
        let mut o: Vec<&mut [f32]> = up_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(up_dt, dim, inter, &up_r, &lhs, &mut o).ok()?;
    }
    let mut h = vec![vec![0f32; inter]; ne];
    for e in 0..ne {
        for k in 0..inter {
            h[e][k] = super::experts::swiglu(gate_o[e][k], up_o[e][k], limit);
        }
    }
    let down_r: Vec<&[u8]> = down_c.iter().map(|c| c.as_ref()).collect();
    let h_r: Vec<&[f32]> = h.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![vec![0f32; dim]; ne];
    {
        let mut o: Vec<&mut [f32]> = out.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(down_dt, inter, dim, &down_r, &h_r, &mut o).ok()?;
    }
    Some(out)
}

/// A block of tokens through its routed experts in one thread-pool region per projection, each
/// expert over the rows the block routed to it. `row_batches[i]` is expert `i`'s inputs, row-major
/// (`nrows_i * dim` floats); the result is expert `i`'s outputs in the same order. Each loaded
/// weight column meets every row of its expert before the next column is loaded, so an expert's
/// weights are read once for all the rows that route to it, not once per row. The pool is
/// published to three times a layer for the whole block. Bit-for-bit the per-row products: each
/// output is the same dot over the same quantised row. `None` if a weight is not in the format the
/// kernels read.
#[allow(clippy::type_complexity)]
fn multirow_one_format(
    experts: &[Arc<Expert>],
    row_batches: &[Vec<f32>],
    dim: usize,
    limit: f32,
) -> Option<Vec<Vec<f32>>> {
    use crate::inference::offload::projection::Projection;
    use crate::tensor::quant_cpu::matmul_bytes_grouped;
    use crate::tensor::quantized::GgmlDType;
    let ne = experts.len();
    if ne == 0 {
        return Some(Vec::new());
    }
    fn qbytes(p: &Projection, want: GgmlDType) -> Option<std::borrow::Cow<'_, [u8]>> {
        match p {
            Projection::Quant(q) if q.dtype() == want => q.data().ok(),
            _ => None,
        }
    }
    let inter = experts[0].w1.dims()[0];
    // The quant format the kernels read is taken from the experts themselves: whatever the
    // model quantised its experts to, not a fixed recipe. A batch of one format only.
    let ex_dtype = |p: &Projection| match p {
        Projection::Quant(q) => Some(q.dtype()),
        _ => None,
    };
    let gate_dt = ex_dtype(&experts[0].w1)?;
    let up_dt = ex_dtype(&experts[0].w3)?;
    let down_dt = ex_dtype(&experts[0].w2)?;
    let (mut gate_c, mut up_c, mut down_c) = (
        Vec::with_capacity(ne),
        Vec::with_capacity(ne),
        Vec::with_capacity(ne),
    );
    for e in experts {
        gate_c.push(qbytes(&e.w1, gate_dt)?);
        up_c.push(qbytes(&e.w3, up_dt)?);
        down_c.push(qbytes(&e.w2, down_dt)?);
    }
    if row_batches.len() != ne || row_batches.iter().any(|b| b.len() % dim.max(1) != 0) {
        return None;
    }
    let nrows_of: Vec<usize> = row_batches.iter().map(|b| b.len() / dim.max(1)).collect();
    if nrows_of.iter().all(|&r| r == 0) {
        return Some(vec![Vec::new(); ne]);
    }
    let lhs: Vec<&[f32]> = row_batches.iter().map(|b| b.as_slice()).collect();
    let gate_r: Vec<&[u8]> = gate_c.iter().map(|c| c.as_ref()).collect();
    let up_r: Vec<&[u8]> = up_c.iter().map(|c| c.as_ref()).collect();
    let down_r: Vec<&[u8]> = down_c.iter().map(|c| c.as_ref()).collect();
    let mut gate_o: Vec<Vec<f32>> = nrows_of.iter().map(|&r| vec![0f32; r * inter]).collect();
    let mut up_o: Vec<Vec<f32>> = nrows_of.iter().map(|&r| vec![0f32; r * inter]).collect();
    {
        let mut o: Vec<&mut [f32]> = gate_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_grouped(gate_dt, dim, inter, &gate_r, &lhs, &mut o).ok()?;
    }
    {
        let mut o: Vec<&mut [f32]> = up_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_grouped(up_dt, dim, inter, &up_r, &lhs, &mut o).ok()?;
    }
    let h: Vec<Vec<f32>> = gate_o
        .iter()
        .zip(&up_o)
        .map(|(g, u)| {
            g.iter()
                .zip(u)
                .map(|(&g, &u)| super::experts::swiglu(g, u, limit))
                .collect()
        })
        .collect();
    let h_r: Vec<&[f32]> = h.iter().map(|v| v.as_slice()).collect();
    let mut out: Vec<Vec<f32>> = nrows_of.iter().map(|&r| vec![0f32; r * dim]).collect();
    {
        let mut o: Vec<&mut [f32]> = out.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_grouped(down_dt, inter, dim, &down_r, &h_r, &mut o).ok()?;
    }
    Some(out)
}

/// The formats of an expert's gate, up and down, `None` for one that is not quantised.
fn formats(e: &Expert) -> Option<[crate::tensor::quantized::GgmlDType; 3]> {
    use crate::inference::offload::projection::Projection;
    let dt = |p: &Projection| match p {
        Projection::Quant(q) => Some(q.dtype()),
        _ => None,
    };
    Some([dt(&e.w1)?, dt(&e.w3)?, dt(&e.w2)?])
}

/// `experts` split into runs of one format each, in the order first seen: the indices of each.
fn by_format(experts: &[Arc<Expert>]) -> Option<Vec<Vec<usize>>> {
    let mut groups: Vec<([crate::tensor::quantized::GgmlDType; 3], Vec<usize>)> = Vec::new();
    for (i, e) in experts.iter().enumerate() {
        let f = formats(e)?;
        match groups.iter_mut().find(|(g, _)| *g == f) {
            Some((_, ids)) => ids.push(i),
            None => groups.push((f, vec![i])),
        }
    }
    Some(groups.into_iter().map(|(_, ids)| ids).collect())
}

/// Run `one` over each format's experts and put the outputs back in the experts' order: experts
/// read from files of different formats run as one batch per format.
fn per_format(
    experts: &[Arc<Expert>],
    one: &dyn Fn(&[Arc<Expert>], &[usize]) -> Option<Vec<Vec<f32>>>,
) -> Option<Vec<Vec<f32>>> {
    let groups = by_format(experts)?;
    if groups.len() <= 1 {
        let all: Vec<usize> = (0..experts.len()).collect();
        return one(experts, &all);
    }
    let mut out = vec![Vec::new(); experts.len()];
    for ids in groups {
        let part: Vec<Arc<Expert>> = ids.iter().map(|&i| experts[i].clone()).collect();
        for (i, o) in ids.iter().zip(one(&part, &ids)?) {
            out[*i] = o;
        }
    }
    Some(out)
}

/// The decode's host experts, as `batched_one_format` computes them, one batch per format.
pub(crate) fn batched(experts: &[Arc<Expert>], x: &[f32], limit: f32) -> Option<Vec<Vec<f32>>> {
    per_format(experts, &|part, _| batched_one_format(part, x, limit))
}

/// A prompt's host experts, as `multirow_one_format` computes them, one batch per format.
pub(crate) fn multirow(
    experts: &[Arc<Expert>],
    row_batches: &[Vec<f32>],
    dim: usize,
    limit: f32,
) -> Option<Vec<Vec<f32>>> {
    if by_format(experts)?.len() <= 1 {
        return multirow_one_format(experts, row_batches, dim, limit);
    }
    per_format(experts, &|part, ids| {
        let rows: Vec<Vec<f32>> = ids
            .iter()
            .map(|&i| row_batches.get(i).cloned())
            .collect::<Option<_>>()?;
        multirow_one_format(part, &rows, dim, limit)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::offload::projection::Projection;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    use crate::tensor::{Device, Tensor};

    /// The host's three ways through quantised experts - a decode's block, a verify's rows, an
    /// expert over a prompt's rows - give the same bits: the card's kernels are held to one product,
    /// and it is the same whichever of these the host took.
    #[test]
    fn the_host_paths_through_an_expert_agree_bit_for_bit() {
        let (dim, inter, limit, n) = (512usize, 256usize, 7.0f32, 5usize);
        let draw = |seed: u32, len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    (((i as u32)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(11)
                        >> 8) as f32
                        / (1 << 24) as f32
                        - 0.5)
                        * scale
                })
                .collect()
        };
        let quant = |w: Vec<f32>, rows: usize, cols: usize, dtype: GgmlDType| {
            let t = Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap();
            Projection::Quant(Arc::new(QTensor::quantize(&t, dtype).unwrap()))
        };
        let experts: Vec<Arc<Expert>> = (0..2u32)
            .map(|j| {
                Arc::new(Expert {
                    w1: quant(
                        draw(10 * j + 1, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w3: quant(
                        draw(10 * j + 2, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w2: quant(
                        draw(10 * j + 3, dim * inter, 0.08),
                        dim,
                        inter,
                        GgmlDType::Q2K,
                    ),
                })
            })
            .collect();
        let x = draw(77, n * dim, 6.0);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let rows: Vec<Vec<f32>> = experts.iter().map(|_| x.clone()).collect();
        let multi = multirow(&experts, &rows, dim, limit).unwrap();
        for (j, e) in experts.iter().enumerate() {
            let xe = Tensor::from_vec(x.clone(), (n, dim), &Device::Cpu).unwrap();
            let whole = crate::inference::offload::on_host(|| e.forward(&xe, limit))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(bits(&whole), bits(&multi[j]), "rows, expert {j}");
            for t in 0..n {
                let one = batched(&experts, &x[t * dim..(t + 1) * dim], limit).unwrap();
                assert_eq!(
                    bits(&one[j]),
                    bits(&whole[t * dim..(t + 1) * dim]),
                    "block, expert {j} row {t}"
                );
            }
        }
    }

    /// A synthetic MoE's prompt, its experts read once for all the tokens that route to them,
    /// gives each token the bits it gets when its experts run for it alone: every pair's product,
    /// and every token's weighted sum taken in the router's order.
    #[test]
    fn a_prompt_batched_by_expert_matches_its_tokens_one_by_one() {
        let (dim, inter, limit) = (512usize, 256usize, 7.0f32);
        let (n_experts, k, tokens) = (5usize, 2usize, 13usize);
        let draw = |seed: u32, len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    (((i as u32)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(13)
                        >> 8) as f32
                        / (1 << 24) as f32
                        - 0.5)
                        * scale
                })
                .collect()
        };
        let quant = |w: Vec<f32>, rows: usize, cols: usize, dtype: GgmlDType| {
            let t = Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap();
            Projection::Quant(Arc::new(QTensor::quantize(&t, dtype).unwrap()))
        };
        let experts: Vec<Arc<Expert>> = (0..n_experts as u32)
            .map(|j| {
                Arc::new(Expert {
                    w1: quant(
                        draw(7 * j + 1, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Q2K,
                    ),
                    w3: quant(
                        draw(7 * j + 2, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Q2K,
                    ),
                    w2: quant(
                        draw(7 * j + 3, dim * inter, 0.08),
                        dim,
                        inter,
                        GgmlDType::Q4K,
                    ),
                })
            })
            .collect();
        let x = draw(91, tokens * dim, 5.0);
        // Token t routes to experts (t, t + 1 + t / 3) mod n, with weights of its own: every expert
        // serves several tokens, some of them adjacent.
        let routing: Vec<Vec<(usize, f32)>> = (0..tokens)
            .map(|t| {
                let a = t % n_experts;
                let b = (t + 1 + t / 3) % n_experts;
                let b = if b == a { (a + 1) % n_experts } else { b };
                vec![(a, 0.6 + t as f32 * 0.01), (b, 0.4 - t as f32 * 0.01)]
            })
            .collect();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();

        // One by one: each token's experts as a decode runs them.
        let mut one_by_one = vec![0f32; tokens * dim];
        let mut pair_rows: Vec<Vec<Vec<f32>>> = Vec::with_capacity(tokens);
        for (t, route) in routing.iter().enumerate() {
            let mine: Vec<Arc<Expert>> = route.iter().map(|&(e, _)| experts[e].clone()).collect();
            let outs = batched(&mine, &x[t * dim..(t + 1) * dim], limit).unwrap();
            for (out, &(_, w)) in outs.iter().zip(route) {
                for (a, b) in one_by_one[t * dim..(t + 1) * dim].iter_mut().zip(out) {
                    *a += w * b;
                }
            }
            pair_rows.push(outs);
        }

        // Batched: every expert once over all the rows that route to it.
        let mut assign: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
        for (t, route) in routing.iter().enumerate() {
            for (slot, &(e, _)) in route.iter().enumerate() {
                assign[e].push((t, slot));
            }
        }
        let batches: Vec<Vec<f32>> = assign
            .iter()
            .map(|rows| {
                rows.iter()
                    .flat_map(|&(t, _)| x[t * dim..(t + 1) * dim].iter().copied())
                    .collect()
            })
            .collect();
        assert!(
            assign.iter().all(|a| a.len() >= 2),
            "every expert serves several tokens"
        );
        let grouped = multirow(&experts, &batches, dim, limit).unwrap();
        let mut at = vec![vec![(0usize, 0usize); k]; tokens];
        for (e, rows) in assign.iter().enumerate() {
            for (r, &(t, slot)) in rows.iter().enumerate() {
                assert_eq!(
                    bits(&grouped[e][r * dim..(r + 1) * dim]),
                    bits(&pair_rows[t][slot]),
                    "expert {e} token {t}"
                );
                at[t][slot] = (e, r);
            }
        }
        let mut together = vec![0f32; tokens * dim];
        for (t, route) in routing.iter().enumerate() {
            for (slot, &(_, w)) in route.iter().enumerate() {
                let (e, r) = at[t][slot];
                for (a, b) in together[t * dim..(t + 1) * dim]
                    .iter_mut()
                    .zip(&grouped[e][r * dim..(r + 1) * dim])
                {
                    *a += w * b;
                }
            }
        }
        assert_eq!(bits(&together), bits(&one_by_one));
    }
}
