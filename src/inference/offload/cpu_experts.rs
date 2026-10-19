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
pub(crate) fn batched(experts: &[Arc<Expert>], x: &[f32], limit: f32) -> Option<Vec<Vec<f32>>> {
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
pub(crate) fn multirow(
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

/// How long this node's cores take to compute one routed expert it does not hold resident: gate and
/// up in `gate_dt`, down in `down_dt`, over `dim` x `inter`, averaged over `iters`. This is the cut
/// planner's `miss_ms`, measured on the node rather than guessed. The weight bytes are a fixed
/// pattern - a GEMV's time is its size, not its values - and the activation drawn once.
///
/// `None` when a kernel cannot run these dims and formats (an unaligned dim, a quant the kernel does
/// not carry): a node whose miss cost cannot be measured yields no number rather than crashing the
/// caller, which is what lets this run on the model-load path. The caller reads `None` as unknown.
pub fn bench_miss_ms(
    dim: usize,
    inter: usize,
    gate_dt: crate::tensor::quantized::GgmlDType,
    down_dt: crate::tensor::quantized::GgmlDType,
    iters: usize,
) -> Option<f64> {
    use crate::tensor::quant_cpu::matmul_bytes_multi;
    let iters = iters.max(1);
    let gate_bytes = vec![0x11u8; inter * (dim / gate_dt.block_size()) * gate_dt.type_size()];
    let down_bytes = vec![0x11u8; dim * (inter / down_dt.block_size()) * down_dt.type_size()];
    let x = vec![0.05f32; dim];
    let h = vec![0.05f32; inter];
    let mut gout = vec![0f32; inter];
    let mut uout = vec![0f32; inter];
    let mut dout = vec![0f32; dim];
    let run = |gout: &mut [f32], uout: &mut [f32], dout: &mut [f32]| -> bool {
        matmul_bytes_multi(gate_dt, dim, inter, &[&gate_bytes], &[&x], &mut [gout]).is_ok()
            && matmul_bytes_multi(gate_dt, dim, inter, &[&gate_bytes], &[&x], &mut [uout]).is_ok()
            && matmul_bytes_multi(down_dt, inter, dim, &[&down_bytes], &[&h], &mut [dout]).is_ok()
    };
    // Warm the caches and the pool, then time. A kernel failure at either phase yields no number.
    if !run(&mut gout, &mut uout, &mut dout) {
        return None;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        if !run(&mut gout, &mut uout, &mut dout) {
            return None;
        }
    }
    Some(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

#[cfg(test)]
mod bench_tests {
    use super::bench_miss_ms;
    use crate::tensor::quantized::GgmlDType;

    /// The bench measures a positive per-miss cost over real, block-aligned expert dims.
    #[test]
    fn bench_measures_a_miss() {
        let ms = bench_miss_ms(5120, 2304, GgmlDType::Iq2Xxs, GgmlDType::Q2K, 8)
            .expect("aligned dims measure");
        assert!(ms > 0.0);
    }

    /// A dim the kernel cannot run (not a multiple of the quant's block) yields no number rather
    /// than a panic - the guard that lets this run on the model-load path.
    #[test]
    fn an_unrunnable_dim_yields_none() {
        assert!(bench_miss_ms(100, 2304, GgmlDType::Iq2Xxs, GgmlDType::Q2K, 4).is_none());
    }
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
