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

/// A block of tokens through its routed experts in one thread-pool region, each expert over the
/// rows the block routed to it. `row_batches[i]` is expert `i`'s inputs, row-major (`nrows_i * dim`
/// floats); the result is expert `i`'s outputs in the same order. A verify runs this once for the
/// whole block, so the pool is published to three times a layer for the block instead of three
/// times a layer per token - the per-layer dispatch is what a decode pays over and over, and the
/// batch pays it once. `None` if a weight is not in the format the kernels read.
#[allow(clippy::type_complexity)]
pub(crate) fn multirow(
    experts: &[Arc<Expert>],
    row_batches: &[Vec<f32>],
    dim: usize,
    limit: f32,
) -> Option<Vec<Vec<f32>>> {
    use crate::inference::offload::projection::Projection;
    use crate::tensor::quant_cpu::matmul_bytes_multi;
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
    // Every (expert, row) is a pair the pool runs as one GEMV: the same weight bytes for each of
    // an expert's rows, that row's activation as the input.
    let (mut gate_r, mut up_r, mut down_r) = (Vec::new(), Vec::new(), Vec::new());
    let mut lhs: Vec<&[f32]> = Vec::new();
    let mut nrows_of = vec![0usize; ne];
    for i in 0..ne {
        let nrows = row_batches[i].len() / dim;
        nrows_of[i] = nrows;
        for r in 0..nrows {
            gate_r.push(gate_c[i].as_ref());
            up_r.push(up_c[i].as_ref());
            down_r.push(down_c[i].as_ref());
            lhs.push(&row_batches[i][r * dim..(r + 1) * dim]);
        }
    }
    let npairs = lhs.len();
    if npairs == 0 {
        return Some(vec![Vec::new(); ne]);
    }
    let mut gate_o = vec![vec![0f32; inter]; npairs];
    let mut up_o = vec![vec![0f32; inter]; npairs];
    {
        let mut o: Vec<&mut [f32]> = gate_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(gate_dt, dim, inter, &gate_r, &lhs, &mut o).ok()?;
    }
    {
        let mut o: Vec<&mut [f32]> = up_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(up_dt, dim, inter, &up_r, &lhs, &mut o).ok()?;
    }
    let mut h = vec![vec![0f32; inter]; npairs];
    for p in 0..npairs {
        for k in 0..inter {
            h[p][k] = super::experts::swiglu(gate_o[p][k], up_o[p][k], limit);
        }
    }
    let h_r: Vec<&[f32]> = h.iter().map(|v| v.as_slice()).collect();
    let mut out_pairs = vec![vec![0f32; dim]; npairs];
    {
        let mut o: Vec<&mut [f32]> = out_pairs.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_bytes_multi(down_dt, inter, dim, &down_r, &h_r, &mut o).ok()?;
    }
    let mut out: Vec<Vec<f32>> = (0..ne)
        .map(|i| Vec::with_capacity(nrows_of[i] * dim))
        .collect();
    let mut p = 0usize;
    for i in 0..ne {
        for _ in 0..nrows_of[i] {
            out[i].extend_from_slice(&out_pairs[p]);
            p += 1;
        }
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
}
