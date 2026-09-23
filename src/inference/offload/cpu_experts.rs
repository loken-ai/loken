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
            let (mut g, mut u) = (gate_o[e][k], up_o[e][k]);
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            h[e][k] = (g / (1.0 + (-g).exp())) * u;
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
            let (mut g, mut u) = (gate_o[p][k], up_o[p][k]);
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            h[p][k] = (g / (1.0 + (-g).exp())) * u;
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
