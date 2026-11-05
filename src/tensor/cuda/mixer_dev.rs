//! Device-resident steps of a latent attention layer over 1 shared key head.
//!
//! Every function takes and gives device buffers, so a layer chains them without a crossing.
//! The kernels are the ones the decode stream launches: the rotation, the gather, the sink softmax.

use super::*;
use crate::tensor::gpu_runtime::driver::{CudaView, PushKernelArg};

const THREADS: usize = 256;

/// One thread per element of `total`.
fn grid_for(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(THREADS).max(1) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_err(name: &str, e: crate::tensor::gpu_runtime::driver::DriverError) -> Error {
    Error(format!("{name} launch: {e}"))
}

/// `x` `[rows, inp]` through the weight `w` `[out, inp]`: `[rows, out]`.
pub fn linear_dev(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    (rows, inp, out): (usize, usize, usize),
) -> Result<CudaSlice<f32>> {
    if x.len() < rows * inp || w.len() < out * inp {
        return Err(Error(format!(
            "linear: {} inputs and {} weights for [{rows}, {inp}] x [{out}, {inp}]",
            x.len(),
            w.len()
        )));
    }
    matmul_nt_f32(dev, x, w, 1, rows, inp, out)
}

/// Rotates the last `rd` lanes of each of the `h` heads of the `n` rows of `x` `[n, h * hd]` in
/// place. Row `t` turns by `cos[t]` and `sin[t]`, each `[rd / 2]` wide.
pub fn rope_tokens(
    dev: &CudaDevice,
    x: &mut CudaSlice<f32>,
    (n, h, hd, rd): (usize, usize, usize, usize),
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
) -> Result<()> {
    let half = rd / 2;
    if x.len() < n * h * hd || rd > hd || cos.len() < n * half || sin.len() < n * half {
        return Err(Error(format!(
            "rope: {} values for {n} rows of {h} heads of {hd}, {rd} rotated, tables of {} and {}",
            x.len(),
            cos.len(),
            sin.len()
        )));
    }
    if half == 0 || n == 0 {
        return Ok(());
    }
    let func = dev.quantized_fn("rope_tail_f32")?;
    let (total, h_i, hd_i, rd_i) = ((n * h * half) as i64, h as i64, hd as i64, rd as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(x);
    b.arg(cos);
    b.arg(sin);
    b.arg(&total);
    b.arg(&h_i);
    b.arg(&hd_i);
    b.arg(&rd_i);
    unsafe { b.launch(grid_for(n * h * half)) }
        .map(|_| ())
        .map_err(|e| launch_err("rope_tail_f32", e))
}

/// Columns `off..off + w` of each row of `src` `[n, src_w]`: `[n, w]`.
pub fn take_cols(
    dev: &CudaDevice,
    src: &CudaSlice<f32>,
    (n, src_w): (usize, usize),
    (off, w): (usize, usize),
) -> Result<CudaSlice<f32>> {
    if off + w > src_w || src.len() < n * src_w {
        return Err(Error(format!(
            "take columns: {off}+{w} of {src_w} over {} values",
            src.len()
        )));
    }
    let out = alloc_f32(dev, "take columns", n * w)?;
    let func = dev.quantized_fn("take_cols_f32")?;
    let (total, w_i, sw, off_i) = ((n * w) as i64, w as i64, src_w as i64, off as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(src);
    b.arg(&out);
    b.arg(&total);
    b.arg(&w_i);
    b.arg(&sw);
    b.arg(&off_i);
    unsafe { b.launch(grid_for(n * w)) }.map_err(|e| launch_err("take_cols_f32", e))?;
    Ok(out)
}

/// Sets columns `off..off + w` of each row of `dst` `[n, dst_w]` from `src` `[n, w]`.
pub fn put_cols(
    dev: &CudaDevice,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    (n, dst_w): (usize, usize),
    (off, w): (usize, usize),
) -> Result<()> {
    if off + w > dst_w || dst.len() < n * dst_w || src.len() < n * w {
        return Err(Error(format!(
            "put columns: {off}+{w} of {dst_w} from {} values into {}",
            src.len(),
            dst.len()
        )));
    }
    let func = dev.quantized_fn("put_cols_f32")?;
    let (total, w_i, dw, off_i) = ((n * w) as i64, w as i64, dst_w as i64, off as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(src);
    b.arg(&*dst);
    b.arg(&total);
    b.arg(&w_i);
    b.arg(&dw);
    b.arg(&off_i);
    unsafe { b.launch(grid_for(n * w)) }
        .map_err(|e| launch_err("put_cols_f32", e))
        .map(|_| ())
}

/// The `parts` laid end to end in one device buffer.
pub fn concat_f32(dev: &CudaDevice, parts: &[CudaView<'_, f32>]) -> Result<CudaSlice<f32>> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = alloc_f32(dev, "concatenated rows", total)?;
    let mut at = 0;
    for p in parts {
        if !p.is_empty() {
            dev.stream()
                .memcpy_dtod(p, &mut out.slice_mut(at..at + p.len()))
                .map_err(|e| Error(format!("concatenated rows copy: {e}")))?;
        }
        at += p.len();
    }
    Ok(out)
}

/// `n` rows of attention with a per-head sink over the rows of one buffer.
///
/// `q` is `[n, h, d]`, `kv` is `[rows, d]` and serves as key and value of the one shared head.
/// `idxs` is `[n, topk]` row numbers into `kv`, negative for an empty slot.
/// Queries go through in chunks of [`ATTENTION_QUERY_CHUNK`].
/// Returns `[n, h, d]`.
#[allow(clippy::too_many_arguments)]
pub fn slots_attention(
    dev: &CudaDevice,
    q: &CudaSlice<f32>,
    kv: &CudaSlice<f32>,
    sink: &CudaSlice<f32>,
    idxs: &CudaSlice<i32>,
    (n, h, d, topk): (usize, usize, usize, usize),
    scale: f32,
) -> Result<CudaSlice<f32>> {
    if q.len() < n * h * d || sink.len() < h || idxs.len() < n * topk || d == 0 {
        return Err(Error(format!(
            "slots attention: {} queries, {} sinks, {} slots for n {n} h {h} d {d} topk {topk}",
            q.len(),
            sink.len(),
            idxs.len()
        )));
    }
    let stream = dev.stream();
    let mut out = alloc_f32(dev, "slots attention", n * h * d)?;
    let chunk = ATTENTION_QUERY_CHUNK.clamp(1, n.max(1));
    let (gather, softmax) = (
        dev.quantized_fn("gather_slots_f32")?,
        dev.quantized_fn("sink_softmax_f32")?,
    );
    for start in (0..n).step_by(chunk) {
        let count = chunk.min(n - start);
        let qc = copy_f32(
            dev,
            "slots attention queries",
            &q.slice(start * h * d..(start + count) * h * d),
        )?;
        let keys = alloc_f32(dev, "slots attention keys", count * topk * d)?;
        let (start_i, count_i, topk_i, d_i, h_i) =
            (start as i64, count as i64, topk as i64, d as i64, h as i64);
        let mut g = stream.launch_builder(&gather);
        g.arg(kv);
        g.arg(idxs);
        g.arg(&keys);
        g.arg(&start_i);
        g.arg(&count_i);
        g.arg(&topk_i);
        g.arg(&d_i);
        unsafe { g.launch(grid_for(count * topk)) }
            .map_err(|e| launch_err("gather_slots_f32", e))?;
        // [count, h, d] against [count, topk, d]: [count, h, topk].
        let logits = matmul_nt_f32(dev, &qc, &keys, count, h, d, topk)?;
        let mut sm = stream.launch_builder(&softmax);
        sm.arg(&logits);
        sm.arg(sink);
        sm.arg(idxs);
        sm.arg(&start_i);
        sm.arg(&count_i);
        sm.arg(&h_i);
        sm.arg(&topk_i);
        sm.arg(&scale);
        unsafe { sm.launch(warp_cfg(count * h)) }.map_err(|e| launch_err("sink_softmax_f32", e))?;
        // [count, h, topk] against [count, topk, d]: [count, h, d].
        let part = matmul_f32(dev, &logits, &keys, count, h, topk, d)?;
        stream
            .memcpy_dtod(
                &part.slice(0..count * h * d),
                &mut out.slice_mut(start * h * d..(start + count) * h * d),
            )
            .map_err(|e| Error(format!("slots attention copy: {e}")))?;
    }
    Ok(out)
}
