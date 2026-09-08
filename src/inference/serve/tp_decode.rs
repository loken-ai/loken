//! Tensor-parallel (TP=2) decode primitives - the reusable core behind the
//! tensor-parallel decode lever (deepseek-r1 +42% vs vLLM).
//!
//! The 2-GPU pipeline runs one GPU per decode token (the other idles). TP runs BOTH GPUs on
//! every layer: q/k/v and gate/up are **column-parallel** (each rank owns a head/intermediate
//! subset), o_proj and down are **row-parallel** (each rank's partial output is summed by a
//! cross-GPU all-reduce). A column->row pair needs no concat - the column output stays split
//! and feeds the row input directly; only the row output is all-reduced back to replicated.
//!
//! Every primitive here is validated cross-GPU (bins `tp_{weight_slice,ffn,attn}_test`):
//! column slice bit-exact, row slice 1.1e-5, TP FFN 3.9e-3, TP attention 2.4e-4 vs single-GPU.
//! All-reduce keystone measured at ~53µs/reduce over the PHB/PCIe pair (`tp_allreduce_bench`).

use crate::tensor::quantized::{QMatMul, QTensor};
use crate::tensor::{Device, Tensor};
use anyhow::Result;
use std::sync::Arc;

use crate::tensor::quant_cpu::QK_K;

/// Column-parallel slice: output rows `[s, e)` of a `[out, in]` quantized weight. Rows are
/// contiguous block-runs, so this is a clean byte-range slice + `qtensor_from_ggml`.
pub fn slice_qtensor_rows(qt: &QTensor, s: usize, e: usize, dev: &Device) -> Result<Arc<QTensor>> {
    let dims = qt.shape().dims().to_vec();
    let (out, inn) = (dims[0], dims[1]);
    debug_assert!(e <= out && s < e);
    let bytes = qt.data()?;
    let row_bytes = bytes.len() / out;
    Ok(Arc::new(QTensor::from_ggml_bytes(
        qt.dtype(),
        &bytes[s * row_bytes..e * row_bytes],
        vec![e - s, inn],
        dev,
    )?))
}

/// Row-parallel slice: input columns `[c0, c1)` (multiples of QK_K) of a `[out, in]` weight.
/// Per output row the in-dimension blocks are contiguous, so gather each row's block sub-range.
pub fn slice_qtensor_cols(
    qt: &QTensor,
    c0: usize,
    c1: usize,
    dev: &Device,
) -> Result<Arc<QTensor>> {
    let dims = qt.shape().dims().to_vec();
    let (out, inn) = (dims[0], dims[1]);
    debug_assert!(c0.is_multiple_of(QK_K) && c1.is_multiple_of(QK_K) && c1 <= inn && c0 < c1);
    let bytes = qt.data()?;
    let row_bytes = bytes.len() / out;
    let block_bytes = row_bytes / (inn / QK_K);
    let (b0, b1) = (c0 / QK_K, c1 / QK_K);
    let mut out_bytes = Vec::with_capacity(out * (b1 - b0) * block_bytes);
    for r in 0..out {
        let rs = r * row_bytes;
        out_bytes.extend_from_slice(&bytes[rs + b0 * block_bytes..rs + b1 * block_bytes]);
    }
    Ok(Arc::new(QTensor::from_ggml_bytes(
        qt.dtype(),
        &out_bytes,
        vec![out, c1 - c0],
        dev,
    )?))
}

/// Column-parallel matmul: a `[out,in]` weight split by output rows across the 2 GPUs.
/// `forward` returns each rank's partial output on its own device (stays split for the
/// following row-parallel op).
pub struct TpColumn {
    pub g0: QMatMul, // rows [0, out/2)  on device 0
    pub g1: QMatMul, // rows [out/2, out) on device 1
    pub split: usize,
    // Optional per-rank bias slice (qwen2 q/k/v projections carry a bias).
    // Column-parallel: bias rows split the same way as the weight rows.
    pub b0: Option<Tensor>, // bias[0, split)   on d0
    pub b1: Option<Tensor>, // bias[split, out) on d1
}
impl TpColumn {
    pub fn from_qtensor(qt: &QTensor, d0: &Device, d1: &Device) -> Result<Self> {
        Self::from_qtensor_bias(qt, None, d0, d1)
    }
    /// Column-parallel build with an optional dequantized bias (F32 `[out]`),
    /// sliced per rank to match the weight rows.
    pub fn from_qtensor_bias(
        qt: &QTensor,
        bias: Option<&Tensor>,
        d0: &Device,
        d1: &Device,
    ) -> Result<Self> {
        let out = qt.shape().dims()[0];
        let split = out / 2;
        let (b0, b1) = match bias {
            Some(b) => (
                Some(b.narrow(0, 0, split)?.to_device(d0)?),
                Some(b.narrow(0, split, out - split)?.to_device(d1)?),
            ),
            None => (None, None),
        };
        Ok(Self {
            g0: QMatMul::from_arc(slice_qtensor_rows(qt, 0, split, d0)?)?,
            g1: QMatMul::from_arc(slice_qtensor_rows(qt, split, out, d1)?)?,
            split,
            b0,
            b1,
        })
    }
    /// x0 on d0, x1 on d1 (replicated input) -> (out0 on d0, out1 on d1).
    pub fn forward(&self, x0: &Tensor, x1: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut o0 = self.g0.forward(x0)?;
        if let Some(b) = &self.b0 {
            o0 = o0.broadcast_add(b)?;
        }
        let mut o1 = self.g1.forward(x1)?;
        if let Some(b) = &self.b1 {
            o1 = o1.broadcast_add(b)?;
        }
        Ok((o0, o1))
    }
}

/// Fused column-parallel projection: several `[out_i, in]` weights sharing the SAME input
/// (q/k/v, or gate/up) byte-concatenated row-wise into ONE matmul per rank. Replaces the
/// N matmul + N bias-add launches of the unfused path with 1 + 1 - pure launch reduction
/// (decode is launch-bound), and one larger mvq has better occupancy than N small ones.
/// All sub-weights must share dtype + input dim (q/k/v and gate/up do in qwen2).
/// GPU0's share of a `dim`-sized axis for an ASYMMETRIC TP split, rounded DOWN to a QK_K(256)
/// boundary so a row-parallel partner's column slice stays block-aligned. `num/den` is GPU0's
/// target fraction - the faster GPU gets more rows so both finish their matmuls together and
/// neither idles at the all-reduce barrier (nsys: 50/50 made the fast GPU0 spin 656ms/60-tok
/// waiting for the ~2x slower GPU1). Falls back to `dim/2` if rounding would zero a side.
pub fn asym_split(dim: usize, num: usize, den: usize) -> usize {
    let s = (dim * num / den / QK_K) * QK_K;
    if s == 0 || s >= dim {
        (dim / 2 / QK_K) * QK_K
    } else {
        s
    }
}

pub struct TpColumnFused {
    pub g0: QMatMul,
    pub g1: QMatMul,
    pub b0: Option<Tensor>,
    pub b1: Option<Tensor>,
    /// per-rank output width of each sub-projection (in concat order). Differ when the split
    /// is asymmetric (GPU0 gets more rows than GPU1).
    pub widths0: Vec<usize>,
    pub widths1: Vec<usize>,
}
impl TpColumnFused {
    /// Symmetric (50/50) build - q/k where the o-proj alignment forbids an asymmetric cut.
    pub fn from_qtensors(
        items: &[(&QTensor, Option<&Tensor>)],
        d0: &Device,
        d1: &Device,
    ) -> Result<Self> {
        Self::from_qtensors_frac(items, 1, 2, d0, d1)
    }
    /// Asymmetric build: each weight is column-parallel row-split at `asym_split(out,num,den)`
    /// (GPU0's share), the per-rank row-slices byte-concatenated into one quantized weight/device.
    pub fn from_qtensors_frac(
        items: &[(&QTensor, Option<&Tensor>)],
        num: usize,
        den: usize,
        d0: &Device,
        d1: &Device,
    ) -> Result<Self> {
        let dt = items[0].0.dtype();
        let inn = items[0].0.shape().dims()[1];
        let (mut bytes0, mut bytes1) = (Vec::<u8>::new(), Vec::<u8>::new());
        let (mut rows0, mut rows1) = (0usize, 0usize);
        let (mut widths0, mut widths1) = (Vec::new(), Vec::new());
        let (mut b0p, mut b1p) = (Vec::<Tensor>::new(), Vec::<Tensor>::new());
        let has_bias = items[0].1.is_some();
        for (qt, bias) in items {
            let dims = qt.shape().dims();
            let (out, in_i) = (dims[0], dims[1]);
            if qt.dtype() != dt || in_i != inn {
                anyhow::bail!("TpColumnFused: mismatched dtype/in-dim");
            }
            if bias.is_some() != has_bias {
                anyhow::bail!("TpColumnFused: mixed bias / no-bias");
            }
            let split = asym_split(out, num, den);
            let data = qt.data()?;
            let row_bytes = data.len() / out;
            bytes0.extend_from_slice(&data[0..split * row_bytes]);
            bytes1.extend_from_slice(&data[split * row_bytes..out * row_bytes]);
            rows0 += split;
            rows1 += out - split;
            widths0.push(split);
            widths1.push(out - split);
            if let Some(b) = bias {
                b0p.push(b.narrow(0, 0, split)?);
                b1p.push(b.narrow(0, split, out - split)?);
            }
        }
        let g0 = QMatMul::from_arc(Arc::new(QTensor::from_ggml_bytes(
            dt,
            &bytes0,
            vec![rows0, inn],
            d0,
        )?))?;
        let g1 = QMatMul::from_arc(Arc::new(QTensor::from_ggml_bytes(
            dt,
            &bytes1,
            vec![rows1, inn],
            d1,
        )?))?;
        let (b0, b1) = if has_bias {
            (
                Some(Tensor::cat(&b0p.iter().collect::<Vec<_>>(), 0)?.to_device(d0)?),
                Some(Tensor::cat(&b1p.iter().collect::<Vec<_>>(), 0)?.to_device(d1)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            g0,
            g1,
            b0,
            b1,
            widths0,
            widths1,
        })
    }
    /// One fused matmul per rank (+bias) -> (out0 on d0, out1 on d1).
    pub fn forward(&self, x0: &Tensor, x1: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut o0 = self.g0.forward(x0)?;
        if let Some(b) = &self.b0 {
            o0 = o0.broadcast_add(b)?;
        }
        let mut o1 = self.g1.forward(x1)?;
        if let Some(b) = &self.b1 {
            o1 = o1.broadcast_add(b)?;
        }
        Ok((o0, o1))
    }
    /// Split rank `r`'s fused output `[seq, sum(widths_r)]` into its contiguous sub-projections.
    pub fn split(&self, o: &Tensor, rank: usize) -> Result<Vec<Tensor>> {
        let widths = if rank == 0 {
            &self.widths0
        } else {
            &self.widths1
        };
        let mut parts = Vec::with_capacity(widths.len());
        let mut off = 0;
        for &w in widths {
            parts.push(o.narrow(1, off, w)?.contiguous()?);
            off += w;
        }
        Ok(parts)
    }
}

/// Row-parallel matmul: a `[out,in]` weight split by input columns across the 2 GPUs.
/// `forward` takes each rank's split input and returns its partial output (to be all-reduced).
pub struct TpRow {
    pub g0: QMatMul, // cols [0, in/2)  on device 0
    pub g1: QMatMul, // cols [in/2, in) on device 1
}
impl TpRow {
    pub fn from_qtensor(qt: &QTensor, d0: &Device, d1: &Device) -> Result<Self> {
        Self::from_qtensor_frac(qt, 1, 2, d0, d1)
    }
    /// Asymmetric col-split at `asym_split(in,num,den)` - must MATCH the row-split of the
    /// column-parallel weight feeding this one (e.g. `down` cols ↔ `gate/up` rows).
    pub fn from_qtensor_frac(
        qt: &QTensor,
        num: usize,
        den: usize,
        d0: &Device,
        d1: &Device,
    ) -> Result<Self> {
        let inn = qt.shape().dims()[1];
        let split = asym_split(inn, num, den);
        Ok(Self {
            g0: QMatMul::from_arc(slice_qtensor_cols(qt, 0, split, d0)?)?,
            g1: QMatMul::from_arc(slice_qtensor_cols(qt, split, inn, d1)?)?,
        })
    }
    /// h0 on d0, h1 on d1 (split input) -> partial outputs (p0 on d0, p1 on d1).
    pub fn forward(&self, h0: &Tensor, h1: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.g0.forward(h0)?, self.g1.forward(h1)?))
    }
}

/// Cross-GPU all-reduce(sum) for TP=2: p0 on d0, p1 on d1 -> the sum, replicated on both.
/// (Naive P2P via facade `to_device` + add. NCCL is the faster path once `libnccl` is
/// installed - see the plan.)
///
/// SYNCHRONIZATION IS LOAD-BEARING. the reference cross-CUDA `to_device` enqueues a peer copy on
/// the destination stream without waiting for the SOURCE device's compute stream to finish
/// producing the tensor. Without the syncs below, the copy races the producing matmul and
/// reads partial/stale data - making the whole forward NON-DETERMINISTIC (the same layer
/// gives different values across runs) and incoherent. Each device must be synchronized
/// before its tensor is read cross-device, and after each peer copy before the result is used.
pub fn all_reduce_sum(
    p0: &Tensor,
    p1: &Tensor,
    d0: &Device,
    d1: &Device,
) -> Result<(Tensor, Tensor)> {
    // Correct (deterministic) but serializing fallback: the reference cross-CUDA peer copy does NOT
    // reliably order against the matmul that produced the source tensor, so we host-synchronize
    // the source device before each peer read. This is a full barrier (no GPU overlap). The fast
    // path is `all_reduce_sum_nccl` below.
    d1.synchronize()?;
    let p1_d0 = p1.to_device(d0)?;
    let s0 = (p0 + p1_d0)?;
    d0.synchronize()?;
    let s1 = s0.to_device(d1)?;
    Ok((s0, s1))
}

/// NCCL all-reduce(sum) for TP=2 - the OVERLAPPING path. Runs `ncclAllReduce` on each rank's
/// own stream (so the collective is stream-ordered with the matmuls and the two GPUs
/// run in parallel - no `device.synchronize()` barrier). `comms` are the 2 per-device comms
/// from `make_comms`. Returns the summed tensor on each device (p0+p1 replicated).
#[cfg(feature = "cuda")]
pub fn all_reduce_sum_nccl(
    p0: &Tensor,
    p1: &Tensor,
    comms: &[cudarc::nccl::safe::Comm],
) -> Result<(Tensor, Tensor)> {
    use cudarc::nccl::safe::{group_end, group_start, ReduceOp};
    // Clone each partial's f32 device buffer into an owned, mutable CudaSlice (outside the
    // NCCL group), so we can all-reduce it in place.
    let (mut c0, dev0, shape0) = clone_cuda_f32(p0)?;
    let (mut c1, dev1, shape1) = clone_cuda_f32(p1)?;
    group_start().map_err(|e| anyhow::anyhow!("nccl group_start: {e:?}"))?;
    comms[0]
        .all_reduce_in_place(&mut c0, &ReduceOp::Sum)
        .map_err(|e| anyhow::anyhow!("nccl ar0: {e:?}"))?;
    comms[1]
        .all_reduce_in_place(&mut c1, &ReduceOp::Sum)
        .map_err(|e| anyhow::anyhow!("nccl ar1: {e:?}"))?;
    group_end().map_err(|e| anyhow::anyhow!("nccl group_end: {e:?}"))?;
    Ok((
        wrap_cuda_f32(c0, &dev0, shape0)?,
        wrap_cuda_f32(c1, &dev1, shape1)?,
    ))
}

/// F16 NCCL all-reduce(sum) for TP=2 - halves the PCIe payload of the small 1-token decode
/// reduction (the collective is latency/byte-bound on the PHB pair, not compute-bound, so
/// fewer bytes ≈ lower latency). Each f32 partial is cast to f16 on its own device (stream-
/// ordered), all-reduced in f16, then cast back to f32 for the residual add.
///
/// PRECISION: the sum runs in f16. For TP=2 a row-parallel partial is one rank's share of the
/// reduction (~half the contraction width), so the two addends are O(activation) magnitude and
/// their f16 sum keeps ~3 decimal digits - well inside decode tolerance. Greedy argmax is
/// unaffected at this scale (validated coherent + deterministic on deepseek-r1 TP=2). The
/// downstream residual/norm/matmul stay f32; only the cross-GPU transport is f16.
#[cfg(feature = "cuda")]
pub fn all_reduce_sum_nccl_f16(
    p0: &Tensor,
    p1: &Tensor,
    comms: &[cudarc::nccl::safe::Comm],
) -> Result<(Tensor, Tensor)> {
    use crate::tensor::DType;
    use cudarc::nccl::safe::{group_end, group_start, ReduceOp};
    let (shape0, shape1) = (p0.shape().clone(), p1.shape().clone());
    let (dev0, dev1) = (p0.device().clone(), p1.device().clone());
    // Cast to f16 on each producing device (stream-ordered with the matmul) and clone into an
    // owned, mutable CudaSlice<f16> for the in-place collective.
    let h0 = p0.to_dtype(DType::F16)?;
    let h1 = p1.to_dtype(DType::F16)?;
    let mut c0 = clone_cuda_f16(&h0)?;
    let mut c1 = clone_cuda_f16(&h1)?;
    group_start().map_err(|e| anyhow::anyhow!("nccl group_start: {e:?}"))?;
    comms[0]
        .all_reduce_in_place(&mut c0, &ReduceOp::Sum)
        .map_err(|e| anyhow::anyhow!("nccl f16 ar0: {e:?}"))?;
    comms[1]
        .all_reduce_in_place(&mut c1, &ReduceOp::Sum)
        .map_err(|e| anyhow::anyhow!("nccl f16 ar1: {e:?}"))?;
    group_end().map_err(|e| anyhow::anyhow!("nccl group_end: {e:?}"))?;
    let r0 = crate::tensor::cuda_ext::tensor_from_f16_slice(c0, shape0, &dev0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let r1 = crate::tensor::cuda_ext::tensor_from_f16_slice(c1, shape1, &dev1)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((r0.to_dtype(DType::F32)?, r1.to_dtype(DType::F32)?))
}

#[cfg(feature = "cuda")]
fn clone_cuda_f16(t: &Tensor) -> Result<crate::tensor::cuda_ext::CudaSlice<half::f16>> {
    use crate::tensor::cuda_ext;
    let s = cuda_ext::f16_slice_of(t)
        .map_err(|e| anyhow::anyhow!("all_reduce f16 expects CUDA f16 tensors: {e}"))?;
    let view = s.view().map_err(|e| anyhow::anyhow!("{e}"))?;
    s.stream()
        .clone_dtod(&view)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Build the 2 single-process NCCL comms (rank 0 = d0, rank 1 = d1) from the reference per-device
/// streams. Create once and reuse for every layer's all-reduce.
#[cfg(feature = "cuda")]
pub fn make_comms(d0: &Device, d1: &Device) -> Result<Vec<cudarc::nccl::safe::Comm>> {
    let s0 = d0
        .as_cuda_device()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .cuda_stream();
    let s1 = d1
        .as_cuda_device()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .cuda_stream();
    cudarc::nccl::safe::Comm::from_devices(vec![s0, s1])
        .map_err(|e| anyhow::anyhow!("nccl comm init: {e:?}"))
}

#[cfg(feature = "cuda")]
fn clone_cuda_f32(
    t: &Tensor,
) -> Result<(
    crate::tensor::cuda_ext::CudaSlice<f32>,
    Device,
    crate::tensor::Shape,
)> {
    use crate::tensor::cuda_ext;
    let dev = t.device().clone();
    let shape = t.shape().clone();
    // Borrow the partial's f32 storage and clone it into an owned, mutable
    // CudaSlice on the device's compute stream (stream-ordered with the
    // producing matmuls), so NCCL can all-reduce it in place.
    let s = cuda_ext::f32_slice_of(t)
        .map_err(|e| anyhow::anyhow!("all_reduce_sum_nccl expects CUDA f32 tensors: {e}"))?;
    let view = s.view().map_err(|e| anyhow::anyhow!("{e}"))?;
    let cloned = s
        .stream()
        .clone_dtod(&view)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((cloned, dev, shape))
}

#[cfg(feature = "cuda")]
fn wrap_cuda_f32(
    slice: crate::tensor::cuda_ext::CudaSlice<f32>,
    dev: &Device,
    shape: crate::tensor::Shape,
) -> Result<Tensor> {
    crate::tensor::cuda_ext::tensor_from_f32_slice(slice, shape, dev)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    // Single-GPU slicing math check (the cross-GPU composition is validated in the bins).
    #[test]
    fn column_slice_is_exact() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()), // no CUDA in CI -> skip
        };
        let (out, inn) = (512usize, 1024usize);
        let w = Tensor::randn(0f32, 1f32, (out, inn), &dev)?;
        let qt = QTensor::quantize(&w, GgmlDType::Q4K)?;
        // QTensor is not Clone - quantize the same weights again (deterministic).
        let full = QMatMul::from_arc(Arc::new(QTensor::quantize(&w, GgmlDType::Q4K)?))?;
        let x = Tensor::randn(0f32, 1f32, (1usize, inn), &dev)?;
        let fo = full.forward(&x)?;
        for &(s, e) in &[(0usize, 256usize), (256usize, 512usize)] {
            let part = QMatMul::from_arc(slice_qtensor_rows(&qt, s, e, &dev)?)?;
            // Host-side max reduction - works on both tensor substrates (the
            // native compat shim has no `max_all`).
            let d = (&part.forward(&x)? - &fo.narrow(1, s, e - s)?)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .fold(0f32, f32::max);
            assert!(d < 1e-4, "col slice {s}..{e} mismatch {d}");
        }
        Ok(())
    }
}
