//! Tensor-parallel decode primitives - the reusable core behind the
//! tensor-parallel decode lever (deepseek-r1 +42% vs vLLM).
//!
//! The multi-GPU pipeline runs one GPU per decode token (the others idle). TP runs EVERY GPU on
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

/// Boundaries of `shares.len()` ranks over an axis of `dim`: `shares.len() + 1` offsets from 0 to
/// `dim`, rank `r` owning `[b[r], b[r + 1])`. Each interior boundary is rounded down to a multiple
/// of `align` (a head, or a QK_K block, so a row-parallel partner's slice stays whole). A rank
/// the rounding would leave empty sends the split to an even one.
pub fn split_points(dim: usize, shares: &[usize], align: usize) -> Vec<usize> {
    let ranks = shares.len();
    let align = align.max(1);
    let total: usize = shares.iter().sum::<usize>().max(1);
    let cut =
        |num: usize, den: usize, i: usize| (dim * num / den / align) * align * usize::from(i > 0);
    let mut points = vec![0usize];
    let mut cum = 0;
    for (i, share) in shares.iter().enumerate() {
        cum += share;
        points.push(if i + 1 == ranks {
            dim
        } else {
            cut(cum, total, 1)
        });
    }
    if points.windows(2).any(|w| w[1] <= w[0]) {
        points = (0..=ranks)
            .map(|i| if i == ranks { dim } else { cut(i, ranks, i) })
            .collect();
    }
    points
}

/// Column-parallel matmul: a `[out,in]` weight split by output rows across the ranks.
/// `forward` returns each rank's partial output on its own device (stays split for the
/// following row-parallel op).
pub struct TpColumn {
    /// Rank `r` holds output rows `[points[r], points[r + 1])` on its own device.
    pub mats: Vec<QMatMul>,
    pub points: Vec<usize>,
    /// Optional per-rank bias slice (qwen2 q/k/v projections carry a bias), split like the rows.
    pub biases: Vec<Option<Tensor>>,
}
impl TpColumn {
    pub fn from_qtensor(qt: &QTensor, devs: &[Device]) -> Result<Self> {
        Self::from_qtensor_bias(qt, None, devs)
    }
    /// Column-parallel build with an optional dequantized bias (F32 `[out]`), sliced per rank
    /// to match the weight rows. The rows are split evenly, `align` rows at a time.
    pub fn from_qtensor_bias(qt: &QTensor, bias: Option<&Tensor>, devs: &[Device]) -> Result<Self> {
        let out = qt.shape().dims()[0];
        let points = split_points(out, &vec![1; devs.len()], 1);
        let mut mats = Vec::with_capacity(devs.len());
        let mut biases = Vec::with_capacity(devs.len());
        for (r, dev) in devs.iter().enumerate() {
            let (s, e) = (points[r], points[r + 1]);
            mats.push(QMatMul::from_arc(slice_qtensor_rows(qt, s, e, dev)?)?);
            biases.push(match bias {
                Some(b) => Some(b.narrow(0, s, e - s)?.to_device(dev)?),
                None => None,
            });
        }
        Ok(Self {
            mats,
            points,
            biases,
        })
    }
    /// The replicated input `xs[r]` on rank `r`'s device -> each rank's output on its device.
    pub fn forward(&self, xs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.mats
            .iter()
            .zip(&self.biases)
            .zip(xs)
            .map(|((m, b), x)| {
                let o = m.forward(x)?;
                Ok(match b {
                    Some(b) => o.broadcast_add(b)?,
                    None => o,
                })
            })
            .collect()
    }
}

/// `dim`'s offsets for an asymmetric split between the ranks, rounded down to a QK_K(256)
/// boundary so a row-parallel partner's column slice stays block-aligned. The faster rank takes
/// the larger share so all finish their matmuls together and none idles at the all-reduce.
/// Falls back to an even split if rounding would empty a rank.
pub fn asym_points(dim: usize, shares: &[usize]) -> Vec<usize> {
    split_points(dim, shares, QK_K)
}

/// Fused column-parallel projection: several `[out_i, in]` weights sharing the SAME input
/// (q/k/v, or gate/up) byte-concatenated row-wise into ONE matmul per rank. Replaces the
/// N matmul + N bias-add launches of the unfused path with 1 + 1 - pure launch reduction
/// (decode is launch-bound), and one larger mvq has better occupancy than N small ones.
/// All sub-weights must share dtype + input dim (q/k/v and gate/up do in qwen2).
pub struct TpColumnFused {
    pub mats: Vec<QMatMul>,
    pub biases: Vec<Option<Tensor>>,
    /// Per-rank output width of each sub-projection (in concat order). They differ when the
    /// split is asymmetric.
    pub widths: Vec<Vec<usize>>,
}
impl TpColumnFused {
    /// Even build on `align`-row boundaries (a head for q/k, where the o-proj alignment forbids
    /// an asymmetric cut).
    pub fn from_qtensors(
        items: &[(&QTensor, Option<&Tensor>)],
        align: usize,
        devs: &[Device],
    ) -> Result<Self> {
        Self::from_qtensors_shares(items, &vec![1; devs.len()], align, devs)
    }
    /// Asymmetric build: each weight is column-parallel row-split at `split_points(out, shares,
    /// align)`, the per-rank row-slices byte-concatenated into one quantized weight per device.
    pub fn from_qtensors_shares(
        items: &[(&QTensor, Option<&Tensor>)],
        shares: &[usize],
        align: usize,
        devs: &[Device],
    ) -> Result<Self> {
        let ranks = devs.len();
        let dt = items[0].0.dtype();
        let inn = items[0].0.shape().dims()[1];
        let mut bytes = vec![Vec::<u8>::new(); ranks];
        let mut rows = vec![0usize; ranks];
        let mut widths = vec![Vec::new(); ranks];
        let mut bias_parts: Vec<Vec<Tensor>> = vec![Vec::new(); ranks];
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
            let points = split_points(out, shares, align);
            let data = qt.data()?;
            let row_bytes = data.len() / out;
            for r in 0..ranks {
                let (s, e) = (points[r], points[r + 1]);
                bytes[r].extend_from_slice(&data[s * row_bytes..e * row_bytes]);
                rows[r] += e - s;
                widths[r].push(e - s);
                if let Some(b) = bias {
                    bias_parts[r].push(b.narrow(0, s, e - s)?);
                }
            }
        }
        let mut mats = Vec::with_capacity(ranks);
        let mut biases = Vec::with_capacity(ranks);
        for (r, dev) in devs.iter().enumerate() {
            mats.push(QMatMul::from_arc(Arc::new(QTensor::from_ggml_bytes(
                dt,
                &bytes[r],
                vec![rows[r], inn],
                dev,
            )?))?);
            biases.push(if has_bias {
                Some(Tensor::cat(&bias_parts[r].iter().collect::<Vec<_>>(), 0)?.to_device(dev)?)
            } else {
                None
            });
        }
        Ok(Self {
            mats,
            biases,
            widths,
        })
    }
    /// One fused matmul per rank (+bias): the replicated input `xs[r]` on rank `r` -> its output.
    pub fn forward(&self, xs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.mats
            .iter()
            .zip(&self.biases)
            .zip(xs)
            .map(|((m, b), x)| {
                let o = m.forward(x)?;
                Ok(match b {
                    Some(b) => o.broadcast_add(b)?,
                    None => o,
                })
            })
            .collect()
    }
    /// Split rank `rank`'s fused output `[seq, sum(widths_r)]` into its contiguous sub-projections.
    pub fn split(&self, o: &Tensor, rank: usize) -> Result<Vec<Tensor>> {
        let widths = &self.widths[rank];
        let mut parts = Vec::with_capacity(widths.len());
        let mut off = 0;
        for &w in widths {
            parts.push(o.narrow(1, off, w)?.contiguous()?);
            off += w;
        }
        Ok(parts)
    }
}

/// Row-parallel matmul: a `[out,in]` weight split by input columns across the ranks.
/// `forward` takes each rank's split input and returns its partial output (to be all-reduced).
pub struct TpRow {
    pub mats: Vec<QMatMul>,
}
impl TpRow {
    /// Even column split on QK_K blocks.
    pub fn from_qtensor(qt: &QTensor, devs: &[Device]) -> Result<Self> {
        Self::from_qtensor_shares(qt, &vec![1; devs.len()], devs)
    }
    /// Asymmetric col-split at `asym_points(in, shares)` - must MATCH the row-split of the
    /// column-parallel weight feeding this one (e.g. `down` cols ↔ `gate/up` rows).
    pub fn from_qtensor_shares(qt: &QTensor, shares: &[usize], devs: &[Device]) -> Result<Self> {
        let inn = qt.shape().dims()[1];
        let points = asym_points(inn, shares);
        let mut mats = Vec::with_capacity(devs.len());
        for (r, dev) in devs.iter().enumerate() {
            mats.push(QMatMul::from_arc(slice_qtensor_cols(
                qt,
                points[r],
                points[r + 1],
                dev,
            )?)?);
        }
        Ok(Self { mats })
    }
    /// Each rank's split input `hs[r]` on its device -> its partial output.
    pub fn forward(&self, hs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.mats
            .iter()
            .zip(hs)
            .map(|(m, h)| Ok(m.forward(h)?))
            .collect()
    }
}

/// Cross-GPU all-reduce(sum): `ps[r]` on rank `r`'s device -> the sum, replicated on every rank.
/// (Naive peer copies via facade `to_device` + add, through rank 0. NCCL is the faster path
/// once `libnccl` is installed.)
///
/// SYNCHRONIZATION IS LOAD-BEARING. the reference cross-CUDA `to_device` enqueues a peer copy on
/// the destination stream without waiting for the SOURCE device's compute stream to finish
/// producing the tensor. Without the syncs below, the copy races the producing matmul and
/// reads partial/stale data - making the whole forward NON-DETERMINISTIC (the same layer
/// gives different values across runs) and incoherent. Each device must be synchronized
/// before its tensor is read cross-device, and after each peer copy before the result is used.
pub fn all_reduce_sum(ps: &[Tensor], devs: &[Device]) -> Result<Vec<Tensor>> {
    // Correct (deterministic) but serializing fallback: the reference cross-CUDA peer copy does NOT
    // reliably order against the matmul that produced the source tensor, so we host-synchronize
    // the source device before each peer read. This is a full barrier (no GPU overlap). The fast
    // path is `all_reduce_sum_nccl` below.
    let mut sum = ps[0].clone();
    for (p, dev) in ps.iter().zip(devs).skip(1) {
        dev.synchronize()?;
        sum = (&sum + p.to_device(&devs[0])?)?;
    }
    devs[0].synchronize()?;
    devs.iter().map(|d| Ok(sum.to_device(d)?)).collect()
}

/// An NCCL failure as the library words it: the result code and NCCL's last message, which
/// names the CUDA error behind it, so an allocation that ran out of memory reads as one and
/// takes the out-of-memory recovery.
#[cfg(feature = "cuda")]
fn nccl_failure(what: &str, e: &cudarc::nccl::result::NcclError) -> anyhow::Error {
    // SAFETY: ncclGetLastError ignores its communicator and returns a pointer to a static,
    // NUL-terminated string owned by the library, or null.
    let detail = unsafe {
        let p = cudarc::nccl::sys::ncclGetLastError(std::ptr::null_mut());
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    anyhow::anyhow!("{what}: {:?}: {detail}", e.0)
}

/// NCCL all-reduce(sum) - the OVERLAPPING path. Runs `ncclAllReduce` on each rank's own stream
/// (so the collective is stream-ordered with the matmuls and the GPUs run in parallel - no
/// `device.synchronize()` barrier). `comms` are the per-device comms from `make_comms`.
/// Returns the summed tensor on each device, replicated.
#[cfg(feature = "cuda")]
pub fn all_reduce_sum_nccl(
    ps: &[Tensor],
    comms: &[cudarc::nccl::safe::Comm],
) -> Result<Vec<Tensor>> {
    use cudarc::nccl::safe::{group_end, group_start, ReduceOp};
    // Clone each partial's f32 device buffer into an owned, mutable CudaSlice (outside the
    // NCCL group), so we can all-reduce it in place.
    let mut owned = ps.iter().map(clone_cuda_f32).collect::<Result<Vec<_>>>()?;
    group_start().map_err(|e| nccl_failure("nccl group_start", &e))?;
    for (comm, (slice, _, _)) in comms.iter().zip(owned.iter_mut()) {
        comm.all_reduce_in_place(slice, &ReduceOp::Sum)
            .map_err(|e| nccl_failure("nccl all-reduce", &e))?;
    }
    group_end().map_err(|e| nccl_failure("nccl group_end", &e))?;
    owned
        .into_iter()
        .map(|(slice, dev, shape)| wrap_cuda_f32(slice, &dev, shape))
        .collect()
}

/// F16 NCCL all-reduce(sum) - halves the PCIe payload of the small 1-token decode reduction
/// (the collective is latency/byte-bound on the PHB pair, not compute-bound, so fewer bytes ≈
/// lower latency). Each f32 partial is cast to f16 on its own device (stream-ordered),
/// all-reduced in f16, then cast back to f32 for the residual add.
///
/// PRECISION: the sum runs in f16. A row-parallel partial is one rank's share of the reduction,
/// so the addends are O(activation) magnitude and their f16 sum keeps ~3 decimal digits - well
/// inside decode tolerance. Greedy argmax is unaffected at this scale (validated coherent +
/// deterministic on deepseek-r1 TP=2). The downstream residual/norm/matmul stay f32; only the
/// cross-GPU transport is f16.
#[cfg(feature = "cuda")]
pub fn all_reduce_sum_nccl_f16(
    ps: &[Tensor],
    comms: &[cudarc::nccl::safe::Comm],
) -> Result<Vec<Tensor>> {
    use crate::tensor::DType;
    use cudarc::nccl::safe::{group_end, group_start, ReduceOp};
    // Cast to f16 on each producing device (stream-ordered with the matmul) and clone into an
    // owned, mutable CudaSlice<f16> for the in-place collective.
    let mut owned = ps
        .iter()
        .map(|p| {
            let h = p.to_dtype(DType::F16)?;
            Ok((clone_cuda_f16(&h)?, p.shape().clone(), p.device().clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    group_start().map_err(|e| nccl_failure("nccl group_start", &e))?;
    for (comm, (slice, _, _)) in comms.iter().zip(owned.iter_mut()) {
        comm.all_reduce_in_place(slice, &ReduceOp::Sum)
            .map_err(|e| nccl_failure("nccl f16 all-reduce", &e))?;
    }
    group_end().map_err(|e| nccl_failure("nccl group_end", &e))?;
    owned
        .into_iter()
        .map(|(slice, shape, dev)| {
            let r = crate::tensor::cuda_ext::tensor_from_f16_slice(slice, shape, &dev)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(r.to_dtype(DType::F32)?)
        })
        .collect()
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

/// Build the single-process NCCL comms (rank `r` = `devs[r]`) from the reference per-device
/// streams. Create once and reuse for every layer's all-reduce.
#[cfg(feature = "cuda")]
pub fn make_comms(devs: &[Device]) -> Result<Vec<cudarc::nccl::safe::Comm>> {
    if !crate::tensor::cuda::runtime::nccl_present() {
        return Err(anyhow::anyhow!("NCCL library not found"));
    }
    let streams = devs
        .iter()
        .map(|d| {
            Ok(d.as_cuda_device()
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .cuda_stream())
        })
        .collect::<Result<Vec<_>>>()?;
    cudarc::nccl::safe::Comm::from_devices(streams).map_err(|e| nccl_failure("nccl comm init", &e))
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
mod rank_tests {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    #[test]
    fn two_ranks_split_where_the_pair_always_did() {
        for (dim, num, den) in [(10240usize, 2usize, 3usize), (27648, 1, 2), (5120, 3, 5)] {
            let old = (dim * num / den / QK_K) * QK_K;
            assert_eq!(asym_points(dim, &[num, den - num]), vec![0, old, dim]);
        }
    }

    #[test]
    fn any_number_of_ranks_cover_the_axis_in_whole_blocks() {
        for ranks in 2..=6usize {
            for shares in [vec![1; ranks], (1..=ranks).collect::<Vec<_>>()] {
                let p = asym_points(17408, &shares);
                assert_eq!(p.len(), ranks + 1);
                assert_eq!((p[0], p[ranks]), (0, 17408));
                assert!(p.windows(2).all(|w| w[1] > w[0]), "{p:?}");
                assert!(p[1..ranks].iter().all(|b| b % QK_K == 0), "{p:?}");
            }
        }
    }

    #[test]
    fn a_split_that_would_empty_a_rank_goes_even() {
        let p = asym_points(1024, &[1, 100, 1]);
        assert_eq!(p, vec![0, 256, 512, 1024]);
    }

    /// Column slices over three host "ranks" give the full projection row by row, and row slices
    /// give partials that sum to it: the arithmetic of a tensor-parallel layer of any width.
    #[test]
    fn three_ranks_reproduce_one_projection() -> Result<()> {
        let devs = vec![Device::Cpu, Device::Cpu, Device::Cpu];
        let (out, inn) = (768usize, 1536usize);
        let w = Tensor::randn(0f32, 1f32, (out, inn), &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1f32, (1usize, inn), &Device::Cpu)?;
        let q = |w: &Tensor| QTensor::quantize(w, GgmlDType::Q4K);
        let full = QMatMul::from_arc(Arc::new(q(&w)?))?.forward(&x)?;
        let want = full.flatten_all()?.to_vec1::<f32>()?;

        let col = TpColumn::from_qtensor(&q(&w)?, &devs)?;
        assert_eq!(col.points, vec![0, 256, 512, 768]);
        let got = col
            .forward(&vec![x.clone(); 3])?
            .iter()
            .map(|t| t.flatten_all()?.to_vec1::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()?
            .concat();
        let gap = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(gap < 1e-3, "column gap {gap}");

        let row = TpRow::from_qtensor_shares(&q(&w)?, &[1, 1, 1], &devs)?;
        let points = asym_points(inn, &[1, 1, 1]);
        let parts: Vec<Tensor> = (0..3)
            .map(|r| {
                x.narrow(1, points[r], points[r + 1] - points[r])
                    .unwrap()
                    .contiguous()
                    .unwrap()
            })
            .collect();
        let partial = row.forward(&parts)?;
        let mut sum = partial[0].clone();
        for p in &partial[1..] {
            sum = (&sum + p)?;
        }
        let got = sum.flatten_all()?.to_vec1::<f32>()?;
        let gap = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(gap < 5e-2, "row gap {gap}");
        Ok(())
    }
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
