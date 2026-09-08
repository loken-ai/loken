// Deterministic bit-identity gate: the single-pass fused down reduce must
// equal the ORIGINAL two-phase reduce EXACTLY (maxabs==0). This isolates the
// one thing that changed - dropping the second pass - from the tiled-GEMM's
// pre-existing tiny delta vs the per-column `dot` (which both carry). It
// is reproducible, free of the model's upstream thread-scheduling flicker.
use super::*;
use crate::tensor::quant_cpu::BlockFormat;
use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, repack_q6k, BlockQ8K};

/// Everything a case is made of, built once and handed to BOTH sides of the comparison.
///
/// Fixture only. It decides what is computed and never what the right answer is: the answer
/// comes from [`twophase`], which is the pre-fusion reduce kept whole. The two must not share
/// more than this - a case that asked the fused path for its inputs would be judging one
/// implementation against itself.
struct Case {
    dtype: GgmlDType,
    n_tokens: usize,
    topk: usize,
    e_count: usize,
    hidden: usize,
    k: usize,
    sti: Vec<u32>,
    eid: Vec<u32>,
    tw: Vec<f32>,
    weights: Arc<QTensor>,
    input: Tensor,
    residual: Option<Tensor>,
}

impl Case {
    fn build(
        keep: &mut Vec<Arc<QTensor>>,
        dtype: GgmlDType,
        n_tokens: usize,
        topk: usize,
        e_count: usize,
        hidden: usize,
        k: usize,
        with_residual: bool,
    ) -> Self {
        let dev = crate::tensor::Device::Cpu;
        let (sti, eid, tw) = build_routing(n_tokens, topk, e_count);
        let m = sti.len();
        // Expert weights [E, hidden, k], deterministic.
        let wdata: Vec<f32> = (0..e_count * hidden * k)
            .map(|i| (((i * 7 + 3) % 53) as f32) * 0.021 - 0.55)
            .collect();
        let wt = Tensor::from_vec(wdata, (e_count, hidden, k), &dev).unwrap();
        let weights = Arc::new(QTensor::quantize(&wt, dtype).unwrap());
        // Hold every Arc alive for the whole test: the repack cache is keyed by the tensor's
        // address, so a stack freed here can hand its address to the next one and be answered
        // out of the first one's entry.
        keep.push(weights.clone());
        // GLU activation per routed slot [m, k].
        let idata: Vec<f32> = (0..m * k)
            .map(|i| (((i * 11 + 5) % 41) as f32) * 0.033 - 0.6)
            .collect();
        let input = Tensor::from_vec(idata, (m, k), &dev).unwrap();
        let residual = with_residual.then(|| {
            let rd: Vec<f32> = (0..n_tokens * hidden)
                .map(|i| (((i * 13 + 1) % 37) as f32) * 0.05 - 0.9)
                .collect();
            Tensor::from_vec(rd, (n_tokens, hidden), &dev).unwrap()
        });
        Self {
            dtype,
            n_tokens,
            topk,
            e_count,
            hidden,
            k,
            sti,
            eid,
            tw,
            weights,
            input,
            residual,
        }
    }

    /// What the reduce answered before it was fused, for this case's format.
    fn twophase(&self) -> Tensor {
        match self.dtype {
            GgmlDType::Q6K => twophase(self, repacked_q6k, repack_q6k::gemm_group_avx2),
            GgmlDType::Q4K => twophase(self, repacked_q4k, repack_q4k::gemm_group_avx2),
            other => unreachable!("no two-phase baseline is kept for {other:?}"),
        }
    }

    fn label(&self) -> String {
        format!(
            "{:?} n={} topk={} E={} hidden={} k={} res={}",
            self.dtype,
            self.n_tokens,
            self.topk,
            self.e_count,
            self.hidden,
            self.k,
            self.residual.is_some()
        )
    }
}

/// The original two-phase reduce: materialise `dscaled`, then reduce in slot order.
///
/// Kept verbatim from before the fusion - it IS the bit-identity baseline, so the arithmetic
/// below must not be simplified, reordered or delegated to the path it judges. What the format
/// changes is only which superblock unpacker runs, so the two formats pass theirs in rather
/// than each carrying its own copy of the reduce; the reduce itself is written once and both
/// answers come out of the same statements in the same order.
fn twophase<B: Sync>(
    case: &Case,
    repacked: fn(&Arc<QTensor>, usize, usize, usize) -> Result<Arc<Vec<B>>>,
    gemm_group: unsafe fn(&[B], &[&[BlockQ8K]], usize, &mut [f32]),
) -> Tensor {
    let (hidden, k, topk) = (case.hidden, case.k, case.topk);
    let (sti, eid, tw) = (&case.sti[..], &case.eid[..], &case.tw[..]);
    let n_real_tokens = case.n_tokens;
    let m = sti.len();
    let bpr = k / 256;
    let e_count = case.weights.shape().dims()[0];
    let gsz = (hidden / 8) * bpr;
    let down_all = repacked(&case.weights, e_count, hidden, bpr).unwrap();
    let hs = case
        .input
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let hq: Vec<Vec<BlockQ8K>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut dscaled = vec![0f32; m * hidden];
    let dptr = SendF32(dscaled.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = hidden / 8;
    let empty: &[BlockQ8K] = &[];
    for (e, j0, j1) in expert_runs(eid) {
        let te = j1 - j0;
        let down_x8 = &down_all[e * gsz..(e + 1) * gsz];
        let ntiles = te.div_ceil(8);
        pool.run(ntiles * ngroups, &|wk| {
            let dptr = &dptr;
            let (tt, cg) = (wk / ngroups, wk % ngroups);
            let r0 = tt * 8;
            let mt = (te - r0).min(8);
            let mut acts: [&[BlockQ8K]; 8] = [empty; 8];
            for (i, a) in acts.iter_mut().enumerate().take(mt) {
                *a = &hq[j0 + r0 + i];
            }
            let bgrp = &down_x8[cg * bpr..(cg + 1) * bpr];
            let mut dl = [0f32; 64];
            unsafe {
                gemm_group(bgrp, &acts[..mt], bpr, &mut dl);
            }
            for i in 0..mt {
                let slot = j0 + r0 + i;
                let twv = tw[sti[slot] as usize];
                for c in 0..8 {
                    unsafe {
                        *dptr.0.add(slot * hidden + cg * 8 + c) = twv * dl[i * 8 + c];
                    }
                }
            }
        });
    }
    let mut slots_of: Vec<Vec<usize>> = vec![Vec::with_capacity(topk); n_real_tokens];
    for j in 0..m {
        let t = (sti[j] as usize) / topk;
        if t < n_real_tokens {
            slots_of[t].push(j);
        }
    }
    let mut out = match case.residual.as_ref() {
        Some(r) => r
            .reshape((n_real_tokens, hidden))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        None => vec![0f32; n_real_tokens * hidden],
    };
    crate::tensor::quant_cpu::pool_for_each_mut(&mut out, 64, &|i, o| {
        let (t, r) = (i / hidden, i % hidden);
        for &j in &slots_of[t] {
            *o += dscaled[j * hidden + r];
        }
    });
    Tensor::from_vec(out, (n_real_tokens, hidden), &case.input.device()).unwrap()
}

fn build_routing(n_tokens: usize, topk: usize, e_count: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut pairs: Vec<(u32, u32)> = Vec::new(); // (expert, flat_pair_idx)
    for t in 0..n_tokens {
        for s in 0..topk {
            let e = ((t * topk + s * 3 + t) % e_count) as u32;
            pairs.push((e, (t * topk + s) as u32));
        }
    }
    pairs.sort_by_key(|p| p.0); // contiguous expert runs, as the kernels require
    let eid: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    let sti: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    let tw: Vec<f32> = (0..n_tokens * topk)
        .map(|i| 0.07 + ((i % 17) as f32) * 0.041)
        .collect();
    (sti, eid, tw)
}

fn maxabs(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(av.len(), bv.len());
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The tiled fused reduce against the baseline, on one case.
fn fused_matches_twophase(
    keep: &mut Vec<Arc<QTensor>>,
    dtype: GgmlDType,
    n_tokens: usize,
    topk: usize,
    e_count: usize,
    hidden: usize,
    k: usize,
    with_residual: bool,
) {
    let case = Case::build(
        keep,
        dtype,
        n_tokens,
        topk,
        e_count,
        hidden,
        k,
        with_residual,
    );
    let twophase = case.twophase();
    let fused = match dtype {
        GgmlDType::Q6K => fused_down_reduce_q6k_tiled(
            &case.input,
            &case.weights,
            &case.sti,
            &case.eid,
            &case.tw,
            case.topk,
            case.n_tokens,
            case.residual.as_ref(),
            case.hidden,
            case.k,
        ),
        GgmlDType::Q4K => fused_down_reduce_q4k_tiled(
            &case.input,
            &case.weights,
            &case.sti,
            &case.eid,
            &case.tw,
            case.topk,
            case.n_tokens,
            case.residual.as_ref(),
            case.hidden,
            case.k,
        ),
        other => unreachable!("no tiled fused reduce for {other:?}"),
    }
    .unwrap()
    .expect("fused produced output");

    let d = maxabs(&twophase, &fused);
    assert_eq!(d, 0.0, "{} maxabs(twophase,fused)={d}", case.label());
}

#[test]
fn q6k_fused_matches_reference() {
    let mut keep = Vec::new();
    for &(n, tk, e) in &[(40usize, 4usize, 8usize), (7, 2, 4), (96, 8, 16), (3, 4, 8)] {
        for &res in &[false, true] {
            fused_matches_twophase(&mut keep, GgmlDType::Q6K, n, tk, e, 512, 512, res);
            fused_matches_twophase(&mut keep, GgmlDType::Q6K, n, tk, e, 256, 256, res);
        }
    }
}

#[test]
fn q6k_decode_gemv_matches_reference() {
    // The NEW decode-path Q6K down GEMV must be bit-identical to the
    // two-phase reference (same kernel, single-row tiles, same
    // accumulation order) - including the few-tokens-per-expert decode
    // shapes the tiled path never sees.
    let mut keep = Vec::new();
    for &(n, tk, e) in &[(1usize, 8usize, 32usize), (2, 4, 8), (3, 4, 8), (7, 2, 4)] {
        for &res in &[false, true] {
            for &(hidden, k) in &[(512usize, 512usize), (1024, 512), (256, 256)] {
                let case = Case::build(&mut keep, GgmlDType::Q6K, n, tk, e, hidden, k, res);
                let twophase = case.twophase();
                let gemv = fused_down_reduce_q6k_gemv(
                    &case.input,
                    &case.weights,
                    &case.sti,
                    &case.eid,
                    &case.tw,
                    case.topk,
                    case.n_tokens,
                    case.residual.as_ref(),
                    case.hidden,
                    case.k,
                )
                .unwrap()
                .expect("q6k gemv produced output");
                let d = maxabs(&twophase, &gemv);
                assert_eq!(d, 0.0, "q6k decode gemv {} maxabs={d}", case.label());
            }
        }
    }
}

#[test]
fn q4k_fused_matches_reference() {
    let mut keep = Vec::new();
    for &(n, tk, e) in &[(40usize, 4usize, 8usize), (7, 2, 4), (96, 8, 16), (3, 4, 8)] {
        for &res in &[false, true] {
            fused_matches_twophase(&mut keep, GgmlDType::Q4K, n, tk, e, 512, 512, res);
            fused_matches_twophase(&mut keep, GgmlDType::Q4K, n, tk, e, 256, 256, res);
        }
    }
}
