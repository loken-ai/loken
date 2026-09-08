//! CPU fallback shim for `moe_cuda` (the CUDA MoE expert-GEMM + routing helpers).
//!
//! Selected via `#[cfg(not(feature = "cuda"))]` in `mod.rs`. Provides tensor-op CPU
//! implementations of the expert grouped-GEMM (the only non-optional ops the
//! arches call directly) and returns `None` for the optional fused routing/conv
//! helpers so the architecture falls back to its portable path.
//!
//! Routing format (must match the CUDA kernels): for the M = n_tokens.topk
//! expanded (token,slot) pairs, `sorted_token_ids[j]` is the flattened pair index
//! at sorted position j (token = idx / topk), and `expert_ids[j]` is its expert
//! (the ids are sorted, so each expert occupies a contiguous run).

use crate::tensor::quantized::{GgmlDType, QMatMul, QTensor};
use crate::tensor::{DType, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// -- per-expert quantized-matmul cache ----------------------------------------
// The expert GEMMs below used to `dequantize()` the ENTIRE `[E, N, K]` expert
// stack to F32 on every token, then matmul only the selected experts - paying
// full-stack F32 dequant + memory traffic each step (the ~100x CPU-MoE slowdown
// vs Ollama). Instead we slice each expert's contiguous quantized-block range
// into its own `QMatMul` ONCE (cached, keyed by the stable QTensor address) and
// run a fused quantized GEMV (`dot` over the GGUF blocks of only the routed
// experts) - exactly what Ollama does. `None` for a tensor whose quant type has
// no CPU dot path -> that call keeps the dequantize-then-matmul fallback.

/// Slice a 3-D expert stack `[E, R, C]` into `E` per-expert `[R, C]` `QMatMul`s.
/// Blocks are laid out expert-major / row-major, so each expert is one
/// contiguous byte range - exposed as ZERO-COPY views sharing the parent's
/// storage: the cache used to hold a full byte-copied duplicate of every
/// expert tensor, doubling the resident weight memory of CPU MoE models.
fn build_expert_qmms(weights: &Arc<QTensor>) -> Result<Option<Vec<QMatMul>>> {
    if !crate::inference::fused_moe::cpu_qmatmul_supported(weights.dtype()) {
        return Ok(None);
    }
    let dims = weights.shape().dims();
    if dims.len() != 3 {
        return Ok(None);
    }
    let e_count = dims[0];
    let mut v = Vec::with_capacity(e_count);
    for i in 0..e_count {
        let qt = crate::tensor::quant_view::expert_view(weights, i)?;
        v.push(QMatMul::from_qtensor(qt)?);
    }
    Ok(Some(v))
}

#[allow(clippy::type_complexity)]
fn expert_cache() -> &'static Mutex<HashMap<(usize, Vec<usize>), Option<Arc<Vec<QMatMul>>>>> {
    static C: OnceLock<Mutex<HashMap<(usize, Vec<usize>), Option<Arc<Vec<QMatMul>>>>>> =
        OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop all cached per-expert `QMatMul`s. MUST be called on model unload: the
/// views hold their parent expert tensor alive (Arc), so a stale entry pins
/// the previous model's full expert stack in memory for the process lifetime  -
/// one model-sized leak per reload.
pub fn clear_expert_cache() {
    expert_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        repack_q4k_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        repack_q6k_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

// Prefill-tiling repack caches: the whole `[E,N,K]` expert stack repacked into the 8-column
// block-interleaved layout the tiled GEMM reads, built once per tensor and reused every
// forward (the repack is a gather - same size as the weight, so this roughly doubles the
// resident expert memory, the cost of the ~2x prefill weight-reuse).
//
// Keyed by the tensor's IDENTITY, not its address. An entry here is a copy and holds nothing
// of the parent, so the parent can be dropped and its address handed to the next allocation  -
// which would then be answered with the previous tensor's repack, silently. The identity only
// counts up, so a stale entry is a leak until `clear_expert_cache`, never a wrong answer.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
type Q4Kx8 = crate::tensor::quant_cpu::repack_q4k::BlockQ4Kx8;
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
type Q6Kx8 = crate::tensor::quant_cpu::repack_q6k::BlockQ6Kx8;

// One entry per stack, and the entry owns the build. The table's lock guards only the table,
// so it is dropped before the build starts and a repack never blocks an unrelated stack; the
// entry's own lock is what makes the build single-flight. A second caller for the same stack  -
// a request arriving while the load-time warm is still running - waits on that entry and is
// handed the same result, instead of repacking the stack a second time beside it. The two
// locks are only ever taken in this order and the build takes no other lock, so there is no
// cycle to deadlock on.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
type RepackCell<T> = Arc<Mutex<Option<Arc<Vec<T>>>>>;

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn repack_q4k_cache() -> &'static Mutex<HashMap<u64, RepackCell<Q4Kx8>>> {
    static C: OnceLock<Mutex<HashMap<u64, RepackCell<Q4Kx8>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn repack_q6k_cache() -> &'static Mutex<HashMap<u64, RepackCell<Q6Kx8>>> {
    static C: OnceLock<Mutex<HashMap<u64, RepackCell<Q6Kx8>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get-or-build the repacked `[E][N/8][bpr]` Q4_K stack for `weights`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn repacked_q4k(
    weights: &Arc<QTensor>,
    e_count: usize,
    n: usize,
    bpr: usize,
) -> Result<Arc<Vec<Q4Kx8>>> {
    use crate::tensor::quant_cpu::{repack_q4k, BlockQ4K};
    let key = weights.id();
    let cell = repack_q4k_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(key)
        .or_default()
        .clone();
    let mut slot = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(a) = slot.as_ref() {
        return Ok(a.clone());
    }
    let d = weights.data()?;
    let blocks: &[BlockQ4K] = unsafe {
        std::slice::from_raw_parts(
            d.as_ptr() as *const BlockQ4K,
            d.len() / std::mem::size_of::<BlockQ4K>(),
        )
    };
    let mut out = Vec::with_capacity(e_count * (n / 8) * bpr);
    for e in 0..e_count {
        out.extend(repack_q4k::repack(
            &blocks[e * n * bpr..(e + 1) * n * bpr],
            n,
            bpr,
        ));
    }
    let a = Arc::new(out);
    *slot = Some(a.clone());
    Ok(a)
}

/// Get-or-build the repacked `[E][hidden/8][bpr]` Q6_K stack for `weights`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn repacked_q6k(
    weights: &Arc<QTensor>,
    e_count: usize,
    n: usize,
    bpr: usize,
) -> Result<Arc<Vec<Q6Kx8>>> {
    use crate::tensor::quant_cpu::{repack_q6k, BlockQ6K};
    let key = weights.id();
    let cell = repack_q6k_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(key)
        .or_default()
        .clone();
    let mut slot = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(a) = slot.as_ref() {
        return Ok(a.clone());
    }
    let d = weights.data()?;
    let blocks: &[BlockQ6K] = unsafe {
        std::slice::from_raw_parts(
            d.as_ptr() as *const BlockQ6K,
            d.len() / std::mem::size_of::<BlockQ6K>(),
        )
    };
    let mut out = Vec::with_capacity(e_count * (n / 8) * bpr);
    for e in 0..e_count {
        out.extend(repack_q6k::repack(
            &blocks[e * n * bpr..(e + 1) * n * bpr],
            n,
            bpr,
        ));
    }
    let a = Arc::new(out);
    *slot = Some(a.clone());
    Ok(a)
}

/// Build the repack of one expert stack ahead of the request that would otherwise build it.
///
/// The host prefill path reads its expert weights through an 8-column block-interleaved
/// layout that is derived from the stack, not stored with it, so whichever call needs it
/// first pays for the whole model. That call is normally the first prefill, inside a user's
/// request; this lets the load warm the same cache entries off the request path. It is the
/// same work and the same result, only earlier - a request that arrives mid-warm blocks on
/// the entry and gets the warm's result rather than repacking beside it.
///
/// A stack whose shape or format the host path has no repack for is silently nothing to do:
/// the caller warms every stack it has and does not need to know which ones qualify.
pub fn prewarm_expert_repack(weights: &Arc<QTensor>) -> Result<()> {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        let dims = weights.shape().dims();
        // [experts, rows, k] - the shape every repack call site reads, and the two
        // divisibilities the repacked layout needs.
        if dims.len() != 3 || !dims[1].is_multiple_of(8) || !dims[2].is_multiple_of(256) {
            return Ok(());
        }
        let (e_count, rows, bpr) = (dims[0], dims[1], dims[2] / 256);
        match weights.dtype() {
            GgmlDType::Q4K => {
                repacked_q4k(weights, e_count, rows, bpr)?;
            }
            GgmlDType::Q6K => {
                repacked_q6k(weights, e_count, rows, bpr)?;
            }
            _ => {}
        }
    }
    #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
    let _ = weights;
    Ok(())
}

/// Get (building + caching on first use) the per-expert `QMatMul`s for `weights`.
///
/// Keyed by the tensor's identity and its shape. This cache's entries do keep their parent
/// alive - the views hold it - so the address would have been sound here; the identity says
/// so without the argument, and matches the repack caches beside it. Entries are dropped via
/// `clear_expert_cache` on model unload.
fn expert_qmms(weights: &Arc<QTensor>) -> Option<Arc<Vec<QMatMul>>> {
    let key = (weights.id() as usize, weights.shape().dims().to_vec());
    let mut cache = expert_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(v) = cache.get(&key) {
        return v.clone();
    }
    let built = build_expert_qmms(weights).ok().flatten().map(Arc::new);
    cache.insert(key, built.clone());
    built
}

// -- optional fused router/conv helpers: None -> caller uses its portable fallback -
pub fn gate_gemv_f32(_xs: &Tensor, _gate_w: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn gate_topk_softmax(
    _xs: &Tensor,
    _gate_w: &Tensor,
    _n: usize,
    _norm: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn gate_topk_sigmoid(
    _xs: &Tensor,
    _gate_w: &Tensor,
    _bias: Option<&Tensor>,
    _n: usize,
    _norm: bool,
    _scale: f64,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn topk_sigmoid_post(
    _logits: &Tensor,
    _bias: Option<&Tensor>,
    _n: usize,
    _norm: bool,
    _scale: f64,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn argsort_small_u32(_flat: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn lfm2_shortconv_f16io(
    _bcx: &Tensor,
    _state: &Tensor,
    _conv_w: &Tensor,
    _d_model: usize,
    _l_cache: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn lfm2_shortconv_f16io_inplace(
    _bcx: &Tensor,
    _state: &Tensor,
    _conv_w: &Tensor,
    _d_model: usize,
    _l_cache: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn zgate_rmsnorm_f16io(
    _o: &Tensor,
    _z: &Tensor,
    _norm_w: &Tensor,
    _eps: f32,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn fused_conv_silu_f16in(
    _qkv: &Tensor,
    _conv_state: &Tensor,
    _w: &Tensor,
    _conv_kernel: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}
pub fn deltanet_gate_f16in(
    _alpha: &Tensor,
    _beta_in: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}

// -- simple norms (non-optional; matches the kernel math) --
pub fn head_rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    // rmsnorm over the last dim (head_dim), per head.
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let n = xf.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    n.broadcast_mul(&w.to_dtype(DType::F32)?)?.to_dtype(dt)
}

pub fn zgate_rmsnorm(o: &Tensor, z: &Tensor, norm_w: &Tensor, eps: f32) -> Result<Tensor> {
    // rmsnorm(o) * silu(z) * norm_w
    let n = head_rmsnorm(o, norm_w, eps)?;
    n * crate::tensor::ops::silu(z)?
}

pub fn l2norm_gqa(x: &Tensor, _kg: usize, _kd: usize, _rep: usize, eps: f32) -> Result<Tensor> {
    // L2-normalize over the last dim.
    let xf = x.to_dtype(DType::F32)?;
    let norm = xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    xf.broadcast_div(&(norm + eps as f64)?)?.to_dtype(x.dtype())
}

// Helper: iterate the contiguous expert runs in the sorted routing.
fn expert_runs(eid: &[u32]) -> Vec<(usize, usize, usize)> {
    // returns (expert, j0, j1) runs over the sorted expert-id slice.
    let mut runs = Vec::new();
    let mut j = 0usize;
    while j < eid.len() {
        let e = eid[j] as usize;
        let j0 = j;
        while j < eid.len() && eid[j] as usize == e {
            j += 1;
        }
        runs.push((e, j0, j));
    }
    runs
}

/// mul_mat_id-style fused gate+up SwiGLU (the llama.cpp/ollama CPU MoE shape):
/// quantize each token's activation ONCE, then ONE rayon region over every
/// (slot, out_row) output - each job runs the gate and up `dot`s directly
/// on the expert row's quantized blocks. Replaces the per-expert QMatMul path,
/// which re-quantized the same activation once per routed expert (kx per
/// layer) and opened 2k tiny parallel regions per layer.
/// `Ok(None)` -> caller falls back (dtype mismatch / no dot path).
fn fused_gate_up_silu_cpu(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quantized::GgmlDType;
    let dt = gate_weights.dtype();
    if up_weights.dtype() != dt {
        return Ok(None);
    }
    let dims = gate_weights.shape().dims();
    if dims.len() != 3 {
        return Ok(None);
    }
    let (n, k) = (dims[1], dims[2]);
    // PREFILL: route Q4_K experts through the tiled per-expert GEMM (weight reuse
    // across the routed-token batch) when the batch amortizes the repack (avg
    // tokens/expert >= 4). Bit-identical to the per-column path below; decode and
    // tiny batches stay per-column (the repack would not pay off).
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        let runs = expert_runs(eid);
        if dt == GgmlDType::Q4K && !runs.is_empty() {
            if sti.len() / runs.len() >= 4 {
                if let Some(h) = fused_gate_up_silu_q4k_tiled(
                    input,
                    gate_weights,
                    up_weights,
                    sti,
                    eid,
                    topk,
                    n,
                    k,
                )? {
                    return Ok(Some(h));
                }
            } else {
                // Decode / few-tokens-per-expert: interleaved 8x8 GEMV (ported
                // llama.cpp kernel, bit-identical to dot).
                if let Some(h) = fused_gate_up_silu_q4k_gemv_v2(
                    input,
                    gate_weights,
                    up_weights,
                    sti,
                    eid,
                    topk,
                    n,
                    k,
                )? {
                    return Ok(Some(h));
                }
            }
        }
    }
    macro_rules! dispatch {
        ($($q:ident => $t:ty),+ $(,)?) => {
            match dt {
                $(GgmlDType::$q => fused_gate_up_silu_t::<$t>(input, gate_weights, up_weights, sti, eid, topk, n, k),)+
                _ => Ok(None),
            }
        }
    }
    use crate::tensor::quant_cpu::*;
    dispatch!(Q4_0 => BlockQ4_0, Q5_0 => BlockQ5_0, Q8_0 => BlockQ8_0,
              Q2K => BlockQ2K, Q3K => BlockQ3K, Q4K => BlockQ4K,
              Q5K => BlockQ5K, Q6K => BlockQ6K, MxFp4 => BlockMxFp4)
}

fn fused_gate_up_silu_t<T: crate::tensor::quant_cpu::BlockFormat>(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    if !k.is_multiple_of(T::BLOCK_LEN)
        || !k.is_multiple_of(
            <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::BLOCK_LEN,
        )
    {
        return Ok(None);
    }
    let m = sti.len();
    let blocks_per_row = k / T::BLOCK_LEN;
    let gd = gate_weights.data()?;
    let ud = up_weights.data()?;
    // typed views over the expert stacks (expert-major / row-major blocks)
    let gw: &[T] = unsafe {
        std::slice::from_raw_parts(gd.as_ptr() as *const T, gd.len() / std::mem::size_of::<T>())
    };
    let uw: &[T] = unsafe {
        std::slice::from_raw_parts(ud.as_ptr() as *const T, ud.len() / std::mem::size_of::<T>())
    };
    // quantize each distinct token row ONCE
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let vblk = <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::BLOCK_LEN;
    let xq: Vec<Vec<T::ActivationBlock>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![
                <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::zeros();
                k / vblk
            ];
            <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    crate::tensor::quant_cpu::pool_for_each_mut(&mut h, 64, &|i, out| {
        let (j, r) = (i / n, i % n);
        let e = eid[j] as usize;
        let t = (sti[j] as usize) / topk;
        let row = (e * n + r) * blocks_per_row;
        let x = &xq[t];
        let g = T::dot(&gw[row..row + blocks_per_row], x);
        let u = T::dot(&uw[row..row + blocks_per_row], x);
        *out = u * (g / (1.0 + (-g).exp()));
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// Send+Sync raw f32 pointer for the disjoint-shard writes in the tiled kernels.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
struct SendF32(*mut f32);
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
unsafe impl Send for SendF32 {}
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
unsafe impl Sync for SendF32 {}

/// Bit-identical tiled prefill for Q4_K gate/up experts. Replaces the per-column
/// `dot` of `fused_gate_up_silu_t` with the repacked 8-column tiled GEMM
/// (`repack_q4k::gemm_group_avx2`, proven `maxabs==0` vs `dot`): each
/// expert's weight is repacked once and reused across its routed-token batch,
/// the reuse the per-column loop misses. Same `xq` (activation quantized once)
/// and the SAME silu formula, so the result is bit-identical to the per-column
/// path - greedy tokens unchanged. `Ok(None)` when the shape cannot tile.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_gate_up_silu_q4k_tiled(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, BlockQ8K};
    if !n.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let m = sti.len();
    if m == 0 {
        return Ok(Some(Tensor::zeros_on((0, n), DType::F32, &input.device())?));
    }
    let bpr = k / 256;
    let e_count = gate_weights.shape().dims()[0];
    let gsz = (n / 8) * bpr; // repacked blocks per expert
    let gate_all = repacked_q4k(gate_weights, e_count, n, bpr)?;
    let up_all = repacked_q4k(up_weights, e_count, n, bpr)?;
    // Quantize each token's activation ONCE (identical to fused_gate_up_silu_t).
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let xq: Vec<Vec<BlockQ8K>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    let hptr = SendF32(h.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = n / 8;
    let empty: &[BlockQ8K] = &[];
    for (e, j0, j1) in expert_runs(eid) {
        let te = j1 - j0;
        let gate_x8 = &gate_all[e * gsz..(e + 1) * gsz];
        let up_x8 = &up_all[e * gsz..(e + 1) * gsz];
        let ntiles = te.div_ceil(8);
        pool.run(ntiles * ngroups, &|wk| {
            let hptr = &hptr;
            let (tt, cg) = (wk / ngroups, wk % ngroups);
            let r0 = tt * 8;
            let mt = (te - r0).min(8);
            let mut acts: [&[BlockQ8K]; 8] = [empty; 8];
            for (i, a) in acts.iter_mut().enumerate().take(mt) {
                let t = sti[j0 + r0 + i] as usize / topk;
                *a = &xq[t];
            }
            let gate_bgrp = &gate_x8[cg * bpr..(cg + 1) * bpr];
            let up_bgrp = &up_x8[cg * bpr..(cg + 1) * bpr];
            let mut gl = [0f32; 64];
            let mut ul = [0f32; 64];
            unsafe {
                repack_q4k::gemm_group_avx2(gate_bgrp, &acts[..mt], bpr, &mut gl);
                repack_q4k::gemm_group_avx2(up_bgrp, &acts[..mt], bpr, &mut ul);
            }
            for i in 0..mt {
                let slot = j0 + r0 + i;
                for c in 0..8 {
                    let g = gl[i * 8 + c];
                    let u = ul[i * 8 + c];
                    // EXACT same formula as the per-column kernel.
                    unsafe {
                        *hptr.0.add(slot * n + cg * 8 + c) = u * (g / (1.0 + (-g).exp()));
                    }
                }
            }
        });
    }
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// DECODE (M=1 / few-tokens-per-expert) Q4_K gate+up: the ported llama.cpp
/// interleaved 8x8 GEMV, reading the SAME plain-scales repack the tiled prefill
/// GEMM uses (`gemv_group_avx2_plain` + the shared `repacked_q4k` cache - one
/// repacked copy per stack instead of two). One SINGLE coarse-chunked pool.run over every
/// (slot, colgroup) pair (matching the plain path's orchestration - per-expert
/// pool.run was a measured dead end). Bit-identical to the per-column path.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_gate_up_silu_q4k_gemv_v2(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, BlockQ8K};
    if !n.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let m = sti.len();
    if m == 0 {
        return Ok(Some(Tensor::zeros_on((0, n), DType::F32, &input.device())?));
    }
    let bpr = k / 256;
    let e_count = gate_weights.shape().dims()[0];
    let gsz = (n / 8) * bpr;
    let gate_all = repacked_q4k(gate_weights, e_count, n, bpr)?;
    let up_all = repacked_q4k(up_weights, e_count, n, bpr)?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let xq: Vec<Vec<BlockQ8K>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    let hptr = SendF32(h.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = n / 8;
    let total = m * ngroups;
    let nthreads = pool.threads.max(1);
    let chunk = total.div_ceil(nthreads).max(1);
    let nchunks = total.div_ceil(chunk);
    pool.run(nchunks, &|c| {
        let hptr = &hptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(total);
        for wk in lo..hi {
            let (slot, cg) = (wk / ngroups, wk % ngroups);
            let e = eid[slot] as usize;
            let t = sti[slot] as usize / topk;
            let aq = &xq[t];
            let gate_grp = &gate_all[e * gsz + cg * bpr..e * gsz + (cg + 1) * bpr];
            let up_grp = &up_all[e * gsz + cg * bpr..e * gsz + (cg + 1) * bpr];
            let mut gl = [0f32; 8];
            let mut ul = [0f32; 8];
            unsafe {
                repack_q4k::gemv_group_avx2_plain(gate_grp, aq, bpr, &mut gl);
                repack_q4k::gemv_group_avx2_plain(up_grp, aq, bpr, &mut ul);
            }
            for cc in 0..8 {
                let g = gl[cc];
                let u = ul[cc];
                unsafe {
                    *hptr.0.add(slot * n + cg * 8 + cc) = u * (g / (1.0 + (-g).exp()));
                }
            }
        }
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// [`fused_gate_up_silu_q4k_gemv_v2`] for the FUSED gate/up stack `[E, 2N, K]`
/// (rows 0..N = gate, N..2N = up per expert): one repacked buffer, the gate row
/// group for output column-group `cg` is group `cg`, the up group is `N/8 + cg`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_gate_up_silu_fused_q4k_gemv_v2(
    input: &Tensor,
    gate_up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, BlockQ8K};
    if !n.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let m = sti.len();
    if m == 0 {
        return Ok(Some(Tensor::zeros_on((0, n), DType::F32, &input.device())?));
    }
    let bpr = k / 256;
    let e_count = gate_up_weights.shape().dims()[0];
    let esz = (2 * n / 8) * bpr;
    let all = repacked_q4k(gate_up_weights, e_count, 2 * n, bpr)?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let xq: Vec<Vec<BlockQ8K>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    let hptr = SendF32(h.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = n / 8;
    let total = m * ngroups;
    let nthreads = pool.threads.max(1);
    let chunk = total.div_ceil(nthreads).max(1);
    let nchunks = total.div_ceil(chunk);
    pool.run(nchunks, &|c| {
        let hptr = &hptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(total);
        for wk in lo..hi {
            let (slot, cg) = (wk / ngroups, wk % ngroups);
            let e = eid[slot] as usize;
            let t = sti[slot] as usize / topk;
            let aq = &xq[t];
            let gate_grp = &all[e * esz + cg * bpr..e * esz + (cg + 1) * bpr];
            let up_grp = &all[e * esz + (ngroups + cg) * bpr..e * esz + (ngroups + cg + 1) * bpr];
            let mut gl = [0f32; 8];
            let mut ul = [0f32; 8];
            unsafe {
                repack_q4k::gemv_group_avx2_plain(gate_grp, aq, bpr, &mut gl);
                repack_q4k::gemv_group_avx2_plain(up_grp, aq, bpr, &mut ul);
            }
            for cc in 0..8 {
                let g = gl[cc];
                let u = ul[cc];
                unsafe {
                    *hptr.0.add(slot * n + cg * 8 + cc) = u * (g / (1.0 + (-g).exp()));
                }
            }
        }
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// Like `fused_gate_up_silu_cpu` but for a FUSED gate‖up expert stack
/// `[E, 2N, K]` (rows 0..N = gate, N..2N = up per expert) - the qwen3-coder /
/// FusedMoeGGUF layout. Same mul_mat_id shape: quantize each token once, ONE
/// work-stealing region over every (slot, out_row), running gate+up `dot`s
/// directly on the expert's quantized rows. Replaces the per-expert concurrent
/// QMatMul path that left most cores idle at M=1 decode.
fn fused_gate_up_silu_fused_cpu(
    input: &Tensor,
    gate_up_weights: &Arc<QTensor>, // [E, 2N, K]
    sti: &[u32],
    eid: &[u32],
    topk: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quantized::GgmlDType;
    let dt = gate_up_weights.dtype();
    let dims = gate_up_weights.shape().dims();
    if dims.len() != 3 || !dims[1].is_multiple_of(2) {
        return Ok(None);
    }
    let (n, k) = (dims[1] / 2, dims[2]);
    // Decode / few-tokens-per-expert: interleaved 8x8 GEMV on the fused stack (same ported
    // llama.cpp kernel as the separate-stack path; bit-identical to the dot below).
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        let runs = expert_runs(eid);
        if dt == GgmlDType::Q4K && !runs.is_empty() && sti.len() / runs.len() < 4 {
            if let Some(h) =
                fused_gate_up_silu_fused_q4k_gemv_v2(input, gate_up_weights, sti, eid, topk, n, k)?
            {
                return Ok(Some(h));
            }
        }
    }
    macro_rules! dispatch {
        ($($q:ident => $t:ty),+ $(,)?) => {
            match dt {
                $(GgmlDType::$q => fused_gate_up_silu_fused_t::<$t>(input, gate_up_weights, sti, eid, topk, n, k),)+
                _ => Ok(None),
            }
        }
    }
    use crate::tensor::quant_cpu::*;
    dispatch!(Q4_0 => BlockQ4_0, Q5_0 => BlockQ5_0, Q8_0 => BlockQ8_0,
              Q2K => BlockQ2K, Q3K => BlockQ3K, Q4K => BlockQ4K,
              Q5K => BlockQ5K, Q6K => BlockQ6K, MxFp4 => BlockMxFp4)
}

fn fused_gate_up_silu_fused_t<T: crate::tensor::quant_cpu::BlockFormat>(
    input: &Tensor,
    gate_up_weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    if !k.is_multiple_of(T::BLOCK_LEN)
        || !k.is_multiple_of(<T::ActivationBlock as BlockFormat>::BLOCK_LEN)
    {
        return Ok(None);
    }
    let m = sti.len();
    let two_n = 2 * n;
    let blocks_per_row = k / T::BLOCK_LEN;
    let wd = gate_up_weights.data()?;
    let w: &[T] = unsafe {
        std::slice::from_raw_parts(wd.as_ptr() as *const T, wd.len() / std::mem::size_of::<T>())
    };
    // quantize each distinct token row ONCE
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let vblk = <T::ActivationBlock as BlockFormat>::BLOCK_LEN;
    let xq: Vec<Vec<T::ActivationBlock>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![<T::ActivationBlock as BlockFormat>::zeros(); k / vblk];
            <T::ActivationBlock as BlockFormat>::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    // NOTE: a two-single-stream variant (sweep all gate rows, then all
    // up rows, to avoid the two far-apart gate/up DRAM streams per output element)
    // was measured NEUTRAL at decode (8.6/8.7 tok/s, unchanged) -> the HW prefetcher
    // handles the 2 streams fine; the CPU-MoE residual vs ollama is NOT this access
    // pattern (nor the dot kernel, which is instruction-identical to llama.cpp).
    crate::tensor::quant_cpu::pool_for_each_mut(&mut h, 64, &|i, out| {
        let (j, r) = (i / n, i % n);
        let e = eid[j] as usize;
        let t = (sti[j] as usize) / topk;
        let g_row = (e * two_n + r) * blocks_per_row; // gate row r
        let u_row = (e * two_n + n + r) * blocks_per_row; // up row r
        let x = &xq[t];
        let g = T::dot(&w[g_row..g_row + blocks_per_row], x);
        let u = T::dot(&w[u_row..u_row + blocks_per_row], x);
        *out = u * (g / (1.0 + (-g).exp()));
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// Public entry: fused gate‖up `[E,2N,K]` SwiGLU (silu) -> `[M, N]`. Falls back to
/// the per-expert path when the quant type has no CPU dot.
pub fn moe_gemm_gguf_gate_up_silu_mul_fused(
    input: &Tensor,
    gate_up_weights: &Arc<QTensor>, // [E, 2N, K]
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    let dev = input.device();
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = expert_ids.to_vec1()?;
    if let Some(h) = fused_gate_up_silu_fused_cpu(input, gate_up_weights, &sti, &eid, topk)? {
        return Ok(h);
    }
    // Fallback: dequantize the fused stack and run per-expert F32 matmul.
    let gu = gate_up_weights.dequantize(&dev)?; // [E, 2N, K]
    let two_n = gu.dim(1)?;
    let n = two_n / 2;
    let m = sorted_token_ids.dim(0)?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?;
    let h = Tensor::zeros_on((m, n), DType::F32, &dev)?;
    for (e, j0, j1) in expert_runs(&eid) {
        let toks: Vec<u32> = (j0..j1).map(|s| sti[s] / topk as u32).collect();
        let sel = Tensor::from_vec(toks, (j1 - j0,), &dev)?;
        let x_e = xs.index_select(&sel, 0)?.contiguous()?;
        let gu_e = gu.i(e)?; // [2N, K]
        let ge = gu_e.narrow(0, 0, n)?; // [N, K]
        let ue = gu_e.narrow(0, n, n)?; // [N, K]
        let g = x_e.matmul_t(&ge)?;
        let u = x_e.matmul_t(&ue)?;
        let he = (u * crate::tensor::ops::silu(&g)?)?;
        h.slice_set(&he, 0, j0)?;
    }
    Ok(h)
}

/// Biased clamped-SwiGLU-OAI variant of the fused gate+up path (gpt-oss).
fn fused_gate_up_oai_t<T: crate::tensor::quant_cpu::BlockFormat>(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    gb: &[f32],
    ub: &[f32],
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
    alpha: f32,
    limit: f32,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    if !k.is_multiple_of(T::BLOCK_LEN)
        || !k.is_multiple_of(<T::ActivationBlock as BlockFormat>::BLOCK_LEN)
    {
        return Ok(None);
    }
    let m = sti.len();
    let blocks_per_row = k / T::BLOCK_LEN;
    let gd = gate_weights.data()?;
    let ud = up_weights.data()?;
    let gw: &[T] = unsafe {
        std::slice::from_raw_parts(gd.as_ptr() as *const T, gd.len() / std::mem::size_of::<T>())
    };
    let uw: &[T] = unsafe {
        std::slice::from_raw_parts(ud.as_ptr() as *const T, ud.len() / std::mem::size_of::<T>())
    };
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let vblk = <T::ActivationBlock as BlockFormat>::BLOCK_LEN;
    let xq: Vec<Vec<T::ActivationBlock>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![<T::ActivationBlock as BlockFormat>::zeros(); k / vblk];
            <T::ActivationBlock as BlockFormat>::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    crate::tensor::quant_cpu::pool_for_each_mut(&mut h, 64, &|i, out| {
        let (j, r) = (i / n, i % n);
        let e = eid[j] as usize;
        let t = (sti[j] as usize) / topk;
        let row = (e * n + r) * blocks_per_row;
        let x = &xq[t];
        let g = T::dot(&gw[row..row + blocks_per_row], x) + gb[e * n + r];
        let u = T::dot(&uw[row..row + blocks_per_row], x) + ub[e * n + r];
        let xg = g.min(limit);
        let gg = u.clamp(-limit, limit);
        let act = xg * (1.0 / (1.0 + (-(xg * alpha)).exp()));
        *out = act * (gg + 1.0);
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// gate/up expert GEMM + SwiGLU: for each sorted slot j, h[j] = silu(gate[e].x).(up[e].x).
pub fn moe_gemm_gguf_gate_up_silu_mul(
    input: &Tensor,              // [n_tok, hidden] F32
    gate_weights: &Arc<QTensor>, // [E, N, K]
    up_weights: &Arc<QTensor>,   // [E, N, K]
    sorted_token_ids: &Tensor,   // [M] u32 (flattened pair idx, sorted by expert)
    expert_ids: &Tensor,         // [M] u32
    topk: usize,
) -> Result<Tensor> {
    let dev = input.device();
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = expert_ids.to_vec1()?;
    if let Some(h) = fused_gate_up_silu_cpu(input, gate_weights, up_weights, &sti, &eid, topk)? {
        return Ok(h);
    }
    let qg = expert_qmms(gate_weights);
    let qu = expert_qmms(up_weights);
    let (gate_all, up_all) = if qg.is_none() || qu.is_none() {
        (
            Some(gate_weights.dequantize(&dev)?),
            Some(up_weights.dequantize(&dev)?),
        )
    } else {
        (None, None)
    };
    let n = gate_weights.shape().dims()[1];
    let m = sorted_token_ids.dim(0)?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?;
    let h = Tensor::zeros_on((m, n), DType::F32, &dev)?;
    for (e, j0, j1) in expert_runs(&eid) {
        let toks: Vec<u32> = (j0..j1).map(|s| sti[s] / topk as u32).collect();
        let sel = Tensor::from_vec(toks, (j1 - j0,), &dev)?;
        let x_e = xs.index_select(&sel, 0)?.contiguous()?; // [te, K]
        let g = match &qg {
            Some(q) => q[e].forward(&x_e)?,
            None => x_e.matmul(&gate_all.as_ref().unwrap().i(e)?.t()?)?,
        };
        let u = match &qu {
            Some(q) => q[e].forward(&x_e)?,
            None => x_e.matmul(&up_all.as_ref().unwrap().i(e)?.t()?)?,
        };
        let he = (u * crate::tensor::ops::silu(&g)?)?; // [te, N]
        h.slice_set(&he, 0, j0)?;
    }
    Ok(h)
}

/// gate/up expert GEMM + gpt-oss clamped SwiGLU-OAI (+ per-expert biases).
pub fn moe_gemm_gguf_gate_up_swiglu_oai(
    input: &Tensor,
    gate_weights: &Arc<QTensor>,
    up_weights: &Arc<QTensor>,
    gate_bias: Option<&Tensor>,
    up_bias: Option<&Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    alpha: f64,
    limit: f64,
) -> Result<Tensor> {
    let dev = input.device();
    let n = gate_weights.shape().dims()[1];
    let k_dim = gate_weights.shape().dims()[2];
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = expert_ids.to_vec1()?;
    if let (Some(gbt), Some(ubt)) = (gate_bias, up_bias) {
        if gate_weights.dtype() == up_weights.dtype() {
            let gb: Vec<f32> = gbt.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let ub: Vec<f32> = ubt.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            use crate::tensor::quant_cpu::*;
            macro_rules! disp { ($($q:ident => $t:ty),+ $(,)?) => { match gate_weights.dtype() {
                $(GgmlDType::$q => fused_gate_up_oai_t::<$t>(input, gate_weights, up_weights, &gb, &ub, &sti, &eid, topk, n, k_dim, alpha as f32, limit as f32)?,)+
                _ => None, } } }
            if let Some(h) = disp!(Q4_0 => BlockQ4_0, Q5_0 => BlockQ5_0, Q8_0 => BlockQ8_0,
                Q2K => BlockQ2K, Q3K => BlockQ3K, Q4K => BlockQ4K,
                Q5K => BlockQ5K, Q6K => BlockQ6K, MxFp4 => BlockMxFp4)
            {
                return Ok(h);
            }
        }
    }
    let qg = expert_qmms(gate_weights);
    let qu = expert_qmms(up_weights);
    let (gate_all, up_all) = if qg.is_none() || qu.is_none() {
        (
            Some(gate_weights.dequantize(&dev)?),
            Some(up_weights.dequantize(&dev)?),
        )
    } else {
        (None, None)
    };
    let m = sorted_token_ids.dim(0)?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?;
    let h = Tensor::zeros_on((m, n), DType::F32, &dev)?;
    for (e, j0, j1) in expert_runs(&eid) {
        let toks: Vec<u32> = (j0..j1).map(|s| sti[s] / topk as u32).collect();
        let sel = Tensor::from_vec(toks, (j1 - j0,), &dev)?;
        let x_e = xs.index_select(&sel, 0)?.contiguous()?;
        let mut g = match &qg {
            Some(q) => q[e].forward(&x_e)?,
            None => x_e.matmul(&gate_all.as_ref().unwrap().i(e)?.t()?)?,
        };
        let mut u = match &qu {
            Some(q) => q[e].forward(&x_e)?,
            None => x_e.matmul(&up_all.as_ref().unwrap().i(e)?.t()?)?,
        };
        if let Some(gb) = gate_bias {
            g = g.broadcast_add(&gb.i(e)?.to_dtype(DType::F32)?)?;
        }
        if let Some(ub) = up_bias {
            u = u.broadcast_add(&ub.i(e)?.to_dtype(DType::F32)?)?;
        }
        let lim = Tensor::full(limit as f32, (1,), &g.device())?;
        let neg = Tensor::full(-limit as f32, (1,), &u.device())?;
        let x = g.broadcast_minimum(&lim)?;
        let gg = u.broadcast_maximum(&neg)?.broadcast_minimum(&lim)?;
        let act = (&x * crate::tensor::ops::sigmoid(&(&x * alpha)?)?)?;
        let he = (act * (gg + 1.0)?)?;
        h.slice_set(&he, 0, j0)?;
    }
    Ok(h)
}

/// Plain expert GEMM (single weight, optional per-slot topk weighting).
/// Fused UP + relu² for gateless MoE (nemotron_h's up->relu²->down): ONE
/// work-stealing region over (slot, out-row), each job = one dot + relu².
/// Replaces both the sequential per-expert loop in `moe_gemm_gguf` (~5 tensor
/// dispatches per expert per token, experts computed one after another) AND
/// the separate relu+sqr pass - the same fusion `fused_gate_up_silu` gives the
/// swiglu archs. `Ok(None)` -> caller falls back to the sequential path.
fn fused_up_relu2_cpu(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quantized::GgmlDType;
    let dt = weights.dtype();
    let dims = weights.shape().dims();
    if dims.len() != 3 {
        return Ok(None);
    }
    let (n, k) = (dims[1], dims[2]);
    macro_rules! dispatch {
        ($($q:ident => $t:ty),+ $(,)?) => {
            match dt {
                $(GgmlDType::$q => fused_up_relu2_t::<$t>(input, weights, sti, eid, topk, n, k),)+
                _ => Ok(None),
            }
        }
    }
    use crate::tensor::quant_cpu::*;
    dispatch!(Q4_0 => BlockQ4_0, Q5_0 => BlockQ5_0, Q8_0 => BlockQ8_0,
              Q2K => BlockQ2K, Q3K => BlockQ3K, Q4K => BlockQ4K,
              Q5K => BlockQ5K, Q6K => BlockQ6K, MxFp4 => BlockMxFp4)
}

fn fused_up_relu2_t<T: crate::tensor::quant_cpu::BlockFormat>(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    topk: usize,
    n: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    if !k.is_multiple_of(T::BLOCK_LEN)
        || !k.is_multiple_of(
            <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::BLOCK_LEN,
        )
    {
        return Ok(None);
    }
    let m = sti.len();
    let blocks_per_row = k / T::BLOCK_LEN;
    let wd = weights.data()?;
    let w: &[T] = unsafe {
        std::slice::from_raw_parts(wd.as_ptr() as *const T, wd.len() / std::mem::size_of::<T>())
    };
    // quantize each distinct token row ONCE
    let xs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let vblk = <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::BLOCK_LEN;
    let xq: Vec<Vec<T::ActivationBlock>> = xs
        .iter()
        .map(|row| {
            let mut q = vec![
                <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::zeros();
                k / vblk
            ];
            <T::ActivationBlock as crate::tensor::quant_cpu::BlockFormat>::quantize(row, &mut q);
            q
        })
        .collect();
    let mut h = vec![0f32; m * n];
    crate::tensor::quant_cpu::pool_for_each_mut(&mut h, 64, &|i, out| {
        let (j, r) = (i / n, i % n);
        let e = eid[j] as usize;
        let t = (sti[j] as usize) / topk;
        let row = (e * n + r) * blocks_per_row;
        let u = T::dot(&w[row..row + blocks_per_row], &xq[t]);
        let rl = if u > 0.0 { u } else { 0.0 };
        *out = rl * rl;
    });
    Ok(Some(Tensor::from_vec(h, (m, n), &input.device())?))
}

/// UP + relu² for gateless MoE: fused work-stealing path with a sequential
/// per-expert fallback (`moe_gemm_gguf` + tensor relu²).
pub fn moe_gemm_gguf_up_relu2(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = experts_ids.to_vec1()?;
    if let Some(h) = fused_up_relu2_cpu(input, weights, &sti, &eid, topk)? {
        return Ok(h);
    }
    let up = moe_gemm_gguf(
        input,
        weights,
        &None,
        sorted_token_ids,
        experts_ids,
        topk,
        false,
        DType::F32,
    )?;
    up.relu()?.sqr()
}

pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &Arc<QTensor>,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    _is_prefill: bool,
    _dtype: DType,
) -> Result<Tensor> {
    let dev = input.device();
    let qw = expert_qmms(weights);
    let w_all = if qw.is_none() {
        Some(weights.dequantize(&dev)?)
    } else {
        None
    }; // [E, N, K]
    let n = weights.shape().dims()[1];
    let m = sorted_token_ids.dim(0)?;
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = experts_ids.to_vec1()?;
    let xs = input.to_dtype(DType::F32)?.contiguous()?;
    let tw: Option<Vec<f32>> = match topk_weights {
        Some(t) => Some(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?),
        None => None,
    };
    let out = Tensor::zeros_on((m, n), DType::F32, &dev)?;
    for (e, j0, j1) in expert_runs(&eid) {
        let toks: Vec<u32> = (j0..j1).map(|s| sti[s] / topk as u32).collect();
        let sel = Tensor::from_vec(toks, (j1 - j0,), &dev)?;
        let x_e = xs.index_select(&sel, 0)?.contiguous()?;
        let mut y = match &qw {
            Some(q) => q[e].forward(&x_e)?,
            None => x_e.matmul(&w_all.as_ref().unwrap().i(e)?.t()?)?,
        }; // [te, N]
        if let Some(w) = &tw {
            let wv: Vec<f32> = (j0..j1).map(|s| w[sti[s] as usize]).collect();
            y = y.broadcast_mul(&Tensor::from_vec(wv, (j1 - j0, 1), &dev)?)?;
        }
        out.slice_set(&y, 0, j0)?;
    }
    Ok(out)
}

/// down expert GEMM + weighted top-k reduction back to [n_tokens, hidden].
/// mul_mat_id-style fused down projection + weighted reduce: each slot's GLU
/// output is quantized ONCE, then one rayon region over (token, hidden_row)
/// outputs - each job sums tw[j].dot(down[e_j][row], h_q[j]) over the
/// token's slots, on top of the residual. `Ok(None)` -> caller falls back.
fn fused_down_reduce_cpu(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    use crate::tensor::quantized::GgmlDType;
    let dt = weights.dtype();
    let dims = weights.shape().dims();
    if dims.len() != 3 {
        return Ok(None);
    }
    let (hidden, k) = (dims[1], dims[2]);
    // PREFILL: tiled per-expert down GEMM (see gate_up rationale). lfm2moe ships
    // both Q4_K and Q6_K `ffn_down` expert stacks across its layers, so tile both.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        let runs = expert_runs(eid);
        if !runs.is_empty() && sti.len() / runs.len() >= 4 {
            let tiled = match dt {
                GgmlDType::Q6K => fused_down_reduce_q6k_tiled(
                    input,
                    weights,
                    sti,
                    eid,
                    tw,
                    topk,
                    n_real_tokens,
                    residual,
                    hidden,
                    k,
                )?,
                GgmlDType::Q4K => fused_down_reduce_q4k_tiled(
                    input,
                    weights,
                    sti,
                    eid,
                    tw,
                    topk,
                    n_real_tokens,
                    residual,
                    hidden,
                    k,
                )?,
                _ => None,
            };
            if let Some(o) = tiled {
                return Ok(Some(o));
            }
        } else if !runs.is_empty() && dt == GgmlDType::Q4K {
            // Decode: interleaved 8x8 GEMV down + reduce (ported llama.cpp kernel).
            if let Some(o) = fused_down_reduce_q4k_gemv_v2(
                input,
                weights,
                sti,
                eid,
                tw,
                topk,
                n_real_tokens,
                residual,
                hidden,
                k,
            )? {
                return Ok(Some(o));
            }
        } else if !runs.is_empty() && dt == GgmlDType::Q6K {
            // Decode: 8-column repacked group dot for Q6K ffn_down experts
            // (single-row tile of the oracle-validated tiled kernel) - the
            // generic per-element fallback below cost ~2x on this path.
            if let Some(o) = fused_down_reduce_q6k_gemv(
                input,
                weights,
                sti,
                eid,
                tw,
                topk,
                n_real_tokens,
                residual,
                hidden,
                k,
            )? {
                return Ok(Some(o));
            }
        }
    }
    macro_rules! dispatch {
        ($($q:ident => $t:ty),+ $(,)?) => {
            match dt {
                $(GgmlDType::$q => fused_down_reduce_t::<$t>(input, weights, sti, eid, tw, topk, n_real_tokens, residual, hidden, k),)+
                _ => Ok(None),
            }
        }
    }
    use crate::tensor::quant_cpu::*;
    dispatch!(Q4_0 => BlockQ4_0, Q5_0 => BlockQ5_0, Q8_0 => BlockQ8_0,
              Q2K => BlockQ2K, Q3K => BlockQ3K, Q4K => BlockQ4K,
              Q5K => BlockQ5K, Q6K => BlockQ6K, MxFp4 => BlockMxFp4)
}

fn fused_down_reduce_t<T: crate::tensor::quant_cpu::BlockFormat>(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    hidden: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    if !k.is_multiple_of(T::BLOCK_LEN)
        || !k.is_multiple_of(<T::ActivationBlock as BlockFormat>::BLOCK_LEN)
    {
        return Ok(None);
    }
    let m = sti.len();
    let blocks_per_row = k / T::BLOCK_LEN;
    let wd = weights.data()?;
    let w: &[T] = unsafe {
        std::slice::from_raw_parts(wd.as_ptr() as *const T, wd.len() / std::mem::size_of::<T>())
    };
    // quantize each slot's GLU row once
    let hs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let vblk = <T::ActivationBlock as BlockFormat>::BLOCK_LEN;
    let hq: Vec<Vec<T::ActivationBlock>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![<T::ActivationBlock as BlockFormat>::zeros(); k / vblk];
            <T::ActivationBlock as BlockFormat>::quantize(row, &mut q);
            q
        })
        .collect();
    // token -> its slot indices
    let mut slots_of: Vec<Vec<usize>> = vec![Vec::with_capacity(topk); n_real_tokens];
    for j in 0..m {
        let t = (sti[j] as usize) / topk;
        if t < n_real_tokens {
            slots_of[t].push(j);
        }
    }
    let mut out = match residual {
        Some(r) => r
            .reshape((n_real_tokens, hidden))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        None => vec![0f32; n_real_tokens * hidden],
    };
    crate::tensor::quant_cpu::pool_for_each_mut(&mut out, 64, &|i, o| {
        let (t, r) = (i / hidden, i % hidden);
        for &j in &slots_of[t] {
            let e = eid[j] as usize;
            let row = (e * hidden + r) * blocks_per_row;
            *o += tw[sti[j] as usize] * T::dot(&w[row..row + blocks_per_row], &hq[j]);
        }
    });
    Ok(Some(Tensor::from_vec(
        out,
        (n_real_tokens, hidden),
        &input.device(),
    )?))
}

/// Bit-identical tiled prefill for Q6_K down experts. The tiled GEMM
/// (`repack_q6k::gemm_group_avx2`, `maxabs==0` vs `dot`) gets the weight-reuse
/// win; the per-token reduction keeps the per-column kernel's EXACT order (each
/// token sums its slots in ascending-slot order). Single traversal: `expert_runs`
/// yields runs in ascending slot index and the outer loop is sequential, so
/// accumulating `tw*dot` straight into `out` lands each token's topk contributions
/// in ascending-slot (== per-column `slots_of`) order - bit-identical to a separate
/// reduce, with the `dscaled` (m*hidden f32) round-trip removed. `hq` quantized once.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_down_reduce_q6k_tiled(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    hidden: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q6k, BlockQ8K};
    if !hidden.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let bpr = k / 256;
    let e_count = weights.shape().dims()[0];
    let gsz = (hidden / 8) * bpr;
    let down_all = repacked_q6k(weights, e_count, hidden, bpr)?;
    // Quantize each slot's GLU row ONCE (identical to fused_down_reduce_t).
    let hs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let hq: Vec<Vec<BlockQ8K>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    // Single-pass fused reduce: accumulate `tw*dot` straight into `out` in
    // ascending-slot order (== per-column `slots_of` order) - bit-identical.
    let mut out = match residual {
        Some(r) => r
            .reshape((n_real_tokens, hidden))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        None => vec![0f32; n_real_tokens * hidden],
    };
    let optr = SendF32(out.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = hidden / 8;
    let empty: &[BlockQ8K] = &[];
    for (e, j0, j1) in expert_runs(eid) {
        let te = j1 - j0;
        let down_x8 = &down_all[e * gsz..(e + 1) * gsz];
        let ntiles = te.div_ceil(8);
        pool.run(ntiles * ngroups, &|wk| {
            let optr = &optr;
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
                repack_q6k::gemm_group_avx2(bgrp, &acts[..mt], bpr, &mut dl);
            }
            for i in 0..mt {
                let slot = j0 + r0 + i;
                let t = (sti[slot] as usize) / topk;
                if t >= n_real_tokens {
                    continue;
                }
                let twv = tw[sti[slot] as usize];
                for c in 0..8 {
                    // tw * dot, SAME operand order and SAME accumulation order
                    // (ascending slot within a token) as the per-column kernel.
                    unsafe {
                        *optr.0.add(t * hidden + cg * 8 + c) += twv * dl[i * 8 + c];
                    }
                }
            }
        });
    }
    Ok(Some(Tensor::from_vec(
        out,
        (n_real_tokens, hidden),
        &input.device(),
    )?))
}

/// Bit-identical tiled prefill for Q4_K down experts - identical structure to
/// `fused_down_reduce_q6k_tiled` (some lfm2moe layers ship a Q4_K `ffn_down`
/// stack rather than Q6_K), using the Q4_K repack + tiled GEMM.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_down_reduce_q4k_tiled(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    hidden: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, BlockQ8K};
    if !hidden.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let bpr = k / 256;
    let e_count = weights.shape().dims()[0];
    let gsz = (hidden / 8) * bpr;
    let down_all = repacked_q4k(weights, e_count, hidden, bpr)?;
    let hs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let hq: Vec<Vec<BlockQ8K>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    // Single-pass fused reduce (see fused_down_reduce_q6k_tiled): accumulate
    // `tw*dot` straight into `out` in ascending-slot order - bit-identical.
    let mut out = match residual {
        Some(r) => r
            .reshape((n_real_tokens, hidden))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        None => vec![0f32; n_real_tokens * hidden],
    };
    let optr = SendF32(out.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = hidden / 8;
    let empty: &[BlockQ8K] = &[];
    for (e, j0, j1) in expert_runs(eid) {
        let te = j1 - j0;
        let down_x8 = &down_all[e * gsz..(e + 1) * gsz];
        let ntiles = te.div_ceil(8);
        pool.run(ntiles * ngroups, &|wk| {
            let optr = &optr;
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
                repack_q4k::gemm_group_avx2(bgrp, &acts[..mt], bpr, &mut dl);
            }
            for i in 0..mt {
                let slot = j0 + r0 + i;
                let t = (sti[slot] as usize) / topk;
                if t >= n_real_tokens {
                    continue;
                }
                let twv = tw[sti[slot] as usize];
                for c in 0..8 {
                    unsafe {
                        *optr.0.add(t * hidden + cg * 8 + c) += twv * dl[i * 8 + c];
                    }
                }
            }
        });
    }
    Ok(Some(Tensor::from_vec(
        out,
        (n_real_tokens, hidden),
        &input.device(),
    )?))
}

/// DECODE Q4_K down + topk-weighted reduce via the ported interleaved 8x8 GEMV.
/// Parallelizes over COLGROUPS (not experts) so each worker owns a disjoint set
/// of output columns and accumulates every expert's tw*dot into them serially -
/// no cross-expert write race, ONE pool.run. Bit-identical to the per-column path.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
/// Decode down-projection + topk reduce for Q6K expert stacks: the ffn_down
/// of several MoE arches ships Q6K, which previously fell to the GENERIC
/// per-output-element path (closure dispatch per element, on rayon - fighting
/// the gemv pool). Mirrors [`fused_down_reduce_q4k_gemv_v2`]: repacked
/// 8-column groups on the gemv pool, with the oracle-validated
/// `repack_q6k::gemm_group_avx2` kernel run as a single-row tile.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn fused_down_reduce_q6k_gemv(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    hidden: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q6k, BlockQ8K};
    if !hidden.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let bpr = k / 256;
    let e_count = weights.shape().dims()[0];
    let gsz = (hidden / 8) * bpr;
    let down_all = repacked_q6k(weights, e_count, hidden, bpr)?;
    let hs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let hq: Vec<Vec<BlockQ8K>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut out = match residual {
        Some(r) => r
            .reshape((n_real_tokens, hidden))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        None => vec![0f32; n_real_tokens * hidden],
    };
    let optr = SendF32(out.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = hidden / 8;
    let m = sti.len();
    let nthreads = pool.threads.max(1);
    let chunk = ngroups.div_ceil(nthreads).max(1);
    let nchunks = ngroups.div_ceil(chunk);
    pool.run(nchunks, &|c| {
        let optr = &optr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(ngroups);
        for cg in lo..hi {
            for slot in 0..m {
                let e = eid[slot] as usize;
                let t = sti[slot] as usize / topk;
                if t >= n_real_tokens {
                    continue;
                }
                let twv = tw[sti[slot] as usize];
                let bgrp = &down_all[e * gsz + cg * bpr..e * gsz + (cg + 1) * bpr];
                let acts: [&[BlockQ8K]; 1] = [&hq[slot]];
                let mut dl = [0f32; 8];
                unsafe { repack_q6k::gemm_group_avx2(bgrp, &acts, bpr, &mut dl) };
                for cc in 0..8 {
                    unsafe {
                        *optr.0.add(t * hidden + cg * 8 + cc) += twv * dl[cc];
                    }
                }
            }
        }
    });
    Ok(Some(Tensor::from_vec(
        out,
        (n_real_tokens, hidden),
        &input.device(),
    )?))
}

fn fused_down_reduce_q4k_gemv_v2(
    input: &Tensor,
    weights: &Arc<QTensor>,
    sti: &[u32],
    eid: &[u32],
    tw: &[f32],
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    hidden: usize,
    k: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::quant_cpu::BlockFormat;
    use crate::tensor::quant_cpu::{gemv_pool, repack_q4k, BlockQ8K};
    if !hidden.is_multiple_of(8) || !k.is_multiple_of(256) {
        return Ok(None);
    }
    let bpr = k / 256;
    let e_count = weights.shape().dims()[0];
    let gsz = (hidden / 8) * bpr;
    let down_all = repacked_q4k(weights, e_count, hidden, bpr)?;
    let hs = input.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?;
    let hq: Vec<Vec<BlockQ8K>> = hs
        .iter()
        .map(|row| {
            let mut q = vec![BlockQ8K::zeros(); bpr];
            BlockQ8K::quantize(row, &mut q);
            q
        })
        .collect();
    let mut out = match residual {
        Some(r) => r
            .reshape((n_real_tokens, hidden))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        None => vec![0f32; n_real_tokens * hidden],
    };
    let optr = SendF32(out.as_mut_ptr());
    let pool = gemv_pool::pool();
    let ngroups = hidden / 8;
    let m = sti.len();
    let nthreads = pool.threads.max(1);
    let chunk = ngroups.div_ceil(nthreads).max(1);
    let nchunks = ngroups.div_ceil(chunk);
    pool.run(nchunks, &|c| {
        let optr = &optr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(ngroups);
        for cg in lo..hi {
            for slot in 0..m {
                let e = eid[slot] as usize;
                let t = sti[slot] as usize / topk;
                if t >= n_real_tokens {
                    continue;
                }
                let twv = tw[sti[slot] as usize];
                let bgrp = &down_all[e * gsz + cg * bpr..e * gsz + (cg + 1) * bpr];
                let mut dl = [0f32; 8];
                unsafe { repack_q4k::gemv_group_avx2_plain(bgrp, &hq[slot], bpr, &mut dl) };
                for cc in 0..8 {
                    unsafe {
                        *optr.0.add(t * hidden + cg * 8 + cc) += twv * dl[cc];
                    }
                }
            }
        }
    });
    Ok(Some(Tensor::from_vec(
        out,
        (n_real_tokens, hidden),
        &input.device(),
    )?))
}

pub fn moe_gemm_gguf_down_reduce(
    input: &Tensor,            // [M, N] F32 (the gate/up GLU output, sorted)
    weights: &Arc<QTensor>,    // [E, hidden, N]
    sorted_token_ids: &Tensor, // [M] u32
    expert_ids: &Tensor,       // [M] u32
    topk_weights: &Tensor,     // [n_tok, topk] F32
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    down_bias: Option<&Tensor>,
) -> Result<Tensor> {
    // The routed FFN runs through this fused kernel, not through matmul_bytes,
    // so the engine's work counter saw 0.06 GMAC/token on a model spending 0.4
    // CPU-seconds on it. Count it where it actually happens.
    {
        use std::sync::atomic::Ordering::Relaxed;
        let (m, n) = (input.dims()[0], input.dims()[1]);
        let hidden = weights.shape().dims()[1];
        crate::tensor::quant_cpu::CPU_MACS
            .fetch_add((m as u64) * (n as u64) * (hidden as u64), Relaxed);
        crate::tensor::quant_cpu::CPU_GEMV_CALLS.fetch_add(1, Relaxed);
        // Only the experts this token selected are read, not the whole table.
        crate::tensor::quant_cpu::CPU_BYTES
            .fetch_add(((m as u64) * (n as u64) * (hidden as u64)) / 2, Relaxed);
    }
    let dev = input.device();
    let hidden = weights.shape().dims()[1];
    let sti: Vec<u32> = sorted_token_ids.to_vec1()?;
    let eid: Vec<u32> = expert_ids.to_vec1()?;
    let tw: Vec<f32> = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    if down_bias.is_none() {
        if let Some(o) = fused_down_reduce_cpu(
            input,
            weights,
            &sti,
            &eid,
            &tw,
            topk,
            n_real_tokens,
            residual,
        )? {
            return Ok(o);
        }
    }
    let qd = expert_qmms(weights);
    let down_all = if qd.is_none() {
        Some(weights.dequantize(&dev)?)
    } else {
        None
    }; // [E, hidden, N]
    let mut out = match residual {
        Some(r) => r.reshape((n_real_tokens, hidden))?.to_dtype(DType::F32)?,
        None => Tensor::zeros_on((n_real_tokens, hidden), DType::F32, &dev)?,
    };
    for (e, j0, j1) in expert_runs(&eid) {
        let h_e = input.narrow(0, j0, j1 - j0)?.contiguous()?; // [te, N]
        let mut d = match &qd {
            Some(q) => q[e].forward(&h_e)?,
            None => h_e.matmul(&down_all.as_ref().unwrap().i(e)?.t()?)?,
        }; // [te, hidden]
        if let Some(db) = down_bias {
            d = d.broadcast_add(&db.i(e)?.to_dtype(DType::F32)?)?;
        }
        let toks: Vec<u32> = (j0..j1).map(|s| sti[s] / topk as u32).collect();
        let wv: Vec<f32> = (j0..j1).map(|s| tw[sti[s] as usize]).collect();
        let d = d.broadcast_mul(&Tensor::from_vec(wv, (j1 - j0, 1), &dev)?)?;
        let sel = Tensor::from_vec(toks, (j1 - j0,), &dev)?;
        out = out.index_add(&sel, &d, 0)?;
    }
    Ok(out)
}

pub fn topk_softmax(
    logits: &Tensor,
    n_expert_used: usize,
    with_norm: bool,
) -> Result<(Tensor, Tensor)> {
    let probs = crate::tensor::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
    let ids = probs
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, n_expert_used)?
        .contiguous()?;
    let mut w = probs.gather(&ids, D::Minus1)?;
    if with_norm {
        w = w.broadcast_div(&w.sum_keepdim(D::Minus1)?)?;
    }
    Ok((w, ids))
}

// -- fused decode kernels (CUDA-only optimizations; CPU uses the slow path) --
// In qwen3_moe_multi these fire ONLY under `... && device.is_cuda()` guards, so on
// a CPU build they're dead code that must still type-check. `add_rms_norm` is
// trivially correct so we implement it; the two `attn_post_qkv_decode*` fused
// q-norm+RoPE kernels have CUDA-specific layout/RoPE-style assumptions, so rather
// than risk a latent numerical mismatch we `bail!` - the unfused tensor path
// (q_norm.forward + rotary_emb.apply, same file) produces the real CPU output.

/// `(a+b, rms_norm(a+b).gamma)` - fused residual-add + RMSNorm over the last dim.
pub fn add_rms_norm(a: &Tensor, b: &Tensor, gamma: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    let xs = (a + b)?;
    let var = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xs
        .broadcast_div(&(var + eps as f64)?.sqrt()?)?
        .broadcast_mul(gamma)?;
    Ok((xs, normed))
}

pub fn attn_post_qkv_decode(
    _qkv: &Tensor,
    _q_norm_w: &Tensor,
    _k_norm_w: &Tensor,
    _rope_cos: &Tensor,
    _rope_sin: &Tensor,
    _n_q: usize,
    _n_kv: usize,
    _hd: usize,
    _rope_pos: usize,
    _rms_eps: f32,
    _q_scale: f32,
    _out_dtype: DType,
    _rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    crate::tensor::bail!(
        "attn_post_qkv_decode is a CUDA decode fast-path; CPU uses the unfused q_norm+RoPE path"
    )
}

pub fn attn_post_qkv_decode_qf32(
    _qkv: &Tensor,
    _q_norm_w: &Tensor,
    _k_norm_w: &Tensor,
    _rope_cos: &Tensor,
    _rope_sin: &Tensor,
    _n_q: usize,
    _n_kv: usize,
    _hd: usize,
    _rope_pos: usize,
    _rms_eps: f32,
    _q_scale: f32,
    _out_dtype: DType,
    _rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    crate::tensor::bail!(
        "attn_post_qkv_decode_qf32 is a CUDA decode fast-path; CPU uses the unfused path"
    )
}

#[cfg(all(test, target_feature = "avx2", target_arch = "x86_64"))]
mod down_reduce_bit_identity;
