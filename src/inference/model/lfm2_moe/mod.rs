//! lfm2moe (LiquidAI LFM2-MoE) - hybrid short-convolution + attention + MoE.
//!
//! Block layout (standard 2-sublayer transformer block):
//!   x = x + operator(operator_norm(x))      // operator = shortconv | attention
//!   x = x + ffn(ffn_norm(x))                 // ffn = dense SwiGLU | MoE SwiGLU
//!
//! - **shortconv** (LFM2 gated short conv, NOT Mamba2 SSM): `in_proj` -> split
//!   (B, C, x) -> `bx = B.x` -> causal depthwise conv1d (kernel `l_cache`) -> `C.conv`
//!   -> `out_proj`. Per-block rolling conv state, reset at `input_pos == 0`.
//! - **attention**: QKV -> per-head QK RMS-norm -> full NEOX rope -> GQA -> softmax.
//! - **MoE**: DeepSeek-style SIGMOID router (`exp_probs_b` bias = selection only,
//!   unbiased weights, normalized over top-k) -> SwiGLU experts `down(silu(gate).up)`.
//! - **dense FFN** (the leading `n_layer_dense_lead` blocks): SwiGLU.
//!
//! Operator kind per block is detected by tensor presence (attn_q ⇒ attention,
//! else shortconv); ffn kind by `block_idx >= leading_dense_block_count`.

use crate::tensor::layer::Embedding;
use crate::tensor::ops::{heads_first, host_f32};
use crate::tensor::quantized::{gguf_file, QMatMul, QTensor};
use crate::tensor::{DType, Device, IndexOp, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;
use std::sync::OnceLock;

fn lfm2_devpos_enabled() -> bool {
    false
}

/// Production CUDA-graph decode (single-GPU). Implies devpos (the device-resident
/// state the graph needs). Gated; default OFF until validated + measured.
fn lfm2_graph_enabled() -> bool {
    false
}

/// Load a matmul weight, keeping native block-quant types the MoE/MMVQ kernels
/// accept; requantize only float / MXFP4 / unsupported types to Q8_0.
/// Optional mmap of the source GGUF: CPU-placed passthrough tensors become
/// zero-copy file views instead of heap copies.
type Mm<'a> = Option<&'a std::sync::Arc<memmap2::Mmap>>;

fn load_q8<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
    mm: Mm<'_>,
) -> Result<QTensor> {
    use crate::tensor::quantized::GgmlDType;
    let passthrough = c
        .tensor_infos
        .get(name)
        .map(|i| crate::inference::moe_cuda::gemm::expert_kernels_serve(i.ggml_dtype))
        .unwrap_or(false);
    if passthrough {
        if let (Some(m), true) = (mm, d.is_cpu()) {
            if let Some(qt) = crate::tensor::quant_view::gguf_mmap_view(c, m, name)? {
                return Ok(qt);
            }
        }
        return c.tensor(r, name, d);
    }
    let cpu = c.tensor(r, name, &Device::Cpu)?;
    let f = cpu.dequantize(&Device::Cpu)?;
    QTensor::quantize_onto(&f, GgmlDType::Q8_0, d)
}

fn ld_f32<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
) -> Result<Tensor> {
    c.tensor(r, name, d)?.dequantize(d)
}

#[derive(Debug, Clone)]
pub struct Lfm2Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub dense_ff: usize,
    pub leading_dense: usize,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f64,
    pub rms_eps: f64,
    pub context_length: usize,
    pub rope_freq_base: f32,
    pub l_cache: usize, // shortconv kernel size
}

impl Lfm2Config {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        // Accept both the MoE arch (`lfm2moe.*`) and the dense arch (`lfm2.*`,
        // e.g. lfm2.5-thinking - same hybrid shortconv+attn block layout, just a
        // dense SwiGLU FFN with no experts -> all blocks load as `Ffn::Dense`).
        let g = |k: &str| {
            ct.metadata
                .get(&format!("lfm2moe.{k}"))
                .or_else(|| ct.metadata.get(&format!("lfm2.{k}")))
        };
        let req_u = |k: &str| -> Result<usize> {
            g(k).and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| crate::tensor::Error::msg(format!("lfm2moe: missing lfm2moe.{k}")))
        };
        let opt_u = |k: &str, d: usize| {
            g(k).and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let opt_f = |k: &str, d: f32| g(k).and_then(|v| v.to_f32().ok()).unwrap_or(d);
        Ok(Self {
            n_layers: req_u("block_count")?,
            d_model: req_u("embedding_length")?,
            n_head: opt_u("attention.head_count", 0),
            head_dim: opt_u("attention.key_length", 64),
            n_expert: opt_u("expert_count", 0),
            n_expert_used: opt_u("expert_used_count", 0),
            expert_ff: opt_u("expert_feed_forward_length", 0),
            dense_ff: opt_u("feed_forward_length", 0),
            leading_dense: opt_u("leading_dense_block_count", 0),
            // lfm2moe hardcodes weight-norm on; scale defaults to 1 (no-op).
            expert_weights_norm: ct
                .metadata
                .get("lfm2moe.expert_weights_norm")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(true),
            expert_weights_scale: {
                let s = opt_f("expert_weights_scale", 1.0) as f64;
                if s == 0.0 {
                    1.0
                } else {
                    s
                }
            },
            rms_eps: opt_f("attention.layer_norm_rms_epsilon", 1e-5) as f64,
            context_length: opt_u("context_length", 32768),
            rope_freq_base: opt_f("rope.freq_base", 1_000_000.0),
            l_cache: opt_u("shortconv.l_cache", 3),
        })
    }
}

/// Per-head RMS norm over head_dim (weight `[head_dim]`).
fn head_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // x: [b, n_head, seq, head_dim] F32
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    xn.broadcast_mul(w)
}

/// Fused CPU per-head RMS norm: one flat parallel pass per (b.head.seq) row,
/// bit-identical to `head_rms` above. The tensor-op form issues ~6 allocating
/// ops per call (sqr, mean, +eps, sqrt, two strided `broadcast_binary` passes);
/// on CPU prefill the q/k-norm on the attention layers was the bulk of the
/// "misc elementwise" time. Replicates head_rms EXACTLY: `ss = Σx²` in element
/// order (== sqr + last-dim reduce from 0.0), `var = ss / hd`, `denom =
/// sqrt(var + eps_f32)` (`+eps` maps to affine with `eps as f32`), then
/// `out = (x / denom) * w` - DIVISION then weight-multiply, not the
/// reciprocal-multiply of `Tensor::rms_norm` (which would differ by ULPs).
fn head_rms_cpu(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let hd = *dims
        .last()
        .ok_or_else(|| crate::tensor::Error("head_rms_cpu on rank-0".into()))?;
    let wv = w.flatten_all()?.to_vec1::<f32>()?;
    let mut out = x.flatten_all()?.to_vec1::<f32>()?;
    let norm_row = |row: &mut [f32]| {
        let mut ss = 0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let denom = (ss / hd as f32 + eps).sqrt();
        for (o, &wt) in row.iter_mut().zip(wv.iter()) {
            *o = (*o / denom) * wt;
        }
    };
    if out.len() == hd {
        norm_row(&mut out);
    } else {
        use rayon::prelude::*;
        out.par_chunks_mut(hd).for_each(norm_row);
    }
    Tensor::from_vec(out, dims, &x.device())
}

/// Per-head RMSNorm via the fused `head_rmsnorm` kernel (1 launch vs ~6 separate tensor
/// ops). x [b, heads, seq, hd] (F16/F32); w [hd] F32. lfm2 attn is launch-bound;
/// this matches qwen3.5's head_rms_fused. portable fallback off-CUDA.
fn head_rms_fused(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    if x.device().is_cuda() {
        // F16-stream fast path: reuse fused_rmsnorm_f16 (F16 in/out, F32-internal
        // accumulate, F32 weight) which normalizes over the last dim (= head_dim)  -
        // identical RMS math but WITHOUT the F16->F32 + F32->F16 cast pair that
        // wrapped the F32 head_rmsnorm kernel. lfm2 decode is launch-bound (nsys
        // ~68% util); QK-norm fires 2x/attn-layer so this drops ~40
        // cast launches/token. Shape is preserved by fused_rmsnorm_f16.
        if x.dtype() == DType::F16 {
            return crate::inference::kernel::fused::fused_rmsnorm_f16(x, w, eps as f32);
        }
        let dims = x.dims().to_vec();
        let hd = *dims.last().unwrap();
        let n = x.elem_count() / hd;
        let st = x.dtype();
        let xf = x.to_dtype(DType::F32)?.reshape((n, hd))?;
        let y = crate::inference::moe_cuda::head_rmsnorm(&xf, w, eps as f32)?;
        return y.reshape(dims)?.to_dtype(st);
    }
    let st = x.dtype();
    // CPU: fused single-pass per-head RMS (bit-identical to head_rms), keeping
    // the same F16↔F32 cast pair. Off-CUDA the q/k-norm was ~6 allocating
    // tensor ops (two strided broadcast_binary) - the bulk of the CPU-prefill
    // "misc elementwise" bucket. See head_rms_cpu for the bit-identity argument.
    if matches!(x.device(), crate::tensor::Device::Cpu) {
        return head_rms_cpu(&x.to_dtype(DType::F32)?, w, eps as f32)?.to_dtype(st);
    }
    head_rms(&x.to_dtype(DType::F32)?, w, eps)?.to_dtype(st)
}

/// LFM2 attention: QKV -> per-head QK RMS-norm -> full rope -> GQA -> softmax.
pub struct Lfm2Attn {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: Tensor, // [head_dim]
    k_norm: Tensor, // [head_dim]
    cos: Tensor,
    sin: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
    // Fixed-size cache: slice_set writes the new token at its position (O(1)/token)
    // instead of ConcatKvCache's per-token cat that re-copied the whole KV (O(N)/
    // token -> O(N²) over a generation; catastrophic at long context). Grows in
    // 4096-token steps beyond the initial budget. Also graph-friendlier.
    kv_cache: crate::tensor::KvCache,
    // Device-position KV ring buffers for the CUDA-graph decode path (gated by
    // LOKEN_LFM2_DEVPOS). Fixed-max [b,n_kv,kv_max,hd]; the write slot lives on
    // device (pos_dev) so a captured graph advances it on replay. None until first
    // decode token allocates them. Validated bit-identical to the KvCache path.
    kbuf: Option<Tensor>,
    vbuf: Option<Tensor>,
    kv_max: usize,
    cos_f16: Tensor, // full [max_seq, head_dim/2] F16 - for device-position rope
    sin_f16: Tensor,
    /// CPU decode fast path (qwen3-coder recipe): f16 GQA KV + one-pass
    /// online-softmax attention; a fused Rust post-QKV pass kills the per-token
    /// reshape/transpose/norm/rope/sdpa dispatch chain. KvCache stays mirrored
    /// (O(1) slice_set append) so prefill/multi-token history is untouched.
    cpu_f16_kv: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    /// (q_norm, k_norm) as flat f32 [head_dim] - lazy, for the fused pass.
    norm_f32: Option<(Vec<f32>, Vec<f32>)>,
    /// Full rope tables as flat f32 [max_seq * head_dim/2] - lazy.
    cos_sin_f32: Option<(Vec<f32>, Vec<f32>)>,
}

impl Lfm2Attn {
    #[allow(clippy::too_many_arguments)]
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Lfm2Config,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let wk_qt = load_q8(c, r, &format!("{p}.attn_k.weight"), device, mm)?;
        // The config gives how many query heads there are and how wide one is; how many
        // key/value heads the layer writes comes from the K projection's own width.
        let (n_head, head_dim) = (cfg.n_head, cfg.head_dim);
        let n_kv_head = (wk_qt.shape().dims()[0] / head_dim.max(1)).max(1);
        // One number for both KV stores: the tensor cache's initial capacity and the ring
        // buffer's fixed maximum are the same window, and they must not drift apart.
        let kv_window = cfg.context_length.min(4096).max(256);
        Ok(Self {
            wq: QMatMul::from_qtensor(load_q8(c, r, &format!("{p}.attn_q.weight"), device, mm)?)?,
            wk: QMatMul::from_qtensor(wk_qt)?,
            wv: QMatMul::from_qtensor(load_q8(c, r, &format!("{p}.attn_v.weight"), device, mm)?)?,
            wo: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.attn_output.weight"),
                device,
                mm,
            )?)?,
            q_norm: ld_f32(c, r, &format!("{p}.attn_q_norm.weight"), device)?,
            k_norm: ld_f32(c, r, &format!("{p}.attn_k_norm.weight"), device)?,
            cos_f16: cos.to_dtype(DType::F16)?,
            sin_f16: sin.to_dtype(DType::F16)?,
            cos,
            sin,
            n_head,
            n_kv_head,
            head_dim,
            rms_eps: cfg.rms_eps,
            kv_cache: crate::tensor::KvCache::new(2, kv_window),
            kbuf: None,
            vbuf: None,
            kv_max: kv_window,
            cpu_f16_kv: None,
            norm_f32: None,
            cos_sin_f32: None,
        })
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.kbuf = None;
        self.vbuf = None;
        if let Some(c) = self.cpu_f16_kv.as_mut() {
            c.reset();
        }
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        input_pos: usize,
        shared_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dev = x.device().clone();

        // CPU decode fast path: keep the 3 GEMVs, then one fused Rust
        // pass (shared-weight head rmsnorm + full-neox rope + q-scale) feeding
        // the f16 GQA cache and its single-pass online-softmax attention.
        // Replaces reshape/transpose/contiguous x3 + head_rms x2 + rope x2 +
        // the 8-dispatch sdpa chain per layer per token. KvCache mirror kept
        // (O(1) slice_set) for prefill/multi-token paths.
        if seq == 1 && b == 1 && dev.is_cpu() {
            let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
            let half = hd / 2;
            let qv = host_f32(&self.wq.forward(x)?)?;
            let kv_ = host_f32(&self.wk.forward(x)?)?;
            let vv = host_f32(&self.wv.forward(x)?)?;
            // Weights and tables are read once per layer, not once per token: neither the
            // per-head norms nor the rope tables change between calls.
            if self.norm_f32.is_none() {
                self.norm_f32 = Some((host_f32(&self.q_norm)?, host_f32(&self.k_norm)?));
            }
            if self.cos_sin_f32.is_none() {
                self.cos_sin_f32 = Some((host_f32(&self.cos)?, host_f32(&self.sin)?));
            }
            let (qn_w, kn_w) = self.norm_f32.as_ref().unwrap();
            let (cos_t, sin_t) = self.cos_sin_f32.as_ref().unwrap();
            let cos = &cos_t[input_pos * half..(input_pos + 1) * half];
            let sin = &sin_t[input_pos * half..(input_pos + 1) * half];
            let eps = self.rms_eps as f32;
            let q_scale = (1.0 / (hd as f64).sqrt()) as f32;
            let mut qf = vec![0f32; nh * hd];
            let mut kf = vec![0f32; nkv * hd];
            let do_head = |src: &[f32], w: &[f32], dst: &mut [f32], scale: f32| {
                let ss: f32 = src.iter().map(|&v| v * v).sum();
                let inv = 1.0f32 / (ss / hd as f32 + eps).sqrt();
                for i in 0..half {
                    let a = src[i] * inv * w[i];
                    let bq = src[i + half] * inv * w[i + half];
                    dst[i] = (a * cos[i] - bq * sin[i]) * scale;
                    dst[i + half] = (a * sin[i] + bq * cos[i]) * scale;
                }
            };
            for h in 0..nh {
                do_head(
                    &qv[h * hd..(h + 1) * hd],
                    qn_w,
                    &mut qf[h * hd..(h + 1) * hd],
                    q_scale,
                );
            }
            for h in 0..nkv {
                do_head(
                    &kv_[h * hd..(h + 1) * hd],
                    kn_w,
                    &mut kf[h * hd..(h + 1) * hd],
                    1.0,
                );
            }
            // NOTE: the tensor `kv_cache` mirror is NOT maintained here at decode
            // - CPU decode attention reads only `cpu_f16_kv` below, and the
            // seed loop already reconstructs the prefill history from `kv_cache`
            // (populated by the seq>1 prefill path) exactly once. Skipping the
            // per-token `from_vec + to_dtype + slice_set` mirror (audit item #2,
            // dead work at decode) - measured. (Speculative/PLD
            // multi-token verify is not on this seq==1 path.)
            if self.cpu_f16_kv.is_none() {
                self.cpu_f16_kv = Some(crate::inference::cache::cpu_f16_kv::CpuF16Kv::new(
                    nh, nkv, hd, None,
                ));
            }
            // Prefill wrote only the tensor cache - seed the f16 store from it
            // once (input_pos already includes the current token's mirror
            // above, so the gap is everything before this token).
            if self.cpu_f16_kv.as_ref().unwrap().len() < input_pos {
                let start = self.cpu_f16_kv.as_ref().unwrap().len();
                if let (Some(kall), Some(vall)) = (self.kv_cache.k()?, self.kv_cache.v()?) {
                    let (kv3, vv3) = (host_f32(&kall)?, host_f32(&vall)?);
                    let total = kall.dim(2)?; // [1, nkv, total, hd]
                    let cache = self.cpu_f16_kv.as_mut().unwrap();
                    let mut kt = vec![0f32; nkv * hd];
                    let mut vt = vec![0f32; nkv * hd];
                    for t in start..input_pos.min(total) {
                        for h in 0..nkv {
                            let src = h * total * hd + t * hd;
                            kt[h * hd..(h + 1) * hd].copy_from_slice(&kv3[src..src + hd]);
                            vt[h * hd..(h + 1) * hd].copy_from_slice(&vv3[src..src + hd]);
                        }
                        cache.append(&kt, &vt)?;
                    }
                }
            }
            let cache = self.cpu_f16_kv.as_mut().unwrap();
            cache.append(&kf, &vv)?;
            let mut out = vec![0f32; nh * hd];
            cache.attention(&qf, 1.0, &mut out)?; // q pre-scaled above
            let out_t = Tensor::from_vec(out, (1, 1, nh * hd), &dev)?;
            let o = self.wo.forward(&out_t)?;
            // Downstream (residual add, operator_norm) runs in the arch's
            // working dtype (F16 on CPU here) - match it.
            return if o.dtype() == x.dtype() {
                Ok(o)
            } else {
                o.to_dtype(x.dtype())
            };
        }

        // Three projections, one rule, so only the head count differs between them.
        let project = |w: &QMatMul, heads: usize| -> Result<Tensor> {
            heads_first(w.forward(x)?, heads, self.head_dim)
        };
        let q = project(&self.wq, self.n_head)?;
        let k = project(&self.wk, self.n_kv_head)?;
        let v = project(&self.wv, self.n_kv_head)?;
        // per-head QK RMS-norm (F32) then back to working dtype
        let _st = q.dtype();
        let q = head_rms_fused(&q, &self.q_norm, self.rms_eps)?;
        let k = head_rms_fused(&k, &self.k_norm, self.rms_eps)?;
        // CUDA-graph devpos path shares ONE device position scalar across rope +
        // KV write (a captured graph advances it on replay). Default path uses the
        // host-narrowed cos/sin + portable KvCache. Both bit-identical.
        let devpos = lfm2_devpos_enabled()
            && dev.is_cuda()
            && q.dtype() == DType::F16
            && input_pos + seq <= self.kv_max;
        // Prefer the model's SHARED pos counter (same device) so a captured graph
        // advances every layer from one buffer; else a fresh per-layer scalar.
        let pos_dev = if devpos {
            match shared_pos {
                Some(p) if p.device().location() == dev.location() => Some(p.clone()),
                _ => Some(Tensor::new(&[input_pos as i32], &dev)?),
            }
        } else {
            None
        };
        let (q, k) = if let Some(pd) = pos_dev.as_ref() {
            // device-position rope off the FULL F16 cos/sin tables
            use crate::inference::kernel::fused::neox_rope_devpos_f16 as rd;
            match (
                rd(&q, &self.cos_f16, &self.sin_f16, pd, self.head_dim)?,
                rd(&k, &self.cos_f16, &self.sin_f16, pd, self.head_dim)?,
            ) {
                (Some(q), Some(k)) => (q, k),
                _ => crate::tensor::bail!("lfm2 devpos rope failed"),
            }
        } else {
            // F16 working dtype: narrow the PRE-CAST F16 tables (zero-copy view;
            // `cast(narrow(cos))` == `narrow(cast(cos))` elementwise) instead of
            // paying a per-token pack-copy + cast launch for cos and sin each.
            let (cos, sin) = if q.dtype() == DType::F16 {
                (
                    self.cos_f16.narrow(0, input_pos, seq)?,
                    self.sin_f16.narrow(0, input_pos, seq)?,
                )
            } else {
                (
                    self.cos.narrow(0, input_pos, seq)?.to_dtype(q.dtype())?,
                    self.sin.narrow(0, input_pos, seq)?.to_dtype(q.dtype())?,
                )
            };
            // Fused full-NEOX rope (one launch vs contiguous+rope) on decode, bit-exact.
            let fr = |t: &Tensor| -> Result<Tensor> {
                if seq == 1 && cos.dtype() == DType::F16 {
                    if let Some(o) = crate::inference::kernel::fused::neox_rope_f16(
                        t,
                        &cos,
                        &sin,
                        self.head_dim,
                    )? {
                        return Ok(o);
                    }
                }
                crate::tensor::ops::rope(&t.contiguous()?, &cos, &sin)
            };
            (fr(&q)?, fr(&k)?)
        };
        // KV cache: device-position ring buffer (gated, CUDA-graph keystone) OR the
        // default the reference KV cache. The devpos path is bit-identical (a plain copy at
        // the same slot) - validates the kernel before the graph capture/replay.
        let (k, v, kv_len) = if let Some(pd) = pos_dev.as_ref() {
            // Writes ALL `seq` positions, so prefill (seq>1) populates the prompt KV
            // and decode (seq==1) appends - the prefill population was the bug.
            if self.kbuf.is_none() {
                self.kbuf = Some(Tensor::zeros_on(
                    (b, self.n_kv_head, self.kv_max, self.head_dim),
                    DType::F16,
                    &dev,
                )?);
                self.vbuf = Some(Tensor::zeros_on(
                    (b, self.n_kv_head, self.kv_max, self.head_dim),
                    DType::F16,
                    &dev,
                )?);
            }
            let (kbuf, vbuf) = (self.kbuf.as_ref().unwrap(), self.vbuf.as_ref().unwrap());
            crate::inference::kernel::fused::kv_write_at_pos(&k, &v, kbuf, vbuf, pd, self.kv_max)?;
            // Decode (seq==1): device-kv_len flash-decode over the FULL ring buffer
            // - kv_len = *pos_dev+1 read on device, so a captured graph attends the
            // correct growing count on replay (the host-narrow below would freeze).
            if seq == 1 && self.head_dim % 32 == 0 {
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                if let Some(o) = crate::inference::kernel::fused::flash_decode_devkvlen(
                    &q.reshape((b, self.n_head, self.head_dim))?,
                    kbuf,
                    vbuf,
                    pd,
                    None,
                    0,
                    scale as f32,
                    b,
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                )? {
                    return self
                        .wo
                        .forward(&o.reshape((b, seq, self.n_head * self.head_dim))?);
                }
            }
            // Prefill (seq>1, not captured): host-narrow + the matmul chain below.
            let kv_len = input_pos + seq;
            (
                kbuf.narrow(2, 0, kv_len)?,
                vbuf.narrow(2, 0, kv_len)?,
                kv_len,
            )
        } else {
            self.kv_cache.append_write(&k, &v)?;
            let kv_len = self.kv_cache.current_seq_len();
            // Decode flash path: read the FULL backing buffers - the kernel takes
            // an explicit kv_len + per-dim strides, so the per-token narrow-COPY
            // of the whole growing KV cache (`append()`'s current_data, 2 copy
            // launches/layer/token) is pure waste. Bit-identical: same rows read.
            if seq == 1 && dev.is_cuda() && self.head_dim == 64 && q.dtype() == DType::F16 {
                if let (Some(kb), Some(vb)) = (
                    self.kv_cache.k_cache().all_data().clone(),
                    self.kv_cache.v_cache().all_data().clone(),
                ) {
                    let scale = 1.0 / (self.head_dim as f64).sqrt();
                    if let Some(o) = crate::inference::kernel::fused::gptoss_flash_decode(
                        &q.reshape((b, self.n_head, self.head_dim))?,
                        &kb,
                        &vb,
                        None,
                        None,
                        scale as f32,
                        b,
                        self.n_head,
                        self.n_kv_head,
                        kv_len,
                        self.head_dim,
                    )? {
                        let out = o.reshape((b, seq, self.n_head * self.head_dim))?;
                        return self.wo.forward(&out);
                    }
                }
            }
            let (Some(k), Some(v)) = (self.kv_cache.k()?, self.kv_cache.v()?) else {
                crate::tensor::bail!("lfm2 kv cache empty after append")
            };
            (k, v, kv_len)
        };
        // How many query heads share one key/value head: a quotient of two fields this
        // block already holds, so it is taken here rather than stored a second time.
        let g = self.n_head / self.n_kv_head;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Fused flash-decode (seq=1, hd=64): one F16 kernel does scores+softmax+V
        // on the KV-cache narrow - no cuBLAS, no [n_head,kv_len] scores in HBM, so
        // the forward becomes CUDA-graph-capturable. lfm2 has no attention sinks
        // (sinks=None -> online softmax seeded empty). Falls back below otherwise.
        if seq == 1 && dev.is_cuda() && self.head_dim == 64 && q.dtype() == DType::F16 {
            if let Some(o) = crate::inference::kernel::fused::gptoss_flash_decode(
                &q.reshape((b, self.n_head, self.head_dim))?,
                &k,
                &v,
                None,
                None,
                scale as f32,
                b,
                self.n_head,
                self.n_kv_head,
                kv_len,
                self.head_dim,
            )? {
                let out = o.reshape((b, seq, self.n_head * self.head_dim))?;
                return self.wo.forward(&out);
            }
        }
        // GQA without repeat_kv: read K/V once (n_kv_head) not n_head copies.
        // repeat_kv maps head H -> kv H/groups, so reshape q
        // [b,n_head,seq,hd] -> [b,n_kv_head,groups*seq,hd] is the matching group.
        let qg_shape = (b, self.n_kv_head, g * seq, self.head_dim);
        // Prefill: stream the attention instead of building the scores. The
        // score tensor grows with chunk length times context length and is read
        // once by the softmax and once by the product with V; accumulating each
        // query row against the keys directly needs none of it. Rows are
        // independent, so this parallelises the same way. Off this path the
        // chain below is unchanged.
        if seq > 1 {
            let qg = q.reshape(qg_shape)?;
            if let Some(o) =
                crate::tensor::ops::flash_attn_prefill(&qg, &k, &v, scale, input_pos, seq)?
            {
                let out = o
                    .reshape((b, self.n_head, seq, self.head_dim))?
                    .transpose(1, 2)?
                    .reshape((b, seq, self.n_head * self.head_dim))?;
                return self.wo.forward(&out);
            }
        }
        // Keep the scores in the matmul's own dtype here: the widen and the
        // scale are folded into the masked softmax below, which has to walk the
        // tensor anyway. Doing them as separate steps wrote the whole score
        // tensor twice before the softmax had read it once.
        let scores_h = if g > 1 {
            let qg = q.reshape(qg_shape)?;
            qg.matmul(&k.transpose(2, 3)?)?
                .reshape((b, self.n_head, seq, kv_len))?
        } else {
            q.matmul(&k.transpose(2, 3)?)?
        };
        let kv = input_pos + seq;
        // Fold the causal mask INTO the softmax rather than adding it first:
        // `broadcast_add` materialises a whole second [b,n_head,seq,kv] f32
        // tensor per attention layer that softmax then re-reads, which is the
        // bulk of this stage at prefill. The fused form is bit-identical on
        // CPU-F32 and falls back to the unfused pair elsewhere, so the GPU path
        // is unchanged.
        let w = if seq > 1 {
            let mut mask = vec![0f32; seq * kv];
            for i in 0..seq {
                for j in (input_pos + i + 1)..kv {
                    mask[i * kv + j] = f32::NEG_INFINITY;
                }
            }
            let m = Tensor::from_vec(mask, (1, 1, seq, kv), &dev)?;
            crate::tensor::ops::softmax_scaled_masked(&scores_h, scale, &m)?
        } else {
            // Decode reduces a single row; the separate steps cost nothing here.
            let scores = (scores_h.to_dtype(DType::F32)? * scale)?;
            crate::tensor::ops::softmax_last_dim(&scores)?
        }
        .to_dtype(v.dtype())?;
        // The weighted sum of the values, then the head axis back behind the sequence so the
        // output projection reads one row per token.
        let out = match g > 1 {
            true => w
                .reshape((b, self.n_kv_head, g * seq, kv_len))?
                .matmul(&v)?
                .reshape((b, self.n_head, seq, self.head_dim))?,
            false => w.matmul(&v)?,
        };
        self.wo.forward(
            &out.transpose(1, 2)?
                .reshape((b, seq, self.n_head * self.head_dim))?,
        )
    }
}

/// Host-side shortconv state for the fused CPU path (`forward_cpu_fused`):
/// plain `Vec<f32>` so the per-token loop never touches the tensor-op dispatch
/// layer. The rolling state holds the last l_cache-1 B.x products.
#[derive(Clone)]
struct CpuShortConvState {
    conv: Vec<f32>, // [d_model * (l_cache-1)]
    w: Vec<f32>,    // conv weights [d_model * l_cache]
}

/// LFM2 gated short-convolution operator.
pub struct Lfm2ShortConv {
    in_proj: QMatMul,  // d_model -> 3*d_model  (B, C, x)
    out_proj: QMatMul, // d_model -> d_model
    conv_w: Tensor,    // [d_model, l_cache] F32
    d_model: usize,
    l_cache: usize,
    device: Device,
    conv_state: Option<Tensor>, // [b, d_model, l_cache-1]
    cpu_state: Option<CpuShortConvState>,
}

impl Lfm2ShortConv {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Lfm2Config,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        Ok(Self {
            in_proj: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.shortconv.in_proj.weight"),
                device,
                mm,
            )?)?,
            out_proj: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.shortconv.out_proj.weight"),
                device,
                mm,
            )?)?,
            conv_w: ld_f32(c, r, &format!("{p}.shortconv.conv.weight"), device)?, // [d_model, l_cache]
            d_model: cfg.d_model,
            l_cache: cfg.l_cache,
            device: device.clone(),
            conv_state: None,
            cpu_state: None,
        })
    }

    fn reset_state(&mut self, b: usize) -> Result<()> {
        self.conv_state = Some(Tensor::zeros_on(
            (b, self.d_model, self.l_cache - 1),
            DType::F32,
            &self.device,
        )?);
        Ok(())
    }

    pub fn forward(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        // CPU: fused host-side loop (no per-op the tensor-op dispatch).
        if b == 1 && !x.device().is_cuda() && self.l_cache >= 2 {
            if input_pos == 0 {
                self.cpu_state = None;
            }
            return self.forward_cpu_fused(x, seq);
        }
        if input_pos == 0 || self.conv_state.is_none() {
            self.reset_state(b)?;
        }
        // PREFILL (seq>1): batch in_proj into ONE GEMM (d_model->3.d_model, the big
        // projection) instead of `seq` per-token GEMVs that re-stream the weight
        // every token. Same per-token-prefill outlier pattern as nemotron. The conv
        // (stateful) + out_proj branches stay per-token (minimal/safe change - the
        // devpos in-place graph path + f16io conv are untouched). Decode (seq==1)
        // keeps the exact per-token path. Mathematically identical (GEMM row == GEMV).
        let proj_all = if seq > 1 {
            Some(self.in_proj.forward(x)?)
        } else {
            None
        }; // [b,seq,3.d_model] F16
        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let bcx_f16 = match &proj_all {
                Some(p) => p.i((.., t, ..))?, // [b, 3.d_model] (pre-batched)
                None => self.in_proj.forward(&x.i((.., t, ..))?)?, // [b, 3*d_model] (F16 stream)
            };
            let cs = self.conv_state.as_ref().unwrap(); // [b, d_model, l_cache-1] F32
                                                        // CUDA-graph devpos path: update conv_state IN PLACE (no per-token
                                                        // realloc -> no MEM_ALLOC node to block replay). Bit-identical. The
                                                        // fixed buffer is read+written by the in-place-safe shift kernel.
            let devpos_conv = lfm2_devpos_enabled()
                && bcx_f16.device().is_cuda()
                && bcx_f16.dtype() == DType::F16
                && true;
            // Fused gated causal depthwise conv + state shift in one launch:
            // bx = B*X; acc = Σ window.conv_w; y = C*acc; state = shift(state, bx).
            // Replaces ~20 tensor ops with 1. The F16-I/O kernel additionally
            // drops the in_proj->F32 and y->F16 casts (lfm2 is launch-bound); it is
            // bit-identical (same F32 math + F16↔F32 conversions). F32 fallback.
            let out = if devpos_conv {
                match crate::inference::moe_cuda::lfm2_shortconv_f16io_inplace(
                    &bcx_f16,
                    cs,
                    &self.conv_w,
                    self.d_model,
                    self.l_cache,
                )? {
                    Some(y) => self.out_proj.forward(&y)?, // conv_state updated in place
                    None => crate::tensor::bail!("lfm2 devpos shortconv failed"),
                }
            } else if let Some((y, new_state)) = crate::inference::moe_cuda::lfm2_shortconv_f16io(
                &bcx_f16,
                cs,
                &self.conv_w,
                self.d_model,
                self.l_cache,
            )? {
                self.conv_state = Some(new_state);
                self.out_proj.forward(&y)? // y is F16
            } else {
                let bcx = bcx_f16.to_dtype(DType::F32)?;
                let (y, new_state) = crate::inference::kernel::fused::fused_lfm2_shortconv(
                    &bcx,
                    cs,
                    &self.conv_w,
                    self.d_model,
                    self.l_cache,
                )?;
                self.conv_state = Some(new_state);
                self.out_proj.forward(&y.to_dtype(x.dtype())?)?
            };
            outs.push(out.unsqueeze(1)?);
        }
        // Decode (seq==1): skip the cat copy - return the single step directly.
        if outs.len() == 1 {
            return Ok(outs.into_iter().next().unwrap());
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)
    }

    /// Fused CPU path: ONE batched in_proj matmul, a plain-Rust f32 loop for
    /// the gated causal conv (bx = B.x; acc = Σ window.w; y = C.acc; state
    /// roll), then ONE out_proj matmul. Same math and op order as the
    /// fused_lfm2_shortconv reference; the conv itself is tiny (d_model.l_cache
    /// mults/token) - the cost on CPU was entirely the tensor-op dispatch chain.
    fn forward_cpu_fused(&mut self, x: &Tensor, seq: usize) -> Result<Tensor> {
        let (d, l) = (self.d_model, self.l_cache);
        if self.cpu_state.is_none() {
            self.cpu_state = Some(CpuShortConvState {
                conv: vec![0f32; d * (l - 1)],
                w: self
                    .conv_w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            });
        }
        // Read the projection in place. The rank-2 host copy used here before
        // materialized the whole projection twice - once flat, once split into a
        // separate buffer per token - on the calling thread, while the rest of
        // the pool had nothing to do. The conv below only ever reads these
        // values, so borrowing the tensor's elements is equivalent.
        let proj_t = self.in_proj.forward(x)?.to_dtype(DType::F32)?;
        let proj = proj_t.f32_data()?;
        let st = self.cpu_state.as_mut().unwrap();
        let mut y_all = vec![0f32; seq * d];
        for t in 0..seq {
            let row = &proj[t * 3 * d..(t + 1) * 3 * d];
            let (bv, cv, xv) = (&row[..d], &row[d..2 * d], &row[2 * d..]);
            let yt = &mut y_all[t * d..(t + 1) * d];
            for c in 0..d {
                let bx = bv[c] * xv[c];
                let s = &mut st.conv[c * (l - 1)..(c + 1) * (l - 1)];
                let w = &st.w[c * l..(c + 1) * l];
                let mut acc = w[l - 1] * bx;
                for j in 0..l - 1 {
                    acc += w[j] * s[j];
                }
                s.rotate_left(1);
                s[l - 2] = bx;
                yt[c] = cv[c] * acc;
            }
        }
        let y = Tensor::from_vec(y_all, (1, seq, d), &x.device())?.to_dtype(x.dtype())?;
        self.out_proj.forward(&y)
    }
}

/// Router shared by MoE blocks: sigmoid gating, bias for selection only,
/// unbiased normalized weights x scale. Returns (topk_w, topk_ids[n_tok,k]).
fn route(
    gate: &crate::tensor::layer::Linear,
    gate_bias: &Option<Tensor>,
    xs: &Tensor,
    n_used: usize,
    norm: bool,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    // NOTE: a fully-fused sigmoid router kernel (moe_cuda::gate_topk_sigmoid)
    // exists but REGRESSED lfm2 -4% (113.1->108.3): its warp-per-token hand-rolled
    // gate GEMV (1 warp over n_expertxhidden) is much slower than the cuBLAS gemv
    // here, outweighing the launch savings. Kept as latent infra; the right
    // redesign fuses only the POST-matmul ops (sigmoid+bias+topk+gather+norm)
    // taking pre-computed logits, leaving the fast matmul to cuBLAS.
    // CUDA-graph devpos path: use the fully-fused NON-cuBLAS gate GEMV. cuBLAS
    // gemv is the lone non-capture-safe op in the lfm2 decode forward (the NaN-on-
    // replay culprit, gemma4-saga class); the hand-rolled warp-per-token GEMV is
    // slower in normal mode (-4%) but capturable, and in a graph the launch saving
    // dominates anyway.
    if lfm2_devpos_enabled() {
        if let Some(out) = crate::inference::moe_cuda::gate_topk_sigmoid(
            xs,
            gate.weight()?,
            gate_bias.as_ref(),
            n_used,
            norm,
            scale,
        )? {
            return Ok(out);
        }
    }
    let logits = gate.forward(xs)?;
    // Fuse all POST-matmul router ops (sigmoid + bias-select top-k + gather +
    // renorm + scale) into one kernel, keeping the fast cuBLAS gemv above.
    if let Some(out) = crate::inference::moe_cuda::topk_sigmoid_post(
        &logits,
        gate_bias.as_ref(),
        n_used,
        norm,
        scale,
    )? {
        return Ok(out);
    }
    let probs = crate::tensor::ops::sigmoid(&logits)?;
    let sel = match gate_bias {
        Some(g) => probs.broadcast_add(g)?,
        None => probs.clone(),
    };
    let (_sv, sidx) = sel.sort_last_dim(false)?;
    let topk_ids = sidx.narrow(D::Minus1, 0, n_used)?.contiguous()?;
    let mut topk_w = probs.gather(&topk_ids, D::Minus1)?;
    if norm {
        let s = topk_w.sum_keepdim(D::Minus1)?;
        // max(s, c) as affine->relu->affine: 3 fused DEVICE kernels instead of the
        // broadcast_maximum host round-trip (3 PCIe transfers per token - measured as a
        // -30% decode regression on lfm2). Exact for s<=c; <=1 ulp for s>c (router-weight
        // normalization floor, far below the f16 expert-compute noise).
        const C: f32 = 6.103515625e-5;
        let s = s.affine(1.0, -C)?.relu()?.affine(1.0, C)?;
        topk_w = topk_w.broadcast_div(&s)?;
    }
    if (scale - 1.0).abs() > 1e-9 {
        topk_w = (topk_w * scale)?;
    }
    Ok((topk_w, topk_ids))
}

/// SwiGLU MoE: router -> silu(gate).up per expert -> weighted down-reduce.
pub struct Lfm2Moe {
    gate: crate::tensor::layer::Linear,
    gate_bias: Option<Tensor>,
    gate_exps: Arc<QTensor>,
    up_exps: Arc<QTensor>,
    down_exps: Arc<QTensor>,
    n_expert_used: usize,
    weights_norm: bool,
    weights_scale: f64,
}

impl Lfm2Moe {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Lfm2Config,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let gate_w = ld_f32(c, r, &format!("{p}.ffn_gate_inp.weight"), device)?; // [n_expert, d_model]
        let gate = crate::tensor::layer::Linear::new(gate_w, None)?;
        let gate_bias = ld_f32(c, r, &format!("{p}.exp_probs_b.bias"), device).ok();
        Ok(Self {
            gate,
            gate_bias,
            gate_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_gate_exps.weight"),
                device,
                mm,
            )?),
            up_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_up_exps.weight"),
                device,
                mm,
            )?),
            down_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_down_exps.weight"),
                device,
                mm,
            )?),
            n_expert_used: cfg.n_expert_used,
            weights_norm: cfg.expert_weights_norm,
            weights_scale: cfg.expert_weights_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_resid(x, None)
    }

    /// Like `forward`, but when `residual` is Some, fuses `residual + moe_out`
    /// into the down_reduce output init (it pre-loads the F32 accumulator with
    /// the residual, then the reduce kernel adds the experts) - dropping the
    /// caller's separate `residual + h` badd launch. lfm2 is launch-bound.
    pub fn forward_resid(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let xs = x
            .reshape((x.elem_count() / hidden, hidden))?
            .to_dtype(DType::F32)?;
        let n_tokens = xs.dim(0)?;
        let resid_flat = match residual {
            Some(r) => Some(r.reshape((r.elem_count() / hidden, hidden))?),
            None => None,
        };
        let (topk_w, topk_ids) = route(
            &self.gate,
            &self.gate_bias,
            &xs,
            self.n_expert_used,
            self.weights_norm,
            self.weights_scale,
        )?;
        let topk_flat = topk_ids.flatten_all()?;
        // Decode: replace the tensor-level sort_last_dim (pipeline-stalling) with a one-warp
        // argsort (bit-identical output). Prefill (m>32) falls back to it.
        let (expert_ids, sorted_token_ids) =
            match crate::inference::moe_cuda::argsort_small_u32(&topk_flat)? {
                Some(out) => out,
                None => topk_flat.sort_last_dim(true)?,
            };
        // SwiGLU: silu(gate(x)) * up(x), per selected expert - fused into ONE
        // launch (gate GEMM + up GEMM + silu + mul), avoiding the two [M.topk, N]
        // gate/up intermediates and a separate silu+mul. Same kernel FusedMoeGGUF
        // uses for the standard-SwiGLU MoE arches; works for prefill + decode.
        // Expert GEMMs: the cuda kernels reject CPU tensors, so on CPU (single
        // binary `--cpu`) delegate to the CPU expert-GEMV twin (bit-identical math).
        let routed = if xs.device().is_cuda() {
            let h = crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul(
                &xs,
                &self.gate_exps,
                &self.up_exps,
                &sorted_token_ids,
                &expert_ids,
                self.n_expert_used,
            )?;
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                resid_flat.as_ref(),
                None,
            )?
        } else {
            let h = crate::inference::moe_cpu::moe_gemm_gguf_gate_up_silu_mul(
                &xs,
                &self.gate_exps,
                &self.up_exps,
                &sorted_token_ids,
                &expert_ids,
                self.n_expert_used,
            )?;
            crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                resid_flat.as_ref(),
                None,
            )?
        }; // [n_tokens, hidden] F32 (+residual if Some)
           // Output in the LAYER-STACK dtype: the residual's when fused (x may
           // arrive pre-cast F32 from the fused ffn_norm), else the input's.
        let out_dt = match residual {
            Some(r) => r.dtype(),
            None => x.dtype(),
        };
        routed.reshape((b, seq, hidden))?.to_dtype(out_dt)
    }
}

/// Dense SwiGLU FFN (leading dense blocks).
pub struct Lfm2Dense {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    dtype: DType,
}

impl Lfm2Dense {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        dtype: DType,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        Ok(Self {
            gate: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_gate.weight"),
                device,
                mm,
            )?)?,
            up: QMatMul::from_qtensor(load_q8(c, r, &format!("{p}.ffn_up.weight"), device, mm)?)?,
            down: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_down.weight"),
                device,
                mm,
            )?)?,
            dtype,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.to_dtype(self.dtype)?;
        // Fused CPU decode SwiGLU: quantize the activation ONCE and run
        // gate+up+silu+mul in a single pass (vs two matmul_f16 that each
        // re-quantize xt into a fresh buffer + 4 whole-row intermediate casts).
        // Falls back below for prefill (m>1), GPU, or non-F16 stream.
        if let Some(h) = self.gate.gate_up_silu_f16(&self.up, &xt)? {
            return self.down.forward(&h);
        }
        // Prefill: keep both projections in the matmul's own dtype and let the
        // gated activation do the widen, the activation, the product and the
        // narrow in one pass. Done as separate steps, each wrote the whole wide
        // intermediate before the next read it.
        let gate_h = self.gate.forward(&xt)?;
        let up_h = self.up.forward(&xt)?;
        let h = crate::tensor::ops::swiglu(&gate_h, &up_h, self.dtype)?;
        self.down.forward(&h)
    }
}

/// Per-block deep-copied state for the exact-prompt prefix cache. lfm2 is a
/// hybrid, so the two operators carry different state: short-conv blocks a
/// rolling window, attention blocks a KV cache. Both are captured - the KV
/// alone cannot reconstruct the conv state, which is exactly why this model
/// cannot trim to an arbitrary position and needs a whole-state snapshot.
enum Lfm2StateSnap {
    ShortConv {
        conv: Option<Tensor>,
        cpu: Option<CpuShortConvState>,
    },
    Attn {
        kv: crate::tensor::KvCache,
        cpu: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
        kbuf: Option<Tensor>,
        vbuf: Option<Tensor>,
    },
}

/// Deep-copy an optional tensor (`affine(1,0)` is the copy idiom here).
fn snap_dc(t: &Option<Tensor>) -> Result<Option<Tensor>> {
    match t {
        Some(x) => Ok(Some(x.affine(1.0, 0.0)?)),
        None => Ok(None),
    }
}

impl Lfm2ShortConv {
    fn snapshot(&self) -> Result<Lfm2StateSnap> {
        Ok(Lfm2StateSnap::ShortConv {
            conv: snap_dc(&self.conv_state)?,
            cpu: self.cpu_state.clone(),
        })
    }

    fn restore(&mut self, s: &Lfm2StateSnap) -> Result<()> {
        if let Lfm2StateSnap::ShortConv { conv, cpu } = s {
            self.conv_state = snap_dc(conv)?;
            self.cpu_state = cpu.clone();
        }
        Ok(())
    }
}

impl Lfm2Attn {
    fn snapshot(&self) -> Result<Lfm2StateSnap> {
        Ok(Lfm2StateSnap::Attn {
            kv: self.kv_cache.deep_copy()?,
            cpu: self.cpu_f16_kv.clone(),
            kbuf: snap_dc(&self.kbuf)?,
            vbuf: snap_dc(&self.vbuf)?,
        })
    }

    fn restore(&mut self, s: &Lfm2StateSnap) -> Result<()> {
        if let Lfm2StateSnap::Attn {
            kv,
            cpu,
            kbuf,
            vbuf,
        } = s
        {
            self.kv_cache = kv.deep_copy()?;
            self.cpu_f16_kv = cpu.clone();
            self.kbuf = snap_dc(kbuf)?;
            self.vbuf = snap_dc(vbuf)?;
        }
        Ok(())
    }
}

enum Operator {
    ShortConv(Lfm2ShortConv),
    Attn(Lfm2Attn),
}
enum Ffn {
    Dense(Lfm2Dense),
    Moe(Lfm2Moe),
}

struct Lfm2Layer {
    op_norm: crate::tensor::layer::RmsNorm,
    ffn_norm: crate::tensor::layer::RmsNorm,
    op: Operator,
    ffn: Ffn,
    device: Device,
    rms_eps: f64,
}

/// Pre-norm for the F16 launch-bound lfm2 path: fuse cast-F32 + rms_norm +
/// cast-F16 (3 launches) into one kernel. lfm2 decode is launch-bound (nsys:
/// ~20% GPU util), so cutting ~160 norm-related launches/token matters. Falls
/// back to the tensor-op cast path off-CUDA / non-F16.
fn norm_fwd(norm: &crate::tensor::layer::RmsNorm, x: &Tensor, eps: f64) -> Result<Tensor> {
    if x.dtype() == DType::F16 && x.device().is_cuda() {
        return crate::inference::kernel::fused::fused_rmsnorm_f16(x, norm.weight(), eps as f32);
    }
    let st = x.dtype();
    norm.forward(&x.to_dtype(DType::F32)?)?.to_dtype(st)
}

/// lfm2moe - hybrid shortconv/attention + dense/MoE, multi-device.
pub struct Lfm2MoeModel {
    embed: Embedding,
    embed_dev: Device,
    layers: Vec<Lfm2Layer>,
    norm: crate::tensor::layer::RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    // Shared device position counter for the CUDA-graph devpos path: one i32 [1]
    // buffer all attention layers read, so a captured graph advances every layer's
    // KV-write/rope from a single counter incremented between replays. None until
    // the devpos path first runs.
    pos_dev: Option<Tensor>,
    // Fixed input-token buffer the captured decode forward embeds; written in place
    // (outside the captured region) before each replay.
    input_tok: Option<Tensor>,
    // When true, the forward does NOT recreate pos_dev (the capture/replay machine
    // owns it and writes it in place outside the captured region).
    graph_managed_pos: bool,
    // Production CUDA-graph decode (LOKEN_LFM2_GRAPH): captured once after
    // prefill, replayed per decode token. out_logits is the captured forward's
    // output (read after each replay). The capture arena stays armed until the
    // graph is dropped (request reset / new prefill).
    #[cfg(feature = "cuda")]
    graph: Option<crate::tensor::cuda_ext::CudaGraph>,
    out_logits: Option<Tensor>,
    ws_pinned: bool,
    // Count of UNCAPTURED warmup decode tokens run before capturing. The bisect
    // proved forward_inner is graph-safe ONLY when capture is primed by DECODE
    // forwards; capturing right after the PREFILL trips a transition wild-ptr ->
    // replay ILLEGAL_ADDRESS. Reset per request.
    graph_warmup: usize,
    /// One resident exact-prompt snapshot (in-memory only, never persisted).
    prefix_cache: Option<Lfm2PrefixCache>,
    /// Widest FFN intermediate (max of dense + per-expert) - prefill-chunk sizing.
    widest_ff: usize,
}

/// A whole-state snapshot taken at position `prompt.len()`, reusable only by the
/// very same prompt: the short-conv state advances per token and cannot be
/// rewound, so no partial-prefix reuse is possible from it.
struct Lfm2PrefixCache {
    prompt: Vec<u32>,
    layers: Vec<Lfm2StateSnap>,
    logits: Tensor,
}

impl Lfm2MoeModel {
    /// Widest FFN intermediate (dense or per-expert) - prefill-chunk sizing.
    pub fn widest_ffn(&self) -> usize {
        self.widest_ff
    }

    /// Where each layer actually lives, in layer order. Each layer keeps the
    /// device it was loaded onto, so this reads the placement back rather than
    /// recomputing the split.
    pub fn layer_device_locations(&self) -> Vec<crate::tensor::DeviceLocation> {
        self.layers.iter().map(|l| l.device.location()).collect()
    }

    /// Capture every block's state plus the prefill logits. Call AFTER
    /// prefilling [0, prompt.len()).
    pub fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> Result<()> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(match &l.op {
                Operator::ShortConv(b) => b.snapshot()?,
                Operator::Attn(b) => b.snapshot()?,
            });
        }
        self.prefix_cache = Some(Lfm2PrefixCache {
            prompt: prompt.to_vec(),
            layers,
            logits: logits.affine(1.0, 0.0)?,
        });
        Ok(())
    }

    /// On an EXACT prompt match, restore every block and return the prefill
    /// logits, so decode resumes at prompt.len() without prefilling. Else None.
    pub fn try_restore_prefix(&mut self, prompt: &[u32]) -> Result<Option<Tensor>> {
        match &self.prefix_cache {
            Some(pc) if pc.prompt.as_slice() == prompt => {}
            _ => return Ok(None),
        }
        let pc = self.prefix_cache.take().unwrap();
        for (l, snap) in self.layers.iter_mut().zip(pc.layers.iter()) {
            match &mut l.op {
                Operator::ShortConv(b) => b.restore(snap)?,
                Operator::Attn(b) => b.restore(snap)?,
            }
        }
        // A restore stands in for a prefill, so it owes the same per-request
        // reset: the capture arena stays armed until the graph is dropped, and
        // replaying it against restored state would read a stale capture.
        #[cfg(feature = "cuda")]
        {
            self.graph = None;
        }
        self.graph_warmup = 0;
        self.out_logits = None;
        let logits = pc.logits.affine(1.0, 0.0)?;
        self.prefix_cache = Some(pc);
        Ok(Some(logits))
    }
    pub fn from_gguf<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let cfg = Lfm2Config::from_gguf(content)?;
        let n_dev = devices.len().max(1);
        let dev_for = |l: usize| -> &Device { &devices[(l * n_dev / cfg.n_layers).min(n_dev - 1)] };
        let max_seq = cfg.context_length.min(32768).max(8192);
        let mut rope: Vec<(Tensor, Tensor)> = Vec::with_capacity(n_dev);
        for d in devices {
            rope.push(crate::inference::model::rope::precomput_freqs_cis_host(
                cfg.head_dim,
                max_seq,
                cfg.rope_freq_base,
                d,
            )?);
        }

        let embed_dev = devices[0].clone();
        let embed_t = content
            .tensor(reader, "token_embd.weight", &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_dtype(DType::F16)?
            .to_device(&embed_dev)?;
        let embed = Embedding::new(embed_t);

        let has = |l: usize, s: &str| content.tensor_infos.contains_key(&format!("blk.{l}.{s}"));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let device = dev_for(i);
            let di = (i * n_dev / cfg.n_layers).min(n_dev - 1);
            let (cos, sin) = &rope[di];
            let op_norm = crate::tensor::layer::RmsNorm::new(
                ld_f32(
                    content,
                    reader,
                    &format!("blk.{i}.attn_norm.weight"),
                    device,
                )?,
                cfg.rms_eps as f32,
            );
            let ffn_norm = crate::tensor::layer::RmsNorm::new(
                ld_f32(content, reader, &format!("blk.{i}.ffn_norm.weight"), device)?,
                cfg.rms_eps as f32,
            );
            let op = if has(i, "attn_q.weight") {
                Operator::Attn(Lfm2Attn::load(
                    content,
                    reader,
                    i,
                    &cfg,
                    cos.clone(),
                    sin.clone(),
                    device,
                    mm,
                )?)
            } else {
                Operator::ShortConv(Lfm2ShortConv::load(content, reader, i, &cfg, device, mm)?)
            };
            let ffn = if i >= cfg.leading_dense && has(i, "ffn_gate_inp.weight") {
                Ffn::Moe(Lfm2Moe::load(content, reader, i, &cfg, device, mm)?)
            } else {
                Ffn::Dense(Lfm2Dense::load(content, reader, i, dtype, device, mm)?)
            };
            if i == 0 || i + 1 == cfg.n_layers {
                tracing::info!(
                    "lfm2moe layer {i} op={} ffn={} on {:?}",
                    matches!(op, Operator::Attn(_))
                        .then(|| "attn")
                        .unwrap_or("shortconv"),
                    matches!(ffn, Ffn::Moe(_)).then(|| "moe").unwrap_or("dense"),
                    device.location()
                );
            }
            layers.push(Lfm2Layer {
                op_norm,
                ffn_norm,
                op,
                ffn,
                device: device.clone(),
                rms_eps: cfg.rms_eps,
            });
        }
        let last = dev_for(cfg.n_layers - 1);
        // lfm2 names the final norm `token_embd_norm`; lm_head is tied to token_embd.
        let norm_w = ld_f32(content, reader, "output_norm.weight", last)
            .or_else(|_| ld_f32(content, reader, "token_embd_norm.weight", last))?;
        let norm = crate::tensor::layer::RmsNorm::new(norm_w, cfg.rms_eps as f32);
        let lm_head = match load_q8(content, reader, "output.weight", last, mm) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => {
                QMatMul::from_qtensor(load_q8(content, reader, "token_embd.weight", last, mm)?)?
            }
        };
        Ok(Self {
            embed,
            embed_dev,
            layers,
            norm,
            lm_head,
            dtype,
            pos_dev: None,
            input_tok: None,
            graph_managed_pos: false,
            #[cfg(feature = "cuda")]
            graph: None,
            out_logits: None,
            ws_pinned: false,
            graph_warmup: 0,
            prefix_cache: None,
            widest_ff: cfg.dense_ff.max(cfg.expert_ff),
        })
    }

    /// Embed the input. On the graph path (graph_managed_pos) use a capture-safe
    /// device-token gather (the reference index_select bakes the index at capture);
    /// otherwise the reference embed lookup.
    fn embed_input(&self, input_ids: &Tensor) -> Result<Tensor> {
        if self.graph_managed_pos {
            if let Some(it) = self.input_tok.as_ref() {
                let table = self.embed.embeddings();
                let d_model = *table.dims().last().unwrap();
                if let Some(g) =
                    crate::inference::kernel::fused::embed_gather_f16(table, it, d_model)?
                {
                    return g.reshape((1, 1, d_model));
                }
            }
        }
        let ids = input_ids.to_device(&self.embed_dev)?;
        let mut x = self.embed.forward(&ids)?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }
        Ok(x)
    }

    /// Public entry: routes single-GPU decode (seq==1) through the CUDA-graph
    /// capture/replay machine when enabled; everything else runs forward_inner.
    pub fn forward(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        let _seq = input_ids.dim(1).unwrap_or(1);
        // CUDA-graph decode (capture/replay) is a CUDA-only optimization; on the
        // CPU build the whole machine is gated out and we always run forward_inner.
        #[cfg(feature = "cuda")]
        {
            let seq = _seq;
            // New request: drop any captured graph + free its arena so the next
            // decode re-captures against fresh prefilled state.
            if input_pos == 0 {
                self.graph_warmup = 0;
                if self.graph.is_some() {
                    self.graph = None;
                    self.out_logits = None;
                    self.graph_managed_pos = false;
                    if let Ok(cd) = self.embed_dev.as_cuda_device() {
                        cd.cuda_stream().context().free_capture_arena();
                    }
                }
            }
            let graph_path = lfm2_graph_enabled()
                && lfm2_devpos_enabled()
                && seq == 1
                && self.embed_dev.is_cuda()
                && input_pos > 0;
            if graph_path {
                return match self.forward_graph_decode(input_ids, input_pos) {
                    Ok(l) => Ok(l),
                    Err(e) => {
                        tracing::warn!("lfm2 graph decode failed ({e}); falling back");
                        self.graph = None;
                        self.out_logits = None;
                        self.graph_managed_pos = false;
                        self.forward_inner(input_ids, input_pos)
                    }
                };
            }
        }
        self.graph_managed_pos = false;
        self.forward_inner(input_ids, input_pos)
    }

    fn forward_inner(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        let mut x = self.embed_input(input_ids)?;
        // Shared device position counter (devpos path): one i32 [1] buffer all
        // attention layers read. Fresh tensor per token here (bit-identical); the
        // capture/replay machine will instead write it in place between replays so
        // the captured graph advances every layer from a single counter.
        if lfm2_devpos_enabled() && self.embed_dev.is_cuda() && !self.graph_managed_pos {
            self.pos_dev = Some(Tensor::new(&[input_pos as i32], &self.embed_dev)?);
        }
        let shared_pos = self.pos_dev.clone();
        for layer in self.layers.iter_mut() {
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            let _st = x.dtype();
            // operator sublayer
            let residual = x.clone();
            let h = norm_fwd(&layer.op_norm, &x, layer.rms_eps)?;
            let h = match &mut layer.op {
                Operator::ShortConv(o) => o.forward(&h, input_pos)?,
                Operator::Attn(o) => {
                    if input_pos == 0 {
                        o.reset();
                    }
                    o.forward(&h, input_pos, shared_pos.as_ref())?
                }
            };
            x = (residual + h)?;
            // ffn sublayer
            let residual = x.clone();
            x = match &layer.ffn {
                Ffn::Dense(f) => {
                    let h = norm_fwd(&layer.ffn_norm, &x, layer.rms_eps)?;
                    (residual + f.forward(&h)?)?
                }
                // MoE: fuse residual+moe_out into down_reduce (drops the badd),
                // and on CUDA/F16 emit the ffn_norm directly in F32 (the MoE FFN
                // consumes F32: router GEMV + expert q8_1 quantize) - kills the
                // per-layer input cast launch. Bit-identical (F16-rounded store).
                Ffn::Moe(f) => {
                    let h = if x.dtype() == DType::F16 && x.device().is_cuda() {
                        crate::inference::kernel::fused::fused_rmsnorm_f16_out_f32(
                            &x,
                            layer.ffn_norm.weight(),
                            layer.rms_eps as f32,
                        )?
                    } else {
                        norm_fwd(&layer.ffn_norm, &x, layer.rms_eps)?
                    };
                    f.forward_resid(&h, Some(&residual))?
                }
            };
        }
        let x = self.norm.forward(&x.to_dtype(DType::F32)?)?;
        let x = x.i((.., seq - 1, ..))?.to_dtype(self.dtype)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }

    /// Production CUDA-graph decode: capture the decode forward ONCE (after
    /// prefill, on the first decode token) into a graph reading the device-
    /// resident input_tok + pos_dev; on every token write those in place (outside
    /// the captured region) and replay. Returns the captured forward's logits
    /// (read after the launch). Single H2D scalar writes + 1 graph launch/token
    /// replaces ~860 kernel launches -> the 1.41x the probe measured.
    #[cfg(feature = "cuda")]
    fn forward_graph_decode(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        use crate::inference::kernel::fused::set_i32_inplace;
        use crate::tensor::cuda_ext;
        let dev = self.embed_dev.clone();
        let cd = dev.as_cuda_device()?;
        let stream = cd.cuda_stream();
        let ctx = stream.context();
        let _ = ctx.bind_to_thread();
        let msg = |s: &str| crate::tensor::Error::msg(s.to_string());
        if self.input_tok.is_none() {
            self.input_tok = Some(Tensor::zeros_on((1usize, 1usize), DType::U32, &dev)?);
        }
        if self.pos_dev.is_none() {
            self.pos_dev = Some(Tensor::zeros_on((1usize,), DType::I32, &dev)?);
        }
        let (input_tok, pos_dev) = (
            self.input_tok.clone().unwrap(),
            self.pos_dev.clone().unwrap(),
        );
        // advance token + position IN PLACE, outside the captured region. Token is
        // copied device->device (no D2H sync); position is a host int -> device.
        let it_src = input_ids.to_device(&dev)?;
        crate::inference::kernel::fused::copy_u32_dev(&it_src, &input_tok)?;
        set_i32_inplace(&pos_dev, input_pos as i32)?;
        self.graph_managed_pos = true;
        // WARMUP: run the first K decode tokens UNCAPTURED so the subsequent
        // capture is primed by DECODE-mode state. The ILLEGAL-bisect proved
        // forward_inner replays bit-perfectly when primed by decode forwards,
        // but capturing on the FIRST decode token (primed by the PREFILL, seq>1)
        // trips a transition wild-pointer -> replay ILLEGAL_ADDRESS. Each warmup
        // token returns its own logits (uses the in-place input_tok/pos_dev).
        if self.graph.is_none() && self.graph_warmup < 3 {
            let logits = self.forward_inner(&input_tok, input_pos)?;
            self.graph_warmup += 1;
            return Ok(logits);
        }
        if self.graph.is_none() {
            if !self.ws_pinned {
                cuda_ext::pin_cublas_workspace(&dev, 64 << 20)?;
                self.ws_pinned = true;
            }
            ctx.begin_capture_arena(256 << 20)
                .map_err(|e| msg(&format!("arena {e:?}")))?;
            ctx.begin_defer_frees();
            let it = input_tok.clone();
            let cap: Result<(cuda_ext::CudaGraph, Tensor)> = (|| {
                cuda_ext::begin_capture(&stream)?;
                let logits = self.forward_inner(&it, input_pos)?;
                let g = cuda_ext::end_capture(&stream)?.ok_or_else(|| msg("no graph"))?;
                Ok((g, logits))
            })();
            ctx.end_defer_frees();
            let (arena_peak, arena_of) = ctx.end_capture_arena(); // disarm but KEEP alive for replays
            tracing::warn!("🟦 lfm2 GRAPH arena peak={}MB overflow={} (large peak -> L2 cache thrash -> slow replay?)", arena_peak >> 20, arena_of);
            let (g, logits) = cap?;
            // DIAG: node-type histogram - MEM_ALLOC/MEM_FREE nodes captured into
            // the graph are the classic replay wild-pointer source (their device
            // addresses diverge on replay -> ILLEGAL_ADDRESS).
            {
                use cuda_ext::CUgraphNodeType as T;
                let nn = g.num_nodes().unwrap_or(0);
                let handles = g.nodes(nn).unwrap_or_default();
                let types = g.node_types(&handles).unwrap_or_default();
                let (mut k, mut ma, mut mf, mut mcpy, mut mset, mut other) = (0, 0, 0, 0, 0, 0);
                for t in &types {
                    match t {
                        T::CU_GRAPH_NODE_TYPE_KERNEL => k += 1,
                        T::CU_GRAPH_NODE_TYPE_MEM_ALLOC => ma += 1,
                        T::CU_GRAPH_NODE_TYPE_MEM_FREE => mf += 1,
                        T::CU_GRAPH_NODE_TYPE_MEMCPY => mcpy += 1,
                        T::CU_GRAPH_NODE_TYPE_MEMSET => mset += 1,
                        _ => other += 1,
                    }
                }
                tracing::warn!("🟦 lfm2 GRAPH node-histogram: nodes={nn} kernel={k} MEM_ALLOC={ma} MEM_FREE={mf} memcpy={mcpy} memset={mset} other={other}");
            }
            // DEFENSE IN DEPTH: a 0-node graph recorded no work - storing it
            // would replay a no-op every token, leaving out_logits frozen ->
            // constant-token output. Err -> the forward() wrapper falls back to
            // eager forward_inner (correct, just unaccelerated).
            if g.num_nodes().unwrap_or(0) == 0 {
                drop(g); // graph references arena VAs - destroy before freeing
                ctx.free_capture_arena();
                return Err(msg(
                    "capture produced an empty graph (0 nodes) - refusing to replay",
                ));
            }
            g.upload().map_err(|e| msg(&format!("upload {e:?}")))?;
            self.graph = Some(g);
            self.out_logits = Some(logits);
        }
        if let Some(g) = self.graph.as_ref() {
            g.launch().map_err(|e| msg(&format!("launch {e:?}")))?;
        }
        // No per-token sync: the launch is queued on the default stream; the engine
        // syncs when it reads the logits for sampling (and the next token's input
        // depends on that sample, so there is no race).
        self.out_logits.clone().ok_or_else(|| msg("no out_logits"))
    }

    /// Run embed + the first `n` layers (no final norm / lm_head); returns the
    /// hidden state. For CUDA-graph NaN bisection - capture forward_prefix(n),
    /// replay, find the first n whose output NaNs on replay.
    pub fn forward_prefix(
        &mut self,
        input_ids: &Tensor,
        input_pos: usize,
        n: usize,
        half: bool,
    ) -> Result<Tensor> {
        let mut x = self.embed_input(input_ids)?;
        if lfm2_devpos_enabled() && self.embed_dev.is_cuda() && !self.graph_managed_pos {
            self.pos_dev = Some(Tensor::new(&[input_pos as i32], &self.embed_dev)?);
        }
        let shared_pos = self.pos_dev.clone();
        for (li, layer) in self.layers.iter_mut().enumerate() {
            // `half`: on the extra layer index n, run only its op (shortconv/attn),
            // skipping the FFN - to split op vs ffn as the NaN source.
            let op_only = half && li == n;
            if li > n || (li == n && !half) {
                break;
            }
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            let residual = x.clone();
            let h = norm_fwd(&layer.op_norm, &x, layer.rms_eps)?;
            let h = match &mut layer.op {
                Operator::ShortConv(o) => o.forward(&h, input_pos)?,
                Operator::Attn(o) => {
                    if input_pos == 0 {
                        o.reset();
                    }
                    o.forward(&h, input_pos, shared_pos.as_ref())?
                }
            };
            x = (residual + h)?;
            if op_only {
                break;
            }
            let residual = x.clone();
            let h = norm_fwd(&layer.ffn_norm, &x, layer.rms_eps)?;
            let h = match &layer.ffn {
                Ffn::Dense(f) => f.forward(&h)?,
                Ffn::Moe(f) => f.forward(&h)?,
            };
            x = (residual + h)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod head_rms_bit_identity {
    // Deterministic gate: the fused CPU per-head RMS (`head_rms_cpu`) must equal
    // the original tensor-op `head_rms` EXACTLY (maxabs==0), across the q/k-norm
    // shapes lfm2moe uses. Reproducible; free of full-model non-determinism.
    use super::{head_rms, head_rms_cpu};
    use crate::tensor::{Device, Tensor};

    fn maxabs(a: &Tensor, b: &Tensor) -> f32 {
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av.len(), bv.len());
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn run(b: usize, heads: usize, seq: usize, hd: usize, eps: f32) {
        let dev = Device::Cpu;
        let n = b * heads * seq * hd;
        let xd: Vec<f32> = (0..n)
            .map(|i| (((i * 7 + 3) % 197) as f32) * 0.017 - 1.6)
            .collect();
        let x = Tensor::from_vec(xd, (b, heads, seq, hd), &dev).unwrap();
        let wd: Vec<f32> = (0..hd).map(|i| 0.5 + ((i % 11) as f32) * 0.07).collect();
        let w = Tensor::from_vec(wd, (hd,), &dev).unwrap();
        // head_rms takes eps as f64 and internally applies it as `eps as f32`
        // (affine); pass the same f32 value widened so both use the identical eps.
        let reference = head_rms(&x, &w, eps as f64).unwrap();
        let fused = head_rms_cpu(&x, &w, eps).unwrap();
        let d = maxabs(&reference, &fused);
        assert_eq!(
            d, 0.0,
            "b={b} heads={heads} seq={seq} hd={hd} eps={eps} maxabs={d}"
        );
    }

    #[test]
    fn fused_matches_tensor_op() {
        for &(b, h, s, hd) in &[
            (1usize, 32usize, 512usize, 64usize), // q-norm prefill
            (1, 8, 512, 64),                      // k-norm prefill (GQA kv heads)
            (1, 32, 1, 64),                       // decode single row
            (1, 8, 37, 64),                       // odd seq
            (2, 4, 3, 128),                       // larger head_dim, small
        ] {
            for &eps in &[1e-5f32, 1e-6, 1e-4] {
                run(b, h, s, hd, eps);
            }
        }
    }
}
