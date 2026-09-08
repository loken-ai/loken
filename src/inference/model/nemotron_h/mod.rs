//! nemotron_h_moe (NVIDIA Nemotron-H) - hybrid Mamba2 + attention + MoE.
//!
//! Each transformer block is exactly one of: an **SSM** (Mamba2) block, an
//! **attention** block, or a **MoE-FFN** block - stacked in a fixed pattern.
//! The block kind is detected from which `blk.N.*` tensors the GGUF carries
//! (robust; avoids parsing the per-layer `attention.head_count_kv` array).
//!
//! Composition:
//! - SSM block  -> the `mamba2` selective-scan math, `ssm_in`/`ssm_out` as
//!   quantized `QMatMul`, F32 recurrence, per-block conv+ssm state.
//! - attention  -> QKV (no bias) + partial NEOX rope (rotate `rope_dim`<head_dim)
//!   + GQA + plain softmax + KV cache.
//! - MoE-FFN    -> non-gated `down(relu(up(x))²)` (LLM_FFN_RELU_SQR) via facade
//!   moe-GEMM primitives + router bias + weight-norm/scale + a shared expert.
//!
//! Structurally complete + compiles; pending llm_engine dispatch wiring and
//! end-to-end numerical validation (the SSM recurrence is the highest risk).

use crate::tensor::layer::Embedding;
use crate::tensor::ops::heads_first;
use crate::tensor::quantized::{gguf_file, QMatMul};
use crate::tensor::{DType, Device, IndexOp, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Mamba2 selective-scan block (`ssm_*` tensors).
    Ssm,
    /// Self-attention block (`attn_q/k/v/output`); present only on the layers
    /// whose `head_count_kv` entry is non-zero.
    Attention,
    /// MoE feed-forward block (`ffn_gate_inp` + routed/shared experts).
    MoeFfn,
}

/// Detect a block's kind from the tensors present in the GGUF.
pub fn detect_layer_kind(ct: &gguf_file::Content, layer: usize) -> LayerKind {
    let has = |suffix: &str| {
        ct.tensor_infos
            .contains_key(&format!("blk.{layer}.{suffix}"))
    };
    if has("ssm_in.weight") {
        LayerKind::Ssm
    } else if has("attn_q.weight") {
        LayerKind::Attention
    } else {
        LayerKind::MoeFfn
    }
}

/// Parsed `nemotron_h_moe.*` hyper-parameters. SSM dims that aren't present as
/// metadata are derived from tensor shapes at load time.
#[derive(Debug, Clone)]
pub struct NemotronHConfig {
    pub n_layers: usize,
    pub embedding_length: usize,
    pub n_head: usize,
    pub head_dim: usize, // attention.key_length
    pub rope_dim: usize, // rope.dimension_count (partial rotary)
    /// MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    /// MoE routing-weight post-processing (nemotron: normalize top-k then xscale).
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f64,
    pub ffn_dim: usize,
    pub rms_eps: f64,
    pub context_length: usize,
    pub rope_freq_base: f32,
    /// Mamba2 SSM params (0 ⇒ derive from tensor shapes at load).
    pub ssm_conv_kernel: usize, // d_conv
    pub ssm_state_size: usize,  // d_state
    pub ssm_group_count: usize, // ngroups
    pub ssm_inner_size: usize,  // d_inner (= expand . d_model)
    pub ssm_time_step_rank: usize,
}

impl NemotronHConfig {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        // One place knows where this architecture's keys live and what a count looks like;
        // the readers below differ only in the type they want and in what absence means.
        let value = |k: &str| ct.metadata.get(&format!("nemotron_h_moe.{k}"));
        let count = |k: &str| value(k).and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let req_u = |k: &str| -> Result<usize> {
            count(k).ok_or_else(|| {
                crate::tensor::Error::msg(format!("nemotron_h: missing nemotron_h_moe.{k}"))
            })
        };
        let opt_u = |k: &str, d: usize| -> usize { count(k).unwrap_or(d) };
        let opt_f =
            |k: &str, d: f32| -> f32 { value(k).and_then(|v| v.to_f32().ok()).unwrap_or(d) };
        Ok(Self {
            n_layers: req_u("block_count")?,
            embedding_length: req_u("embedding_length")?,
            n_head: opt_u("attention.head_count", 0),
            head_dim: opt_u("attention.key_length", 128),
            rope_dim: opt_u("rope.dimension_count", 0),
            n_expert: opt_u("expert_count", 0),
            n_expert_used: opt_u("expert_used_count", 0),
            n_expert_shared: opt_u("expert_shared_count", 0),
            expert_weights_norm: value("expert_weights_norm")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(false),
            expert_weights_scale: opt_f("expert_weights_scale", 1.0) as f64,
            ffn_dim: opt_u(
                "expert_feed_forward_length",
                opt_u("feed_forward_length", 0),
            ),
            rms_eps: opt_f("attention.layer_norm_rms_epsilon", 1e-5) as f64,
            context_length: opt_u("context_length", 8192),
            rope_freq_base: opt_f("rope.freq_base", 10000.0),
            ssm_conv_kernel: opt_u("ssm.conv_kernel", 0),
            ssm_state_size: opt_u("ssm.state_size", 0),
            ssm_group_count: opt_u("ssm.group_count", 0),
            ssm_inner_size: opt_u("ssm.inner_size", 0),
            ssm_time_step_rank: opt_u("ssm.time_step_rank", 0),
        })
    }

    /// The per-block kinds, in stack order.
    pub fn layer_kinds(&self, ct: &gguf_file::Content) -> Vec<LayerKind> {
        (0..self.n_layers)
            .map(|i| detect_layer_kind(ct, i))
            .collect()
    }
}

use crate::tensor::quantized::QTensor;
use std::io::{Read, Seek};

/// Load a matmul weight as Q8_0 (requantizing MXFP4/BF16/F16 like the gpt-oss
/// path - the CUDA MMVQ kernels need a supported quant + QMatMul dequantizes
/// float weights to F32 otherwise). Already-quantized weights load as-is.
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
    let cpu = c.tensor(r, name, &crate::tensor::Device::Cpu)?;
    let f = cpu.dequantize(&crate::tensor::Device::Cpu)?;
    QTensor::quantize_onto(&f, GgmlDType::Q8_0, d)
}
fn ld_f32<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
) -> Result<Tensor> {
    c.tensor(r, name, d)?.dequantize(d)?.to_dtype(DType::F32)
}

use crate::tensor::layer::Linear;
use std::sync::Arc;

/// nemotron_h MoE-FFN block: non-gated `down(relu(up(x))²)` (LLM_FFN_RELU_SQR),
/// 128 experts / 6 used, router with bias, weight-norm + scale, plus a shared
/// (always-on) expert. Uses the reference pub MoE-GEMM primitives directly (the
/// FusedMoeGGUF SwiGLU path doesn't fit a non-gated FFN).
pub struct NemotronMoeBlock {
    gate: Linear,              // router: hidden -> n_expert (F32)
    gate_bias: Option<Tensor>, // exp_probs_b [n_expert] F32
    up_exps: Arc<QTensor>,     // [n_expert, ffn, hidden]
    down_exps: Arc<QTensor>,   // [n_expert, hidden, ffn]
    up_shexp: QMatMul,         // shared expert up: hidden -> shexp_ffn
    down_shexp: QMatMul,       // shared expert down: shexp_ffn -> hidden
    n_expert_used: usize,
    weights_norm: bool,
    weights_scale: f64,
    dtype: DType,
}

impl NemotronMoeBlock {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &NemotronHConfig,
        dtype: DType,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let gate_ws = ld_f32(c, r, &format!("{p}.ffn_gate_inp.weight"), device)?;
        Ok(Self {
            gate: Linear::new(gate_ws, None)?,
            gate_bias: ld_f32(c, r, &format!("{p}.exp_probs_b.bias"), device).ok(),
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
            up_shexp: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_up_shexp.weight"),
                device,
                mm,
            )?)?,
            down_shexp: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_down_shexp.weight"),
                device,
                mm,
            )?)?,
            n_expert_used: cfg.n_expert_used,
            weights_norm: cfg.expert_weights_norm,
            weights_scale: cfg.expert_weights_scale,
            dtype,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let xs = x.reshape((b * seq, hidden))?.to_dtype(DType::F32)?;
        let n_tokens = xs.dim(0)?;
        // Router (DeepSeek-V3 style, SIGMOID gating): probs = sigmoid(gate(x));
        // the exp_probs_b bias affects SELECTION only - the weights come from the
        // UNBIASED probs, then are normalized over the top-k and scaled.
        let logits = self.gate.forward(&xs)?; // [n_tok, n_expert] F32
                                              // Fuse all POST-matmul router ops (sigmoid + bias-select top-k + gather +
                                              // renorm + scale) into one kernel, keeping the cuBLAS gemv above. Same
                                              // win as lfm2 (+30%): erases the per-layer sort_last_dim.
        let (topk_w, topk_ids) = match crate::inference::moe_cuda::topk_sigmoid_post(
            &logits,
            self.gate_bias.as_ref(),
            self.n_expert_used,
            self.weights_norm,
            self.weights_scale,
        )? {
            Some(out) => out,
            None => {
                let probs = crate::tensor::ops::sigmoid(&logits)?; // [n_tok, n_expert]
                let sel = match &self.gate_bias {
                    Some(gb) => probs.broadcast_add(gb)?,
                    None => probs.clone(),
                };
                let (_sv, sidx) = sel.sort_last_dim(false)?;
                let topk_ids = sidx
                    .narrow(D::Minus1, 0, self.n_expert_used)?
                    .contiguous()?; // [n_tok,k]
                let mut topk_w = probs.gather(&topk_ids, D::Minus1)?; // [n_tok,k]
                if self.weights_norm {
                    let s0 = topk_w.sum_keepdim(D::Minus1)?;
                    let floor = Tensor::full(6.103_515_6e-5_f32, s0.dims(), &s0.device())?;
                    let s = s0.maximum(&floor)?;
                    topk_w = topk_w.broadcast_div(&s)?;
                }
                if (self.weights_scale - 1.0).abs() > 1e-9 {
                    topk_w = (topk_w * self.weights_scale)?;
                }
                (topk_w, topk_ids)
            }
        };
        let topk_flat = topk_ids.flatten_all()?;
        // Decode: one-warp argsort (bit-identical to sort_last_dim); tensor-op fallback for prefill.
        let (expert_ids, sorted_token_ids) =
            match crate::inference::moe_cuda::argsort_small_u32(&topk_flat)? {
                Some(out) => out,
                None => topk_flat.sort_last_dim(true)?,
            };
        // routed: up -> relu² -> down (weighted by topk_w, reduced over top-k)
        // Expert GEMMs: cuda kernels reject CPU tensors -> delegate to the CPU
        // expert-GEMV twin on CPU (single binary `--cpu`), bit-identical math.
        let on_cuda = xs.device().is_cuda();
        let h = if on_cuda {
            let up = crate::inference::moe_cuda::moe_gemm_gguf(
                &xs,
                &self.up_exps,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.n_expert_used,
                seq > 1,
                self.dtype,
            )?;
            match crate::inference::kernel::fused::relu2_f16(&up)? {
                Some(y) => y,
                None => up.relu()?.sqr()?,
            }
        } else {
            // Fused up+relu² work-stealing path: the sequential
            // per-expert `moe_gemm_gguf` loop + separate relu.sqr was the
            // profiled 65%-of-token MoE cost's overhead component.
            crate::inference::moe_cpu::moe_gemm_gguf_up_relu2(
                &xs,
                &self.up_exps,
                &sorted_token_ids,
                &expert_ids,
                self.n_expert_used,
            )?
        };
        let routed = if on_cuda {
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                None,
                None,
            )?
        } else {
            crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                None,
                None,
            )?
        }; // [n_tokens, hidden] F32
           // shared expert (dense, all tokens): down(relu(up(x))²). relu².cast via
           // one F16 launch (F32-internal -> bit-identical), saving cast+relu+sqr+cast
           // dispatches/layer (the qwen3.5 dispatch-fusion lesson applied to nemotron).
        let xt = xs.to_dtype(self.dtype)?;
        let up_sh = self.up_shexp.forward(&xt)?; // F16
        let act = match crate::inference::kernel::fused::relu2_f16(&up_sh)? {
            Some(y) => y,
            None => up_sh
                .to_dtype(DType::F32)?
                .relu()?
                .sqr()?
                .to_dtype(self.dtype)?,
        };
        let sh = self.down_shexp.forward(&act)?; // F16
                                                 // Merge shared expert into routed (F32) and write F16 in one launch:
                                                 // f16(routed + f32(sh)) - was cast+add+cast (3 tensor-op dispatches), bit-exact.
        let merged = match crate::inference::kernel::fused::add_to_f16(&routed, &sh)? {
            Some(o) => o,
            None => routed
                .broadcast_add(&sh.to_dtype(DType::F32)?)?
                .to_dtype(x.dtype())?,
        };
        merged.reshape((b, seq, hidden))
    }
}

/// Consecutive spans of a tensor's last dim, in the order they were packed into it.
///
/// The in-projection writes `z ‖ (x,B,C) ‖ dt` and the convolution writes `x ‖ B ‖ C`; both
/// are read the same way - take a width, advance by it, take the next - so the offsets are
/// counted here once instead of being spelled out at every cut, where an arithmetic slip
/// would hand the recurrence somebody else's channels.
fn split_last<const N: usize>(t: &Tensor, widths: [usize; N]) -> Result<[Tensor; N]> {
    let mut at = 0usize;
    let mut spans = Vec::with_capacity(N);
    for width in widths {
        spans.push(t.narrow(D::Minus1, at, width)?);
        at += width;
    }
    spans
        .try_into()
        .map_err(|_| crate::tensor::Error::msg("split_last: span count mismatch"))
}

/// nemotron_h attention block: QKV (no bias) -> partial NEOX rope -> KV cache ->
/// GQA -> scaled scores -> causal mask -> plain softmax -> output proj.
pub struct NemotronAttn {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    cos: Tensor,
    sin: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_dim: usize,
    // Fixed-size O(1)/token cache (slice_set) - was ConcatKvCache, whose per-token
    // cat re-copied the whole KV (O(N²) over a generation). Same proven swap as
    // lfm2/qwen3.5. nemotron has few attention layers (mostly Mamba2) so the gain
    // is smaller, but it is free + safe (bit-exact).
    kv_cache: crate::tensor::KvCache,
}

impl NemotronAttn {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &NemotronHConfig,
        head_dim: usize,
        rotary_dim: usize,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let wk_qt = load_q8(c, r, &format!("{p}.attn_k.weight"), device, mm)?;
        // How many key/value heads a layer writes varies from one attention block to the
        // next, so it is derived from the K projection's own width rather than read from the
        // config, which carries only the query head count.
        let n_head = cfg.n_head;
        let n_kv_head = (wk_qt.shape().dims()[0] / head_dim.max(1)).max(1);
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
            cos,
            sin,
            n_head,
            n_kv_head,
            head_dim,
            rotary_dim,
            // Small initial capacity: the reference KV cache allocates the full initial
            // size on first append and grows in those steps; nemotron is 2-GPU
            // tight, so keep the pre-alloc small to preserve prefill headroom.
            kv_cache: crate::tensor::KvCache::new(2, cfg.context_length.min(2048).max(256)),
        })
    }

    /// Partial NEOX rope: rotate the first `rotary_dim` dims, pass the rest.
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Fused partial-NEOX rope (one launch vs narrow+contig+rope+cat+contig),
        // bit-identical. Falls back to the tensor-op op chain off the fast path.
        if let Some(o) = crate::inference::kernel::fused::neox_rope_f16(
            x,
            cos,
            sin,
            self.rotary_dim.min(self.head_dim),
        )? {
            return Ok(o);
        }
        if self.rotary_dim >= self.head_dim {
            return crate::tensor::ops::rope(x, cos, sin);
        }
        let rot = x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        let pass = x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
        let rot = crate::tensor::ops::rope(&rot, cos, sin)?;
        // cat over a transposed [b,h,seq,hd] layout yields a non-contiguous
        // result; matmul needs a contiguous lhs.
        Tensor::cat(&[&rot, &pass], D::Minus1)?.contiguous()
    }

    pub fn forward(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dev = x.device().clone();
        // Three projections, one rule, so only the head count differs between them.
        let project = |w: &QMatMul, heads: usize| -> Result<Tensor> {
            heads_first(w.forward(x)?, heads, self.head_dim)
        };
        let q = project(&self.wq, self.n_head)?;
        let k = project(&self.wk, self.n_kv_head)?;
        let v = project(&self.wv, self.n_kv_head)?;
        let rows = |table: &Tensor| -> Result<Tensor> {
            table.narrow(0, input_pos, seq)?.to_dtype(q.dtype())
        };
        let (cos, sin) = (rows(&self.cos)?, rows(&self.sin)?);
        let q = self.rope(&q, &cos, &sin)?;
        let k = self.rope(&k, &cos, &sin)?;
        let (k, v) = self.kv_cache.append(&k, &v)?; // [b, n_kv_head, kv_len, hd]
        let n_kv = k.dim(1)?;
        let kv_len = k.dim(2)?;
        // How many query heads share one key/value head: a quotient of two fields this
        // block already holds, so it is taken here rather than stored a second time.
        let g = self.n_head / self.n_kv_head;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // NOTE: the fused flash-decode kernel (used by lfm2) REGRESSED nemotron
        // -11% (126->112 tok/s): nemotron is GPU-bound (nsys ~40% util) with few,
        // wide (hd=128) attention layers, so the one-warp-per-head kernel loses to
        // cuBLAS batched gemv. Kept on the cuBLAS chain. Measured.
        // GQA without repeat_kv: read K/V once (n_kv_head) not n_head copies.
        // Keep the scores in the matmul's own dtype here: the widen and the
        // scale fold into the masked softmax below, which walks the tensor
        // anyway. Done as separate steps they wrote the whole score tensor
        // twice before the softmax read it once.
        let scores_h = if g > 1 {
            let qg = q.reshape((b, n_kv, g * seq, self.head_dim))?;
            qg.matmul(&k.transpose(2, 3)?)?
                .reshape((b, self.n_head, seq, kv_len))?
        } else {
            q.matmul(&k.transpose(2, 3)?)?
        };
        let kv = input_pos + seq;
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
            let scores = (scores_h.to_dtype(DType::F32)? * scale)?;
            crate::tensor::ops::softmax_last_dim(&scores)?
        }
        .to_dtype(v.dtype())?;
        // The weighted sum of the values, then the head axis back behind the sequence so the
        // output projection reads one row per token.
        let out = match g > 1 {
            true => w
                .reshape((b, n_kv, g * seq, kv_len))?
                .matmul(&v)?
                .reshape((b, self.n_head, seq, self.head_dim))?,
            false => w.matmul(&v)?,
        };
        self.wo.forward(
            &out.transpose(1, 2)?
                .reshape((b, seq, self.n_head * self.head_dim))?,
        )
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }
    /// Deep-copied KV state for the prefix cache. See `Cache::deep_copy`.
    fn snapshot(&self) -> Result<NemoStateSnap> {
        Ok(NemoStateSnap::Attn(self.kv_cache.deep_copy()?))
    }
    fn restore(&mut self, s: &NemoStateSnap) -> Result<()> {
        if let NemoStateSnap::Attn(kv) = s {
            self.kv_cache = kv.deep_copy()?;
        }
        Ok(())
    }
}

/// One Mamba2 SSM block (nemotron_h `ssm_*` tensors). Ported from the reference
/// `mamba2::Mamba2Block` with the in/out projections as quantized `QMatMul`
/// and the recurrence run in F32. Holds its own conv + ssm state (reset when
/// `input_pos == 0`). Processes tokens one at a time (correct for prefill +
/// decode; the chunked parallel scan is a later optimization).
/// Host-side Mamba2 state + weight caches for the fused CPU recurrence
/// (`forward_cpu_fused`): plain `Vec<f32>` so the per-token loop never touches
/// the tensor-op dispatch layer (the per-token tensor-op path is ~40 small ops/layer,
/// each a single-threaded dispatch - the dominant CPU decode cost).
#[derive(Clone)]
struct CpuMambaState {
    conv: Vec<f32>, // rolling window [d_xbc * d_conv] (last d_conv inputs)
    h: Vec<f32>,    // ssm state [nheads * headdim * d_state]
    w: Vec<f32>,    // conv weights [d_xbc * d_conv]
    wb: Vec<f32>,   // conv bias [d_xbc]
    a: Vec<f32>,    // [nheads]
    d: Vec<f32>,    // [nheads]
    dtb: Vec<f32>,  // dt_bias [nheads]
    norm: Vec<f32>, // group-norm weight, flattened [d_inner]
}

pub struct NemotronSsmBlock {
    in_proj: QMatMul,  // ssm_in:  d_model -> d_inner + d_xbc + nheads
    out_proj: QMatMul, // ssm_out: d_inner -> d_model
    conv1d_w: Tensor,  // [d_xbc, 1, d_conv] F32
    conv1d_b: Tensor,  // [d_xbc] F32
    a: Tensor,         // [nheads] F32  (= -exp(ssm_a))
    d: Tensor,         // [nheads] F32
    dt_bias: Tensor,   // [nheads] F32
    norm_w: Tensor,    // gated GROUP RMSNorm weight [ngroups, group_size] F32
    d_inner: usize,
    d_state: usize,
    d_xbc: usize,
    headdim: usize,
    nheads: usize,
    ngroups: usize,
    d_conv: usize,
    rms_eps: f64,
    conv_state: Option<Tensor>, // [b, d_xbc, d_conv]
    ssm_h: Option<Tensor>,      // [b, nheads, headdim, d_state]
    cpu_state: Option<CpuMambaState>,
    device: Device,
}

impl NemotronSsmBlock {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &NemotronHConfig,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let in_proj =
            QMatMul::from_qtensor(load_q8(c, r, &format!("{p}.ssm_in.weight"), device, mm)?)?;
        let out_proj =
            QMatMul::from_qtensor(load_q8(c, r, &format!("{p}.ssm_out.weight"), device, mm)?)?;
        let conv1d_w = ld_f32(c, r, &format!("{p}.ssm_conv1d.weight"), device)?; // [d_xbc, d_conv]
        let conv1d_b = ld_f32(c, r, &format!("{p}.ssm_conv1d.bias"), device)?.flatten_all()?; // [d_xbc]
                                                                                              // ssm_a/d/dt are stored as [nheads, 1] column vectors -> flatten to
                                                                                              // [nheads]. ssm_a in the GGUF is ALREADY -exp(A_log) (all negative);
                                                                                              // the recurrence uses it raw in exp(softplus(dt).A) - no extra transform.
        let a = ld_f32(c, r, &format!("{p}.ssm_a"), device)?.flatten_all()?;
        let d = ld_f32(c, r, &format!("{p}.ssm_d"), device)?.flatten_all()?;
        let dt_bias = ld_f32(c, r, &format!("{p}.ssm_dt.bias"), device)?.flatten_all()?;
        // group-wise gated RMSNorm: weight is [ngroups, d_inner/ngroups].
        let norm_w = ld_f32(c, r, &format!("{p}.ssm_norm.weight"), device)?;

        let d_inner = cfg.ssm_inner_size;
        let d_state = cfg.ssm_state_size;
        let ngroups = cfg.ssm_group_count;
        let d_conv = cfg.ssm_conv_kernel;
        let nheads = a.dims1()?; // ssm_a is [nheads]
        let headdim = d_inner / nheads.max(1);
        let d_xbc = d_inner + 2 * ngroups * d_state;
        Ok(Self {
            in_proj,
            out_proj,
            conv1d_w,
            conv1d_b,
            a,
            d,
            dt_bias,
            norm_w,
            d_inner,
            d_state,
            d_xbc,
            headdim,
            nheads,
            ngroups,
            d_conv,
            rms_eps: cfg.rms_eps,
            conv_state: None,
            ssm_h: None,
            cpu_state: None,
            device: device.clone(),
        })
    }

    fn reset_state(&mut self, b: usize) -> Result<()> {
        self.conv_state = Some(Tensor::zeros_on(
            (b, self.d_xbc, self.d_conv),
            DType::F32,
            &self.device,
        )?);
        self.ssm_h = Some(Tensor::zeros_on(
            (b, self.nheads, self.headdim, self.d_state),
            DType::F32,
            &self.device,
        )?);
        Ok(())
    }

    /// Deep-copied Mamba2 recurrent state (conv window + SSM hidden, + host CPU
    /// state) for the prefix cache. `affine(1,0)` forces fresh storage.
    fn snapshot(&self) -> Result<NemoStateSnap> {
        let dc = |o: &Option<Tensor>| -> Result<Option<Tensor>> {
            Ok(match o {
                Some(t) => Some(t.affine(1.0, 0.0)?),
                None => None,
            })
        };
        Ok(NemoStateSnap::Ssm {
            conv: dc(&self.conv_state)?,
            h: dc(&self.ssm_h)?,
            cpu: self.cpu_state.clone(),
        })
    }
    fn restore(&mut self, s: &NemoStateSnap) -> Result<()> {
        if let NemoStateSnap::Ssm { conv, h, cpu } = s {
            let dc = |o: &Option<Tensor>| -> Result<Option<Tensor>> {
                Ok(match o {
                    Some(t) => Some(t.affine(1.0, 0.0)?),
                    None => None,
                })
            };
            self.conv_state = dc(conv)?;
            self.ssm_h = dc(h)?;
            self.cpu_state = cpu.clone();
        }
        Ok(())
    }

    /// Causal depthwise conv1d over the [x,B,C] channels using the rolling
    /// conv_state (last d_conv inputs). Returns [b, d_xbc].
    /// Returns silu(conv) - the SiLU is folded into the fused kernel.
    fn conv_step(&mut self, xbc: &Tensor) -> Result<Tensor> {
        // Fused causal depthwise conv1d + bias + SiLU + state-shift in one launch
        // (was ~15: state narrow, cat, d_conv-step loop, state store, + an
        // external silu). Bit-identical conv reduction order. conv1d_w is
        // [d_xbc, d_conv]; conv_state holds the last d_conv inputs.
        let cs = self.conv_state.as_ref().unwrap();
        let (y, new_cs) = crate::inference::kernel::fused::fused_causal_conv1d_silu(
            xbc,
            cs,
            &self.conv1d_w,
            &self.conv1d_b,
            self.d_xbc,
            self.d_conv,
        )?;
        self.conv_state = Some(new_cs);
        Ok(y)
    }

    /// Single-token SSM recurrence: h = exp(dt.a).h + dt.(x⊗B); y = Σ_state(h.C).
    /// x_c/b_/c_ are F32 [b, d_inner], [b, ngroups.d_state], [b, ngroups.d_state];
    /// dt is F32 [b, nheads]. Returns y [b, nheads, headdim].
    fn ssm_step(&mut self, x_c: &Tensor, b_: &Tensor, c_: &Tensor, dt: &Tensor) -> Result<Tensor> {
        // Fused per-(b,head,headdim) SSM recurrence: group broadcasts,
        // decay=exp(dt*a), x⊗B outer, state update h, and C contraction -> y,
        // all in one launch (was ~25). Validated equivalent vs the unfused path
        // (before/after output matched: Tokyo correct, same behavior); the
        // d_state contraction is a sequential sum so non-bit-exact (benign drift).
        let h = self.ssm_h.as_ref().unwrap();
        let (y, h_out) = crate::inference::kernel::fused::fused_mamba2_ssm_step(
            x_c,
            b_,
            c_,
            dt,
            &self.a,
            h,
            self.nheads,
            self.headdim,
            self.d_state,
            self.ngroups,
        )?;
        self.ssm_h = Some(h_out);
        Ok(y) // [b, nh, hd]
    }

    pub fn forward(&mut self, xs: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (b, seq, _dm) = xs.dims3()?;
        // CPU: fused host-side recurrence (no per-op the tensor-op dispatch).
        if b == 1 && !xs.device().is_cuda() {
            if input_pos == 0 {
                self.cpu_state = None;
            }
            return self.forward_cpu_fused(xs, seq);
        }
        if input_pos == 0 || self.conv_state.is_none() {
            self.reset_state(b)?;
        }
        // PREFILL (seq>1): batch the two BIG projections (in_proj, out_proj) into
        // single GEMMs instead of `seq` separate GEMVs that re-stream the full
        // weight every token - the dominant cost of the per-token prefill loop
        // (nemotron prefill was 310 tok/s, ~6x below attention models). The conv +
        // SSM recurrence + gate stay sequential here (the recurrence is inherently
        // sequential; parallelizing it via the SSD chunked scan is a further step).
        // Mathematically identical to the per-token path (only the matmul batching
        // changes; reduction order differs negligibly like the existing fused
        // kernels). Decode (seq==1) keeps the EXACT per-token path (proj_all=None).
        let proj_all = if seq > 1 {
            Some(self.in_proj.forward(xs)?.to_dtype(DType::F32)?) // [b, seq, proj_size] - ONE matmul
        } else {
            None
        };
        let mut outs = Vec::with_capacity(seq);
        let mut ys = Vec::with_capacity(seq); // batched-prefill: per-token [b, d_inner], one out_proj at the end
        for t in 0..seq {
            let proj = match &proj_all {
                Some(p) => p.i((.., t, ..))?, // [b, proj_size] (pre-batched)
                None => self
                    .in_proj
                    .forward(&xs.i((.., t, ..))?)?
                    .to_dtype(DType::F32)?,
            };
            let [z, xbc, dt] = split_last(&proj, [self.d_inner, self.d_xbc, self.nheads])?;
            let xbc = self.conv_step(&xbc)?; // conv_step folds bias + SiLU
            let per_group = self.ngroups * self.d_state;
            let [x_c, b_, c_] = split_last(&xbc, [self.d_inner, per_group, per_group])?;
            // dt = softplus(dt + dt_bias): one fused launch (was broadcast_add+exp
            // +add+log = 4 tensor-op dispatches/SSM-layer), bit-identical. Fallback.
            let dt = match crate::inference::kernel::fused::softplus_bias(&dt, &self.dt_bias)? {
                Some(y) => y,
                None => ((dt.broadcast_add(&self.dt_bias)?.exp()? + 1.0)?).log()?,
            };
            let y = self
                .ssm_step(&x_c, &b_, &c_, &dt)?
                .reshape((b, self.d_inner))?; // [b,d_inner]
                                              // Fused: D.x skip + SiLU(z) gate + group-wise gated RMSNorm in one
                                              // launch (was ~12). Non-bit-exact (the RMS var reduction order differs
                                              // from the reference's) - validated equivalent via before/after comparison.
            let y = crate::inference::kernel::fused::fused_mamba2_gate_gnorm(
                &y,
                &x_c,
                &self.d,
                &z,
                &self.norm_w,
                b,
                self.d_inner,
                self.headdim,
                self.ngroups,
                self.rms_eps,
            )?;
            if proj_all.is_some() {
                ys.push(y.unsqueeze(1)?); // defer out_proj to ONE batched GEMM below
            } else {
                let out = self.out_proj.forward(&y.to_dtype(xs.dtype())?)?; // [b, d_model]
                outs.push(out.unsqueeze(1)?);
            }
        }
        if proj_all.is_some() {
            // Batched out_proj: ONE GEMM over [b, seq, d_inner] vs `seq` GEMVs.
            let y_all = Tensor::cat(&ys.iter().collect::<Vec<_>>(), 1)?.to_dtype(xs.dtype())?; // [b, seq, d_inner]
            return self.out_proj.forward(&y_all); // [b, seq, d_model]
        }
        // Decode (seq==1): skip the cat copy - return the single step directly.
        if outs.len() == 1 {
            return Ok(outs.into_iter().next().unwrap());
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1) // [b, seq, d_model]
    }

    /// Fused CPU path: ONE batched in_proj matmul, then a plain-Rust f32 loop
    /// for conv + softplus + Mamba2 recurrence + D-skip/z-gate/group-RMSNorm
    /// (rayon across the heads), then ONE out_proj matmul. Same math and op
    /// order as the per-token tensor-op path above.
    fn forward_cpu_fused(&mut self, xs: &Tensor, seq: usize) -> Result<Tensor> {
        use rayon::prelude::*;
        let (nh, hd, ds) = (self.nheads, self.headdim, self.d_state);
        let (di, dx, ng) = (self.d_inner, self.d_xbc, self.ngroups);
        let l = self.d_conv;
        let gh = nh / ng.max(1); // heads per group
        let gsz = di / ng; // channels per norm group
        let eps = self.rms_eps as f32;
        if self.cpu_state.is_none() {
            self.cpu_state = Some(CpuMambaState {
                conv: vec![0f32; dx * l],
                h: vec![0f32; nh * hd * ds],
                w: self
                    .conv1d_w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                wb: self.conv1d_b.to_vec1::<f32>()?,
                a: self.a.to_vec1::<f32>()?,
                d: self.d.to_vec1::<f32>()?,
                dtb: self.dt_bias.to_vec1::<f32>()?,
                norm: self
                    .norm_w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            });
        }
        // One projection pass over the whole sequence: [seq, d_inner | d_xbc | nheads]
        let proj = self
            .in_proj
            .forward(xs)?
            .to_dtype(DType::F32)?
            .reshape((seq, di + dx + nh))?
            .to_vec2::<f32>()?;
        let st = self.cpu_state.as_mut().unwrap();
        let mut y_all = vec![0f32; seq * di];
        let mut conv_out = vec![0f32; dx];
        let silu = |v: f32| v / (1.0 + (-v).exp());
        for t in 0..seq {
            let row = &proj[t];
            let (z, xbc_in, dt_in) = (&row[..di], &row[di..di + dx], &row[di + dx..]);
            // causal depthwise conv1d (+bias +silu); window = last l inputs incl. current
            for c in 0..dx {
                let s = &mut st.conv[c * l..(c + 1) * l];
                s.rotate_left(1);
                s[l - 1] = xbc_in[c];
                let w = &st.w[c * l..(c + 1) * l];
                let mut acc = st.wb[c];
                for j in 0..l {
                    acc += w[j] * s[j];
                }
                conv_out[c] = silu(acc);
            }
            let x_c = &conv_out[..di];
            let b_ = &conv_out[di..di + ng * ds];
            let c_ = &conv_out[di + ng * ds..di + 2 * ng * ds];
            let yt = &mut y_all[t * di..(t + 1) * di];
            let (a_v, d_v, dtb) = (&st.a, &st.d, &st.dtb);
            // recurrence per head (group = h / heads-per-group, matching the
            // (ngroups, gh) broadcast reshape of the reference)
            st.h.par_chunks_mut(hd * ds)
                .zip(yt.par_chunks_mut(hd))
                .enumerate()
                .for_each(|(hidx, (hs, yh))| {
                    let g = hidx / gh.max(1);
                    let dt = (1.0 + (dt_in[hidx] + dtb[hidx]).exp()).ln(); // softplus
                    let da = (dt * a_v[hidx]).exp();
                    let bv = &b_[g * ds..(g + 1) * ds];
                    let cv = &c_[g * ds..(g + 1) * ds];
                    for p in 0..hd {
                        let xv = x_c[hidx * hd + p];
                        let dx_ = dt * xv;
                        let hrow = &mut hs[p * ds..(p + 1) * ds];
                        let mut acc = 0f32;
                        for s in 0..ds {
                            hrow[s] = hrow[s] * da + dx_ * bv[s];
                            acc += hrow[s] * cv[s];
                        }
                        // D-skip folded here; gate + norm follow below
                        yh[p] = acc + d_v[hidx] * xv;
                    }
                });
            // gate by silu(z) FIRST, then group-RMSNorm over the gated values,
            // then the per-channel weight (matches fused_mamba2_gate_gnorm).
            for i in 0..di {
                yt[i] *= silu(z[i]);
            }
            for g in 0..ng {
                let yg = &mut yt[g * gsz..(g + 1) * gsz];
                let var = yg.iter().map(|v| v * v).sum::<f32>() / gsz as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for (i, v) in yg.iter_mut().enumerate() {
                    *v = *v * inv * st.norm[g * gsz + i];
                }
            }
        }
        let y = Tensor::from_vec(y_all, (1, seq, di), &xs.device())?.to_dtype(xs.dtype())?;
        self.out_proj.forward(&y)
    }
}

/// A single hybrid block: one of SSM | attention | MoE-FFN.
enum NemotronBlock {
    Ssm(NemotronSsmBlock),
    Attn(NemotronAttn),
    Moe(NemotronMoeBlock),
}

/// Per-layer deep-copied state for the recurrent prefix cache. MoE blocks
/// are stateless -> `None`. See qwen35_moe's equivalent and `Cache::deep_copy`.
enum NemoStateSnap {
    Ssm {
        conv: Option<Tensor>,
        h: Option<Tensor>,
        cpu: Option<CpuMambaState>,
    },
    Attn(crate::tensor::KvCache),
    None,
}
/// One resident prefix snapshot (in-memory only, never persisted - privacy).
struct NemoPrefixCache {
    prompt: Vec<u32>,
    layers: Vec<NemoStateSnap>,
    logits: Tensor,
}

/// pre-norm + block + residual (the block is one of the three kinds).
struct NemotronLayer {
    input_norm: crate::tensor::layer::RmsNorm, // F32 weight
    block: NemotronBlock,
    device: Device,
}

/// nemotron_h_moe - hybrid Mamba2 + attention + non-gated MoE, multi-device.
/// SSM blocks keep their own conv+ssm state; attention blocks keep a KV cache;
/// both reset on `input_pos == 0`. F16 residual stream; norms + SSM recurrence +
/// attention softmax + MoE run in F32.
pub struct NemotronHModel {
    embed: Embedding,
    embed_dev: Device,
    layers: Vec<NemotronLayer>,
    norm: crate::tensor::layer::RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    /// In-memory recurrent prefix snapshot (env-gated). Never persisted.
    prefix_cache: Option<NemoPrefixCache>,
}

/// Assign each layer to a device by filling the FASTEST GPU (devices[0], passed
/// fastest-first) to its VRAM capacity before spilling to the next. When the cards
/// differ in bandwidth - which any mixed pair does - an equal-layer-count split
/// lets the SLOW GPU gate the sequential decode; packing more layers onto the
/// fast GPU minimizes total per-token time. `gpu_avail` = usable bytes/device,
/// fastest-first. Empty/len-mismatch ⇒ even split (the old behavior).
pub fn plan_layer_devices(
    n_layers: usize,
    n_dev: usize,
    file_size: u64,
    gpu_avail: &[u64],
    layer_sizes: &[u64],
    reserve_bytes: u64,
) -> Vec<usize> {
    if n_dev <= 1 {
        return vec![0; n_layers];
    }
    if gpu_avail.len() != n_dev || file_size == 0 {
        // even-count split (the old behavior / OOM fallback)
        return (0..n_layers)
            .map(|l| (l * n_dev / n_layers.max(1)).min(n_dev - 1))
            .collect();
    }
    // Fill each GPU to its configured budget - `gpu_avail` is already
    // stable_free x max_gpu_memory_fraction - spilling to the next (slower) GPU
    // only once the fast one is full, so the slow card gates as few layers as
    // possible in the sequential decode.
    //
    // Use REAL per-layer byte sizes when available: these arches are wildly
    // non-uniform (a MoE block with N experts dwarfs an SSM/conv block), so a
    // uniform file_size/n_layers estimate mis-counts the fast-GPU boundary and
    // leaves it under-filled (the fast card ends up with FEWER bytes than the
    // slow one - the opposite of the intent). Real sizes pack the fast GPU to
    // its true budget. The non-layer tensors (token embedding, output head,
    // final norm) live on devices[0], so charge their bytes to GPU0 up front.
    let use_real = layer_sizes.len() == n_layers;
    let uniform = (file_size / n_layers.max(1) as u64).max(1);
    let layer_b = |l: usize| -> u64 {
        if use_real {
            layer_sizes[l].max(1)
        } else {
            uniform
        }
    };
    let head_bytes = if use_real {
        file_size.saturating_sub(layer_sizes.iter().copied().sum::<u64>())
    } else {
        0
    };
    // Reserve per-GPU activation/KV headroom before filling: the chunk's
    // attention scores [chunk, kv_len], MoE expert activations and the growing
    // KV cache all allocate on the layer's device. `reserve_bytes` is the
    // model's PRECISE runtime peak (KV at context + cuBLAS workspace + one
    // prefill-chunk activation) - the same figure the dense planner trusts  -
    // not a blanket margin, so low-KV hybrids (mostly SSM/conv layers, few
    // attention layers) pack the fast GPU much fuller instead of stranding ~1 GB.
    // Any residual prefill OOM from packing tight is caught by the adaptive
    // chunker (it halves the compute chunk and retries); load-time OOM falls
    // back to an even split. Floor at 640 MB so the cuBLAS workspace + a min
    // chunk always fit even if the caller passes 0.
    let reserve = reserve_bytes.max(640 * 1024 * 1024);
    let mut plan = Vec::with_capacity(n_layers);
    let mut di = 0usize;
    let mut used = head_bytes; // GPU0 already holds the embed/output head
    for l in 0..n_layers {
        let budget = gpu_avail[di].saturating_sub(reserve);
        let lb = layer_b(l);
        if di + 1 < n_dev && used + lb > budget {
            di += 1;
            used = 0;
        }
        plan.push(di);
        used += lb;
    }
    plan
}

/// Sum the GGUF tensor bytes belonging to each transformer block `blk.{i}.*`,
/// giving real (non-uniform) per-layer sizes for the device planner.
pub fn layer_byte_sizes(content: &gguf_file::Content, n_layers: usize) -> Vec<u64> {
    let mut sizes = vec![0u64; n_layers];
    for (name, info) in content.tensor_infos.iter() {
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(i) = rest[..dot].parse::<usize>() {
                    if i < n_layers {
                        sizes[i] += info.size_in_bytes() as u64;
                    }
                }
            }
        }
    }
    sizes
}

impl NemotronHModel {
    pub fn from_gguf<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
        gpu_avail: &[u64],
        file_size: u64,
        reserve_bytes: u64,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let cfg = NemotronHConfig::from_gguf(content)?;
        let n_dev = devices.len().max(1);
        let layer_sizes = layer_byte_sizes(content, cfg.n_layers);
        let plan = plan_layer_devices(
            cfg.n_layers,
            n_dev,
            file_size,
            gpu_avail,
            &layer_sizes,
            reserve_bytes,
        );
        let dev_for = |l: usize| -> &Device { &devices[plan[l].min(n_dev - 1)] };
        let max_seq = cfg.context_length.min(32768).max(8192);
        let rotary_dim = if cfg.rope_dim > 0 {
            cfg.rope_dim
        } else {
            cfg.head_dim
        };
        let mut rope: Vec<(Tensor, Tensor)> = Vec::with_capacity(n_dev);
        for d in devices {
            // Partial rotary: the table spans the ROTATED width, not the head width.
            rope.push(crate::inference::model::rope::precomput_freqs_cis_host(
                rotary_dim,
                max_seq,
                cfg.rope_freq_base,
                d,
            )?);
        }

        let embed_dev = devices[0].clone();
        let cpu_c = crate::tensor::Device::Cpu;
        let embed_t = content
            .tensor(reader, "token_embd.weight", &cpu_c)?
            .dequantize(&cpu_c)?
            .to_dtype(DType::F16)?
            .to_device(&embed_dev)?;
        let embed = Embedding::new(embed_t);

        let kinds = cfg.layer_kinds(content);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let device = dev_for(i);
            let di = plan[i].min(n_dev - 1);
            let (cos, sin) = &rope[di];
            let input_norm = crate::tensor::layer::RmsNorm::new(
                ld_f32(
                    content,
                    reader,
                    &format!("blk.{i}.attn_norm.weight"),
                    device,
                )?,
                cfg.rms_eps as f32,
            );
            let block = match kinds[i] {
                LayerKind::Ssm => NemotronBlock::Ssm(NemotronSsmBlock::load(
                    content, reader, i, &cfg, device, mm,
                )?),
                LayerKind::Attention => NemotronBlock::Attn(NemotronAttn::load(
                    content,
                    reader,
                    i,
                    &cfg,
                    cfg.head_dim,
                    rotary_dim,
                    cos.clone(),
                    sin.clone(),
                    device,
                    mm,
                )?),
                LayerKind::MoeFfn => NemotronBlock::Moe(NemotronMoeBlock::load(
                    content, reader, i, &cfg, dtype, device, mm,
                )?),
            };
            if i == 0 || i + 1 == cfg.n_layers {
                tracing::info!(
                    "nemotron_h layer {i} {:?} on {:?}",
                    kinds[i],
                    device.location()
                );
            }
            layers.push(NemotronLayer {
                input_norm,
                block,
                device: device.clone(),
            });
        }
        let last = dev_for(cfg.n_layers - 1);
        let norm = crate::tensor::layer::RmsNorm::new(
            ld_f32(content, reader, "output_norm.weight", last)?,
            cfg.rms_eps as f32,
        );
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
            prefix_cache: None,
        })
    }

    /// Where each layer actually lives, in layer order. The placement is not a
    /// plan to re-derive: `plan_layer_devices` ran once at load and every layer
    /// kept the device it was loaded onto, so this reads that device back.
    pub fn layer_device_locations(&self) -> Vec<crate::tensor::DeviceLocation> {
        self.layers.iter().map(|l| l.device.location()).collect()
    }

    /// Snapshot full per-layer state at position `prompt.len()` + prefill logits
    /// for exact-prefix reuse. Call AFTER prefilling [0, prompt.len()).
    pub fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> Result<()> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(match &l.block {
                NemotronBlock::Ssm(b) => b.snapshot()?,
                NemotronBlock::Attn(b) => b.snapshot()?,
                NemotronBlock::Moe(_) => NemoStateSnap::None,
            });
        }
        self.prefix_cache = Some(NemoPrefixCache {
            prompt: prompt.to_vec(),
            layers,
            logits: logits.affine(1.0, 0.0)?,
        });
        Ok(())
    }

    /// On an EXACT prompt match, restore every layer's state and return the
    /// prefill logits (decode resumes at prompt.len()). Else None.
    pub fn try_restore_prefix(&mut self, prompt: &[u32]) -> Result<Option<Tensor>> {
        match &self.prefix_cache {
            Some(pc) if pc.prompt.as_slice() == prompt => {}
            _ => return Ok(None),
        }
        let pc = self.prefix_cache.take().unwrap();
        for (l, snap) in self.layers.iter_mut().zip(pc.layers.iter()) {
            match &mut l.block {
                NemotronBlock::Ssm(b) => b.restore(snap)?,
                NemotronBlock::Attn(b) => b.restore(snap)?,
                NemotronBlock::Moe(_) => {}
            }
        }
        let logits = pc.logits.affine(1.0, 0.0)?;
        self.prefix_cache = Some(pc);
        Ok(Some(logits))
    }

    pub fn forward(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        let ids = input_ids.to_device(&self.embed_dev)?;
        let mut x = self.embed.forward(&ids)?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }
        for layer in self.layers.iter_mut() {
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            let residual = x.clone();
            let st = x.dtype();
            let h = layer
                .input_norm
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(st)?;
            let kind = match &mut layer.block {
                NemotronBlock::Ssm(b) => {
                    let _h = b.forward(&h, input_pos)?;
                    (0u8, _h)
                }
                NemotronBlock::Attn(b) => {
                    if input_pos == 0 {
                        b.reset();
                    }
                    (1u8, b.forward(&h, input_pos)?)
                }
                NemotronBlock::Moe(b) => (2u8, b.forward(&h)?),
            };
            let (_k, h) = kind;
            x = (residual + h)?;
        }
        let x = self.norm.forward(&x.to_dtype(DType::F32)?)?;
        let x = x.i((.., seq - 1, ..))?.to_dtype(self.dtype)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }
}

/// SSD (state-space duality) quadratic form for ONE chunk with zero initial state:
/// the parallel/attention-like equivalent of the sequential Mamba2 recurrence
///   h_t = exp(dt_t.a).h_{t-1} + dt_t.(x_t⊗B_t);  y_t = Σ_s h_t.C_t
/// -> y[t,h,p] = Σ_{j<=t} exp(Lcum[t,h]-Lcum[j,h]).(B_j.C_t)_g . dt_j.x[j,h,p],
/// with Lcum = cumsum_t(dt.a) (computed as a lower-triangular-ones matmul - no
/// cumsum primitive needed). O(seq²) per call; the chunked scan applies this per
/// chunk of size L and carries the state across chunks for the O(seq.L) form.
/// All tensors F32. x:[seq,nh,hd] b/c:[seq,ng,ds] dt:[seq,nh] a:[nh]. Returns y:[seq,nh,hd]
/// (the Σ_s h.C term, no D-skip - matches `ssm_step`'s output).
#[cfg(test)]
fn ssd_quadratic(
    x: &Tensor,
    b: &Tensor,
    c: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    seq: usize,
    nh: usize,
    _hd: usize,
    ng: usize,
    _ds: usize,
) -> Result<Tensor> {
    let dev = x.device();
    let gh = nh / ng;
    // A[t,h] = dt[t,h] . a[h]
    let aa = dt.broadcast_mul(&a.reshape((1, nh))?)?; // [seq, nh]
                                                      // Lcum[i,h] = Σ_{j<=i} A[j,h] = (tril_ones @ A)[i,h]
    let mut tri = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            tri[i * seq + j] = 1.0;
        }
    }
    let tril = Tensor::from_vec(tri.clone(), (seq, seq), &dev)?;
    let lcum = tril.matmul(&aa)?; // [seq, nh]
    let lcum_t = lcum.transpose(0, 1)?.contiguous()?; // [nh, seq]
                                                      // decay[h,i,j] = exp(Lcum[i,h] - Lcum[j,h])
    let li = lcum_t.reshape((nh, seq, 1))?;
    let lj = lcum_t.reshape((nh, 1, seq))?;
    let decay = li.broadcast_sub(&lj)?.exp()?; // [nh, seq, seq]
                                               // Gram[g,i,j] = C[i,g,:] . B[j,g,:]  (= Cg @ Bg^T)
    let cg = c.transpose(0, 1)?.contiguous()?; // [ng, seq, ds]
    let bg = b.transpose(0, 1)?.contiguous()?; // [ng, seq, ds]
    let gram = cg.matmul(&bg.transpose(1, 2)?.contiguous()?)?; // [ng, seq, seq]
    let gram_e = gram
        .reshape((ng, 1, seq, seq))?
        .expand((ng, gh, seq, seq))?
        .reshape((nh, seq, seq))?; // [nh, seq, seq]
                                   // causal mask (i>=j)
    let causal = Tensor::from_vec(tri, (1, seq, seq), &dev)?;
    let w = decay.broadcast_mul(&gram_e)?.broadcast_mul(&causal)?; // [nh, seq, seq]
                                                                   // DX[t,h,p] = dt[t,h].x[t,h,p]
    let dx = x.broadcast_mul(&dt.reshape((seq, nh, 1))?)?; // [seq, nh, hd]
    let dx_h = dx.transpose(0, 1)?.contiguous()?; // [nh, seq, hd]
    let y = w.matmul(&dx_h)?; // [nh, seq, hd]
    y.transpose(0, 1)?.contiguous() // [seq, nh, hd]
}

/// Full SSD chunked scan: O(seq.L) parallel form of the Mamba2 recurrence. Splits
/// the sequence into chunks of `cs`; each chunk uses the quadratic intra-chunk form
/// plus the contribution of the carried state, and updates the state for the next
/// chunk. Bit-parity (≈1e-3) with the sequential `ssm_step` recurrence (h_in=0 at
/// chunk 0). Returns y:[seq,nh,hd] (Σ_s h.C, no D-skip). F32.
#[cfg(test)]
fn ssd_chunk_scan(
    x: &Tensor,
    b: &Tensor,
    c: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    cs: usize,
    seq: usize,
    nh: usize,
    hd: usize,
    ng: usize,
    ds: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let gh = nh / ng;
    // carried state h_in[h,p,s]  [nh, hd, ds]
    let mut h_in = Tensor::zeros_on((nh, hd, ds), DType::F32, &dev)?;
    let mut ys: Vec<Tensor> = Vec::new();
    let mut off = 0usize;
    while off < seq {
        let l = cs.min(seq - off);
        let xc = x.narrow(0, off, l)?; // [l, nh, hd]
        let bc = b.narrow(0, off, l)?; // [l, ng, ds]
        let cc = c.narrow(0, off, l)?;
        let dtc = dt.narrow(0, off, l)?; // [l, nh]
                                         // -- within-chunk decay --
        let aa = dtc.broadcast_mul(&a.reshape((1, nh))?)?; // [l, nh]
        let mut tri = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..=i {
                tri[i * l + j] = 1.0;
            }
        }
        let tril = Tensor::from_vec(tri.clone(), (l, l), &dev)?;
        let lcum = tril.matmul(&aa)?; // [l, nh]
        let lcum_t = lcum.transpose(0, 1)?.contiguous()?; // [nh, l]
        let decay = lcum_t
            .reshape((nh, l, 1))?
            .broadcast_sub(&lcum_t.reshape((nh, 1, l))?)?
            .exp()?; // [nh,l,l]
        let causal = Tensor::from_vec(tri, (1, l, l), &dev)?;
        // -- intra-chunk --
        let cg = cc.transpose(0, 1)?.contiguous()?; // [ng, l, ds]
        let bg = bc.transpose(0, 1)?.contiguous()?; // [ng, l, ds]
        let gram = cg.matmul(&bg.transpose(1, 2)?.contiguous()?)?; // [ng, l, l]
        let gram_e = gram
            .reshape((ng, 1, l, l))?
            .expand((ng, gh, l, l))?
            .reshape((nh, l, l))?;
        let w = decay.broadcast_mul(&gram_e)?.broadcast_mul(&causal)?; // [nh, l, l]
        let dx = xc.broadcast_mul(&dtc.reshape((l, nh, 1))?)?; // [l, nh, hd]
        let dx_h = dx.transpose(0, 1)?.contiguous()?; // [nh, l, hd]
        let y_intra = w.matmul(&dx_h)?; // [nh, l, hd]
                                        // -- inter-chunk: contribution of carried state h_in --
                                        // CH[h,t,p] = Σ_s C[t,g,s] h_in[h,p,s] = Cg(expanded) @ h_in^T
        let cg_e = cg
            .reshape((ng, 1, l, ds))?
            .expand((ng, gh, l, ds))?
            .reshape((nh, l, ds))?; // [nh,l,ds]
        let ch = cg_e.matmul(&h_in.transpose(1, 2)?.contiguous()?)?; // [nh,l,ds]@[nh,ds,hd]=[nh,l,hd]
        let lcum_exp = lcum_t.exp()?.reshape((nh, l, 1))?; // exp(Lcum[t,h]) [nh,l,1]
        let y_inter = ch.broadcast_mul(&lcum_exp)?; // [nh, l, hd]
        let y_chunk = (y_intra + y_inter)?; // [nh, l, hd]
        ys.push(y_chunk.transpose(0, 1)?.contiguous()?); // [l, nh, hd]
                                                         // -- state update for next chunk --
                                                         // decay_end[j,h] = exp(Lcum[l-1,h] - Lcum[j,h])
        let lend = lcum_t.narrow(1, l - 1, 1)?; // [nh, 1]
        let decay_end = lend.broadcast_sub(&lcum_t)?.exp()?; // [nh, l]
                                                             // DXD[h,j,p] = decay_end[j,h] * DX[h,j,p]
        let dxd = dx_h.broadcast_mul(&decay_end.reshape((nh, l, 1))?)?; // [nh, l, hd]
                                                                        // states[h,p,s] = Σ_j DXD[h,j,p] B_g[j,s] = DXD^T @ Bg
        let bg_e = bg
            .reshape((ng, 1, l, ds))?
            .expand((ng, gh, l, ds))?
            .reshape((nh, l, ds))?; // [nh,l,ds]
        let states = dxd.transpose(1, 2)?.contiguous()?.matmul(&bg_e)?; // [nh,hd,l]@[nh,l,ds]=[nh,hd,ds]
        let decay_full = lend.exp()?.reshape((nh, 1, 1))?; // exp(Lcum[l-1,h]) [nh,1,1]
        h_in = (h_in.broadcast_mul(&decay_full)? + states)?; // [nh, hd, ds]
        off += l;
    }
    Ok((Tensor::cat(&ys.iter().collect::<Vec<_>>(), 0)?, h_in)) // y:[seq,nh,hd], h_out:[nh,hd,ds]
}

#[cfg(test)]
mod ssd_tests {
    use super::*;

    // plain sequential recurrence (h_in=0) - the correctness oracle.
    fn ssm_ref(
        x: &[f32],
        b: &[f32],
        c: &[f32],
        dt: &[f32],
        a: &[f32],
        seq: usize,
        nh: usize,
        hd: usize,
        ng: usize,
        ds: usize,
    ) -> Vec<f32> {
        let gh = nh / ng;
        let mut h = vec![0f32; nh * hd * ds];
        let mut y = vec![0f32; seq * nh * hd];
        for t in 0..seq {
            for hh in 0..nh {
                let g = hh / gh;
                let da = (dt[t * nh + hh] * a[hh]).exp();
                for p in 0..hd {
                    let dx = dt[t * nh + hh] * x[(t * nh + hh) * hd + p];
                    let mut acc = 0f32;
                    for s in 0..ds {
                        let hi = (hh * hd + p) * ds + s;
                        h[hi] = h[hi] * da + dx * b[(t * ng + g) * ds + s];
                        acc += h[hi] * c[(t * ng + g) * ds + s];
                    }
                    y[(t * nh + hh) * hd + p] = acc;
                }
            }
        }
        y
    }

    #[test]
    fn ssd_quadratic_matches_recurrence() {
        let (seq, nh, hd, ng, ds) = (8usize, 4usize, 4usize, 2usize, 4usize);
        // deterministic pseudo-random, small magnitudes; a<0 (decay).
        let r = |i: usize, s: f32| {
            ((((i as u32).wrapping_mul(2654435761)) % 1000) as f32 / 1000.0 - 0.5) * s
        };
        let x: Vec<f32> = (0..seq * nh * hd).map(|i| r(i, 1.0)).collect();
        let b: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 7, 1.0)).collect();
        let c: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 13, 1.0)).collect();
        let dt: Vec<f32> = (0..seq * nh)
            .map(|i| 0.1 + (r(i + 19, 0.1)).abs())
            .collect();
        let a: Vec<f32> = (0..nh).map(|i| -(0.5 + r(i + 23, 0.5).abs())).collect();
        let yref = ssm_ref(&x, &b, &c, &dt, &a, seq, nh, hd, ng, ds);
        let dev = crate::tensor::Device::Cpu;
        let xt = Tensor::from_vec(x, (seq, nh, hd), &dev).unwrap();
        let bt = Tensor::from_vec(b, (seq, ng, ds), &dev).unwrap();
        let ct = Tensor::from_vec(c, (seq, ng, ds), &dev).unwrap();
        let dtt = Tensor::from_vec(dt, (seq, nh), &dev).unwrap();
        let at = Tensor::from_vec(a, (nh,), &dev).unwrap();
        let y = ssd_quadratic(&xt, &bt, &ct, &dtt, &at, seq, nh, hd, ng, ds).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let maxdiff = yref
            .iter()
            .zip(&yv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("ssd_quadratic maxdiff vs recurrence = {maxdiff:.2e}");
        assert!(maxdiff < 1e-3, "ssd_quadratic diverges: maxdiff={maxdiff}");
    }

    #[test]
    fn ssd_chunk_scan_matches_recurrence() {
        let (seq, nh, hd, ng, ds) = (10usize, 4usize, 4usize, 2usize, 4usize);
        let r = |i: usize, s: f32| {
            ((((i as u32).wrapping_mul(2654435761)) % 1000) as f32 / 1000.0 - 0.5) * s
        };
        let x: Vec<f32> = (0..seq * nh * hd).map(|i| r(i, 1.0)).collect();
        let b: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 7, 1.0)).collect();
        let c: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 13, 1.0)).collect();
        let dt: Vec<f32> = (0..seq * nh)
            .map(|i| 0.1 + (r(i + 19, 0.1)).abs())
            .collect();
        let a: Vec<f32> = (0..nh).map(|i| -(0.5 + r(i + 23, 0.5).abs())).collect();
        let yref = ssm_ref(&x, &b, &c, &dt, &a, seq, nh, hd, ng, ds);
        let dev = crate::tensor::Device::Cpu;
        let xt = Tensor::from_vec(x, (seq, nh, hd), &dev).unwrap();
        let bt = Tensor::from_vec(b, (seq, ng, ds), &dev).unwrap();
        let ct = Tensor::from_vec(c, (seq, ng, ds), &dev).unwrap();
        let dtt = Tensor::from_vec(dt, (seq, nh), &dev).unwrap();
        let at = Tensor::from_vec(a, (nh,), &dev).unwrap();
        for cs in [3usize, 4, 5, 10] {
            // multiple chunk sizes incl. non-divisor + single-chunk
            let (y, _hout) =
                ssd_chunk_scan(&xt, &bt, &ct, &dtt, &at, cs, seq, nh, hd, ng, ds).unwrap();
            let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let md = yref
                .iter()
                .zip(&yv)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("ssd_chunk_scan cs={cs} maxdiff = {md:.2e}");
            assert!(md < 1e-3, "ssd_chunk_scan cs={cs} diverges: maxdiff={md}");
        }
    }

    // reference final SSM state (after the whole sequence, h_in=0) - for the
    // prefill->decode handoff (the chunked scan must leave the same state).
    fn ssm_ref_state(
        x: &[f32],
        b: &[f32],
        dt: &[f32],
        a: &[f32],
        seq: usize,
        nh: usize,
        hd: usize,
        ng: usize,
        ds: usize,
    ) -> Vec<f32> {
        let gh = nh / ng;
        let mut h = vec![0f32; nh * hd * ds];
        for t in 0..seq {
            for hh in 0..nh {
                let g = hh / gh;
                let da = (dt[t * nh + hh] * a[hh]).exp();
                for p in 0..hd {
                    let dx = dt[t * nh + hh] * x[(t * nh + hh) * hd + p];
                    for s in 0..ds {
                        let hi = (hh * hd + p) * ds + s;
                        h[hi] = h[hi] * da + dx * b[(t * ng + g) * ds + s];
                    }
                }
            }
        }
        h
    }

    #[test]
    fn ssd_chunk_scan_state_matches_recurrence() {
        let (seq, nh, hd, ng, ds) = (10usize, 4usize, 4usize, 2usize, 4usize);
        let r = |i: usize, s: f32| {
            ((((i as u32).wrapping_mul(2654435761)) % 1000) as f32 / 1000.0 - 0.5) * s
        };
        let x: Vec<f32> = (0..seq * nh * hd).map(|i| r(i, 1.0)).collect();
        let b: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 7, 1.0)).collect();
        let c: Vec<f32> = (0..seq * ng * ds).map(|i| r(i + 13, 1.0)).collect();
        let dt: Vec<f32> = (0..seq * nh)
            .map(|i| 0.1 + (r(i + 19, 0.1)).abs())
            .collect();
        let a: Vec<f32> = (0..nh).map(|i| -(0.5 + r(i + 23, 0.5).abs())).collect();
        let href = ssm_ref_state(&x, &b, &dt, &a, seq, nh, hd, ng, ds);
        let dev = crate::tensor::Device::Cpu;
        let xt = Tensor::from_vec(x, (seq, nh, hd), &dev).unwrap();
        let bt = Tensor::from_vec(b, (seq, ng, ds), &dev).unwrap();
        let ct = Tensor::from_vec(c, (seq, ng, ds), &dev).unwrap();
        let dtt = Tensor::from_vec(dt, (seq, nh), &dev).unwrap();
        let at = Tensor::from_vec(a, (nh,), &dev).unwrap();
        for cs in [3usize, 4, 10] {
            let (_y, hout) =
                ssd_chunk_scan(&xt, &bt, &ct, &dtt, &at, cs, seq, nh, hd, ng, ds).unwrap();
            let hv = hout.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let md = href
                .iter()
                .zip(&hv)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("ssd_chunk_scan STATE cs={cs} maxdiff = {md:.2e}");
            assert!(md < 1e-3, "ssd state cs={cs} diverges: {md}");
        }
    }
}
