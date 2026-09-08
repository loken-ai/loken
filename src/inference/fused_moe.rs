//! Fused mixture-of-experts: the router, and the batched expert projections it feeds.
//!
//! The layer layout is adapted from vllm.rs `models/layers/moe.rs` (see NOTICE.md); the CUDA
//! kernels it drives are in `crate::inference::moe_cuda`.
use crate::tensor::layer::Linear;
use crate::tensor::ops::Activation;
use crate::tensor::quantized::{GgmlDType, QMatMul, QTensor};
use crate::tensor::IndexOp;
use crate::tensor::{DType, Device, Result, Tensor, D};
use std::sync::{Arc, OnceLock};

/// Per-expert quantized matmuls for the CPU MoE fast path. Built lazily on
/// first CPU forward by slicing each expert's contiguous quantized-block range
/// out of the `[E, R, C]` expert stacks and wrapping it in a `QMatMul`. This
/// lets decode run a **fused quantized GEMV** (`dot` over the GGUF blocks of
/// only the top-k selected experts) - like Ollama - instead of dequantizing the
/// entire expert stack to F32 every token. Shared across `Clone`s via `Arc`.
#[derive(Debug, Clone)]
pub(crate) struct CpuExperts {
    gate: Vec<QMatMul>, // each [N, K] (empty when gate_up is fused)
    up: Vec<QMatMul>,   // each [N, K] (empty when gate_up is fused)
    /// Fused gate‖up: each [2N, K]. When set, decode quantizes the expert input
    /// once and issues one GEMV instead of two, then splits the [.,2N] output.
    gate_up: Option<Vec<QMatMul>>,
    down: Vec<QMatMul>, // each [hidden, N]
}

/// Slice a 3-D quantized expert stack `[E, R, C]` into `E` per-expert `[rows, C]`
/// `QMatMul`s, taking rows `[start_row, start_row+rows)` of each expert. The
/// quantized blocks are laid out expert-major / row-major, so each sub-range is
/// contiguous in the raw byte buffer - no dequantization, just byte slicing.
fn split_experts_rows(full: &QTensor, start_row: usize, rows: usize) -> Result<Vec<QMatMul>> {
    let dims = full.shape().dims();
    let (e_count, _r_total, c) = (dims[0], dims[1], dims[2]);
    let dtype = full.dtype();
    let bs = dtype.block_size();
    let ts = dtype.type_size();
    let row_bytes = (c / bs) * ts;
    let expert_bytes = _r_total * row_bytes;
    let data = full.data()?; // Cow<[u8]> over the whole tensor (borrowed on CPU)
    let mut v = Vec::with_capacity(e_count);
    for e in 0..e_count {
        let base = e * expert_bytes + start_row * row_bytes;
        let slice = &data[base..base + rows * row_bytes];
        let qt = crate::tensor::quantized::QTensor::from_ggml_bytes(
            dtype,
            slice,
            vec![rows, c],
            &Device::Cpu,
        )?;
        // gpt-oss MXFP4 experts run natively: the CPU quant engine has an AVX2 MxFp4xQ8_0
        // dot (the 4-bit codes shuffle through the KVALUES_MXFP4 LUT). Keeping
        // the experts as MXFP4 reads half the bytes of the old Q8_0 workaround and
        // avoids the requant + the 2x RAM.
        v.push(QMatMul::from_qtensor(qt)?);
    }
    Ok(v)
}

/// Whether the CPU has a dot product that reads this format's blocks directly.
///
/// Written out rather than derived from the block size on purpose. It is an allow-list: a
/// format added to the table without a CPU dot must fall back to dequantise-then-matmul, and a
/// derivation would have it claim a path that does not exist. `Q8_1` is absent because it is an
/// activation format - nothing stores weights in it; `MxFp4` is here because the gpt-oss
/// experts reach a dot through a dequantise into `Q8_0`.
pub(crate) fn cpu_qmatmul_supported(dtype: GgmlDType) -> bool {
    use GgmlDType::*;
    matches!(
        dtype,
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | MxFp4
    )
}

/// The router's choices, grouped so that each expert's rows are one contiguous run.
///
/// `slots[i]` is the flattened `(token, choice)` pair the i-th row of the expert input is
/// built from, and `experts[i]` is the expert that row is multiplied by. Sorting the choices
/// by expert id is what turns a scattered assignment into one GEMM per run, and every expert
/// call takes the three together - so they travel together, and the calls hang off them.
struct ExpertPlan {
    slots: Tensor,
    experts: Tensor,
    topk: usize,
}

impl ExpertPlan {
    /// Group `[num_tokens, topk]` expert ids by expert.
    ///
    /// The sort is stable in all three arms, so the rows of one expert keep the order the
    /// router put them in and the arms differ only in where the work happens.
    fn group(choices: &Tensor, topk: usize) -> Result<Self> {
        let flat = choices.flatten_all()?.to_dtype(DType::U32)?;
        let n = flat.dim(0)?;
        let (experts, slots) =
            if let Some(sorted) = crate::inference::moe_cuda::argsort_small_u32(&flat)? {
                // Decode shape (n <= 32): one warp sorts it in place on the device. The generic
                // sort bounces through the host on CUDA - a pipeline stall at decode cadence and
                // a CUDA-graph capture blocker, since the copy sits inside the captured forward.
                sorted
            } else if n > 4096 {
                // Past the device sort's ceiling - qwen3-coder reaches it at seq ≈ 512, where
                // seq x topk = 4096, and the launch returns CUDA_ERROR_INVALID_VALUE beyond it.
                // Pull the keys back, sort them with their positions, hand both back: a few ms
                // per layer even at 128K (~1M keys), and only prefill is ever this wide.
                let device = flat.device();
                let mut keys: Vec<u32> = flat.to_device(&Device::Cpu)?.to_vec1()?;
                let mut order: Vec<u32> = (0..n as u32).collect();
                order.sort_by_key(|&i| keys[i as usize]);
                keys.sort();
                (
                    Tensor::from_vec(keys, (n,), &Device::Cpu)?.to_device(&device)?,
                    Tensor::from_vec(order, (n,), &Device::Cpu)?.to_device(&device)?,
                )
            } else {
                flat.sort_last_dim(true)?
            };
        Ok(Self {
            slots,
            experts,
            topk,
        })
    }

    /// One expert GEMM over the plan: every row multiplied by its own expert's weights.
    /// `scale` carries the router's weight per row when this launch is the one that reduces
    /// the rows of a token back together, and is `None` when it only expands them.
    #[cfg(feature = "cuda")]
    fn gemm(
        &self,
        input: &Tensor,
        weights: &QTensor,
        scale: &Option<Tensor>,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf(
            input,
            weights,
            scale,
            &self.slots,
            &self.experts,
            self.topk,
            is_prefill,
            dtype,
        )
    }

    /// Gate GEMM, up GEMM and `silu(gate).up` in one launch.
    #[cfg(feature = "cuda")]
    fn gate_up_silu_mul(&self, input: &Tensor, gate: &QTensor, up: &QTensor) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul(
            input,
            gate,
            up,
            &self.slots,
            &self.experts,
            self.topk,
        )
    }

    /// Gate GEMM, up GEMM, per-expert bias and the clamped SwiGLU in one launch (gpt-oss).
    #[cfg(feature = "cuda")]
    fn gate_up_swiglu_oai(
        &self,
        input: &Tensor,
        gate: &QTensor,
        up: &QTensor,
        gate_bias: Option<&Tensor>,
        up_bias: Option<&Tensor>,
        alpha: f64,
        limit: f64,
    ) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf_gate_up_swiglu_oai(
            input,
            gate,
            up,
            gate_bias,
            up_bias,
            &self.slots,
            &self.experts,
            self.topk,
            alpha,
            limit,
        )
    }

    /// How many tokens a block of expert rows covers: the plan holds `topk` of them each.
    fn tokens_in(&self, rows: &Tensor) -> Result<usize> {
        Ok(rows.dim(0)? / self.topk)
    }

    /// Down projection and the scatter-add back to one row per token: each row is scaled by
    /// the score its expert was chosen with and added into the token it came from, with a
    /// residual and a per-expert bias folded into the same accumulation when there are any.
    #[cfg(feature = "cuda")]
    fn device_down_reduce(
        &self,
        input: &Tensor,
        weights: &QTensor,
        scale: &Tensor,
        residual: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf_down_reduce(
            input,
            weights,
            &self.slots,
            &self.experts,
            scale,
            self.topk,
            self.tokens_in(input)?,
            residual,
            bias,
        )
    }

    /// [`Self::gate_up_silu_mul`] on the host, against a fused gate‖up stack.
    fn host_gate_up_silu_mul_fused(
        &self,
        input: &Tensor,
        gate_up: &Arc<QTensor>,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_gate_up_silu_mul_fused(
            input,
            gate_up,
            &self.slots,
            &self.experts,
            self.topk,
        )
    }

    /// [`Self::gate_up_silu_mul`] on the host, against separate gate and up stacks.
    fn host_gate_up_silu_mul(
        &self,
        input: &Tensor,
        gate: &Arc<QTensor>,
        up: &Arc<QTensor>,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_gate_up_silu_mul(
            input,
            gate,
            up,
            &self.slots,
            &self.experts,
            self.topk,
        )
    }

    /// [`Self::device_down_reduce`] on the host.
    fn host_down_reduce(
        &self,
        input: &Tensor,
        weights: &Arc<QTensor>,
        scale: &Tensor,
        residual: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
            input,
            weights,
            &self.slots,
            &self.experts,
            scale,
            self.topk,
            self.tokens_in(input)?,
            residual,
            bias,
        )
    }
}

#[derive(Debug, Clone)]
pub struct FusedMoeGGUF {
    pub gate: Linear,
    /// Separate gate weights - populated only when the load-time gate+up
    /// fusion failed (quant-type mismatch, shape mismatch, or build
    /// error). Mutually exclusive with `gate_up_experts`.
    pub gate_experts: Option<Arc<QTensor>>,
    /// Separate up weights - see `gate_experts`.
    pub up_experts: Option<Arc<QTensor>>,
    pub down_experts: Arc<QTensor>,
    /// Per-expert byte-concat of gate_experts and up_experts along the
    /// output dim. When present, the forward path runs ONE moe_gemm_gguf
    /// call against this fused tensor instead of two - saving one kernel
    /// launch and one quantize+input-load per MoE layer per token.
    /// Shape: [num_experts, 2 * moe_intermediate_size, hidden_size].
    /// Mutually exclusive with `gate_experts`/`up_experts`.
    pub gate_up_experts: Option<Arc<QTensor>>,
    pub act: Activation,
    /// gpt-oss (OPENAI_MOE) SwiGLU-OAI gating. When `Some((alpha, limit))`,
    /// the gate/up combine is
    ///   `(gate.σ(alpha.gate)).(1 + up)`  with  gate clamped to `<=limit`
    ///   and up clamped to `[-limit, limit]`
    /// instead of the default `silu(gate).up`. Matches ggml `swiglu_oai`
    /// (alpha≈1.702, limit≈7). Disables the SiLU kernel-fuse fast path,
    /// which hardcodes plain silu.
    pub swiglu_oai: Option<(f64, f64)>,
    /// gpt-oss biased MoE. All `[n_expert, ...]` F32, applied per selected
    /// expert (gpt-oss puts a bias on the router and every expert projection):
    /// - `gate_inp_bias` `[n_expert]`: added to the router logits pre-topk.
    /// - `gate_exps_bias` / `up_exps_bias` `[n_expert, n_ff]`: added to the
    ///   gate/up projections before the GLU activation.
    /// - `down_exps_bias` `[n_expert, hidden]`: folded as `Σ_slot w.db` into the
    ///   FFN output after the down-projection reduction.
    /// All default `None` (dense / un-biased MoE is unaffected).
    pub gate_inp_bias: Option<Tensor>,
    pub gate_exps_bias: Option<Tensor>,
    pub up_exps_bias: Option<Tensor>,
    pub down_exps_bias: Option<Tensor>,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    // all_reduce: AllReduce,
    // world_size: usize,
    pub dtype: DType,
    /// Lazily-built per-expert quantized matmuls for the CPU fast path. `Arc` so
    /// `Clone`s of the layer share the one-time build; `None` inside the lock
    /// once we determine the quant type has no CPU `dot` path (fall back to
    /// dequantize). Untouched on CUDA.
    pub(crate) cpu_experts: Arc<OnceLock<Option<CpuExperts>>>,
}

impl FusedMoeGGUF {
    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        self.forward_inner(xs, is_prefill, None)
    }

    /// Forward + add residual fused into the down kernel's atomicAdd
    /// reduction. The residual must have the same shape as the FFN
    /// output (`[batch, seq_len, hidden]`) and the same dtype as the
    /// model's working dtype. The fused down+reduce path
    /// fast path is taken, this fold is free; when it isn't, we fall
    /// back to a separate add at the end.
    pub fn forward_with_residual(
        &self,
        xs: &Tensor,
        residual: &Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        self.forward_inner(xs, is_prefill, Some(residual))
    }

    fn forward_inner(
        &self,
        xs: &Tensor,
        is_prefill: bool,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let block = xs.dims3()?;
        let (batch, seq_len, hidden_dim) = block;
        let num_tokens = batch * seq_len;
        let original_dtype = xs.dtype();
        // One f32 row per token: what the router and every expert launch below read. The
        // block's own shape and dtype are restored on the way out.
        let xs = xs.reshape((num_tokens, hidden_dim))?.to_dtype(DType::F32)?;

        // The expert projections below are CUDA kernels; a layer placed on the CPU runs the
        // same routing over tensor ops instead.
        if !xs.device().is_cuda() {
            return self.forward_cpu(&xs, block, residual, original_dtype);
        }
        #[cfg(not(feature = "cuda"))]
        {
            // A build with no kernels compiled in has nothing below this point; the device
            // test above has already sent every call to the host path.
            self.forward_cpu(&xs, block, residual, original_dtype)
        }
        #[cfg(feature = "cuda")]
        {
            // Routing is timed here, where its boundary actually is: the gate matmul, the
            // softmax and the top-k selection, up to the point the experts are known. The
            // caller subtracts it from the block to get the expert compute.
            let route_t0 = crate::inference::place::layer_perf::stages::enabled().then(|| {
                let _ = xs.device().synchronize();
                std::time::Instant::now()
            });
            let (topk_weights, topk_ids) = {
                // Multi-block F32 GEMV gate matmul - default-on, falls back to cublas SGEMV
                // via gate.forward when the shape isn't supported. Scoring and selecting in
                // ONE kernel was measured slower than this pair in production, a single warp
                // under-saturating the GPU at batch=1 decode; that kernel stays in tree,
                // callable from `moe_cuda::gate_topk_softmax`, but nothing dispatches to it.
                let router_logits =
                    match crate::inference::moe_cuda::gate_gemv_f32(&xs, self.gate.weight()?)? {
                        Some(t) => t,
                        None => self.gate.forward(&xs)?,
                    };
                let logits = router_logits.to_dtype(DType::F32)?;
                // gpt-oss router bias (added to the logits before softmax/topk).
                let logits = match &self.gate_inp_bias {
                    Some(b) => logits.broadcast_add(&b.to_dtype(DType::F32)?)?,
                    None => logits,
                };
                // One kernel scores and selects for the expert counts it covers; anything
                // else takes the same two steps as tensor ops.
                if logits.device().is_cuda()
                    && matches!(logits.dim(D::Minus1)?, 32 | 64 | 128 | 256)
                {
                    crate::inference::moe_cuda::topk_softmax(
                        &logits,
                        self.num_experts_per_tok,
                        self.norm_topk_prob,
                    )?
                } else {
                    self.top_k(&crate::tensor::ops::softmax_last_dim(&logits)?)?
                }
            };
            let plan = ExpertPlan::group(&topk_ids, self.num_experts_per_tok)?;
            if let Some(t0) = route_t0 {
                let _ = xs.device().synchronize();
                crate::inference::place::layer_perf::stages::note_router_us(
                    t0.elapsed().as_micros() as u64,
                );
            }

            // How the gate and up projections reach the GLU:
            //   1. one launch that writes what the two combine into - needs the two stacks
            //      apart, and saves four launches per layer per token,
            //   2. one launch over a load-time gate‖up concat, then narrow + activation,
            //   3. two launches, then the activation.
            // A stack pair kept apart is what the fused launches need; when the load fused
            // them into one tensor, or left one of them out, the projections run separately.
            let apart = self.gate_experts.as_ref().zip(self.up_experts.as_ref());
            let down_inputs = match (apart, self.swiglu_oai) {
                // gpt-oss: gate GEMM, up GEMM, per-expert bias and the clamped SwiGLU in one
                // launch - no [M, N] intermediates, no separate bias adds, no epilogue pass.
                (Some((gate, up)), Some((alpha, limit))) => plan.gate_up_swiglu_oai(
                    &xs,
                    gate,
                    up,
                    self.gate_exps_bias.as_ref(),
                    self.up_exps_bias.as_ref(),
                    alpha,
                    limit,
                )?,
                // Plain SiLU with the stacks apart: silu(gate . x) . (up . x) in one launch.
                (Some((gate, up)), None) if matches!(self.act, Activation::Silu) => {
                    plan.gate_up_silu_mul(&xs, gate, up)?
                }
                _ => {
                    let (gate, up) = if let Some(gate_up) = self.gate_up_experts.as_ref() {
                        // One launch against the concatenated stack, then split the halves.
                        let both = plan.gemm(&xs, gate_up, &None, is_prefill, self.dtype)?;
                        let n = both.dim(D::Minus1)? / 2;
                        (
                            both.narrow(D::Minus1, 0, n)?.contiguous()?,
                            both.narrow(D::Minus1, n, n)?.contiguous()?,
                        )
                    } else {
                        let (gate_w, up_w) = apart.ok_or_else(|| {
                            crate::tensor::Error::msg(
                                "FusedMoeGGUF: the layer has neither a fused gate‖up stack \
                                 nor a gate and an up stack",
                            )
                        })?;
                        (
                            plan.gemm(&xs, gate_w, &None, is_prefill, self.dtype)?,
                            plan.gemm(&xs, up_w, &None, is_prefill, self.dtype)?,
                        )
                    };
                    // gpt-oss per-expert gate/up biases (added before the GLU). Expert GEMM
                    // output row j is expanded pair j, whose expert is topk_ids.flatten()[j]
                    // - a direct gather, no sort permutation. The fused launches above fold
                    // the same bias into their epilogue instead.
                    let eids = if self.gate_exps_bias.is_some() || self.up_exps_bias.is_some() {
                        Some(topk_ids.flatten_all()?)
                    } else {
                        None
                    };
                    let gbias = match (&self.gate_exps_bias, &eids) {
                        (Some(gb), Some(e)) => Some(gb.index_select(e, 0)?.to_dtype(gate.dtype())?),
                        _ => None,
                    };
                    let ubias = match (&self.up_exps_bias, &eids) {
                        (Some(ub), Some(e)) => Some(ub.index_select(e, 0)?.to_dtype(up.dtype())?),
                        _ => None,
                    };
                    match self.swiglu_oai {
                        Some((alpha, limit)) => {
                            crate::inference::kernel::fused::fused_swiglu_oai_bias(
                                &gate,
                                &up,
                                gbias.as_ref(),
                                ubias.as_ref(),
                                alpha,
                                limit,
                            )?
                        }
                        None => {
                            let gate = match &gbias {
                                Some(gb) => gate.broadcast_add(gb)?,
                                None => gate,
                            };
                            let up = match &ubias {
                                Some(ub) => up.broadcast_add(ub)?,
                                None => up,
                            };
                            (up * self.act.apply(&gate)?)?
                        }
                    }
                }
            };

            // Down projection and the scatter-add in one launch: the [M.topk, hidden]
            // per-(token, expert) intermediate is never written and the sum over a token's
            // experts is the kernel's own atomicAdd, which leaves the output already one row
            // per token. A residual and the gpt-oss per-expert down bias ride in the same
            // accumulation rather than through a chain of tensor ops - per-token temporaries
            // cannot be captured in a CUDA graph, the pointer moving between replays, and
            // cost about six launches a layer besides.
            let residual = residual
                .map(|r| r.reshape((num_tokens, hidden_dim)))
                .transpose()?;
            let ys = self.down_reduce(&plan, &down_inputs, &topk_weights, residual.as_ref())?;
            ys.to_dtype(original_dtype)?.reshape(block)
        } // end #[cfg(feature="cuda")]
    }

    /// The router's choice, from its scores: for every token the `num_experts_per_tok`
    /// experts scoring highest, and the score each was chosen with.
    ///
    /// The ids come back in descending score order - the order the expert visit and the
    /// reduction both follow - and `norm_topk_prob` rescales the kept scores to sum to one.
    /// Both forwards select this way; only how they score differs.
    fn top_k(&self, scores: &Tensor) -> Result<(Tensor, Tensor)> {
        // One descending sort answers both halves - ranking the scores also ranks the experts
        // they belong to - so the kept scores are read off the sort rather than gathered back.
        let (ranked, by_rank) = scores.sort_last_dim(false)?;
        let kept = |t: &Tensor| -> Result<Tensor> {
            t.narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()
        };
        let ids = kept(&by_rank)?;
        let weights = kept(&ranked)?;
        let weights = if self.norm_topk_prob {
            weights.broadcast_div(&weights.sum_keepdim(D::Minus1)?)?
        } else {
            weights
        };
        Ok((weights, ids))
    }

    /// The layer's down projection over a routing plan, and the scatter-add that ends it.
    ///
    /// Where the expert rows live decides which implementation runs; the plan, the weights and
    /// the arithmetic are the same one either way.
    fn down_reduce(
        &self,
        plan: &ExpertPlan,
        rows: &Tensor,
        scale: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let bias = self.down_exps_bias.as_ref();
        #[cfg(feature = "cuda")]
        if rows.device().is_cuda() {
            return plan.device_down_reduce(rows, &self.down_experts, scale, residual, bias);
        }
        plan.host_down_reduce(rows, &self.down_experts, scale, residual, bias)
    }

    /// CPU MoE forward: a host-only grouped expert FFN. Mirrors the CUDA path's
    /// math (router -> top-k -> per-expert gate/up GLU -> down -> weighted sum) but
    /// dequantizes the expert weights and runs the matmuls grouped by expert.
    /// Correctness-first (used on the `cpu` build where the custom kernels are
    /// absent); not perf-tuned. Handles separate or fused gate/up, the gpt-oss
    /// SwiGLU-OAI gate + per-expert biases, and norm_topk_prob.
    ///
    /// `block` is the `[batch, seq_len, hidden]` the rows of `xs` came from and go back to.
    fn forward_cpu(
        &self,
        xs: &Tensor, // [num_tokens, hidden] F32
        block: (usize, usize, usize),
        residual: Option<&Tensor>,
        original_dtype: DType,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = block;
        let num_tokens = batch * seq_len;
        let dev = xs.device();
        let topk = self.num_experts_per_tok;

        // Router logits (+ optional gpt-oss router bias) -> softmax/sigmoid -> top-k.
        let logits = self.gate.forward(xs)?.to_dtype(DType::F32)?;
        let logits = match &self.gate_inp_bias {
            Some(b) => logits.broadcast_add(&b.to_dtype(DType::F32)?)?,
            None => logits,
        };
        // gpt-oss uses sigmoid gating (swiglu_oai set); everyone else softmax.
        let routing = if self.swiglu_oai.is_some() {
            crate::tensor::ops::sigmoid(&logits)?
        } else {
            crate::tensor::ops::softmax_last_dim(&logits)?
        };
        let (topk_w, topk_ids) = self.top_k(&routing)?;

        // Per-expert quantized matmuls (built once, cached). When present we run
        // a fused quantized GEMV over only the selected experts' GGUF blocks  -
        // the same thing Ollama does - instead of dequantizing the entire expert
        // stack to F32 every token (the ~100x CPU-MoE slowdown). `None` means the
        // quant type has no CPU dot path: fall back to dequantize-then-matmul.
        let cpu_ex = self
            .cpu_experts
            .get_or_init(|| self.build_cpu_experts().ok().flatten())
            .as_ref();

        // mul_mat_id fast-path (qwen3-coder & co): standard SwiGLU, FUSED gate‖up
        // [E,2N,K], no per-expert bias, quant-supported. Quantize each token ONCE
        // and run one work-stealing region over (slot,row) for gate/up AND for
        // down+reduce - fills all cores at M=1 decode, where the per-expert
        // CONCURRENT path below leaves ~topk cores busy (the 16.9 vs 20.5 GB/s gap
        // vs ollama). gpt-oss (swiglu_oai) and separate gate/up keep the path below.
        if cpu_ex.is_some()
            && matches!(self.act, Activation::Silu)
            && self.swiglu_oai.is_none()
            && self.gate_exps_bias.is_none()
            && self.up_exps_bias.is_none()
            && self.down_exps_bias.is_none()
        {
            // sort the (token,slot) pairs by expert id -> contiguous per-expert runs.
            // GH_PROF: per-stage µs sums over the fast path (same knob as the
            // generic-transformer hooks - FusedMoeGGUF bypasses those, so
            // qwen3-coder-class MoE decode was otherwise unprofilable).
            let prof_t0: Option<std::time::Instant> = if std::env::var("GH_PROF").is_ok() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let plan = ExpertPlan::group(&topk_ids, topk)?;
            let prof_t_sort = prof_t0.map(|_| std::time::Instant::now());
            let h = if let Some(gate_up) = self.gate_up_experts.as_ref() {
                Some(plan.host_gate_up_silu_mul_fused(xs, gate_up)?)
            } else if let (Some(g), Some(u)) =
                (self.gate_experts.as_ref(), self.up_experts.as_ref())
            {
                Some(plan.host_gate_up_silu_mul(xs, g, u)?)
            } else {
                None
            };
            let prof_t_gu = prof_t0.map(|_| std::time::Instant::now());
            if let Some(h) = h {
                // The residual is folded into the reduction, as on the device.
                let out = self.down_reduce(&plan, &h, &topk_w, residual)?;
                if let (Some(t0), Some(ts), Some(tg)) = (prof_t0, prof_t_sort, prof_t_gu) {
                    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
                    static SORT_US: AtomicU64 = AtomicU64::new(0);
                    static GU_US: AtomicU64 = AtomicU64::new(0);
                    static DOWN_US: AtomicU64 = AtomicU64::new(0);
                    static CALLS: AtomicUsize = AtomicUsize::new(0);
                    SORT_US.fetch_add(ts.duration_since(t0).as_micros() as u64, Ordering::Relaxed);
                    GU_US.fetch_add(tg.duration_since(ts).as_micros() as u64, Ordering::Relaxed);
                    DOWN_US.fetch_add(tg.elapsed().as_micros() as u64, Ordering::Relaxed);
                    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(100) {
                        tracing::info!(
                            "🟩 GH_PROF moe (sum µs over {n} calls): sort={} gate_up={} down_reduce={}",
                            SORT_US.load(Ordering::Relaxed), GU_US.load(Ordering::Relaxed),
                            DOWN_US.load(Ordering::Relaxed));
                    }
                }
                return out.to_dtype(original_dtype)?.reshape(block);
            }
        }

        // Dequantize fallback stacks - only materialized when there is no
        // quantized CPU path (`cpu_ex` is `None`).
        let (gate_all, up_all, down_all) = if cpu_ex.is_none() {
            let (g, u) = if let Some(gu) = self.gate_up_experts.as_ref() {
                let gu = gu.dequantize(&dev)?; // [E, 2N, K]
                let n = gu.dim(1)? / 2;
                (
                    gu.narrow(1, 0, n)?.contiguous()?,
                    gu.narrow(1, n, n)?.contiguous()?,
                )
            } else {
                (
                    self.gate_experts.as_ref().unwrap().dequantize(&dev)?,
                    self.up_experts.as_ref().unwrap().dequantize(&dev)?,
                )
            };
            (Some(g), Some(u), Some(self.down_experts.dequantize(&dev)?))
        } else {
            (None, None, None)
        };
        let num_experts = self.down_experts.shape().dims()[0];

        let ids: Vec<u32> = topk_ids.flatten_all()?.to_vec1()?;
        let wts: Vec<f32> = topk_w.flatten_all()?.to_vec1()?;
        let xrows = xs.to_dtype(DType::F32)?.contiguous()?; // [num_tokens, hidden]

        // Build per-expert token routing first (sequential, cheap: a scan over
        // the top-k id list). Only experts that actually received a token make
        // the work-list.
        let mut routed: Vec<(usize, Vec<u32>, Vec<f32>)> = Vec::new();
        for e in 0..num_experts {
            let mut tok_idx: Vec<u32> = Vec::new();
            let mut wrow: Vec<f32> = Vec::new();
            for t in 0..num_tokens {
                for s in 0..topk {
                    if ids[t * topk + s] as usize == e {
                        tok_idx.push(t as u32);
                        wrow.push(wts[t * topk + s]);
                    }
                }
            }
            if !tok_idx.is_empty() {
                routed.push((e, tok_idx, wrow));
            }
        }

        // Compute each routed expert's weighted contribution. At decode
        // (num_tokens=1) every expert GEMV has M=1, so the reference in-matmul
        // N-parallelism produces only a few rayon tasks per call - running the
        // ~topk active experts CONCURRENTLY is what actually fills all cores
        // (gdb showed the matmul on one thread while workers parked). Each
        // closure is pure (reads immutable weights, builds fresh tensors), so
        // the only shared write - the index_add reduction - stays sequential
        // below. Helps every CPU MoE arch (qwen3-coder/lfm2/gpt-oss/nemotron/
        // qwen3.5).
        let compute = |e: usize, tok_idx: &[u32], wrow: &[f32]| -> Result<(Tensor, Tensor)> {
            let te = tok_idx.len();
            let sel = Tensor::from_vec(tok_idx.to_vec(), (te,), &dev)?;
            let x_e = xrows.index_select(&sel, 0)?.contiguous()?; // [te, hidden]
            let (mut g, mut u) = if let Some(cx) = cpu_ex {
                if let Some(gu) = cx.gate_up.as_ref() {
                    // One GEMV over the fused [2N, K] expert: quantize x_e once,
                    // split the [te, 2N] output into gate ‖ up.
                    let out = gu[e].forward(&x_e)?;
                    let n = out.dim(1)? / 2;
                    (
                        out.narrow(1, 0, n)?.contiguous()?,
                        out.narrow(1, n, n)?.contiguous()?,
                    )
                } else {
                    // Fused quantized GEMV against expert e's gate/up blocks.
                    (cx.gate[e].forward(&x_e)?, cx.up[e].forward(&x_e)?) // [te, N]
                }
            } else {
                let ge = gate_all.as_ref().unwrap().i(e)?; // [N, K]
                let ue = up_all.as_ref().unwrap().i(e)?;
                (x_e.matmul_t(&ge)?, x_e.matmul_t(&ue)?) // [te, N]
            };
            let h = match self.swiglu_oai {
                Some((alpha, limit)) => {
                    // x=min(gate,limit); g_=clamp(up,-limit,limit); (x.σ(alpha.x)).(1+g_).
                    // On CPU one fused pass folds the per-expert gate/up bias and the
                    // whole clamp/sigmoid chain, replacing ~13 elementwise tensor
                    // passes (bit-identical; profiled at ~22% of prefill). The CUDA
                    // kernel keeps its device-side affine->relu clamp folds.
                    let gb = self.gate_exps_bias.as_ref().map(|b| b.i(e)).transpose()?;
                    let ub = self.up_exps_bias.as_ref().map(|b| b.i(e)).transpose()?;
                    if !x_e.device().is_cuda() {
                        crate::tensor::ops::swiglu_oai(
                            &g,
                            &u,
                            gb.as_ref(),
                            ub.as_ref(),
                            alpha,
                            limit,
                        )?
                    } else {
                        if let Some(gb) = &gb {
                            g = g.broadcast_add(&gb.to_dtype(DType::F32)?)?;
                        }
                        if let Some(ub) = &ub {
                            u = u.broadcast_add(&ub.to_dtype(DType::F32)?)?;
                        }
                        let l = limit as f32;
                        let x = g.affine(-1.0, l)?.relu()?.affine(-1.0, l)?; // min(g, l)
                        let gg = u
                            .affine(1.0, l)?
                            .relu()?
                            .affine(1.0, -l)? // max(u, -l)
                            .affine(-1.0, l)?
                            .relu()?
                            .affine(-1.0, l)?; // min(., l)
                        let act = (&x * crate::tensor::ops::sigmoid(&(&x * alpha)?)?)?;
                        (act * (gg + 1.0)?)?
                    }
                }
                None => {
                    // gpt-oss per-expert gate/up bias (before the GLU).
                    if let Some(gb) = &self.gate_exps_bias {
                        g = g.broadcast_add(&gb.i(e)?.to_dtype(DType::F32)?)?;
                    }
                    if let Some(ub) = &self.up_exps_bias {
                        u = u.broadcast_add(&ub.i(e)?.to_dtype(DType::F32)?)?;
                    }
                    (u * self.act.apply(&g)?)?
                }
            };
            let h = h.contiguous()?;
            let mut d = if let Some(cx) = cpu_ex {
                cx.down[e].forward(&h)? // [te, hidden]
            } else {
                let de = down_all.as_ref().unwrap().i(e)?; // [hidden, N]
                h.matmul_t(&de)? // [te, hidden]
            };
            if let Some(db) = &self.down_exps_bias {
                d = d.broadcast_add(&db.i(e)?.to_dtype(DType::F32)?)?;
            }
            let wv = Tensor::from_vec(wrow.to_vec(), (te, 1), &dev)?;
            let d = d.broadcast_mul(&wv)?;
            Ok((sel, d))
        };

        let contribs: Vec<Result<(Tensor, Tensor)>> = if routed.len() > 1 {
            use rayon::prelude::*;
            routed
                .par_iter()
                .map(|(e, tok_idx, wrow)| compute(*e, tok_idx, wrow))
                .collect()
        } else {
            routed
                .iter()
                .map(|(e, tok_idx, wrow)| compute(*e, tok_idx, wrow))
                .collect()
        };

        let mut out = Tensor::zeros_on((num_tokens, hidden_dim), DType::F32, &dev)?;
        for c in contribs {
            let (sel, d) = c?;
            out = out.index_add(&sel, &d, 0)?;
        }

        // Nothing here accumulates into an output buffer the way the fused reduction above
        // does, so a residual is added at the block's own width, once the rows are back in
        // shape and dtype.
        let ys = out.reshape(block)?.to_dtype(original_dtype)?;
        match residual {
            Some(r) => ys + r,
            None => Ok(ys),
        }
    }

    /// Build the per-expert `QMatMul` cache for the CPU fast path. Returns
    /// `Ok(None)` when the expert quant type has no CPU `dot` matmul (so the
    /// caller keeps the dequantize-then-F32-matmul fallback). One-time cost; the
    /// result is cached in `self.cpu_experts`.
    fn build_cpu_experts(&self) -> Result<Option<CpuExperts>> {
        let down_q = self.down_experts.as_ref();
        if !cpu_qmatmul_supported(down_q.dtype()) {
            return Ok(None);
        }
        let (gate, up, gate_up) = if let Some(gu) = self.gate_up_experts.as_ref() {
            if !cpu_qmatmul_supported(gu.dtype()) {
                return Ok(None);
            }
            // [E, 2N, K]: rows 0..N = gate, rows N..2N = up. Keep the experts
            // FUSED so decode quantizes the input once and issues one GEMV.
            let two_n = gu.shape().dims()[1];
            (
                Vec::new(),
                Vec::new(),
                Some(split_experts_rows(gu, 0, two_n)?),
            )
        } else {
            let g = self.gate_experts.as_ref().unwrap();
            let u = self.up_experts.as_ref().unwrap();
            if !cpu_qmatmul_supported(g.dtype()) || !cpu_qmatmul_supported(u.dtype()) {
                return Ok(None);
            }
            let n = g.shape().dims()[1];
            (
                split_experts_rows(g, 0, n)?,
                split_experts_rows(u, 0, n)?,
                None,
            )
        };
        // down: [E, hidden, N] - all rows.
        let hidden = down_q.shape().dims()[1];
        let down = split_experts_rows(down_q, 0, hidden)?;
        Ok(Some(CpuExperts {
            gate,
            up,
            gate_up,
            down,
        }))
    }
}
