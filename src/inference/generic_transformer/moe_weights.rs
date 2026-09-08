//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

// ------------------------------------------------------------
// Routing plan
// ------------------------------------------------------------

/// Group the router's `(token, choice)` pairs by expert, so each expert's rows are one
/// contiguous run.
///
/// Returns the expert every row is multiplied by, then the flattened pair each row is built
/// from. Sorting by expert id is what turns a scattered assignment into one GEMM per run, and
/// both arms sort stably, so the rows of one expert keep the order the router put them in.
fn sort_by_expert(choices: &Tensor) -> Result<(Tensor, Tensor)> {
    let flat = choices.flatten_all()?;
    let n_pairs = flat.dim(0)?;
    if n_pairs <= 4096 {
        return flat.sort_last_dim(true);
    }
    // Past the tensor sort's ceiling - only a wide prefill reaches it. Pull the keys back,
    // sort them alongside their positions, and hand both sides back where they came from.
    let device = flat.device().clone();
    let host = crate::tensor::Device::Cpu;
    let mut keys: Vec<u32> = flat.to_device(&host)?.to_vec1()?;
    let mut order: Vec<u32> = (0..n_pairs as u32).collect();
    order.sort_by_key(|&i| keys[i as usize]);
    keys.sort();
    Ok((
        Tensor::from_vec(keys, (n_pairs,), &host)?.to_device(&device)?,
        Tensor::from_vec(order, (n_pairs,), &host)?.to_device(&device)?,
    ))
}

/// The router's choices, grouped so that each expert's rows are one contiguous run.
///
/// `slots[i]` is the flattened `(token, choice)` pair the i-th expert row is built from, and
/// `experts[i]` the expert that row is multiplied by. Every expert launch below takes those two
/// together with the per-token choice count - so they travel together, and the launches hang
/// off them rather than repeating the triple at each call.
struct ExpertRouting {
    slots: Tensor,
    experts: Tensor,
    topk: usize,
}

impl ExpertRouting {
    fn group(choices: &Tensor, topk: usize) -> Result<Self> {
        let (experts, slots) = sort_by_expert(choices)?;
        Ok(Self {
            slots,
            experts,
            topk,
        })
    }

    /// One launch over the fused gate‖up stack: both projections, the GELU combine and the
    /// concatenation the down projection reads. `imma` picks the per-pair-tile integer-MMA
    /// kernel over the dp4a one; the arithmetic is the same either way.
    #[cfg(feature = "cuda")]
    fn gate_up_gelu(&self, xs: &Tensor, stack: &QTensor, imma: bool) -> Result<Tensor> {
        if imma {
            crate::inference::moe_cuda::moe_q4k_imma_m8_gate_up_gelu_mul_concat(
                xs,
                stack,
                &self.slots,
                &self.experts,
                self.topk,
            )
        } else {
            crate::inference::moe_cuda::moe_gemm_gguf_gate_up_gelu_mul_concat(
                xs,
                stack,
                &self.slots,
                &self.experts,
                self.topk,
            )
        }
    }

    /// The same, against gate and up kept apart, combined by SiLU.
    #[cfg(feature = "cuda")]
    fn gate_up_silu(&self, xs: &Tensor, gate: &QTensor, up: &QTensor) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul(
            xs,
            gate,
            up,
            &self.slots,
            &self.experts,
            self.topk,
        )
    }

    /// Down projection and the weighted scatter-add that ends it: each row scaled by the
    /// score its expert was chosen with, added into the token it came from.
    #[cfg(feature = "cuda")]
    fn down_reduce(
        &self,
        rows: &Tensor,
        stack: &QTensor,
        weights: &Tensor,
        n_tokens: usize,
        imma: bool,
    ) -> Result<Tensor> {
        if imma {
            crate::inference::moe_cuda::moe_q4k_imma_m8_down_reduce(
                rows,
                stack,
                &self.slots,
                &self.experts,
                weights,
                self.topk,
                n_tokens,
                None,
            )
        } else {
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce(
                rows,
                stack,
                &self.slots,
                &self.experts,
                weights,
                self.topk,
                n_tokens,
                None,
                None,
            )
        }
    }

    /// [`Self::gate_up_silu`] on the host.
    fn host_gate_up_silu(
        &self,
        xs: &Tensor,
        gate: &Arc<QTensor>,
        up: &Arc<QTensor>,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_gate_up_silu_mul(
            xs,
            gate,
            up,
            &self.slots,
            &self.experts,
            self.topk,
        )
    }

    /// [`Self::down_reduce`] on the host.
    fn host_down_reduce(
        &self,
        rows: &Tensor,
        stack: &Arc<QTensor>,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
            rows,
            stack,
            &self.slots,
            &self.experts,
            weights,
            self.topk,
            n_tokens,
            None,
            None,
        )
    }
}

// ------------------------------------------------------------
// Per-layer struct
// ------------------------------------------------------------

/// Gemma4-MoE per-layer weights: top-K-of-N expert FFN.
/// Active per token: K experts x (gate+up+down) - top-8 of 128
/// for gemma4 26B A4B.
pub struct MoeWeights {
    /// Router projection - wrapped as QMatMul at load time so the
    /// forward path doesn't need to clone the underlying QTensor
    /// (which doesn't impl Clone).
    pub gate_inp: crate::tensor::quantized::QMatMul,
    /// Per-channel scale on the router input (Gemma4-specific): `[hidden]`.
    pub gate_inp_scale: Tensor,
    /// Stacked expert gate||up weights `[n_experts, 2*expert_ffn_dim, hidden]`
    /// (the stored layout - GGUF dims are reversed).
    pub gate_up_exps: std::sync::Arc<crate::tensor::quantized::QTensor>,
    /// Stacked expert down weights `[n_experts, hidden, expert_ffn_dim]`.
    pub down_exps: std::sync::Arc<crate::tensor::quantized::QTensor>,
    /// Standard MoE (granitemoe): SEPARATE gate & up expert stacks `[n_exp, ffn,
    /// hidden]`, kept quantized on-device for the GPU SiLU kernel
    /// (`moe_gemm_gguf_gate_up_silu_mul`, which takes gate/up separately). None
    /// for gemma4-MoE (fused gate||up + GELU).
    pub gate_exps_sep: Option<std::sync::Arc<crate::tensor::quantized::QTensor>>,
    pub up_exps_sep: Option<std::sync::Arc<crate::tensor::quantized::QTensor>>,
    /// Per-expert scalar on the down output: `[n_experts]`.
    pub down_exps_scale: Tensor,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub expert_ffn_dim: usize,
    /// Standard MoE (granitemoe): router reads the already-normed hidden directly
    /// - skip the internal rms_norm*gate_inp_scale (which is Gemma4-only).
    pub router_prenormed: bool,
    /// LAYOUT flag: TRUE iff experts were loaded as SEPARATE gate/up stacks (vs a
    /// pre-fused gate||up). Named use_silu for historical reasons; does NOT indicate
    /// the activation (granite-moe has fused gate||up yet uses SiLU). Use `gelu`.
    pub use_silu: bool,
    /// Real expert activation: GELU (gemma4-MoE, config.use_gelu) vs SiLU (granite /
    /// qwen*moe / olmoe). Decoupled from use_silu (layout) so a fused-gate||up SiLU
    /// model (granite) is not mis-run through the wrong expert path.
    pub gelu: bool,
    /// Renormalize the top-K router weights to sum to 1 (`norm_topk_prob`). True for
    /// most MoE (granite, qwen*moe, gemma4); FALSE for OLMoE (raw softmax-over-all
    /// weights, no renorm) - getting this wrong scales the expert mix wrong.
    pub norm_topk: bool,
    /// CPU-only cache: dequantized F32 expert tensors. Populated for layers
    /// placed on CPU since `moe_gemm_gguf` is CUDA-only. Avoids per-call
    /// dequant cost on what is already a slow CPU path.
    pub gate_up_exps_f32: Option<std::sync::Arc<Tensor>>,
    pub down_exps_f32: Option<std::sync::Arc<Tensor>>,
}

impl MoeWeights {
    /// The expert stacks this layer's prefill reads on the host, if its prefill goes there.
    ///
    /// The host expert loop reads its weights through a repacked layout it builds on first
    /// use, and a layer whose card is absent still goes there. Naming the stacks here lets the
    /// load warm that layout instead of leaving it to the first request.
    pub fn host_prefill_stacks(&self) -> Vec<std::sync::Arc<crate::tensor::quantized::QTensor>> {
        if !self.use_silu {
            return Vec::new();
        }
        let mut stacks = Vec::with_capacity(3);
        match (self.gate_exps_sep.as_ref(), self.up_exps_sep.as_ref()) {
            (Some(g), Some(u)) => {
                stacks.push(g.clone());
                stacks.push(u.clone());
            }
            // A layer that fused gate and up at load is repacked as the one stack it is.
            _ => stacks.push(self.gate_up_exps.clone()),
        }
        stacks.push(self.down_exps.clone());
        stacks
    }

    /// Gemma4-MoE forward.
    ///
    /// Input: `attn_out` shape `[b, seq, hidden]`. Returns the MoE branch
    /// output (post-norm-2 NOT applied here; caller wraps).
    ///
    /// Per llama.cpp gemma4-iswa.cpp:
    ///   tmp     = rms_norm(attn_out, eps) * (1 / sqrt(n_embd))
    ///   tmp     = tmp * gate_inp_scale            // per-channel
    ///   logits  = ffn_gate_inp(tmp)               // [n_tokens, n_experts]
    ///   topk_p, topk_i = softmax_topk(logits, k = n_experts_used)
    ///   moe_in  = pre_ffw_norm_2(attn_out)        // applied by caller
    ///   for each `(token, expert)` in routed pairs:
    ///       `gu  = gate_up_exps[expert] @ moe_in`
    ///       `g, u = gu.split(2)`
    ///       `y   = down_exps[expert] @ (gelu(g) * u)`
    ///       `y  *= down_exps_scale[expert]`
    ///       `out[token] += topk_p * y`
    pub(super) fn forward(
        &self,
        moe_in_pre_norm_2: &Tensor, // already pre_ffw_norm_2-applied
        attn_out: &Tensor,          // for the gate's own RMS-norm input
        n_embd: usize,
        rms_eps: f32,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = moe_in_pre_norm_2.dims3()?;
        let xs =
            moe_in_pre_norm_2.reshape((moe_in_pre_norm_2.elem_count() / hidden_dim, hidden_dim))?;
        let num_tokens = xs.dim(0)?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != crate::tensor::DType::F32 {
            xs.to_dtype(crate::tensor::DType::F32)?
        } else {
            xs
        };

        // -- Router input: rms_norm(attn_out) * gate_inp_scale_folded
        // The constant 1/sqrt(n_embd) was folded into gate_inp_scale at
        // load time. Use the reference fused rms_norm kernel which does
        // mean(x^2) + sqrt + eps + divide + scale-by-alpha in a single
        // launch (vs ~4 launches for the manual version).
        let _ = n_embd;
        let attn_flat = attn_out.reshape((attn_out.elem_count() / hidden_dim, hidden_dim))?;
        let attn_f32 = if attn_flat.dtype() == crate::tensor::DType::F32 {
            attn_flat
        } else {
            attn_flat.to_dtype(crate::tensor::DType::F32)?
        };
        // Fused (rms_norm + quantize + qmatmul) - saves 1 launch vs the
        // rms_norm.forward + gate_inp.forward sequence (which internally
        // does quantize+matmul = 2 launches; total 3 -> 2). Falls back
        // when shape is outside the single-block limit (hidden > 16384)
        // or the gate_inp weight isn't a QTensor.
        let logits_f32 = if self.router_prenormed {
            // Standard MoE (granitemoe): `xs` is already the ffn_norm'd hidden  -
            // route it straight through gate_inp (no internal rms_norm / scale).
            self.gate_inp
                .forward(&xs)?
                .to_dtype(crate::tensor::DType::F32)?
        } else {
            {
                #[cfg(feature = "cuda")]
                {
                    if attn_f32.device().is_cuda()
                        && attn_f32.dim(crate::tensor::D::Minus1)? <= 16384
                    {
                        use crate::tensor::quantized::QMatMul;
                        if let QMatMul::QTensor(ref wmm) = self.gate_inp {
                            crate::inference::moe_cuda::rms_norm_then_qmatmul(
                                &attn_f32,
                                &self.gate_inp_scale,
                                wmm.as_ref(),
                                rms_eps,
                            )
                            .ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    None::<Tensor>
                }
            }
            .map(Ok)
            .unwrap_or_else(|| {
                let normed =
                    crate::tensor::ops::rms_norm(&attn_f32, &self.gate_inp_scale, rms_eps)?;
                let logits = self.gate_inp.forward(&normed)?;
                logits.to_dtype(crate::tensor::DType::F32)
            })?
        };

        // Measured directly, not as a residual: a residual absorbs the synchronisations
        // the profiler itself adds at every stage boundary, and then reports them as work.
        let t_topk = crate::inference::place::layer_perf::stages::enabled().then(|| {
            let _ = logits_f32.device().synchronize();
            std::time::Instant::now()
        });
        // -- Softmax + top-K.
        // One kernel scores and selects for the expert widths it covers (gemma4-26B routes
        // over 128); every other width, and every build compiled without kernels, takes the
        // two steps as tensor ops. Which of the two ran is what the stage profile records.
        #[cfg(feature = "cuda")]
        let (topk_weights, topk_ids) = {
            let fused = logits_f32.device().is_cuda()
                && matches!(
                    logits_f32.dim(crate::tensor::D::Minus1)?,
                    32 | 64 | 128 | 256
                );
            crate::inference::place::layer_perf::stages::note_topk_path(fused);
            if fused {
                crate::inference::moe_cuda::topk_softmax(
                    &logits_f32,
                    self.n_experts_used,
                    self.norm_topk,
                )?
            } else {
                self.top_k(&logits_f32)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (topk_weights, topk_ids) = self.top_k(&logits_f32)?;

        // -- Sort (expert, token) pairs.
        if let Some(t) = t_topk {
            let _ = topk_ids.device().synchronize();
            crate::inference::place::layer_perf::stages::add(9, t.elapsed().as_micros() as u64);
        }
        // Routing on THIS path, which is not the one `fused_moe` times: a mixture has two
        // forwards in this tree and only one of them carries the fast argsort.
        let t_route = crate::inference::place::layer_perf::stages::enabled().then(|| {
            let _ = topk_ids.device().synchronize();
            std::time::Instant::now()
        });
        let routing = ExpertRouting::group(&topk_ids, self.n_experts_used)?;

        // -- Fold the per-expert down-output scale into topk_weights.
        // llama.cpp gemma4-iswa.cpp applies it after the down matmul
        // before the topk-weighted reduction:
        //   experts[t,k,h] *= down_exps_scale[topk_ids[t,k]]
        //   experts[t,k,h] *= topk_weights[t,k]
        // Folding scale into weights gives the same final result and
        // keeps a single weighted reduction.
        let scale_per_pair = self
            .down_exps_scale
            .gather(&topk_ids.flatten_all()?, 0)?
            .reshape(topk_weights.shape())?;
        let topk_weights_scaled = (topk_weights.clone() * scale_per_pair)?;

        // SiLU MoE hybrid: the GPU multi-token PREFILL kernels (WMMA + mmvq) both
        // mis-numeric these separate Q4K experts (verified: per-layer GPU-vs-CPU
        // diverges at layer 5+ for M>1). Single-token DECODE via mmvq is EXACT
        // (matched CPU <1% at every layer). So run prefill on the CPU expert path
        // (one-time, correct) and decode on GPU (the tok/s bottleneck, fast+exact).
        let moe_device = xs.device().clone();
        // Which arrangement, not which hardware. True only on the host either way -
        // the two spellings agree on every device - but the question being asked here
        // is the shape of the expert loop, so it is asked in those terms.
        // A SiLU mixture's prefill used to be sent here whatever card the layer sat on, because
        // the tiled down-projection answered wrongly above sixty-four routed rows. It read the
        // quantised activation by SORTED POSITION while both its producers wrote that row by
        // SLOT, and it dropped every expert past the fourth in a tile of eight. Neither was
        // caught, because that kernel had no parity test - it has one now, and the host detour
        // it justified cost about eight times the prefill.
        let on_cpu = !xs.device().runs_as_card();
        let expert_rows = if on_cpu {
            self.forward_cpu(&xs, &topk_ids, &topk_weights_scaled, num_tokens, hidden_dim)?
        } else {
            if let Some(t) = t_route {
                let _ = moe_device.synchronize();
                crate::inference::place::layer_perf::stages::add(3, t.elapsed().as_micros() as u64);
            }
            self.forward_cuda(&xs, &routing, &topk_weights_scaled, num_tokens, is_prefill)?
        };
        // Back to the block's own device, width and dtype. The host expert path answers on
        // the host even for a layer that lives on a card, so that the hybrid prefill above
        // can run there; the caller's residual add expects the layer's device back.
        let expert_rows = match expert_rows.device().same_device(&moe_device) {
            true => expert_rows,
            false => expert_rows.to_device(&moe_device)?,
        };
        expert_rows
            .to_dtype(original_dtype)?
            .reshape((batch, seq_len, hidden_dim))
    }

    /// The router's choice, from its logits: for every token the `n_experts_used` experts
    /// scoring highest, and the probability each was chosen with.
    ///
    /// The ids come back in descending probability order - the order the expert visit and the
    /// reduction both follow - and `norm_topk` rescales the kept probabilities to sum to one.
    /// Both forwards select this way, and so does the fallback of the fused selection above.
    fn top_k(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let probs = crate::tensor::ops::softmax_last_dim(logits)?;
        let topk_ids = probs
            .arg_sort_last_dim(false)?
            .narrow(crate::tensor::D::Minus1, 0, self.n_experts_used)?
            .contiguous()?;
        let mut topk_weights = probs.gather(&topk_ids, crate::tensor::D::Minus1)?;
        if self.norm_topk {
            topk_weights =
                topk_weights.broadcast_div(&topk_weights.sum_keepdim(crate::tensor::D::Minus1)?)?;
        }
        Ok((topk_weights, topk_ids))
    }

    /// Device path: the gate‖up projections and their combine, then the down projection and
    /// the top-k weighted reduction, each as one launch over the routing plan.
    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        xs: &Tensor,
        routing: &ExpertRouting,
        topk_weights: &Tensor,
        num_tokens: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Auto-on: per-pair-tile IMMA M=8 for Q4_K MoE prefill - measured
        // +33% over dp4a on gemma4:26b.
        let weight_is_q4k = self.gate_up_exps.dtype() == crate::tensor::quantized::GgmlDType::Q4K;
        let k_is_aligned = self
            .gate_up_exps
            .shape()
            .dims()
            .last()
            .map(|k| k % 256 == 0)
            .unwrap_or(false);
        let imma_m8_disabled = false;
        // size_m guard: IMMA M=8 wins for size_m >= ~256 (each block does 8
        // pairs, so we need >=32 blocks of work to amortize the 1-warp
        // dispatch overhead vs dp4a's per-(token,expert) granularity).
        // For tiny prefills (size_m<64) dp4a's denser dispatch wins.
        let size_m_est = num_tokens * self.n_experts_used;
        // The IMMA kernels are built on the `m16n8k32` integer MMA, which arrived with
        // Ampere: below that this family ships no code at all, so the card decides here
        // alongside the shape.
        let has_imma = xs
            .device()
            .as_cuda_device()
            .is_ok_and(|d| d.has_ampere_tensor_cores());
        let use_imma_m8_auto = is_prefill
            && weight_is_q4k
            && k_is_aligned
            && size_m_est >= 64
            && !imma_m8_disabled
            && has_imma;

        // The two matrix products the expert block is made of, timed apart, and the whole
        // call around them: what the block costs beyond its arithmetic is the difference.
        let st = crate::inference::place::layer_perf::stages::enabled();
        let t_all = st.then(|| {
            let _ = xs.device().synchronize();
            std::time::Instant::now()
        });
        let t_gu = st.then(|| {
            let _ = xs.device().synchronize();
            std::time::Instant::now()
        });
        let down_inputs = if self.use_silu {
            // Standard MoE (granitemoe): SEPARATE gate/up experts (original quant),
            // one moe_gemm_gguf each (prefill+decode), then SiLU.mul in Rust. Uses
            // the untouched Q4K experts - no fused Q8 requant (which mangles the 3D
            // stack) and SiLU not the gemma4 kernel's baked GELU.
            let g = self.gate_exps_sep.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg("use_silu MoE missing separate gate experts")
            })?;
            let u = self.up_exps_sep.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg("use_silu MoE missing separate up experts")
            })?;
            // Decode-only here: the fused
            // single-launch SiLU.mul mmvq kernel (gate GEMM + up GEMM + SiLU.mul in
            // one pass) is exact for M=1 and faster than two separate GEMMs.
            routing.gate_up_silu(xs, g, u)?
        } else {
            routing.gate_up_gelu(xs, &self.gate_up_exps, use_imma_m8_auto)?
        };
        if let Some(t) = t_gu {
            let _ = xs.device().synchronize();
            crate::inference::place::layer_perf::stages::add(6, t.elapsed().as_micros() as u64);
        }
        let t_dn = st.then(std::time::Instant::now);
        // Down + topk-weighted reduce. For Q4_K MoE prefill the IMMA M=8
        // down kernel (same pattern as gate||up) takes over from dp4a.
        let down_q4k_aligned = self.down_exps.dtype() == crate::tensor::quantized::GgmlDType::Q4K
            && self
                .down_exps
                .shape()
                .dims()
                .last()
                .map(|k| k % 256 == 0)
                .unwrap_or(false);
        let down_no = false;
        // Same size_m guard as gate||up: tiny prefills lose to dp4a.
        let down_use_imma_m8 = is_prefill && down_q4k_aligned && size_m_est >= 64 && !down_no;
        let out = routing.down_reduce(
            &down_inputs,
            &self.down_exps,
            topk_weights,
            num_tokens,
            down_use_imma_m8,
        );
        if let Some(t) = t_dn {
            let _ = xs.device().synchronize();
            crate::inference::place::layer_perf::stages::add(7, t.elapsed().as_micros() as u64);
        }
        if let Some(t) = t_all {
            crate::inference::place::layer_perf::stages::add(8, t.elapsed().as_micros() as u64);
        }
        out
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cuda(
        &self,
        _xs: &Tensor,
        _routing: &ExpertRouting,
        _w: &Tensor,
        _nt: usize,
        _ip: bool,
    ) -> Result<Tensor> {
        crate::tensor::bail!("CUDA path not compiled in")
    }

    /// CPU path: per-expert F32 batched matmul over the tokens routed to
    /// each expert, then scatter back into [num_tokens, hidden]. Slow,
    /// but only used for layers the heterogeneous planner offloaded to
    /// CPU. Expects `gate_up_exps_f32` and `down_exps_f32` populated.
    fn forward_cpu(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        // Inputs may be GPU-resident (SiLU-MoE hybrid: prefill runs here even when
        // the layer is on GPU). The expert math below is host-side (to_vec1), so
        // pull the small per-token tensors to CPU; the caller moves ys back.
        let xs_cpu = xs.to_device(&Device::Cpu)?;
        let topk_ids_cpu = topk_ids.to_device(&Device::Cpu)?;
        let topk_weights_cpu = topk_weights.to_device(&Device::Cpu)?;
        let xs = &xs_cpu;
        let topk_ids = &topk_ids_cpu;
        let topk_weights = &topk_weights_cpu;

        // Quantized fast path (SiLU MoE: granite / qwen*moe): run the per-expert GGUF
        // dot kernels directly on the ORIGINAL quantized experts. Two wins over the
        // F32 reference below: (1) perf - no BF16 dequant + no per-expert w.transpose(0,1)
        // (the reference spent ~60% of granite-moe prefill materializing that transpose);
        // (2) precision - the reference matmuls a BF16 dequant of the experts, this uses
        // the untouched Q4K with the same Q8-activation dot llama.cpp uses, staying
        // numerically closer to the reference engine. down_exps_scale is already folded
        // into topk_weights by the caller. Gated on the REAL activation (!gelu); GELU MoE
        // (gemma4) keeps the F32 reference below.
        if !self.gelu {
            if let (Some(g), Some(u)) = (self.gate_exps_sep.as_ref(), self.up_exps_sep.as_ref()) {
                // The same grouping the device path plans, over the host copies of the ids.
                let routing = ExpertRouting::group(
                    &topk_ids.to_dtype(crate::tensor::DType::U32)?,
                    self.n_experts_used,
                )?;
                // SEPARATE gate/up (granitemoe): mirror the CUDA use_silu path -
                // the original untouched Q4K experts, NOT the lossy Q8 requant of
                // the fused stack. Then SiLU(gate)*up and the down+topk reduce.
                let h = routing.host_gate_up_silu(xs, g, u)?;
                return routing.host_down_reduce(&h, &self.down_exps, topk_weights, num_tokens);
            }
        }

        let gu_full = self.gate_up_exps_f32.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg(
                "MoE CPU path: gate_up_exps_f32 not cached. Layer was placed on CPU \
                 but the dequantized expert cache wasn't built at load time.",
            )
        })?;
        let dn_full = self
            .down_exps_f32
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("MoE CPU path: down_exps_f32 not cached."))?;
        let topk_ids_v: Vec<u32> = topk_ids.flatten_all()?.to_vec1()?;
        let topk_w_v: Vec<f32> = topk_weights.flatten_all()?.to_vec1()?;
        let xs_v: Vec<f32> = xs.flatten_all()?.to_vec1()?;

        // Group tokens by expert.
        let k = self.n_experts_used;
        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for tok in 0..num_tokens {
            for j in 0..k {
                let pair = tok * k + j;
                let e = topk_ids_v[pair] as usize;
                let w = topk_w_v[pair];
                buckets[e].push((tok, w));
            }
        }

        let mut out = vec![0.0f32; num_tokens * hidden_dim];
        for (e, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            // Stack the tokens this expert sees: [m, hidden]
            let m = bucket.len();
            let mut x_sub = Vec::with_capacity(m * hidden_dim);
            for &(tok, _) in bucket {
                let s = tok * hidden_dim;
                x_sub.extend_from_slice(&xs_v[s..s + hidden_dim]);
            }
            let x_t = Tensor::from_vec(x_sub, (m, hidden_dim), &xs.device())?;
            // gate||up @ x.T : weight shape is [n_experts, 2*ffn, hidden].
            // We want x @ W^T, so narrow to expert e then matmul.
            // Cached weights are BF16 - promote to F32 for the matmul to
            // keep the precision the GPU path provides.
            let w_gu = gu_full
                .narrow(0, e, 1)?
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?; // [2*ffn, hidden]
            let w_dn = dn_full
                .narrow(0, e, 1)?
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?; // [hidden, ffn]
            let gu = x_t.matmul(&w_gu.transpose(0, 1)?)?; // [m, 2*ffn]
            let n = gu.dim(crate::tensor::D::Minus1)? / 2;
            let gate = gu.narrow(crate::tensor::D::Minus1, 0, n)?.contiguous()?;
            let up = gu.narrow(crate::tensor::D::Minus1, n, n)?.contiguous()?;
            let inner = if self.use_silu {
                (up * gate.silu()?)?
            } else {
                (up * gate.gelu()?)?
            }; // [m, ffn]
            let y = inner.matmul(&w_dn.transpose(0, 1)?)?; // [m, hidden]
            let y_v: Vec<f32> = y.flatten_all()?.to_vec1()?;
            for (i, &(tok, w)) in bucket.iter().enumerate() {
                let src = i * hidden_dim;
                let dst = tok * hidden_dim;
                for h in 0..hidden_dim {
                    out[dst + h] += w * y_v[src + h];
                }
            }
        }
        Tensor::from_vec(out, (num_tokens, hidden_dim), &xs.device())
    }
}
