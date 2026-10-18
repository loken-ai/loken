//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// The rows of a mixture routed: what the expert products and the block's return need.
pub(super) struct Routed {
    xs: Tensor,
    topk_ids: Tensor,
    topk_weights_scaled: Tensor,
    num_tokens: usize,
    batch: usize,
    seq_len: usize,
    hidden_dim: usize,
    original_dtype: crate::tensor::DType,
    moe_device: Device,
    t_route: Option<std::time::Instant>,
}

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
        // On a card the sort stays there: a host round trip would stop the stream at every
        // routed layer and keep the decode from being captured.
        #[cfg(feature = "cuda")]
        if flat.device().is_cuda() && flat.dtype() == crate::tensor::DType::U32 {
            return crate::inference::moe_cuda::stable_sort_u32(&flat);
        }
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
/// Run `launch`, and with the stage profile on, record the time the card took between its
/// start and its end, from events on the card's stream (a wait the profile pays for; inside a
/// graph being recorded, the events are stamped each time the graph runs).
#[cfg(feature = "cuda")]
fn device_timed<T>(
    name: &'static str,
    on: &Tensor,
    launch: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !crate::inference::offload::stage_prof_enabled() || !on.device().is_cuda() {
        return launch();
    }
    let Ok(dev) = on.device().as_cuda_device() else {
        return launch();
    };
    let stream = dev.0.stream().clone();
    if crate::tensor::cuda_ext::capture_active(&on.device()) {
        return crate::inference::offload::mark_stage_in_capture(name, &stream, launch);
    }
    let ctx = stream.context();
    let event = || {
        ctx.new_event(Some(
            crate::tensor::gpu_runtime::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ))
    };
    let (Ok(start), Ok(end)) = (event(), event()) else {
        return launch();
    };
    if start.record(&stream).is_err() {
        return launch();
    }
    let out = launch();
    if end.record(&stream).is_ok() && end.synchronize().is_ok() {
        if let Ok(ms) = start.elapsed_ms(&end) {
            crate::inference::offload::prof_record(name, (ms as f64 * 1e6) as u64);
        }
    }
    out
}

pub(super) struct ExpertRouting {
    pub(super) slots: Tensor,
    pub(super) experts: Tensor,
    pub(super) topk: usize,
}

impl ExpertRouting {
    pub(super) fn group(choices: &Tensor, topk: usize) -> Result<Self> {
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
    pub(super) fn gate_up_silu(&self, xs: &Tensor, gate: &QTensor, up: &QTensor) -> Result<Tensor> {
        device_timed("expert kernels: gate and up on the device", xs, || {
            crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul(
                xs,
                gate,
                up,
                &self.slots,
                &self.experts,
                self.topk,
            )
        })
    }

    /// [`Self::gate_up_silu`] over the experts of the two stacks and, numbered after them, those
    /// of `more`, in one launch.
    #[cfg(feature = "cuda")]
    pub(super) fn gate_up_silu_over(
        &self,
        xs: &Tensor,
        gate: &QTensor,
        up: &QTensor,
        more: Option<crate::inference::moe_cuda::MoreExperts<(&QTensor, &QTensor)>>,
    ) -> Result<Tensor> {
        device_timed("expert kernels: gate and up on the device", xs, || {
            crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul_over(
                xs,
                gate,
                up,
                more,
                &self.slots,
                &self.experts,
                self.topk,
            )
        })
    }

    /// The down projection and weighted fold of `n_tokens` tokens over the experts of `stack`
    /// and, numbered after them, those of `more`, in one launch.
    #[cfg(feature = "cuda")]
    /// [`Self::down_reduce_over`] up to the pairs' rows, `[pairs, hidden]`: the caller sums a
    /// row's pairs with whatever else it adds.
    pub(super) fn down_pairs_over(
        &self,
        rows: &Tensor,
        stack: &QTensor,
        more: Option<crate::inference::moe_cuda::MoreExperts<&QTensor>>,
        weights: &Tensor,
    ) -> Result<Tensor> {
        device_timed("expert kernels: down on the device", rows, || {
            crate::inference::moe_cuda::moe_gemm_gguf_down_pairs_over(
                rows,
                stack,
                more,
                &self.slots,
                &self.experts,
                weights,
                self.topk,
            )
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn down_reduce_over(
        &self,
        rows: &Tensor,
        stack: &QTensor,
        more: Option<&QTensor>,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        device_timed("expert kernels: down on the device", rows, || {
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce_over(
                rows,
                stack,
                more,
                &self.slots,
                &self.experts,
                weights,
                self.topk,
                n_tokens,
                None,
                None,
                None,
            )
        })
    }

    /// Gate and up kept apart, each with its per-expert bias, combined by the clamped
    /// SwiGLU: one launch.
    #[cfg(feature = "cuda")]
    fn gate_up_swiglu_clamped(
        &self,
        xs: &Tensor,
        gate: &QTensor,
        up: &QTensor,
        biases: (Option<&Tensor>, Option<&Tensor>),
        (alpha, limit): (f64, f64),
    ) -> Result<Tensor> {
        crate::inference::moe_cuda::moe_gemm_gguf_gate_up_swiglu_oai(
            xs,
            gate,
            up,
            biases.0,
            biases.1,
            &self.slots,
            &self.experts,
            self.topk,
            alpha,
            limit,
        )
    }

    /// Down projection and the weighted scatter-add that ends it: each row scaled by the
    /// score its expert was chosen with, added into the token it came from. A per-expert
    /// `bias` joins each row before its score scales it.
    #[cfg(feature = "cuda")]
    pub(super) fn down_reduce(
        &self,
        rows: &Tensor,
        stack: &QTensor,
        weights: &Tensor,
        n_tokens: usize,
        imma: bool,
        bias: Option<&Tensor>,
        add_after: Option<&Tensor>,
    ) -> Result<Tensor> {
        device_timed("expert kernels: down on the device", rows, || {
            if imma && bias.is_none() && add_after.is_none() {
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
                    bias,
                    add_after,
                )
            }
        })
    }

    /// [`Self::gate_up_silu`] on the host.
    pub(super) fn host_gate_up_silu(
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

    /// [`Self::gate_up_swiglu_clamped`] on the host.
    fn host_gate_up_swiglu_clamped(
        &self,
        xs: &Tensor,
        gate: &Arc<QTensor>,
        up: &Arc<QTensor>,
        biases: (Option<&Tensor>, Option<&Tensor>),
        (alpha, limit): (f64, f64),
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_gate_up_swiglu_oai(
            xs,
            gate,
            up,
            biases.0,
            biases.1,
            &self.slots,
            &self.experts,
            self.topk,
            alpha,
            limit,
        )
    }

    /// [`Self::down_reduce`] on the host.
    pub(super) fn host_down_reduce(
        &self,
        rows: &Tensor,
        stack: &Arc<QTensor>,
        weights: &Tensor,
        n_tokens: usize,
        bias: Option<&Tensor>,
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
            bias,
        )
    }
}

/// Sigmoid expert routing: every expert scores `sigmoid(logit)`, the experts are chosen on
/// the scores plus a selection bias, and each chosen expert weighs by its unbiased score,
/// optionally renormalised over the chosen ones and scaled.
#[derive(Debug, Clone)]
pub struct SigmoidRouting {
    /// Added to the scores for the choice only: `[n_experts]` F32.
    pub selection_bias: Option<Tensor>,
    /// The chosen weights are divided by their sum, floored at the smallest normal F16.
    pub normalize: bool,
    /// Multiplies the chosen weights.
    pub scale: f64,
}

impl SigmoidRouting {
    /// Floor of the renormalising sum, the smallest normal F16.
    const SUM_FLOOR: f32 = 6.103_515_6e-5;

    /// `(weights, ids)` `[tokens, k]` from the router `logits` `[tokens, n_experts]` F32.
    pub fn select(&self, logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        if let Some(out) = crate::inference::moe_cuda::topk_sigmoid_post(
            logits,
            self.selection_bias.as_ref(),
            k,
            self.normalize,
            self.scale,
        )? {
            return Ok(out);
        }
        let probs = crate::tensor::ops::sigmoid(logits)?;
        let chosen_on = match &self.selection_bias {
            Some(b) => probs.broadcast_add(&b.to_device(&probs.device())?)?,
            None => probs.clone(),
        };
        let ids = chosen_on
            .arg_sort_last_dim(false)?
            .narrow(crate::tensor::D::Minus1, 0, k)?
            .contiguous()?;
        let mut weights = probs.gather(&ids, crate::tensor::D::Minus1)?;
        if self.normalize {
            let sum = weights.sum_keepdim(crate::tensor::D::Minus1)?;
            let floor = Tensor::full(Self::SUM_FLOOR, sum.dims(), &sum.device())?;
            weights = weights.broadcast_div(&sum.maximum(&floor)?)?;
        }
        if self.scale != 1.0 {
            weights = (weights * self.scale)?;
        }
        Ok((weights, ids))
    }
}

/// The expert every token of a mixture runs beside the routed ones: SwiGLU when it has a
/// gate, `relu(up)^2` otherwise, its output scaled by `sigmoid(x . scalar_gate)` when the
/// file gives that gate.
pub struct SharedExpert {
    pub gate: Option<QMatMul>,
    pub up: QMatMul,
    pub down: QMatMul,
    /// `[hidden, 1]` F32.
    pub scalar_gate: Option<Tensor>,
}

impl SharedExpert {
    /// `routed` `[tokens, hidden]` F32 plus the expert over `xs`. A decoded token on a card
    /// runs the elementwise stages around the projections as two launches.
    pub fn forward_added(
        &self,
        xs: &Tensor,
        routed: &Tensor,
        projected: Option<Tensor>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let (Some(gate), Some(w)) = (&self.gate, &self.scalar_gate) {
            // A decoded token or a window's rows: the pair of projections (fused for one
            // row), the gate, the down projection and the gated sum, each over every row.
            let rows = xs.dims().first().copied().unwrap_or(0);
            if xs.device().is_cuda()
                && rows >= 1
                && rows <= crate::inference::engine::decode_step::VERIFY_ROWS
            {
                if let Some(down) = projected {
                    if let Some(sum) = crate::inference::moe_cuda::shexp_tail(routed, &down, xs, w)?
                    {
                        return Ok(sum);
                    }
                    return routed + self.finish(xs, down)?;
                }
                // The pair in one launch where the card has it, else as two projections
                // and the launch that joins them.
                let fused = QMatMul::fused_silu_pair(gate, &self.up, xs)?;
                let (up, g) = match fused {
                    Some(_) => (None, None),
                    None => (Some(self.up.forward(xs)?), Some(gate.forward(xs)?)),
                };
                let h = match (fused, &up, &g) {
                    (Some(h), _, _) => Some(h),
                    (None, Some(up), Some(g)) => crate::inference::moe_cuda::shexp_swiglu(g, up)?,
                    _ => None,
                };
                if let Some(h) = h {
                    let down = self.down.forward(&h)?;
                    if let Some(sum) = crate::inference::moe_cuda::shexp_tail(routed, &down, xs, w)?
                    {
                        return Ok(sum);
                    }
                    return routed + self.finish(xs, down)?;
                }
                let (Some(up), Some(g)) = (up, g) else {
                    return routed + self.forward(xs)?;
                };
                let up = up.to_dtype(crate::tensor::DType::F32)?;
                let g = g.to_dtype(crate::tensor::DType::F32)?;
                let h = (crate::tensor::ops::silu(&g)? * up)?;
                return routed + self.finish(xs, self.down.forward(&h)?)?;
            }
        }
        routed + self.forward(xs)?
    }

    /// The expert's projections for a decoded token on a card, which do not depend on the routed
    /// experts: the down projection's output, to be joined by [`Self::forward_added`]. `None`
    /// where that path does not serve the expert.
    #[cfg(feature = "cuda")]
    pub fn project(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        let (Some(gate), Some(_)) = (&self.gate, &self.scalar_gate) else {
            return Ok(None);
        };
        if !xs.device().is_cuda() || xs.dims().first() != Some(&1) {
            return Ok(None);
        }
        if let Some(h) = QMatMul::fused_silu_pair(gate, &self.up, xs)? {
            return Ok(Some(self.down.forward(&h)?));
        }
        let up = self.up.forward(xs)?;
        let g = gate.forward(xs)?;
        match crate::inference::moe_cuda::shexp_swiglu(&g, &up)? {
            Some(h) => Ok(Some(self.down.forward(&h)?)),
            None => Ok(None),
        }
    }

    /// No card, no projections ahead of a wait.
    #[cfg(not(feature = "cuda"))]
    pub fn project(&self, _: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// The down projection's output `out` scaled by the scalar gate, F32.
    fn finish(&self, xs: &Tensor, out: Tensor) -> Result<Tensor> {
        let out = out.to_dtype(crate::tensor::DType::F32)?;
        match &self.scalar_gate {
            Some(w) => out.broadcast_mul(&crate::tensor::ops::sigmoid(&xs.matmul(w)?)?),
            None => Ok(out),
        }
    }

    /// The expert over `xs` `[tokens, hidden]` F32, `[tokens, hidden]` F32.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = self.up.forward(xs)?.to_dtype(crate::tensor::DType::F32)?;
        let h = match &self.gate {
            Some(g) => {
                let g = g.forward(xs)?.to_dtype(crate::tensor::DType::F32)?;
                (crate::tensor::ops::silu(&g)? * up)?
            }
            None => up.relu()?.sqr()?,
        };
        let out = self.down.forward(&h)?.to_dtype(crate::tensor::DType::F32)?;
        match &self.scalar_gate {
            Some(w) => {
                let gate = crate::tensor::ops::sigmoid(&xs.matmul(w)?)?;
                out.broadcast_mul(&gate)
            }
            None => Ok(out),
        }
    }
}

// ------------------------------------------------------------
// Per-layer struct
// ------------------------------------------------------------

/// The routed experts one launch reads, all on one device: the stacks, the per-expert biases
/// and the decode slot table, the experts numbered from zero within the set.
#[derive(Clone, Copy)]
pub(super) struct ExpertSet<'a> {
    pub gate: Option<&'a QTensor>,
    pub up: Option<&'a QTensor>,
    pub gate_up: Option<&'a QTensor>,
    pub down: &'a QTensor,
    pub gate_bias: Option<&'a Tensor>,
    pub up_bias: Option<&'a Tensor>,
    pub down_bias: Option<&'a Tensor>,
    pub decode_slots: Option<&'a Tensor>,
    pub n_experts: usize,
}

/// Routed experts of a layer held on one card: the experts `first..first + count` with their
/// per-expert biases, numbered from zero on that card.
pub struct ExpertShard {
    pub dev: Device,
    pub first: usize,
    pub count: usize,
    /// `[gate, up, down]`.
    pub stacks: [QTensor; 3],
    pub gate_bias: Option<Tensor>,
    pub up_bias: Option<Tensor>,
    pub down_bias: Option<Tensor>,
    pub decode_slots: Tensor,
}

impl ExpertShard {
    /// The experts `first..first + count` of `layer`, whose stacks are in host memory, on `dev`.
    /// The layer reads its gate and up apart.
    #[cfg(feature = "cuda")]
    pub fn from_host(layer: &MoeWeights, first: usize, count: usize, dev: &Device) -> Result<Self> {
        let (Some(gate), Some(up)) = (layer.gate_exps_sep.as_ref(), layer.up_exps_sep.as_ref())
        else {
            crate::tensor::bail!("an expert shard reads gate and up apart");
        };
        let range = |stack: &std::sync::Arc<QTensor>| {
            super::expert_tier::experts_range(stack, first, count, dev)
        };
        let rows = |t: &Option<Tensor>| -> Result<Option<Tensor>> {
            t.as_ref()
                .map(|t| t.narrow(0, first, count)?.to_device(dev))
                .transpose()
        };
        let used = layer.n_experts_used as u32;
        Ok(Self {
            dev: dev.clone(),
            first,
            count,
            stacks: [range(gate)?, range(up)?, range(&layer.down_exps)?],
            gate_bias: rows(&layer.gate_exps_bias)?,
            up_bias: rows(&layer.up_exps_bias)?,
            down_bias: rows(&layer.down_exps_bias)?,
            decode_slots: Tensor::from_vec(
                (0..used).collect::<Vec<_>>(),
                (used as usize,),
                &Device::Cpu,
            )?
            .to_device(dev)?,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_host(_: &MoeWeights, _: usize, _: usize, _: &Device) -> Result<Self> {
        crate::tensor::bail!("an expert shard lives on a card")
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn set(&self) -> ExpertSet<'_> {
        ExpertSet {
            gate: Some(&self.stacks[0]),
            up: Some(&self.stacks[1]),
            gate_up: None,
            down: &self.stacks[2],
            gate_bias: self.gate_bias.as_ref(),
            up_bias: self.up_bias.as_ref(),
            down_bias: self.down_bias.as_ref(),
            decode_slots: Some(&self.decode_slots),
            n_experts: self.count,
        }
    }
}

/// Where a split layer's card's experts start among the router's, and how many the router has.
#[derive(Clone, Copy, Debug)]
pub struct LocalExperts {
    pub first: usize,
    pub total: usize,
}

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
    /// Bias added to the router logits before the top-k: `[n_experts]`.
    pub gate_inp_bias: Option<Tensor>,
    /// Per-expert biases of the gate, up and down projections: `[n_experts, width]`.
    /// Present only with `swiglu_clamp`.
    pub gate_exps_bias: Option<Tensor>,
    pub up_exps_bias: Option<Tensor>,
    pub down_exps_bias: Option<Tensor>,
    /// Gate and up combine through the clamped SwiGLU, `(alpha, limit)`, reading the
    /// separate stacks: `x = min(gate, limit)`, `x.sigmoid(alpha.x).(1 + clamp(up))`.
    pub swiglu_clamp: Option<(f64, f64)>,
    /// Stacked expert gate||up weights `[n_experts, 2*expert_ffn_dim, hidden]`
    /// (the stored layout - GGUF dims are reversed). `None` on a clamped-SwiGLU layer,
    /// which reads gate and up apart.
    pub gate_up_exps: Option<std::sync::Arc<crate::tensor::quantized::QTensor>>,
    /// Stacked expert down weights `[n_experts, hidden, expert_ffn_dim]`.
    pub down_exps: std::sync::Arc<crate::tensor::quantized::QTensor>,
    /// Standard MoE (granitemoe): SEPARATE gate & up expert stacks `[n_exp, ffn,
    /// hidden]`, kept quantized on-device for the GPU SiLU kernel
    /// (`moe_gemm_gguf_gate_up_silu_mul`, which takes gate/up separately). None
    /// for gemma4-MoE (fused gate||up + GELU).
    pub gate_exps_sep: Option<std::sync::Arc<crate::tensor::quantized::QTensor>>,
    pub up_exps_sep: Option<std::sync::Arc<crate::tensor::quantized::QTensor>>,
    /// Per-expert scalar on the down output, `[n_experts]`; `None` when the checkpoint has
    /// none, every expert then being scaled by one.
    pub down_exps_scale: Option<Tensor>,
    /// `0..n_experts_used` on the layer's device: at a single-token decode the routed pairs
    /// are already the rows in slot order, so this stands in for the sort that would
    /// otherwise name them.
    pub decode_slots: Option<Tensor>,
    /// The routed experts split by id range across cards, the layer's own card among them;
    /// empty when the layer's own stacks hold every expert.
    pub shards: Vec<ExpertShard>,
    /// On one card of a split layer: the stacks hold the router's experts from this one on, and
    /// the pairs of the other cards' experts weigh zero here.
    pub local: Option<LocalExperts>,
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
    /// The router scores experts by sigmoid instead of softmax. `None` for softmax.
    pub sigmoid_routing: Option<SigmoidRouting>,
    /// The experts have no gate: `down(relu(up(x))^2)` over `up_exps_sep`.
    pub up_relu2: bool,
    /// An expert every token runs, added to the routed ones.
    pub shared_expert: Option<SharedExpert>,
    /// CPU-only cache: dequantized F32 expert tensors. Populated for layers
    /// placed on CPU since `moe_gemm_gguf` is CUDA-only. Avoids per-call
    /// dequant cost on what is already a slow CPU path.
    pub gate_up_exps_f32: Option<std::sync::Arc<Tensor>>,
    /// The routed experts in host memory with a cache of them on the card.
    pub tier: Option<std::sync::Arc<super::expert_tier::LayerTier>>,
    pub down_exps_f32: Option<std::sync::Arc<Tensor>>,
}

impl MoeWeights {
    /// Whether the routed expert stacks sit in host memory, where the expert loop then runs
    /// whatever device the rest of the layer is on.
    pub fn experts_on_host(&self) -> bool {
        !self.down_exps.device().runs_as_card()
    }

    /// Whether a decoded token waits on the host for this layer's experts: they sit in host
    /// memory and the cache on the card does not hold them all.
    pub fn decodes_through_host(&self) -> bool {
        self.experts_on_host() && !self.tier.as_ref().is_some_and(|t| t.is_resident())
    }

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
            _ => stacks.extend(self.gate_up_exps.clone()),
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
    /// [`Self::forward`] over the rows of `moe_in`. While a verify holds `DecodeOrder` the rows
    /// take the decode kernels, so a verified row routes and sums exactly as the step it
    /// stands for: together through a tier on a card (the kernels work a routed pair at a time,
    /// and a token's pairs are summed in rank order), one at a time elsewhere. The whole batch
    /// at once otherwise.
    pub(super) fn forward_rows(
        &self,
        moe_in: &Tensor,
        attn_out: &Tensor,
        n_embd: usize,
        rms_eps: f32,
    ) -> Result<Tensor> {
        let seq = moe_in.dim(1)?;
        // A layer whose experts pass through a cache takes the rows together: it routes every
        // row alone, and its card runs each row's pairs as it would the decoded token's.
        let decode_order = crate::tensor::quantized::DecodeOrder::active();
        // A tier, cached or resident, runs every row's pairs in the decode kernels on the card.
        let on_card = moe_in.device().runs_as_card() && !self.up_relu2 && self.tier.is_some();
        // A cached layer's rows go through the cache as decoded tokens, prompt or verify.
        let cached =
            moe_in.device().runs_as_card() && self.tier.as_ref().is_some_and(|t| !t.is_resident());
        if seq == 1 || !decode_order || on_card {
            let as_prefill = seq > 1 && !decode_order && !cached;
            return self.forward(moe_in, attn_out, n_embd, rms_eps, as_prefill);
        }
        // Each row in storage of its own: `forward` reads its input from the start of the
        // storage.
        let rows = (0..seq)
            .map(|r| {
                self.forward(
                    &moe_in.narrow(1, r, 1)?.contiguous()?,
                    &attn_out.narrow(1, r, 1)?.contiguous()?,
                    n_embd,
                    rms_eps,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
    }

    pub(super) fn forward(
        &self,
        moe_in_pre_norm_2: &Tensor, // already pre_ffw_norm_2-applied
        attn_out: &Tensor,          // for the gate's own RMS-norm input
        n_embd: usize,
        rms_eps: f32,
        is_prefill: bool,
    ) -> Result<Tensor> {
        self.forward_with(
            moe_in_pre_norm_2,
            attn_out,
            n_embd,
            rms_eps,
            is_prefill,
            None,
        )
    }

    /// Whether [`Self::forward_folding`] can join `residual` to the block's output inside the
    /// last launch: one decoded token on a card, the experts on it too, in F32 throughout, and
    /// nothing added to the expert rows between the fold and the residual.
    pub(super) fn can_fold_residual(&self, moe_in: &Tensor, residual: &Tensor) -> bool {
        use crate::tensor::DType::F32;
        moe_in.dtype() == F32
            && residual.dtype() == F32
            && residual.is_contiguous()
            && moe_in.dims().len() == 3
            && moe_in.dims()[..2] == [1, 1]
            && residual.elem_count() == moe_in.elem_count()
            && moe_in.device().runs_as_card()
            && !self.experts_on_host()
            && !self.up_relu2
            && self.shared_expert.is_none()
            && self.decode_slots.is_some()
    }

    /// [`Self::forward_rows`] for a single token, its output already added to `residual`
    /// (`can_fold_residual` says when that is possible).
    pub(super) fn forward_folding(
        &self,
        moe_in: &Tensor,
        attn_out: &Tensor,
        n_embd: usize,
        rms_eps: f32,
        residual: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with(moe_in, attn_out, n_embd, rms_eps, false, Some(residual))
    }

    fn forward_with(
        &self,
        moe_in_pre_norm_2: &Tensor,
        attn_out: &Tensor,
        n_embd: usize,
        rms_eps: f32,
        is_prefill: bool,
        add_after: Option<&Tensor>,
    ) -> Result<Tensor> {
        let routed = self.route(moe_in_pre_norm_2, attn_out, n_embd, rms_eps)?;
        // The shared expert's projections, when a recorded decode queued them ahead of the wait.
        let mut shared_projected: Option<Tensor> = None;
        let expert_rows = self.experts_of(&routed, is_prefill, add_after, &mut shared_projected)?;
        self.finish_rows(&routed, expert_rows, shared_projected)
    }

    /// The rows routed: the rows in F32, the router's choice and its weights, and the shape and
    /// dtype the block returns in.
    fn route(
        &self,
        moe_in_pre_norm_2: &Tensor,
        attn_out: &Tensor,
        n_embd: usize,
        rms_eps: f32,
    ) -> Result<Routed> {
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
            // A biased or sigmoid router scores in full F32 on a card: the GEMV kernel, not
            // the dense matmul whose reduced-precision mode can reorder near-tied experts.
            let exact_logits = || -> Result<Tensor> {
                let exact = match &self.gate_inp {
                    crate::tensor::quantized::QMatMul::Tensor(w) => {
                        crate::inference::moe_cuda::gate_gemv_f32(&xs, w)?
                    }
                    crate::tensor::quantized::QMatMul::Bf16(w) => {
                        crate::inference::moe_cuda::bf16_matvec(&xs, w)?
                    }
                    _ => None,
                };
                let logits = match exact {
                    Some(l) => l,
                    None => self.gate_inp.forward(&xs)?,
                };
                logits.to_dtype(crate::tensor::DType::F32)
            };
            match self.gate_inp_bias.as_ref() {
                None if self.sigmoid_routing.is_some() => exact_logits()?,
                // The rows of a verify score one at a time, as a decoded token does: a library
                // matrix product picks its arithmetic by shape.
                // The rows of a verify each as the one-row call: the row-wise gate kernel over
                // the rows at once, or the quantised gate's mat-vecs, which take the rows one by
                // one under the decode order.
                None if num_tokens > 1 && crate::tensor::quantized::DecodeOrder::active() => {
                    exact_logits()?
                }
                // The cards of a split layer each route the same rows: each row's logits are
                // summed as a one-row call sums them, which no card's kernel choice changes, so
                // every card picks the same experts.
                None if self.local.is_some() => {
                    let _order = crate::tensor::quantized::DecodeOrder::enter();
                    exact_logits()?
                }
                None => self
                    .gate_inp
                    .forward(&xs)?
                    .to_dtype(crate::tensor::DType::F32)?,
                Some(bias) => {
                    exact_logits()?.broadcast_add(&bias.to_dtype(crate::tensor::DType::F32)?)?
                }
            }
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
        // One kernel scores and selects for the expert widths it covers, 32 to 512; every
        // other width, and every build compiled without kernels, takes the
        // two steps as tensor ops. Which of the two ran is what the stage profile records.
        #[cfg(feature = "cuda")]
        let (topk_weights, topk_ids) = if let Some(sigmoid) = self.sigmoid_routing.as_ref() {
            sigmoid.select(&logits_f32, self.n_experts_used)?
        } else {
            let fused = logits_f32.device().is_cuda()
                && crate::inference::moe_cuda::topk_softmax_serves(
                    logits_f32.dim(crate::tensor::D::Minus1)?,
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
        let (topk_weights, topk_ids) = match self.sigmoid_routing.as_ref() {
            Some(sigmoid) => sigmoid.select(&logits_f32, self.n_experts_used)?,
            None => self.top_k(&logits_f32)?,
        };

        // -- Sort (expert, token) pairs.
        if let Some(t) = t_topk {
            let _ = topk_ids.device().synchronize();
            crate::inference::place::layer_perf::stages::add(9, t.elapsed().as_micros() as u64);
        }
        // Routing: from the top-k choice to the expert-grouped plan.
        let t_route = crate::inference::place::layer_perf::stages::enabled().then(|| {
            let _ = topk_ids.device().synchronize();
            std::time::Instant::now()
        });

        // -- Fold the per-expert down-output scale into topk_weights.
        // llama.cpp gemma4-iswa.cpp applies it after the down matmul
        // before the topk-weighted reduction:
        //   experts[t,k,h] *= down_exps_scale[topk_ids[t,k]]
        //   experts[t,k,h] *= topk_weights[t,k]
        // Folding scale into weights gives the same final result and
        // keeps a single weighted reduction.
        let topk_weights_scaled = match self.down_exps_scale.as_ref() {
            Some(scale) => {
                let scale_per_pair = scale
                    .gather(&topk_ids.flatten_all()?, 0)?
                    .reshape(topk_weights.shape())?;
                (topk_weights.clone() * scale_per_pair)?
            }
            None => topk_weights.clone(),
        };

        // SiLU MoE hybrid: the GPU multi-token PREFILL kernels (WMMA + mmvq) both
        // mis-numeric these separate Q4K experts (verified: per-layer GPU-vs-CPU
        // diverges at layer 5+ for M>1). Single-token DECODE via mmvq is EXACT
        // (matched CPU <1% at every layer). So run prefill on the CPU expert path
        // (one-time, correct) and decode on GPU (the tok/s bottleneck, fast+exact).
        let moe_device = xs.device().clone();
        Ok(Routed {
            xs,
            topk_ids,
            topk_weights_scaled,
            num_tokens,
            batch,
            seq_len,
            hidden_dim,
            original_dtype,
            moe_device,
            t_route,
        })
    }

    /// The expert products of the routed rows, `[num_tokens, hidden]`, by whichever path the
    /// layer's placement takes.
    fn experts_of(
        &self,
        routed: &Routed,
        is_prefill: bool,
        add_after: Option<&Tensor>,
        shared_projected: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (xs, topk_ids, topk_weights_scaled) =
            (&routed.xs, &routed.topk_ids, &routed.topk_weights_scaled);
        let (num_tokens, hidden_dim, t_route) =
            (routed.num_tokens, routed.hidden_dim, routed.t_route);
        // Which arrangement, not which hardware. True only on the host either way -
        // the two spellings agree on every device - but the question being asked here
        // is the shape of the expert loop, so it is asked in those terms.
        // A SiLU mixture's prefill used to be sent here whatever card the layer sat on, because
        // the tiled down-projection answered wrongly above sixty-four routed rows. It read the
        // quantised activation by SORTED POSITION while both its producers wrote that row by
        // SLOT, and it dropped every expert past the fourth in a tile of eight. Neither was
        // caught, because that kernel had no parity test - it has one now, and the host detour
        // it justified cost about eight times the prefill.
        let on_cpu = !xs.device().runs_as_card() || self.experts_on_host();
        if add_after.is_some()
            && (self.up_relu2 || on_cpu || self.shared_expert.is_some() || num_tokens != 1)
        {
            crate::tensor::bail!("MoeWeights: this layer cannot fold a residual into its fold");
        }
        // The shared expert's projections, when a recorded decode queued them ahead of the wait.
        let expert_rows = if let Some(local) = self.local.filter(|_| xs.device().runs_as_card()) {
            self.experts_local(
                xs,
                topk_ids,
                topk_weights_scaled,
                num_tokens,
                is_prefill,
                t_route,
                local,
            )?
        } else if !self.shards.is_empty() && xs.device().runs_as_card() {
            self.experts_split(
                xs,
                topk_ids,
                topk_weights_scaled,
                num_tokens,
                is_prefill,
                t_route,
            )?
        } else if self.up_relu2 {
            self.forward_up_relu2(xs, topk_ids, topk_weights_scaled, num_tokens)?
        } else if let Some(tier) = self
            .tier
            .as_ref()
            .filter(|t| t.is_resident() && xs.device().runs_as_card())
        {
            // Every expert is on the card: the layer runs as one that was loaded there.
            self.forward_cuda(
                xs,
                topk_ids,
                topk_weights_scaled,
                num_tokens,
                is_prefill,
                t_route,
                add_after,
                Some(tier.card_stacks()),
            )?
        } else if let (Some(tier), true, 1) =
            (self.tier.as_ref(), xs.device().runs_as_card(), num_tokens)
        {
            // A recorded decode queues the shared expert's projections while the host answers.
            if tier.records(&xs.device()) {
                let pending = tier.graph_begin(self, xs, topk_ids, topk_weights_scaled)?;
                if let Some(shared) = &self.shared_expert {
                    *shared_projected = shared.project(xs)?;
                }
                tier.graph_end(pending)?
            } else {
                tier.decode(self, xs, topk_ids, topk_weights_scaled)?
            }
        } else if let Some(tier) = self.tier.as_ref().filter(|t| {
            xs.device().runs_as_card() && t.stream_pays(num_tokens, self.n_experts_used)
        }) {
            // A wide batch reads nearly every expert: the layer's experts come to the card,
            // beside the work before them, and the whole layer runs there as if it lived there.
            let dev = xs.device().clone();
            let stacks = super::wide::staged("tier fill wait", &dev, false, || tier.stacks())?;
            let out = super::wide::staged("tier experts", &dev, false, || {
                self.forward_cuda(
                    xs,
                    topk_ids,
                    topk_weights_scaled,
                    num_tokens,
                    true,
                    t_route,
                    None,
                    Some(&*stacks),
                )
            });
            // The experts the batch used warm the cache the decode will read.
            #[cfg(feature = "cuda")]
            if let Ok(card) = dev.as_cuda_device() {
                let stream = card.0.stream().clone();
                if let Err(e) = tier.keep_hottest_of_prompt(topk_ids, &stream) {
                    tracing::warn!("the cache is not warmed from the prompt: {e}");
                }
            }
            tier.release_stacks()?;
            out?
        } else if let Some(tier) = self.tier.as_ref().filter(|t| {
            xs.device().runs_as_card()
                && num_tokens > 1
                && t.rows_through_cache(num_tokens, self.n_experts_used)
        }) {
            // Too few rows to pay for the layer's experts to come: the rows go through the cache
            // together, as the decoded tokens they stand for would one after the other. The rows
            // of a recorded verify go through the kernels of the recorded decode.
            if tier.records(&xs.device()) && num_tokens <= tier.graph_rows() {
                tier.decode_rows_graph(self, xs, topk_ids, topk_weights_scaled)?
            } else {
                tier.decode_rows(self, xs, topk_ids, topk_weights_scaled)?
            }
        } else if on_cpu {
            self.forward_cpu(xs, topk_ids, topk_weights_scaled, num_tokens, hidden_dim)?
        } else {
            self.forward_cuda(
                xs,
                topk_ids,
                topk_weights_scaled,
                num_tokens,
                is_prefill,
                t_route,
                add_after,
                None,
            )?
        };
        Ok(expert_rows)
    }

    /// The expert products back on the block's device, the shared expert added, in the block's
    /// width and dtype.
    fn finish_rows(
        &self,
        routed: &Routed,
        expert_rows: Tensor,
        shared_projected: Option<Tensor>,
    ) -> Result<Tensor> {
        let (xs, moe_device) = (&routed.xs, &routed.moe_device);
        let (original_dtype, batch, seq_len, hidden_dim) = (
            routed.original_dtype,
            routed.batch,
            routed.seq_len,
            routed.hidden_dim,
        );
        // Back to the block's own device, width and dtype. The host expert path answers on
        // the host even for a layer that lives on a card, so that the hybrid prefill above
        // can run there; the caller's residual add expects the layer's device back.
        let expert_rows = match expert_rows.device().same_device(moe_device) {
            true => expert_rows,
            false => expert_rows.to_device(moe_device)?,
        };
        let expert_rows = match &self.shared_expert {
            Some(shared) => shared.forward_added(
                xs,
                &expert_rows.to_dtype(crate::tensor::DType::F32)?,
                shared_projected,
            )?,
            None => expert_rows,
        };
        expert_rows
            .to_dtype(original_dtype)?
            .reshape((batch, seq_len, hidden_dim))
    }

    /// Experts without a gate: `down(relu(up(x))^2)` for every chosen expert, weighed and
    /// summed per token. `[tokens, hidden]` F32.
    fn forward_up_relu2(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
    ) -> Result<Tensor> {
        let up = self
            .up_exps_sep
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("gateless experts without an up stack"))?;
        #[cfg(feature = "cuda")]
        if xs.device().runs_as_card() {
            let routing = ExpertRouting::group(topk_ids, self.n_experts_used)?;
            let h = crate::inference::moe_cuda::moe_gemm_gguf(
                xs,
                up,
                &None,
                &routing.slots,
                &routing.experts,
                self.n_experts_used,
                num_tokens > 1,
                crate::tensor::DType::F16,
            )?;
            let h = match crate::inference::kernel::fused::relu2_f16(&h)? {
                Some(y) => y,
                None => h.relu()?.sqr()?,
            };
            return routing.down_reduce(
                &h.to_dtype(crate::tensor::DType::F32)?,
                &self.down_exps,
                topk_weights,
                num_tokens,
                false,
                None,
                None,
            );
        }
        let host = Device::Cpu;
        let (xs, ids, weights) = (
            xs.to_device(&host)?,
            topk_ids.to_device(&host)?,
            topk_weights.to_device(&host)?,
        );
        let routing = ExpertRouting::group(
            &ids.to_dtype(crate::tensor::DType::U32)?,
            self.n_experts_used,
        )?;
        let h = crate::inference::moe_cpu::moe_gemm_gguf_up_relu2(
            &xs,
            up,
            &routing.slots,
            &routing.experts,
            self.n_experts_used,
        )?;
        routing.host_down_reduce(&h, &self.down_exps, &weights, num_tokens, None)
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

    /// The routed pairs of `topk_ids` computed by the shards that hold their experts, every shard
    /// launched before any is waited on, and summed on the layer's card. A shard reads the ids of
    /// its own experts and weighs every other pair zero.
    #[cfg(feature = "cuda")]
    fn experts_split(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
        is_prefill: bool,
        t_route: Option<std::time::Instant>,
    ) -> Result<Tensor> {
        let dev = &xs.device();
        let ids: Vec<u32> = topk_ids
            .to_dtype(crate::tensor::DType::U32)?
            .flatten_all()?
            .to_vec1()?;
        let weights: Vec<f32> = topk_weights
            .to_dtype(crate::tensor::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let mut sum: Option<Tensor> = None;
        let mut parts = Vec::with_capacity(self.shards.len());
        let k = self.n_experts_used;
        for shard in &self.shards {
            if shard.count < k {
                crate::tensor::bail!("a shard holds fewer experts than a row routes to");
            }
            // A pair of another shard's expert reads an expert of this shard its row does not
            // route to, at weight zero: the kernels that group a row's pairs by expert take each
            // expert of a row once.
            let mut local = vec![0u32; ids.len()];
            let mut weight = vec![0f32; ids.len()];
            for (row, (ids, ws)) in ids.chunks(k).zip(weights.chunks(k)).enumerate() {
                let mine = |e: u32| {
                    (e as usize)
                        .checked_sub(shard.first)
                        .filter(|&l| l < shard.count)
                };
                let taken: Vec<usize> = ids.iter().filter_map(|&e| mine(e)).collect();
                let mut spare = (0..shard.count).filter(|l| !taken.contains(l));
                for (j, (&e, &w)) in ids.iter().zip(ws).enumerate() {
                    let at = row * k + j;
                    match mine(e) {
                        Some(l) => (local[at], weight[at]) = (l as u32, w),
                        None => local[at] = spare.next().unwrap_or(0) as u32,
                    }
                }
            }
            let local = Tensor::from_vec(local, topk_ids.shape().clone(), &Device::Cpu)?
                .to_dtype(topk_ids.dtype())?
                .to_device(&shard.dev)?;
            let weight = Tensor::from_vec(weight, topk_weights.shape().clone(), &Device::Cpu)?
                .to_dtype(topk_weights.dtype())?
                .to_device(&shard.dev)?;
            let rows = if shard.dev.same_device(dev) {
                xs.clone()
            } else {
                xs.to_device(&shard.dev)?
            };
            parts.push(self.experts_cuda(
                &rows,
                &local,
                &weight,
                num_tokens,
                is_prefill,
                t_route.filter(|_| shard.dev.same_device(dev)),
                None,
                shard.set(),
            )?);
        }
        for part in parts {
            let part = if part.device().same_device(dev) {
                part
            } else {
                part.to_device(dev)?
            };
            sum = Some(match sum {
                None => part,
                Some(s) => (s + part)?,
            });
        }
        sum.ok_or_else(|| crate::tensor::Error::msg("a split layer has no shard"))
    }

    /// The routed pairs of `topk_ids` this card's experts compute, every other pair at weight
    /// zero: the card's part of a split layer's sum.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn experts_local(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
        is_prefill: bool,
        t_route: Option<std::time::Instant>,
        local: LocalExperts,
    ) -> Result<Tensor> {
        use super::expert_tier::address_of;
        use crate::tensor::DType;
        let k = self.n_experts_used;
        if k > crate::inference::moe_cuda::tp_local::MAX_K || self.n_experts < k {
            crate::tensor::bail!(
                "a card of a split layer holds fewer experts than a row routes to"
            );
        }
        let dev = xs.device();
        let ids = topk_ids.to_dtype(DType::U32)?.contiguous()?;
        let weights = topk_weights.to_dtype(DType::F32)?.contiguous()?;
        let local_ids = Tensor::zeros_on(ids.shape().clone(), DType::U32, &dev)?;
        let local_weights = Tensor::zeros_on(weights.shape().clone(), DType::F32, &dev)?;
        let rows = ids.elem_count() / k;
        let stream = dev.as_cuda_device()?.cuda_stream().cu_stream() as i64;
        unsafe {
            crate::inference::moe_cuda::tp_local::pairs(
                address_of::<u32>(&ids)? as *const u32,
                address_of::<f32>(&weights)? as *const f32,
                address_of::<u32>(&local_ids)? as *mut u32,
                address_of::<f32>(&local_weights)? as *mut f32,
                rows as i32,
                k as i32,
                local.first as i32,
                self.n_experts as i32,
                stream,
            );
        }
        self.experts_cuda(
            xs,
            &local_ids.to_dtype(topk_ids.dtype())?,
            &local_weights.to_dtype(topk_weights.dtype())?,
            num_tokens,
            is_prefill,
            t_route,
            None,
            self.own_experts(None),
        )
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn experts_local(
        &self,
        _xs: &Tensor,
        _topk_ids: &Tensor,
        _topk_weights: &Tensor,
        _num_tokens: usize,
        _is_prefill: bool,
        _t_route: Option<std::time::Instant>,
        _local: LocalExperts,
    ) -> Result<Tensor> {
        crate::tensor::bail!("a split layer runs on cards")
    }

    #[cfg(not(feature = "cuda"))]
    fn experts_split(
        &self,
        _xs: &Tensor,
        _topk_ids: &Tensor,
        _topk_weights: &Tensor,
        _num_tokens: usize,
        _is_prefill: bool,
        _t_route: Option<std::time::Instant>,
    ) -> Result<Tensor> {
        crate::tensor::bail!("a split layer runs on cards")
    }

    /// This layer's own experts, or the stacks `staged` (`[gate, up, down]`) in their place.
    pub(super) fn own_experts<'a>(&'a self, staged: Option<&'a [QTensor; 3]>) -> ExpertSet<'a> {
        ExpertSet {
            gate: staged.map(|s| &s[0]).or(self.gate_exps_sep.as_deref()),
            up: staged.map(|s| &s[1]).or(self.up_exps_sep.as_deref()),
            gate_up: self.gate_up_exps.as_deref(),
            down: staged.map_or(&*self.down_exps, |s| &s[2]),
            gate_bias: self.gate_exps_bias.as_ref(),
            up_bias: self.up_exps_bias.as_ref(),
            down_bias: self.down_exps_bias.as_ref(),
            decode_slots: self.decode_slots.as_ref(),
            n_experts: self.n_experts,
        }
    }

    /// Device path: the gate‖up projections and their combine, then the down projection and
    /// the top-k weighted reduction, each as one launch over the routing plan.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_cuda(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
        is_prefill: bool,
        t_route: Option<std::time::Instant>,
        add_after: Option<&Tensor>,
        staged: Option<&[QTensor; 3]>,
    ) -> Result<Tensor> {
        self.experts_cuda(
            xs,
            topk_ids,
            topk_weights,
            num_tokens,
            is_prefill,
            t_route,
            add_after,
            self.own_experts(staged),
        )
    }

    /// [`Self::forward_cuda`] over the experts `set` holds, on the device `xs` lives on: the
    /// routed ids number the set's experts.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn experts_cuda(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        num_tokens: usize,
        is_prefill: bool,
        t_route: Option<std::time::Instant>,
        add_after: Option<&Tensor>,
        set: ExpertSet<'_>,
    ) -> Result<Tensor> {
        let note_route = || {
            if let Some(t) = t_route {
                let _ = xs.device().synchronize();
                crate::inference::place::layer_perf::stages::add(3, t.elapsed().as_micros() as u64);
            }
        };
        // A wide prefill takes the expert-batched tile kernels, grouping the rows on the
        // device; decode and the layers they do not serve take the sorted routing below.
        let batched = match is_prefill {
            true => self.batched_routing(xs, topk_ids, num_tokens, set)?,
            false => None,
        };
        if let Some(plan) = batched {
            if add_after.is_some() {
                crate::tensor::bail!("MoeWeights: the batched path does not fold a residual");
            }
            note_route();
            return self.batched_experts(xs, &plan, topk_weights, set);
        }
        // One token: the rows are the routed pairs as the router listed them, and every kernel
        // below treats a pair on its own, so nothing needs grouping by expert.
        let routing = match (set.decode_slots, num_tokens) {
            (Some(slots), 1) if topk_ids.dtype() == crate::tensor::DType::U32 => ExpertRouting {
                slots: slots.clone(),
                experts: topk_ids.flatten_all()?,
                topk: self.n_experts_used,
            },
            _ => ExpertRouting::group(topk_ids, self.n_experts_used)?,
        };
        note_route();
        let routing = &routing;
        // Auto-on: per-pair-tile IMMA M=8 for Q4_K MoE prefill - measured
        // +33% over dp4a on gemma4:26b.
        let weight_is_q4k = set
            .gate_up
            .is_some_and(|w| w.dtype() == crate::tensor::quantized::GgmlDType::Q4K);
        let k_is_aligned = set
            .gate_up
            .and_then(|w| w.shape().dims().last().copied())
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
            let g: &QTensor = set.gate.ok_or_else(|| {
                crate::tensor::Error::msg("use_silu MoE missing separate gate experts")
            })?;
            let u: &QTensor = set.up.ok_or_else(|| {
                crate::tensor::Error::msg("use_silu MoE missing separate up experts")
            })?;
            // Decode-only here: the fused
            // single-launch SiLU.mul mmvq kernel (gate GEMM + up GEMM + SiLU.mul in
            // one pass) is exact for M=1 and faster than two separate GEMMs.
            match self.swiglu_clamp {
                Some(clamp) => {
                    routing.gate_up_swiglu_clamped(xs, g, u, (set.gate_bias, set.up_bias), clamp)?
                }
                None => routing.gate_up_silu(xs, g, u)?,
            }
        } else {
            let stack = set.gate_up.ok_or_else(|| {
                crate::tensor::Error::msg("MoE layer without a fused gate||up stack")
            })?;
            routing.gate_up_gelu(xs, stack, use_imma_m8_auto)?
        };
        if let Some(t) = t_gu {
            let _ = xs.device().synchronize();
            crate::inference::place::layer_perf::stages::add(6, t.elapsed().as_micros() as u64);
        }
        let t_dn = st.then(std::time::Instant::now);
        // Down + topk-weighted reduce. For Q4_K MoE prefill the IMMA M=8
        // down kernel (same pattern as gate||up) takes over from dp4a.
        let down_stack: &QTensor = set.down;
        let down_q4k_aligned = down_stack.dtype() == crate::tensor::quantized::GgmlDType::Q4K
            && down_stack
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
            down_stack,
            topk_weights,
            num_tokens,
            down_use_imma_m8,
            set.down_bias,
            add_after,
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

    /// The routing grouped for the expert-batched tile kernels, or `None` when this call is
    /// not one they serve: a GELU mixture, a fused gate‖up stack, a weight format they do not
    /// read, or too few rows per expert to beat the decode kernels.
    #[cfg(feature = "cuda")]
    fn batched_routing(
        &self,
        xs: &Tensor,
        topk_ids: &Tensor,
        num_tokens: usize,
        set: ExpertSet<'_>,
    ) -> Result<Option<crate::inference::moe_cuda::MmqRouting>> {
        use crate::inference::moe_cuda::{moe_mmq_pays, moe_mmq_serves, MmqRouting};
        let (Some(gate), Some(up)) = (set.gate, set.up) else {
            return Ok(None);
        };
        let down_stack: &QTensor = set.down;
        let hidden = xs.dim(crate::tensor::D::Minus1)?;
        // The tile kernels fold gate and up biases into the clamped SwiGLU only.
        let biases_served =
            self.swiglu_clamp.is_some() || (set.gate_bias.is_none() && set.up_bias.is_none());
        let served = !self.gelu
            && biases_served
            && gate.dtype() == up.dtype()
            && moe_mmq_serves(gate.dtype(), hidden)
            && moe_mmq_serves(down_stack.dtype(), self.expert_ffn_dim);
        // A card of a split layer chooses as the whole layer would, so that the cards' sum is
        // the whole layer's.
        let experts = self.local.map_or(set.n_experts, |l| l.total);
        if !served || !moe_mmq_pays(num_tokens * self.n_experts_used, experts) {
            return Ok(None);
        }
        let choices = topk_ids.reshape((num_tokens, self.n_experts_used))?;
        Ok(Some(MmqRouting::from_choices(&choices, set.n_experts)?))
    }

    /// The SiLU expert block through the expert-batched tile kernels.
    #[cfg(feature = "cuda")]
    fn batched_experts(
        &self,
        xs: &Tensor,
        plan: &crate::inference::moe_cuda::MmqRouting,
        topk_weights: &Tensor,
        set: ExpertSet<'_>,
    ) -> Result<Tensor> {
        use crate::inference::place::layer_perf::stages;
        let (Some(gate), Some(up)) = (set.gate, set.up) else {
            crate::tensor::bail!("MoeWeights: the batched path needs gate and up apart");
        };
        let timing = stages::enabled();
        let t0 = timing.then(|| {
            let _ = xs.device().synchronize();
            std::time::Instant::now()
        });
        let glu = match self.swiglu_clamp {
            Some((alpha, limit)) => crate::inference::moe_cuda::Glu::SwigluOai {
                gate_bias: set.gate_bias,
                up_bias: set.up_bias,
                alpha: alpha as f32,
                limit: limit as f32,
            },
            None => crate::inference::moe_cuda::Glu::Silu,
        };
        let hs = plan.gate_up(xs, gate, Some(up), glu)?;
        let t1 = t0.map(|t0| {
            let _ = xs.device().synchronize();
            stages::add(stages::GATE_UP, t0.elapsed().as_micros() as u64);
            std::time::Instant::now()
        });
        let out = plan.down_fold_biased(&hs, set.down, topk_weights, None, set.down_bias)?;
        if let Some(t1) = t1 {
            let _ = xs.device().synchronize();
            stages::add(stages::DOWN, t1.elapsed().as_micros() as u64);
        }
        if let Some(t0) = t0 {
            stages::add(stages::EXPERT_CALL, t0.elapsed().as_micros() as u64);
        }
        Ok(out)
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cuda(
        &self,
        _xs: &Tensor,
        _topk_ids: &Tensor,
        _w: &Tensor,
        _nt: usize,
        _ip: bool,
        _t_route: Option<std::time::Instant>,
        _add_after: Option<&Tensor>,
        _staged: Option<&[QTensor; 3]>,
    ) -> Result<Tensor> {
        crate::tensor::bail!("CUDA path not compiled in")
    }

    /// One expert's fused gate||up `[2*ffn, hidden]` and down `[hidden, ffn]` weights as F32.
    ///
    /// A stack in host memory is read in place: only the routed expert's blocks are decoded,
    /// so nothing is held between calls. A stack on a card has no host blocks to slice; its
    /// layer built a dense cache at load, and that cache is read here.
    fn expert_dense(&self, e: usize) -> Result<(Tensor, Tensor)> {
        if let (Some(gu), Some(dn)) = (self.gate_up_exps_f32.as_ref(), self.down_exps_f32.as_ref())
        {
            let one = |t: &Tensor| -> Result<Tensor> {
                t.narrow(0, e, 1)?
                    .squeeze(0)?
                    .to_dtype(crate::tensor::DType::F32)
            };
            return Ok((one(gu)?, one(dn)?));
        }
        let gu = self.gate_up_exps.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg("MoE CPU path: the layer has no fused gate||up stack")
        })?;
        let dense = |stack: &std::sync::Arc<crate::tensor::quantized::QTensor>| -> Result<Tensor> {
            // CONVERSION: one routed expert is decoded to F32 for the call and dropped.
            crate::tensor::quant_view::expert_view(stack, e)?.dequantize(&Device::Cpu)
        };
        Ok((dense(gu)?, dense(&self.down_exps)?))
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
                let h = match self.swiglu_clamp {
                    Some(clamp) => routing.host_gate_up_swiglu_clamped(
                        xs,
                        g,
                        u,
                        (self.gate_exps_bias.as_ref(), self.up_exps_bias.as_ref()),
                        clamp,
                    )?,
                    None => routing.host_gate_up_silu(xs, g, u)?,
                };
                return routing.host_down_reduce(
                    &h,
                    &self.down_exps,
                    topk_weights,
                    num_tokens,
                    self.down_exps_bias.as_ref(),
                );
            }
        }

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
            let (w_gu, w_dn) = self.expert_dense(e)?; // [2*ffn, hidden], [hidden, ffn]
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

#[cfg(test)]
mod clamped_swiglu_tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QMatMul, QTensor};

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// `x.sigmoid(alpha.x).(1 + u)` with `x = min(g, limit)` and `u` clamped to
    /// `[-limit, limit]`.
    fn clamped_swiglu(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
        let x = g.min(limit);
        let u = u.clamp(-limit, limit);
        x / (1.0 + (-alpha * x).exp()) * (1.0 + u)
    }

    /// `w` `[rows, k]` times `x` `[k]`.
    fn matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
        let k = x.len();
        (0..rows)
            .map(|r| {
                w[r * k..(r + 1) * k]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    /// `x.sigmoid(x)`.
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// A fused gate||up stack on the host is read in place: the host expert loop decodes only
    /// the routed experts, and gives exactly what a dense cache of the same stacks gives.
    #[test]
    fn a_host_stack_serves_the_expert_loop_without_a_dense_cache() {
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 32usize, 32usize, 3usize);
        let dev = Device::Cpu;
        let stack = |seed, rows| {
            let w = Tensor::from_vec(
                values(experts * rows * hidden, seed, 1.0),
                (experts, rows, hidden),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let down = {
            let w = Tensor::from_vec(
                values(experts * hidden * ffn, 33, 1.0),
                (experts, hidden, ffn),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let gate_up = stack(31, 2 * ffn);
        let router =
            Tensor::from_vec(values(experts * hidden, 34, 0.5), (experts, hidden), &dev).unwrap();
        let build = |cache: bool| {
            let dense = |q: &QTensor| Arc::new(q.dequantize(&dev).unwrap());
            MoeWeights {
                gate_inp: QMatMul::Tensor(router.clone()),
                gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
                gate_inp_bias: None,
                gate_exps_bias: None,
                up_exps_bias: None,
                down_exps_bias: None,
                swiglu_clamp: None,
                gate_up_exps: Some(Arc::new(stack(31, 2 * ffn))),
                down_exps: Arc::new(
                    QTensor::quantize_onto(&down.dequantize(&dev).unwrap(), GgmlDType::Q8_0, &dev)
                        .unwrap(),
                ),
                gate_exps_sep: None,
                up_exps_sep: None,
                down_exps_scale: None,
                decode_slots: None,
                n_experts: experts,
                shards: Vec::new(),
                local: None,
                n_experts_used: used,
                expert_ffn_dim: ffn,
                router_prenormed: true,
                use_silu: false,
                gelu: true,
                norm_topk: true,
                sigmoid_routing: None,
                up_relu2: false,
                shared_expert: None,
                gate_up_exps_f32: cache.then(|| dense(&gate_up)),
                down_exps_f32: cache.then(|| dense(&down)),
                tier: None,
            }
        };
        let input =
            Tensor::from_vec(values(tokens * hidden, 35, 1.0), (1, tokens, hidden), &dev).unwrap();
        let run = |m: &MoeWeights| {
            m.forward(&input, &input, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(run(&build(false)), run(&build(true)));
    }

    #[test]
    fn sigmoid_routed_experts_match_the_scalar_reference() {
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 32usize, 32usize, 3usize);
        let (scale, floor) = (2.5f64, 6.103_515_6e-5f32);
        let dev = Device::Cpu;
        let stack = |seed| {
            let w = Tensor::from_vec(
                values(experts * ffn * hidden, seed, 1.0),
                (experts, ffn, hidden),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let (gate, up, down) = (stack(21), stack(22), stack(23));
        let router = values(experts * hidden, 24, 0.5);
        let selection = values(experts, 25, 0.8);
        for normalize in [true, false] {
            let moe = MoeWeights {
                gate_inp: QMatMul::Tensor(
                    Tensor::from_vec(router.clone(), (experts, hidden), &dev).unwrap(),
                ),
                gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
                gate_inp_bias: None,
                gate_exps_bias: None,
                up_exps_bias: None,
                down_exps_bias: None,
                swiglu_clamp: None,
                gate_up_exps: None,
                down_exps: Arc::new(stack(23)),
                gate_exps_sep: Some(Arc::new(stack(21))),
                up_exps_sep: Some(Arc::new(stack(22))),
                down_exps_scale: None,
                decode_slots: None,
                n_experts: experts,
                shards: Vec::new(),
                local: None,
                n_experts_used: used,
                expert_ffn_dim: ffn,
                router_prenormed: true,
                use_silu: true,
                gelu: false,
                norm_topk: true,
                sigmoid_routing: Some(SigmoidRouting {
                    selection_bias: Some(
                        Tensor::from_vec(selection.clone(), (experts,), &dev).unwrap(),
                    ),
                    normalize,
                    scale,
                }),
                up_relu2: false,
                shared_expert: None,
                gate_up_exps_f32: None,
                tier: None,
                down_exps_f32: None,
            };
            let xs = values(tokens * hidden, 26, 1.0);
            let input = Tensor::from_vec(xs.clone(), (1, tokens, hidden), &dev).unwrap();
            let got = moe
                .forward(&input, &input, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let deq = |q: &QTensor| {
                q.dequantize(&dev)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let (gw, uw, dw) = (deq(&gate), deq(&up), deq(&down));
            let mut want = vec![0f32; tokens * hidden];
            for tok in 0..tokens {
                let x = &xs[tok * hidden..(tok + 1) * hidden];
                let probs: Vec<f32> = matvec(&router, x, experts)
                    .iter()
                    .map(|l| 1.0 / (1.0 + (-l).exp()))
                    .collect();
                // Chosen on the biased scores, weighed by the unbiased ones.
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&a, &b| {
                    (probs[b] + selection[b]).total_cmp(&(probs[a] + selection[a]))
                });
                let chosen = &order[..used];
                let sum: f32 = chosen.iter().map(|&e| probs[e]).sum();
                for &e in chosen {
                    let mut w = probs[e];
                    if normalize {
                        w /= sum.max(floor);
                    }
                    w *= scale as f32;
                    let g = matvec(&gw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                    let u = matvec(&uw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                    let h: Vec<f32> = (0..ffn).map(|i| silu(g[i]) * u[i]).collect();
                    let d = matvec(&dw[e * hidden * ffn..(e + 1) * hidden * ffn], &h, hidden);
                    for c in 0..hidden {
                        want[tok * hidden + c] += w * d[c];
                    }
                }
            }
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // The expert kernels quantise their activations to 8 bits.
            assert!(
                worst <= 2e-2 * peak,
                "normalize={normalize}: max diff {worst} of {peak}"
            );
        }
    }

    #[test]
    fn gateless_experts_and_a_shared_expert_match_the_scalar_reference() {
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 32usize, 32usize, 3usize);
        let dev = Device::Cpu;
        let stack = |seed| {
            let w = Tensor::from_vec(
                values(experts * ffn * hidden, seed, 1.0),
                (experts, ffn, hidden),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let dense = |seed, rows, cols| {
            let v = values(rows * cols, seed, 0.5);
            let w = Tensor::from_vec(v.clone(), (rows, cols), &dev).unwrap();
            let q = QTensor::quantize_onto(&w, GgmlDType::F32, &dev).unwrap();
            (
                super::super::projection::QMatMul::from_qtensor(q).unwrap(),
                v,
            )
        };
        let (up, down) = (stack(31), stack(32));
        let router = values(experts * hidden, 33, 0.5);
        let (sh_up, sh_up_w) = dense(34, ffn, hidden);
        let (sh_down, sh_down_w) = dense(35, hidden, ffn);
        let moe = MoeWeights {
            gate_inp: QMatMul::Tensor(
                Tensor::from_vec(router.clone(), (experts, hidden), &dev).unwrap(),
            ),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
            gate_inp_bias: None,
            gate_exps_bias: None,
            up_exps_bias: None,
            down_exps_bias: None,
            swiglu_clamp: None,
            gate_up_exps: None,
            down_exps: Arc::new(stack(32)),
            gate_exps_sep: None,
            up_exps_sep: Some(Arc::new(stack(31))),
            down_exps_scale: None,
            decode_slots: None,
            n_experts: experts,
            shards: Vec::new(),
            local: None,
            n_experts_used: used,
            expert_ffn_dim: ffn,
            router_prenormed: true,
            use_silu: false,
            gelu: false,
            norm_topk: true,
            sigmoid_routing: Some(SigmoidRouting {
                selection_bias: None,
                normalize: false,
                scale: 1.0,
            }),
            up_relu2: true,
            shared_expert: Some(SharedExpert {
                gate: None,
                up: sh_up,
                down: sh_down,
                scalar_gate: None,
            }),
            gate_up_exps_f32: None,
            tier: None,
            down_exps_f32: None,
        };
        let xs = values(tokens * hidden, 36, 1.0);
        let input = Tensor::from_vec(xs.clone(), (1, tokens, hidden), &dev).unwrap();
        let got = moe
            .forward(&input, &input, hidden, 1e-5, tokens > 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let deq = |q: &QTensor| {
            q.dequantize(&dev)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (uw, dw) = (deq(&up), deq(&down));
        let relu2 = |v: f32| v.max(0.0) * v.max(0.0);
        let mut want = vec![0f32; tokens * hidden];
        for tok in 0..tokens {
            let x = &xs[tok * hidden..(tok + 1) * hidden];
            let probs: Vec<f32> = matvec(&router, x, experts)
                .iter()
                .map(|l| 1.0 / (1.0 + (-l).exp()))
                .collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            for &e in &order[..used] {
                let h: Vec<f32> = matvec(&uw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn)
                    .into_iter()
                    .map(relu2)
                    .collect();
                let d = matvec(&dw[e * hidden * ffn..(e + 1) * hidden * ffn], &h, hidden);
                for c in 0..hidden {
                    want[tok * hidden + c] += probs[e] * d[c];
                }
            }
            let h: Vec<f32> = matvec(&sh_up_w, x, ffn).into_iter().map(relu2).collect();
            let d = matvec(&sh_down_w, &h, hidden);
            for c in 0..hidden {
                want[tok * hidden + c] += d[c];
            }
        }
        let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // The expert kernels quantise their activations to 8 bits.
        assert!(worst <= 2e-2 * peak, "max diff {worst} of {peak}");
    }

    #[test]
    fn biased_clamped_experts_match_the_scalar_reference() {
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 32usize, 32usize, 3usize);
        let (alpha, limit) = (1.702f64, 1.5f64);
        let dev = Device::Cpu;
        let stack = |seed| {
            let w = Tensor::from_vec(
                values(experts * ffn * hidden, seed, 1.0),
                (experts, ffn, hidden),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let (gate, up, down) = (stack(11), stack(12), stack(13));
        let router = values(experts * hidden, 14, 0.5);
        let router_bias = values(experts, 15, 0.5);
        let (gb, ub, db) = (
            values(experts * ffn, 16, 0.5),
            values(experts * ffn, 17, 0.5),
            values(experts * hidden, 18, 0.5),
        );
        let t =
            |v: &[f32], shape: (usize, usize)| Tensor::from_vec(v.to_vec(), shape, &dev).unwrap();
        let moe = MoeWeights {
            gate_inp: QMatMul::Tensor(t(&router, (experts, hidden))),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
            gate_inp_bias: Some(Tensor::from_vec(router_bias.clone(), (experts,), &dev).unwrap()),
            gate_exps_bias: Some(t(&gb, (experts, ffn))),
            up_exps_bias: Some(t(&ub, (experts, ffn))),
            down_exps_bias: Some(t(&db, (experts, hidden))),
            swiglu_clamp: Some((alpha, limit)),
            gate_up_exps: None,
            down_exps: Arc::new(down),
            gate_exps_sep: Some(Arc::new(gate)),
            up_exps_sep: Some(Arc::new(up)),
            down_exps_scale: None,
            decode_slots: None,
            n_experts: experts,
            shards: Vec::new(),
            local: None,
            n_experts_used: used,
            expert_ffn_dim: ffn,
            router_prenormed: true,
            use_silu: true,
            gelu: false,
            norm_topk: true,
            sigmoid_routing: None,
            up_relu2: false,
            shared_expert: None,
            gate_up_exps_f32: None,
            tier: None,
            down_exps_f32: None,
        };
        let xs = values(tokens * hidden, 19, 1.0);
        let input = Tensor::from_vec(xs.clone(), (1, tokens, hidden), &dev).unwrap();
        let got = moe
            .forward(&input, &input, hidden, 1e-5, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let deq = |q: &Arc<QTensor>| {
            q.dequantize(&dev)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (gw, uw, dw) = (
            deq(moe.gate_exps_sep.as_ref().unwrap()),
            deq(moe.up_exps_sep.as_ref().unwrap()),
            deq(&moe.down_exps),
        );
        let mut want = vec![0f32; tokens * hidden];
        let mut clamped = 0usize;
        for tok in 0..tokens {
            let x = &xs[tok * hidden..(tok + 1) * hidden];
            let logits: Vec<f32> = matvec(&router, x, experts)
                .iter()
                .zip(&router_bias)
                .map(|(l, b)| l + b)
                .collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let chosen = &order[..used];
            // Softmax over the chosen logits: the top-k of the full softmax, renormalised.
            let m = logits[chosen[0]];
            let z: f32 = chosen.iter().map(|&e| (logits[e] - m).exp()).sum();
            for &e in chosen {
                let w = (logits[e] - m).exp() / z;
                let g = matvec(&gw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                let u = matvec(&uw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                let h: Vec<f32> = (0..ffn)
                    .map(|i| {
                        let (gi, ui) = (g[i] + gb[e * ffn + i], u[i] + ub[e * ffn + i]);
                        clamped += usize::from(gi > limit as f32 || ui.abs() > limit as f32);
                        clamped_swiglu(gi, ui, alpha as f32, limit as f32)
                    })
                    .collect();
                let d = matvec(&dw[e * hidden * ffn..(e + 1) * hidden * ffn], &h, hidden);
                for c in 0..hidden {
                    want[tok * hidden + c] += w * (d[c] + db[e * hidden + c]);
                }
            }
        }
        assert!(clamped > 0, "the inputs must reach the clamp");
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // The expert kernels quantise their activations to 8 bits.
        assert!(worst <= 2e-2 * scale, "max diff {worst} of {scale}");
    }
}

/// What a batch of tokens keeps live on the card per token while its experts run there: for each
/// routed expert a row of the hidden width in f32, the activation quantised to 8 bits, and the
/// gate, the up and the combined outputs in f32; and the hidden rows the layer around them holds.
pub fn prefill_bytes_per_token(used: usize, hidden: usize, ffn: usize) -> usize {
    used * (hidden * 4 + hidden + hidden / 8 + ffn * 4 * 3) + hidden * 4 * 3
}
