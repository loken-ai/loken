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
        #[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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

    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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

    /// Gate and up kept apart, combined by the bounded SwiGLU at `limit`: one launch.
    #[cfg(feature = "gpu")]
    fn gate_up_swiglu_limited(
        &self,
        xs: &Tensor,
        gate: &QTensor,
        up: &QTensor,
        limit: f32,
    ) -> Result<Tensor> {
        // The card kernels read the activations as 8 bit blocks, and the bound turns the small
        // difference to the host into a large one for this codebook format.
        if gate.dtype() == crate::tensor::quantized::GgmlDType::Iq2Xxs
            || up.dtype() == crate::tensor::quantized::GgmlDType::Iq2Xxs
        {
            crate::tensor::bail!("a bounded SwiGLU over IQ2_XXS experts runs on the host");
        }
        device_timed("expert kernels: gate and up on the device", xs, || {
            crate::inference::moe_cuda::moe_gemm_gguf_gate_up_swiglu_limited(
                xs,
                gate,
                up,
                &self.slots,
                &self.experts,
                self.topk,
                limit,
            )
        })
    }

    /// [`Self::gate_up_swiglu_limited`] on the host.
    fn host_gate_up_swiglu_limited(
        &self,
        xs: &Tensor,
        gate: &Arc<QTensor>,
        up: &Arc<QTensor>,
        limit: f32,
    ) -> Result<Tensor> {
        crate::inference::moe_cpu::moe_gemm_gguf_gate_up_swiglu_limited(
            xs,
            gate,
            up,
            &self.slots,
            &self.experts,
            self.topk,
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

/// How a biased-choice router scores an expert from its logit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RouterScore {
    /// `sigmoid(logit)`.
    Sigmoid,
    /// `sqrt(softplus(logit / temperature))`.
    SqrtSoftplus { temperature: f32 },
}

/// Softplus is the identity above this input, as torch's default threshold makes it.
const SOFTPLUS_LINEAR_ABOVE: f32 = 20.0;

/// Added to the sum of the chosen sqrt-softplus scores before they are divided by it.
const SQRT_SOFTPLUS_SUM_EPS: f32 = 1e-20;

/// `softplus(z)`, the identity above [`SOFTPLUS_LINEAR_ABOVE`].
fn softplus(z: f32) -> f32 {
    if z > SOFTPLUS_LINEAR_ABOVE {
        z
    } else {
        z.exp().ln_1p()
    }
}

/// One token's sqrt-softplus choice over its router `row`: the `k` experts of the highest
/// score plus `bias`, highest first, the lower id first among equals.
///
/// Each chosen expert weighs by its unbiased score.
/// With `normalize` and `k > 1`, the weights are divided by their sum plus
/// [`SQRT_SOFTPLUS_SUM_EPS`].
/// The weights are not scaled here.
pub(crate) fn sqrt_softplus_choice(
    row: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    temperature: f32,
    normalize: bool,
) -> (Vec<usize>, Vec<f32>) {
    let scores: Vec<f32> = row
        .iter()
        .map(|&z| softplus(z / temperature).sqrt())
        .collect();
    let biased = |e: usize| scores[e] + bias.map_or(0.0, |b| b[e]);
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        biased(b)
            .partial_cmp(&biased(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let chosen: Vec<usize> = order[..k.min(row.len())].to_vec();
    let mut w: Vec<f32> = chosen.iter().map(|&e| scores[e]).collect();
    if normalize && k > 1 {
        let s: f32 = w.iter().sum::<f32>() + SQRT_SOFTPLUS_SUM_EPS;
        for wi in w.iter_mut() {
            *wi /= s;
        }
    }
    (chosen, w)
}

/// Routing that gives up some of a token's experts for speed.
///
/// Every field's zero is exact routing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Truncation {
    /// Keep at most this many of a token's experts, the heaviest. `0` keeps them all.
    pub keep: usize,
    /// Drop the lightest experts while the weight they carry together stays below this share
    /// of the token's total. `0` drops none.
    pub drop_mass: f32,
    /// A chosen expert whose share of the token's weight is below this and that is not
    /// resident is replaced, or dropped. `0` touches none.
    pub miss_below: f32,
    /// Drop such an expert instead of replacing it with the best resident one.
    pub miss_drops: bool,
}

impl Truncation {
    /// Whether a field gives up exactness.
    pub fn active(&self) -> bool {
        self.keep > 0 || self.drop_mass > 0.0 || self.miss_below > 0.0
    }
}

/// [`sqrt_softplus_choice`] under `truncation`, with the slots it saved.
///
/// The `k` experts of the highest biased score are chosen first.
/// A light chosen expert that is not `resident` gives its slot and its score to the best-ranked
/// resident expert beyond the choice, or is dropped with `miss_drops`.
/// The slots are then ordered heaviest first, `keep` cuts the lightest, and `drop_mass` drops
/// the lightest while their share stays under it.
/// The weights are the kept slots' scores, normalised as the plain choice normalises them.
///
/// A saved slot is one of an expert that is not resident and that the truncation removed or
/// replaced.
pub(crate) fn sqrt_softplus_truncated_choice(
    row: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    temperature: f32,
    normalize: bool,
    truncation: Truncation,
    resident: &dyn Fn(usize) -> bool,
) -> (Vec<usize>, Vec<f32>, u64) {
    let scores: Vec<f32> = row
        .iter()
        .map(|&z| softplus(z / temperature).sqrt())
        .collect();
    let biased = |e: usize| scores[e] + bias.map_or(0.0, |b| b[e]);
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        biased(b)
            .partial_cmp(&biased(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let chosen = k.min(row.len());
    let mut slots: Vec<(usize, f32)> = order[..chosen].iter().map(|&e| (e, scores[e])).collect();
    let mut saved = 0u64;
    if truncation.miss_below > 0.0 {
        let total: f32 = slots.iter().map(|s| s.1).sum::<f32>() + SQRT_SOFTPLUS_SUM_EPS;
        let heaviest = slots.iter().map(|s| s.1).fold(f32::MIN, f32::max);
        // The next expert beyond the choice that may take a slot.
        let mut next = chosen;
        let mut out = Vec::with_capacity(chosen);
        for &(e, s) in &slots {
            let light = s / total < truncation.miss_below && s < heaviest;
            if !light || resident(e) {
                out.push((e, s));
            } else if truncation.miss_drops {
                saved += 1;
            } else {
                while next < order.len() && !resident(order[next]) {
                    next += 1;
                }
                match order.get(next) {
                    Some(&r) => {
                        saved += 1;
                        out.push((r, s));
                        next += 1;
                    }
                    None => out.push((e, s)),
                }
            }
        }
        slots = out;
    }
    slots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if truncation.keep > 0 && truncation.keep < slots.len() {
        saved += slots[truncation.keep..]
            .iter()
            .filter(|s| !resident(s.0))
            .count() as u64;
        slots.truncate(truncation.keep);
    }
    if truncation.drop_mass > 0.0 {
        let total: f32 = slots.iter().map(|s| s.1).sum::<f32>() + SQRT_SOFTPLUS_SUM_EPS;
        let mut dropped = 0f32;
        while slots.len() > 1 && (dropped + slots[slots.len() - 1].1) / total < truncation.drop_mass
        {
            if let Some(gone) = slots.pop() {
                dropped += gone.1;
                saved += !resident(gone.0) as u64;
            }
        }
    }
    let ids: Vec<usize> = slots.iter().map(|s| s.0).collect();
    let mut w: Vec<f32> = slots.iter().map(|s| s.1).collect();
    if normalize && k > 1 {
        let s: f32 = w.iter().sum::<f32>() + SQRT_SOFTPLUS_SUM_EPS;
        for wi in w.iter_mut() {
            *wi /= s;
        }
    }
    (ids, w, saved)
}

/// Biased-choice expert routing: every expert scores its logit by [`RouterScore`], the experts
/// are chosen on the scores plus a selection bias, and each chosen expert weighs by its
/// unbiased score, optionally renormalised over the chosen ones and scaled.
#[derive(Debug, Clone)]
pub struct SigmoidRouting {
    /// Added to the scores for the choice only: `[n_experts]` F32.
    pub selection_bias: Option<Tensor>,
    /// The chosen weights are divided by their sum.
    ///
    /// A sigmoid sum is floored at the smallest normal F16.
    /// A sqrt-softplus sum is raised by [`SQRT_SOFTPLUS_SUM_EPS`], and 1 chosen expert keeps
    /// its score.
    pub normalize: bool,
    /// Multiplies the chosen weights.
    pub scale: f64,
    /// The score of an expert's logit.
    pub score: RouterScore,
}

impl SigmoidRouting {
    /// Floor of the renormalising sum, the smallest normal F16.
    const SUM_FLOOR: f32 = 6.103_515_6e-5;

    /// `(weights, ids)` `[tokens, k]` from the router `logits` `[tokens, n_experts]` F32.
    pub fn select(&self, logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        if let RouterScore::SqrtSoftplus { temperature } = self.score {
            return self.select_sqrt_softplus(logits, k, temperature);
        }
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

    /// [`Self::select`] under the sqrt-softplus score, each row chosen on the host by
    /// [`sqrt_softplus_choice`] and its weights scaled in F32.
    ///
    /// The result lands on the device of `logits`: weights F32, ids U32.
    fn select_sqrt_softplus(
        &self,
        logits: &Tensor,
        k: usize,
        temperature: f32,
    ) -> Result<(Tensor, Tensor)> {
        let host = Device::Cpu;
        let (tokens, n_experts) = logits.dims2()?;
        if k > n_experts {
            crate::tensor::bail!("{k} experts chosen of {n_experts}");
        }
        let rows: Vec<f32> = logits
            .to_device(&host)?
            .to_dtype(crate::tensor::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let bias: Option<Vec<f32>> = self
            .selection_bias
            .as_ref()
            .map(|b| {
                b.to_device(&host)?
                    .to_dtype(crate::tensor::DType::F32)?
                    .flatten_all()?
                    .to_vec1()
            })
            .transpose()?;
        let scale = self.scale as f32;
        let mut weights = Vec::with_capacity(tokens * k);
        let mut ids = Vec::with_capacity(tokens * k);
        for row in rows.chunks(n_experts) {
            let (chosen, w) =
                sqrt_softplus_choice(row, bias.as_deref(), k, temperature, self.normalize);
            ids.extend(chosen.iter().map(|&e| e as u32));
            weights.extend(w.iter().map(|&wi| wi * scale));
        }
        let device = logits.device();
        Ok((
            Tensor::from_vec(weights, (tokens, k), &host)?.to_device(&device)?,
            Tensor::from_vec(ids, (tokens, k), &host)?.to_device(&device)?,
        ))
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
    /// The gate and up combine through `offload::experts::swiglu` bounded at this limit: the
    /// gate clamped from above, the up value on both sides.
    pub swiglu_limit: Option<f32>,
}

/// `offload::experts::swiglu` of `g` and `u` value by value at `limit`, F32 on the device of `g`.
fn bounded_swiglu(g: &Tensor, u: &Tensor, limit: f32) -> Result<Tensor> {
    let host = Device::Cpu;
    let shape = g.shape().clone();
    let gv: Vec<f32> = g.to_device(&host)?.flatten_all()?.to_vec1()?;
    let uv: Vec<f32> = u.to_device(&host)?.flatten_all()?.to_vec1()?;
    let h: Vec<f32> = gv
        .iter()
        .zip(&uv)
        .map(|(&g, &u)| crate::inference::offload::experts::swiglu(g, u, limit))
        .collect();
    Tensor::from_vec(h, shape, &host)?.to_device(&g.device())
}

impl SharedExpert {
    /// `routed` `[tokens, hidden]` F32 plus the expert over `xs`. A decoded token on a card
    /// runs the elementwise stages around the projections as two launches.
    ///
    /// A bounded SwiGLU takes the plain forward on every device.
    pub fn forward_added(
        &self,
        xs: &Tensor,
        routed: &Tensor,
        projected: Option<Tensor>,
    ) -> Result<Tensor> {
        if self.swiglu_limit.is_some() {
            return routed + self.forward(xs)?;
        }
        #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
    pub fn project(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        let (Some(gate), Some(_), None) = (&self.gate, &self.scalar_gate, self.swiglu_limit) else {
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
    #[cfg(not(feature = "gpu"))]
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
                match self.swiglu_limit {
                    Some(limit) => bounded_swiglu(&g, &up, limit)?,
                    None => (crate::tensor::ops::silu(&g)? * up)?,
                }
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
    #[cfg(feature = "gpu")]
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

    #[cfg(not(feature = "gpu"))]
    pub fn from_host(_: &MoeWeights, _: usize, _: usize, _: &Device) -> Result<Self> {
        crate::tensor::bail!("an expert shard lives on a card")
    }

    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
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

/// The routed pairs a card holding the experts `first..first + count` computes: every id as the
/// card numbers it, and the weight of each pair, zero for a pair of another card's expert.
///
/// `ids` and `weights` hold `k` pairs per row. A pair of another card's expert reads an expert
/// of this card its row does not route to, at weight zero, so that the kernels which group a
/// row's pairs by expert take each expert of a row once.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn shard_pairs(
    ids: &[u32],
    weights: &[f32],
    k: usize,
    first: usize,
    count: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut local = vec![0u32; ids.len()];
    let mut weight = vec![0f32; ids.len()];
    for (row, (ids, ws)) in ids.chunks(k).zip(weights.chunks(k)).enumerate() {
        let mine = |e: u32| (e as usize).checked_sub(first).filter(|&l| l < count);
        let taken: Vec<usize> = ids.iter().filter_map(|&e| mine(e)).collect();
        let mut spare = (0..count).filter(|l| !taken.contains(l));
        for (j, (&e, &w)) in ids.iter().zip(ws).enumerate() {
            let at = row * k + j;
            match mine(e) {
                Some(l) => (local[at], weight[at]) = (l as u32, w),
                None => local[at] = spare.next().unwrap_or(0) as u32,
            }
        }
    }
    (local, weight)
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
    /// Gate and up combine through the bounded SwiGLU at this limit, reading the separate
    /// stacks: `silu(min(gate, limit)) . clamp(up, -limit, limit)`.
    ///
    /// The steps are those of `offload::experts::swiglu`.
    /// A layer holds this or `swiglu_clamp`, never both.
    pub swiglu_limit: Option<f32>,
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
    /// The routed experts as projections read in place, block-scaled fp8 or fp4 among them.
    ///
    /// When present, the routed experts run through it on the host and the expert stacks are
    /// not read.
    pub block_scaled: Option<std::sync::Arc<super::block_scaled_experts::BlockScaledExperts>>,
    pub down_exps_f32: Option<std::sync::Arc<Tensor>>,
}

impl MoeWeights {
    /// Whether the routed expert stacks sit in host memory, where the expert loop then runs
    /// whatever device the rest of the layer is on.
    pub fn experts_on_host(&self) -> bool {
        !self.down_exps.device().runs_as_card()
    }

    /// Whether `rows` rows routed together read nearly every expert of this layer from the
    /// host's memory: the experts sit there, the card holds them in no cache that has them all,
    /// and the routed pairs outnumber the experts.
    pub fn reads_every_expert_on_host(&self, rows: usize) -> bool {
        self.decodes_through_host() && rows * self.n_experts_used >= self.n_experts
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
    /// stands for: together on a card (the kernels work a routed pair at a time, and a token's
    /// pairs are summed in rank order), one at a time elsewhere. The whole batch
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
        // On a card every row's pairs run in the decode kernels, the experts read once for the
        // rows that share them, whether the layer's experts live there or pass through a tier.
        let on_card = moe_in.device().runs_as_card() && !self.up_relu2;
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
            // An F32 dense gate on a card scores a decoded token and each row of a verify with
            // the row-wise GEMV, all the rows in one launch: a row's logits do not depend on
            // the rows beside it.
            let row_gemv = match &self.gate_inp {
                crate::tensor::quantized::QMatMul::Tensor(w)
                    if self.gate_inp_bias.is_none()
                        && xs.device().is_cuda()
                        && (num_tokens == 1 || crate::tensor::quantized::DecodeOrder::active()) =>
                {
                    crate::inference::moe_cuda::gate_gemv_f32(&xs, w)?
                }
                _ => None,
            };
            if let Some(logits) = row_gemv {
                logits
            } else {
                match self.gate_inp_bias.as_ref() {
                    None if self.sigmoid_routing.is_some() => exact_logits()?,
                    // The rows of a verify each as the decoded token it stands for: the quantised
                    // gate's mat-vecs take the rows one by one under the decode order, and another
                    // dense gate scores each row alone, as a library matrix product picks its
                    // arithmetic by shape.
                    None if num_tokens > 1 && crate::tensor::quantized::DecodeOrder::active() => {
                        match &self.gate_inp {
                            crate::tensor::quantized::QMatMul::QTensor(_) => exact_logits()?,
                            gate => {
                                let rows = (0..num_tokens)
                                    .map(|r| gate.forward(&xs.narrow(0, r, 1)?))
                                    .collect::<Result<Vec<_>>>()?;
                                Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?
                                    .to_dtype(crate::tensor::DType::F32)?
                            }
                        }
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
            }
        } else {
            {
                #[cfg(feature = "gpu")]
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
                #[cfg(not(feature = "gpu"))]
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
        #[cfg(feature = "gpu")]
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
        #[cfg(not(feature = "gpu"))]
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
            && (self.up_relu2
                || on_cpu
                || self.shared_expert.is_some()
                || self.block_scaled.is_some()
                || num_tokens != 1)
        {
            crate::tensor::bail!("MoeWeights: this layer cannot fold a residual into its fold");
        }
        // The shared expert's projections, when a recorded decode queued them ahead of the wait.
        let expert_rows = if let Some(experts) = self.block_scaled.as_ref() {
            // Block-scaled experts run on the host, whatever device the rows are on.
            let ids: Vec<u32> = topk_ids
                .to_device(&Device::Cpu)?
                .to_dtype(crate::tensor::DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            let weights: Vec<f32> = topk_weights_scaled
                .to_device(&Device::Cpu)?
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            experts.forward(xs, &ids, &weights, self.n_experts_used)?
        } else if let Some(local) = self.local.filter(|_| xs.device().runs_as_card()) {
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
            #[cfg(feature = "gpu")]
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
        #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
            let (local, weight) = shard_pairs(&ids, &weights, k, shard.first, shard.count);
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
    #[cfg(feature = "gpu")]
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

    #[cfg(not(feature = "gpu"))]
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

    #[cfg(not(feature = "gpu"))]
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
    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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
        // The IMMA kernels are integer matrix products: a card without integer matrix
        // instructions has no code for them, so the card decides here alongside the shape.
        let has_imma = xs
            .device()
            .as_cuda_device()
            .is_ok_and(|d| d.has_integer_matrix_cores());
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
            match (self.swiglu_clamp, self.swiglu_limit) {
                (Some(clamp), _) => {
                    routing.gate_up_swiglu_clamped(xs, g, u, (set.gate_bias, set.up_bias), clamp)?
                }
                (None, Some(limit)) => routing.gate_up_swiglu_limited(xs, g, u, limit)?,
                (None, None) => routing.gate_up_silu(xs, g, u)?,
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
        let down_use_imma_m8 =
            is_prefill && down_q4k_aligned && size_m_est >= 64 && !down_no && has_imma;
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
    #[cfg(feature = "gpu")]
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
        // The tile kernels have no bounded SwiGLU.
        if self.swiglu_limit.is_some() {
            return Ok(None);
        }
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
    #[cfg(feature = "gpu")]
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

    #[cfg(not(feature = "gpu"))]
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
                let h = match (self.swiglu_clamp, self.swiglu_limit) {
                    (Some(clamp), _) => routing.host_gate_up_swiglu_clamped(
                        xs,
                        g,
                        u,
                        (self.gate_exps_bias.as_ref(), self.up_exps_bias.as_ref()),
                        clamp,
                    )?,
                    (None, Some(limit)) => routing.host_gate_up_swiglu_limited(xs, g, u, limit)?,
                    (None, None) => routing.host_gate_up_silu(xs, g, u)?,
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
                swiglu_limit: None,
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
                block_scaled: None,
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

    /// The cards of an expert-parallel layer each compute the pairs of their own experts on the
    /// host here, and the sum of their products is the products of one holder of every expert,
    /// for a decoded token, a few rows and a prompt, with ranges of unequal width.
    #[cfg(feature = "gpu")]
    #[test]
    fn shares_of_the_experts_sum_to_the_whole_layer_on_the_host() {
        use super::super::expert_tier::experts_range;
        let (experts, used, hidden, ffn) = (8usize, 2usize, 64usize, 64usize);
        let dev = Device::Cpu;
        let stack = |seed, rows, cols| {
            let w = Tensor::from_vec(
                values(experts * rows * cols, seed, 1.0),
                (experts, rows, cols),
                &dev,
            )
            .unwrap();
            Arc::new(QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap())
        };
        let (gate, up, down) = (
            stack(41, ffn, hidden),
            stack(42, ffn, hidden),
            stack(43, hidden, ffn),
        );
        let router =
            Tensor::from_vec(values(experts * hidden, 44, 0.5), (experts, hidden), &dev).unwrap();
        let build = |first: usize, count: usize| {
            let part = |s: &Arc<QTensor>| Arc::new(experts_range(s, first, count, &dev).unwrap());
            MoeWeights {
                gate_inp: QMatMul::Tensor(router.clone()),
                gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
                gate_inp_bias: None,
                gate_exps_bias: None,
                up_exps_bias: None,
                down_exps_bias: None,
                swiglu_clamp: None,
                swiglu_limit: None,
                gate_up_exps: None,
                down_exps: part(&down),
                gate_exps_sep: Some(part(&gate)),
                up_exps_sep: Some(part(&up)),
                down_exps_scale: None,
                decode_slots: None,
                n_experts: count,
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
                down_exps_f32: None,
                tier: None,
                block_scaled: None,
            }
        };
        let whole = build(0, experts);
        let cut = experts * 5 / 8;
        let shares = [
            (0, cut, build(0, cut)),
            (cut, experts - cut, build(cut, experts - cut)),
        ];
        for tokens in [1usize, 3, 40] {
            let xs = Tensor::from_vec(
                values(tokens * hidden, 45 + tokens as u64, 1.0),
                (tokens, hidden),
                &dev,
            )
            .unwrap();
            let logits = whole.gate_inp.forward(&xs).unwrap();
            let (weights, ids) = whole.top_k(&logits).unwrap();
            let want: Vec<f32> = whole
                .forward_cpu(&xs, &ids, &weights, tokens, hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let id_v: Vec<u32> = ids
                .to_dtype(DType::U32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
            let mut got = vec![0f32; want.len()];
            for (first, count, share) in &shares {
                let (local, weight) = shard_pairs(&id_v, &w_v, used, *first, *count);
                let shape = ids.shape().clone();
                let local = Tensor::from_vec(local, shape.clone(), &dev).unwrap();
                let weight = Tensor::from_vec(weight, shape, &dev).unwrap();
                let part: Vec<f32> = share
                    .forward_cpu(&xs, &local, &weight, tokens, hidden)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for (g, p) in got.iter_mut().zip(part) {
                    *g += p;
                }
            }
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 1e-5 * scale,
                "{tokens} rows: off by {worst} of {scale}"
            );
        }
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
                swiglu_limit: None,
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
                    score: RouterScore::Sigmoid,
                }),
                up_relu2: false,
                shared_expert: None,
                gate_up_exps_f32: None,
                tier: None,
                block_scaled: None,
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
            swiglu_limit: None,
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
                score: RouterScore::Sigmoid,
            }),
            up_relu2: true,
            shared_expert: Some(SharedExpert {
                gate: None,
                up: sh_up,
                down: sh_down,
                scalar_gate: None,
                swiglu_limit: None,
            }),
            gate_up_exps_f32: None,
            tier: None,
            block_scaled: None,
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
            swiglu_limit: None,
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
            block_scaled: None,
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

    /// A layer under the bounded SwiGLU gives the experts of the scalar reference whose SwiGLU is
    /// `offload::experts::swiglu`, the gate clamped from above and the up value on both sides.
    #[test]
    fn bounded_experts_match_the_offload_swiglu() {
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 32usize, 32usize, 3usize);
        let limit = 1.5f32;
        let dev = Device::Cpu;
        let stack = |seed, rows, cols| {
            let w = Tensor::from_vec(
                values(experts * rows * cols, seed, 1.0),
                (experts, rows, cols),
                &dev,
            )
            .unwrap();
            QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()
        };
        let (gate, up, down) = (
            stack(21, ffn, hidden),
            stack(22, ffn, hidden),
            stack(23, hidden, ffn),
        );
        let router = values(experts * hidden, 24, 0.5);
        let t =
            |v: &[f32], shape: (usize, usize)| Tensor::from_vec(v.to_vec(), shape, &dev).unwrap();
        let moe = MoeWeights {
            gate_inp: QMatMul::Tensor(t(&router, (experts, hidden))),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
            gate_inp_bias: None,
            gate_exps_bias: None,
            up_exps_bias: None,
            down_exps_bias: None,
            swiglu_clamp: None,
            swiglu_limit: Some(limit),
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
            block_scaled: None,
            down_exps_f32: None,
        };
        let xs = values(tokens * hidden, 25, 1.0);
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
            let logits = matvec(&router, x, experts);
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let chosen = &order[..used];
            let m = logits[chosen[0]];
            let z: f32 = chosen.iter().map(|&e| (logits[e] - m).exp()).sum();
            for &e in chosen {
                let w = (logits[e] - m).exp() / z;
                let g = matvec(&gw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                let u = matvec(&uw[e * ffn * hidden..(e + 1) * ffn * hidden], x, ffn);
                let h: Vec<f32> = (0..ffn)
                    .map(|i| {
                        clamped += usize::from(g[i] > limit || u[i].abs() > limit);
                        crate::inference::offload::experts::swiglu(g[i], u[i], limit)
                    })
                    .collect();
                let d = matvec(&dw[e * hidden * ffn..(e + 1) * hidden * ffn], &h, hidden);
                for c in 0..hidden {
                    want[tok * hidden + c] += w * d[c];
                }
            }
        }
        assert!(clamped > 0, "the inputs must reach the bound");
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // The expert kernels quantise their activations to 8 bits.
        assert!(worst <= 2e-2 * scale, "max diff {worst} of {scale}");
    }

    /// A bounded shared expert gives what the offload's `Expert` gives over the same dense
    /// weights at the same bound, and differs from the unbounded one.
    #[test]
    fn a_bounded_shared_expert_matches_the_offload_expert() {
        let (hidden, ffn, tokens) = (32usize, 16usize, 3usize);
        let limit = 0.5f32;
        let dev = Device::Cpu;
        let w = |seed, rows, cols| {
            Tensor::from_vec(values(rows * cols, seed, 1.0), (rows, cols), &dev).unwrap()
        };
        let (gate, up, down) = (w(51, ffn, hidden), w(52, ffn, hidden), w(53, hidden, ffn));
        let shared = |limit: Option<f32>| SharedExpert {
            gate: Some(
                super::super::projection::QMatMul::from_qtensor(
                    QTensor::quantize(&gate, GgmlDType::F32).unwrap(),
                )
                .unwrap(),
            ),
            up: super::super::projection::QMatMul::from_qtensor(
                QTensor::quantize(&up, GgmlDType::F32).unwrap(),
            )
            .unwrap(),
            down: super::super::projection::QMatMul::from_qtensor(
                QTensor::quantize(&down, GgmlDType::F32).unwrap(),
            )
            .unwrap(),
            scalar_gate: None,
            swiglu_limit: limit,
        };
        let xs =
            Tensor::from_vec(values(tokens * hidden, 54, 1.0), (tokens, hidden), &dev).unwrap();
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = flat(shared(Some(limit)).forward(&xs).unwrap());
        let free = flat(shared(None).forward(&xs).unwrap());
        let old = crate::inference::offload::experts::Expert::dense(
            gate.clone(),
            down.clone(),
            up.clone(),
        );
        let want = flat(old.forward(&xs, limit).unwrap());
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-5 * scale.max(1.0),
            "max diff {worst} of {scale}"
        );
        let moved = got.iter().zip(&free).filter(|(a, b)| a != b).count();
        assert!(moved > 0, "the bound moves nothing");
    }

    /// A layer holding block-scaled experts runs them on its own routing: its output is the bits
    /// of the expert set over the router's choice and weights.
    #[test]
    fn a_layer_of_block_scaled_experts_runs_them_on_its_routing() {
        use super::super::block_scaled_experts::{BlockScaledExpert, BlockScaledExperts};
        use crate::tensor::blockscaled::fp8::Fp8Weight;
        use crate::tensor::blockscaled::projection::Projection;
        let (experts, used, hidden, ffn, tokens) = (4usize, 2usize, 64usize, 32usize, 3usize);
        let dev = Device::Cpu;
        let fp8 = |out: usize, inp: usize, seed: usize| {
            let block = 32;
            let bytes: Vec<u8> = (0..out * inp)
                .map(|i| match ((i * 29 + 7 + seed * 53) % 256) as u8 {
                    b if b & 0x7F == 0x7F => b - 1,
                    b => b,
                })
                .collect();
            let tiles = out.div_ceil(block) * inp.div_ceil(block);
            let scales: Vec<u8> = (0..tiles).map(|i| (118 + ((i + seed) % 4)) as u8).collect();
            let mapped = |b: &[u8]| {
                let mut m = memmap2::MmapMut::map_anon(b.len().max(1)).unwrap();
                m[..b.len()].copy_from_slice(b);
                crate::tensor::mapped::MappedBytes::new(
                    Arc::new(m.make_read_only().unwrap()),
                    0,
                    b.len(),
                )
                .unwrap()
            };
            Projection::Fp8(Arc::new(
                Fp8Weight::new(mapped(&bytes), mapped(&scales), out, inp, block).unwrap(),
            ))
        };
        let set = Arc::new(BlockScaledExperts {
            experts: (0..experts)
                .map(|e| BlockScaledExpert {
                    gate: fp8(ffn, hidden, 3 * e),
                    up: fp8(ffn, hidden, 3 * e + 1),
                    down: fp8(hidden, ffn, 3 * e + 2),
                })
                .collect(),
            swiglu_limit: 0.0,
        });
        // The stacks are not read: an empty down stack stands where the layer keeps one.
        let unread = Tensor::zeros((experts, hidden, ffn), DType::F32).unwrap();
        let router = values(experts * hidden, 41, 0.5);
        let moe = MoeWeights {
            gate_inp: QMatMul::Tensor(Tensor::from_vec(router, (experts, hidden), &dev).unwrap()),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
            gate_inp_bias: None,
            gate_exps_bias: None,
            up_exps_bias: None,
            down_exps_bias: None,
            swiglu_clamp: None,
            swiglu_limit: None,
            gate_up_exps: None,
            down_exps: Arc::new(QTensor::quantize_onto(&unread, GgmlDType::Q8_0, &dev).unwrap()),
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
            use_silu: true,
            gelu: false,
            norm_topk: true,
            sigmoid_routing: None,
            up_relu2: false,
            shared_expert: None,
            gate_up_exps_f32: None,
            tier: None,
            block_scaled: Some(set.clone()),
            down_exps_f32: None,
        };
        let xs = values(tokens * hidden, 42, 1.0);
        let input = Tensor::from_vec(xs.clone(), (1, tokens, hidden), &dev).unwrap();
        let got = moe
            .forward(&input, &input, hidden, 1e-5, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let routed = moe.route(&input, &input, hidden, 1e-5).unwrap();
        let ids: Vec<u32> = routed
            .topk_ids
            .to_dtype(DType::U32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let weights: Vec<f32> = routed
            .topk_weights_scaled
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let rows = Tensor::from_vec(xs, (tokens, hidden), &dev).unwrap();
        let want = set
            .forward(&rows, &ids, &weights, used)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert!(want.iter().any(|v| *v != 0.0), "the experts give nothing");
        assert_eq!(bits(&got), bits(&want));
    }

    /// The host's bounded SwiGLU under a bound past every product gives the bits of the unbounded
    /// one, and under a small bound keeps every value within `limit` times the largest SiLU a
    /// clamped gate reaches.
    #[test]
    fn the_host_bounded_swiglu_moves_only_what_the_bound_reaches() {
        let (experts, hidden, ffn, tokens, used) = (3usize, 64usize, 32usize, 4usize, 2usize);
        let limit = 0.75f32;
        let dev = Device::Cpu;
        let stack = |seed| {
            let w = Tensor::from_vec(
                values(experts * ffn * hidden, seed, 1.0),
                (experts, ffn, hidden),
                &dev,
            )
            .unwrap();
            Arc::new(QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap())
        };
        let (gate, up) = (stack(31), stack(32));
        let xs =
            Tensor::from_vec(values(tokens * hidden, 33, 1.0), (tokens, hidden), &dev).unwrap();
        let ids: Vec<u32> = (0..tokens * used).map(|p| (p % experts) as u32).collect();
        let routing =
            ExpertRouting::group(&Tensor::from_vec(ids, (tokens, used), &dev).unwrap(), used)
                .unwrap();
        let run = |limit: f32| {
            routing
                .host_gate_up_swiglu_limited(&xs, &gate, &up, limit)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let (free, past, bounded) = (run(0.0), run(f32::MAX), run(limit));
        assert_eq!(
            bits(&past),
            bits(&free),
            "a bound no product reaches moves nothing"
        );
        assert_ne!(bits(&bounded), bits(&free), "the bound moves no value");
        // SiLU is at least its minimum, about -0.28, and at most its value at the bound.
        let silu = |g: f32| g / (1.0 + (-g).exp());
        let reach = silu(limit).max(-silu(-1.278_464_5)) * limit;
        assert!(
            bounded.iter().all(|v| v.abs() <= reach * (1.0 + 1e-6)),
            "a value past {reach}"
        );
    }
}

/// What a batch of tokens keeps live on the card per token while its experts run there: for each
/// routed expert a row of the hidden width in f32, the activation quantised to 8 bits, and the
/// gate, the up and the combined outputs in f32; and the hidden rows the layer around them holds.
pub fn prefill_bytes_per_token(used: usize, hidden: usize, ffn: usize) -> usize {
    used * (hidden * 4 + hidden + hidden / 8 + ffn * 4 * 3) + hidden * 4 * 3
}

#[cfg(test)]
mod sigmoid_router_tests {
    use super::*;

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    /// The router of a mixture that scores by sigmoid, chooses on the score plus a bias, weighs
    /// by the score alone, renormalises over the chosen and scales, written out as its
    /// reference library writes it.
    #[test]
    fn the_sigmoid_router_chooses_on_the_biased_score_and_weighs_by_the_plain_one() {
        let (tokens, experts, used) = (6usize, 16usize, 4usize);
        let dev = Device::Cpu;
        let scores = values(tokens * experts, 41, 3.0);
        let bias = values(experts, 42, 0.5);
        let routing = SigmoidRouting {
            selection_bias: Some(Tensor::from_vec(bias.clone(), (experts,), &dev).unwrap()),
            normalize: true,
            scale: 2.5,
            score: RouterScore::Sigmoid,
        };
        let input = Tensor::from_vec(scores.clone(), (tokens, experts), &dev).unwrap();
        let (weights, ids) = routing.select(&input, used).unwrap();
        let weights = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ids = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let mut biased_apart = 0;
        for t in 0..tokens {
            let probs: Vec<f32> = scores[t * experts..(t + 1) * experts]
                .iter()
                .map(|l| 1.0 / (1.0 + (-l).exp()))
                .collect();
            let by = |score: &dyn Fn(usize) -> f32| {
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
                let mut kept = order[..used].to_vec();
                kept.sort_unstable();
                kept
            };
            let want = by(&|e| probs[e] + bias[e]);
            if want != by(&|e| probs[e]) {
                biased_apart += 1;
            }
            let chosen: Vec<(usize, f32)> = ids[t * used..(t + 1) * used]
                .iter()
                .map(|&e| e as usize)
                .zip(weights[t * used..(t + 1) * used].iter().copied())
                .collect();
            let mut got: Vec<usize> = chosen.iter().map(|c| c.0).collect();
            got.sort_unstable();
            assert_eq!(got, want, "token {t}: the experts chosen");
            let sum: f32 = want.iter().map(|&e| probs[e]).sum();
            for (e, w) in chosen {
                let expected = probs[e] / (sum + 1e-20) * 2.5;
                assert!(
                    (w - expected).abs() <= 1e-6 * expected.max(1.0),
                    "token {t} expert {e}: {w} vs {expected}"
                );
            }
        }
        assert!(biased_apart > 0, "the bias changed no choice");
    }
}

#[cfg(test)]
mod sqrt_softplus_router_tests {
    use super::*;
    use crate::inference::model::deepseek_v41::moe::Moe;
    use crate::inference::offload::experts::Expert;
    use crate::inference::offload::store::ExpertSet;

    const DIM: usize = 16;
    const INTER: usize = 8;
    const ROUTED: usize = 16;
    const TOKENS: usize = 7;

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    fn drawn(seed: u64, dims: &[usize], amp: f32) -> Tensor {
        Tensor::from_vec(values(dims.iter().product(), seed, amp), dims, &Device::Cpu).unwrap()
    }

    fn expert(seed: u64) -> Expert {
        Expert::dense(
            drawn(seed, &[INTER, DIM], 0.05),
            drawn(seed + 1, &[DIM, INTER], 0.05),
            drawn(seed + 2, &[INTER, DIM], 0.05),
        )
    }

    /// The block whose gate the generic router must equal.
    ///
    /// `gate_amp` and `bias_amp` set the spread of the logits and of the selection bias.
    struct Case {
        used: usize,
        temperature: f32,
        route_scale: f32,
        norm_topk: bool,
        gate_amp: f32,
        bias_amp: f32,
    }

    fn reference(case: &Case) -> Moe {
        Moe {
            gate_weight: drawn(1, &[ROUTED, DIM], case.gate_amp),
            gate_bias: drawn(2, &[ROUTED], case.bias_amp),
            experts: ExpertSet::Resident(
                (0..ROUTED)
                    .map(|i| std::sync::Arc::new(expert(100 + 3 * i as u64)))
                    .collect(),
            ),
            shared: expert(10),
            has_shared: true,
            n_routed: ROUTED,
            n_activated: case.used,
            dim: DIM,
            gate_temp: case.temperature,
            route_scale: case.route_scale,
            swiglu_limit: 10.0,
            norm_topk: case.norm_topk,
            last_routing: std::sync::Mutex::new(Vec::new()),
            predicted: std::sync::Mutex::new(Vec::new()),
            last_routed: std::sync::Mutex::new(Vec::new()),
            decode_routes: Default::default(),
            saved: Default::default(),
            truncation: Default::default(),
            resident_override: Default::default(),
            observer: std::sync::RwLock::new(None),
            offload: std::sync::RwLock::new(None),
            scripted: Default::default(),
        }
    }

    /// Run the block's own gate and the generic router on the same logits.
    ///
    /// Returns, per token, the experts and the bits of their weights from each, and the logits.
    #[allow(clippy::type_complexity)]
    fn both(case: &Case) -> (Vec<Vec<(usize, u32)>>, Vec<Vec<(usize, u32)>>, Vec<f32>) {
        let moe = reference(case);
        let x = drawn(7, &[1, TOKENS, DIM], 1.0);
        moe.forward(&x).unwrap();
        let old: Vec<Vec<(usize, u32)>> = moe
            .last_routing
            .lock()
            .unwrap()
            .iter()
            .map(|row| row.iter().map(|&(e, w)| (e, w.to_bits())).collect())
            .collect();

        let rows = x.reshape((TOKENS, DIM)).unwrap();
        let logits = crate::inference::offload::linear(&rows, &moe.gate_weight).unwrap();
        let routing = SigmoidRouting {
            selection_bias: Some(moe.gate_bias.clone()),
            normalize: case.norm_topk,
            scale: case.route_scale as f64,
            score: RouterScore::SqrtSoftplus {
                temperature: case.temperature,
            },
        };
        let (weights, ids) = routing.select(&logits, case.used).unwrap();
        let weights = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ids = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let new: Vec<Vec<(usize, u32)>> = ids
            .chunks(case.used)
            .zip(weights.chunks(case.used))
            .map(|(ids, ws)| {
                ids.iter()
                    .zip(ws)
                    .map(|(&e, w)| (e as usize, w.to_bits()))
                    .collect()
            })
            .collect();
        let logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (old, new, logits)
    }

    /// The generic router chooses the experts the block's own gate chooses, in its order, with
    /// weights equal bit for bit: renormalised and scaled.
    #[test]
    fn the_sqrt_softplus_router_equals_the_blocks_own_gate_bit_for_bit() {
        let case = Case {
            used: 4,
            temperature: 1.0,
            route_scale: 1.5,
            norm_topk: true,
            gate_amp: 0.2,
            bias_amp: 0.1,
        };
        let (old, new, _) = both(&case);
        assert_eq!(old.len(), TOKENS);
        assert_eq!(new, old);
    }

    /// The same without renormalising, at another temperature and scale.
    #[test]
    fn the_sqrt_softplus_router_without_renormalising_equals_the_blocks_gate() {
        let case = Case {
            used: 3,
            temperature: 0.7,
            route_scale: 2.5,
            norm_topk: false,
            gate_amp: 0.3,
            bias_amp: 0.2,
        };
        let (old, new, _) = both(&case);
        assert_eq!(new, old);
    }

    /// 1 chosen expert keeps its score whether or not the router renormalises.
    #[test]
    fn a_single_chosen_expert_keeps_its_score() {
        let case = Case {
            used: 1,
            temperature: 1.0,
            route_scale: 1.0,
            norm_topk: true,
            gate_amp: 0.2,
            bias_amp: 0.1,
        };
        let (old, new, logits) = both(&case);
        assert_eq!(new, old);
        for (t, row) in new.iter().enumerate() {
            let (e, w) = row[0];
            let z = logits[t * ROUTED + e];
            assert_eq!(w, softplus(z).sqrt().to_bits(), "token {t}");
        }
    }

    /// Logits past the softplus threshold take its linear branch, and still equal the gate.
    #[test]
    fn logits_past_the_softplus_threshold_equal_the_blocks_gate() {
        let case = Case {
            used: 4,
            temperature: 0.05,
            route_scale: 1.5,
            norm_topk: true,
            gate_amp: 1.0,
            bias_amp: 0.1,
        };
        let (old, new, logits) = both(&case);
        assert!(
            logits
                .iter()
                .any(|&z| z / case.temperature > SOFTPLUS_LINEAR_ABOVE),
            "no logit reaches the linear branch"
        );
        assert_eq!(new, old);
    }

    /// The bias moves the choice and never the weights: a chosen expert weighs by its
    /// unbiased score.
    #[test]
    fn the_bias_steers_the_choice_and_not_the_weights() {
        let row = [0.0f32, 1.0, 2.0, 3.0];
        let bias = [10.0f32, 0.0, 0.0, 0.0];
        let (chosen, w) = sqrt_softplus_choice(&row, Some(&bias), 2, 1.0, false);
        assert_eq!(chosen, vec![0, 3]);
        assert_eq!(w[0].to_bits(), softplus(0.0).sqrt().to_bits());
        assert_eq!(w[1].to_bits(), softplus(3.0).sqrt().to_bits());
        let (unbiased, _) = sqrt_softplus_choice(&row, None, 2, 1.0, false);
        assert_eq!(unbiased, vec![3, 2]);
    }

    /// Equal biased scores keep the lower expert id first.
    #[test]
    fn equal_scores_keep_the_lower_id_first() {
        let row = [1.0f32, 2.0, 2.0, 1.0];
        let (chosen, _) = sqrt_softplus_choice(&row, None, 3, 1.0, true);
        assert_eq!(chosen, vec![1, 2, 0]);
    }

    /// The routing of the 4-expert case under `truncation`, with `resident` the experts in
    /// memory, from the block's own gate and from the generic truncated choice.
    ///
    /// Returns, per token, the experts and the bits of their scaled weights from each, then
    /// the slots each saved.
    #[allow(clippy::type_complexity)]
    fn truncated_both(
        truncation: Truncation,
        resident: &std::collections::HashSet<usize>,
    ) -> (Vec<Vec<(usize, u32)>>, Vec<Vec<(usize, u32)>>, u64, u64) {
        let case = Case {
            used: 4,
            temperature: 1.0,
            route_scale: 1.5,
            norm_topk: true,
            gate_amp: 0.2,
            bias_amp: 0.1,
        };
        use crate::inference::model::deepseek_v41::moe::Truncation as V41Truncation;
        let moe = reference(&case);
        *moe.truncation.write().unwrap() = V41Truncation {
            keep: truncation.keep,
            drop_mass: truncation.drop_mass,
            miss_below: truncation.miss_below,
            miss_drops: truncation.miss_drops,
        };
        *moe.resident_override.write().unwrap() = Some(resident.clone());
        let x = drawn(7, &[1, TOKENS, DIM], 1.0);
        moe.forward(&x).unwrap();
        let old: Vec<Vec<(usize, u32)>> = moe
            .last_routing
            .lock()
            .unwrap()
            .iter()
            .map(|row| row.iter().map(|&(e, w)| (e, w.to_bits())).collect())
            .collect();
        let old_saved = moe.saved.load(std::sync::atomic::Ordering::Relaxed);

        let rows = x.reshape((TOKENS, DIM)).unwrap();
        let logits = crate::inference::offload::linear(&rows, &moe.gate_weight)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let bias = moe
            .gate_bias
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let in_memory = |e: usize| resident.contains(&e);
        let mut new_saved = 0;
        // As the routed feed-forward routes: the truncated choice only when a field is on.
        let new: Vec<Vec<(usize, u32)>> = logits
            .iter()
            .map(|row| {
                let (ids, w, saved) = match truncation.active() {
                    true => sqrt_softplus_truncated_choice(
                        row,
                        Some(bias.as_slice()),
                        case.used,
                        case.temperature,
                        case.norm_topk,
                        truncation,
                        &in_memory,
                    ),
                    false => {
                        let (ids, w) = sqrt_softplus_choice(
                            row,
                            Some(bias.as_slice()),
                            case.used,
                            case.temperature,
                            case.norm_topk,
                        );
                        (ids, w, 0)
                    }
                };
                new_saved += saved;
                ids.iter()
                    .zip(&w)
                    .map(|(&e, &wi)| (e, (wi * case.route_scale).to_bits()))
                    .collect()
            })
            .collect();
        (old, new, old_saved, new_saved)
    }

    /// Off, the truncated choice routes as the plain one does. Keeping 1 expert leaves each
    /// token the heaviest, alone, with the whole route scale. Each case gives the experts, the
    /// weight bits and the saved slots of the block's own gate.
    #[test]
    fn truncation_keeps_the_heaviest_and_weights_it_whole() {
        let none = std::collections::HashSet::new();
        let (_, exact, _, _) = truncated_both(Truncation::default(), &none);
        let (plain, _, _) = both(&Case {
            used: 4,
            temperature: 1.0,
            route_scale: 1.5,
            norm_topk: true,
            gate_amp: 0.2,
            bias_amp: 0.1,
        });
        assert_eq!(exact, plain, "zeros changed the routing");

        for keep in [1, 2, 3] {
            let t = Truncation {
                keep,
                ..Default::default()
            };
            let (old, new, old_saved, new_saved) = truncated_both(t, &none);
            assert_eq!(new, old, "keep {keep}");
            assert_eq!(new_saved, old_saved, "keep {keep}");
        }
        let one = Truncation {
            keep: 1,
            ..Default::default()
        };
        let (_, kept, _, saved) = truncated_both(one, &none);
        assert_eq!(saved, (TOKENS * 3) as u64, "every cut slot was a read");
        for (kept, full) in kept.iter().zip(&exact) {
            assert_eq!(kept.len(), 1);
            let heaviest = full
                .iter()
                .max_by(|a, b| f32::from_bits(a.1).total_cmp(&f32::from_bits(b.1)))
                .unwrap();
            assert_eq!(kept[0].0, heaviest.0);
            assert!((f32::from_bits(kept[0].1) - 1.5).abs() < 1e-6);
        }
    }

    /// A light expert that is not resident gives its slot to a resident one, or is dropped.
    /// The heaviest of a token is never touched. Each case gives the experts, the weight bits
    /// and the saved slots of the block's own gate.
    #[test]
    fn a_missed_light_expert_is_replaced_by_a_resident_one() {
        let none = std::collections::HashSet::new();
        let (_, exact, _, _) = truncated_both(Truncation::default(), &none);
        let resident: std::collections::HashSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
        let miss = Truncation {
            miss_below: 1.1,
            ..Default::default()
        };
        let (old, got, old_saved, new_saved) = truncated_both(miss, &resident);
        assert_eq!(got, old);
        assert_eq!(new_saved, old_saved);
        for (got, full) in got.iter().zip(&exact) {
            assert_eq!(got.len(), full.len());
            let heaviest = full
                .iter()
                .max_by(|a, b| f32::from_bits(a.1).total_cmp(&f32::from_bits(b.1)))
                .unwrap()
                .0;
            assert!(got.iter().any(|g| g.0 == heaviest));
            for g in got.iter().filter(|g| g.0 != heaviest) {
                assert!(resident.contains(&g.0), "expert {} is not resident", g.0);
            }
        }
        let drops = Truncation {
            miss_drops: true,
            ..miss
        };
        let (old, got, old_saved, new_saved) = truncated_both(drops, &resident);
        assert_eq!(got, old);
        assert_eq!(new_saved, old_saved);
        for got in got {
            assert!(!got.is_empty() && got.len() <= 4);
        }
    }

    /// Shares a quality calibration tries, a dropped mass, and both at once give the experts,
    /// the weight bits and the saved slots of the block's own gate.
    #[test]
    fn every_truncation_routes_as_the_blocks_own_gate() {
        let resident: std::collections::HashSet<usize> = (0..ROUTED).step_by(2).collect();
        let cases = [
            Truncation {
                miss_below: 0.5 / 4.0,
                ..Default::default()
            },
            Truncation {
                miss_below: 1.0 / 4.0,
                ..Default::default()
            },
            Truncation {
                drop_mass: 0.3,
                ..Default::default()
            },
            Truncation {
                keep: 3,
                drop_mass: 0.2,
                miss_below: 0.75 / 4.0,
                miss_drops: false,
            },
        ];
        for t in cases {
            let (old, new, old_saved, new_saved) = truncated_both(t, &resident);
            assert_eq!(new, old, "{t:?}");
            assert_eq!(new_saved, old_saved, "{t:?}");
        }
    }
}
