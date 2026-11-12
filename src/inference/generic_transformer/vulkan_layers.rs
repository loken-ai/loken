//! The layers a placement gives a Vulkan device.
//!
//! Each Vulkan segment of the plan holds its layers in [`VulkanStack`]s built from the file's
//! blocks, one stack for each run of layers that attend.
//! A layer of the segment that does not attend, as a recurrent one, runs on the host.
//! The host keeps its own build of the stacks' layers, which no forward runs; the layer loop
//! hands a stack's rows to it and takes the rows it returns. The stack's caches
//! are positional, so a forward from `index_pos` overwrites what a cut cache dropped.
//!
//! Without the `vulkan` feature the set is always empty.

use super::*;
#[cfg(feature = "vulkan")]
use crate::tensor::quantized::GgmlDType;

/// A tensor of the file, as stored: its format, rows, columns and bytes.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(super) type RawRead<'a> = dyn Fn(&str) -> Result<Option<(crate::tensor::quantized::GgmlDType, usize, usize, Vec<u8>)>>
    + 'a;

#[cfg(feature = "vulkan")]
struct Segment {
    from: usize,
    to: usize,
    /// The Vulkan device that runs the segment.
    device: usize,
    stack: crate::inference::kernel::vulkan::VulkanStack,
}

/// The model's Vulkan segments, by the layers they run.
#[derive(Default)]
pub(crate) struct VulkanLayers {
    #[cfg(feature = "vulkan")]
    segments: Vec<Segment>,
    /// The logits of the last row of the step that just left the last layer, when the segment
    /// that ran it holds the output head.
    #[cfg(feature = "vulkan")]
    logits: Option<Vec<f32>>,
}

impl VulkanLayers {
    /// Whether no layer runs through Vulkan.
    pub(crate) fn is_empty(&self) -> bool {
        #[cfg(feature = "vulkan")]
        return self.segments.is_empty();
        #[cfg(not(feature = "vulkan"))]
        true
    }

    /// The Vulkan device that runs layer `layer`, when one does.
    pub(crate) fn device_of(&self, layer: usize) -> Option<usize> {
        #[cfg(feature = "vulkan")]
        return self
            .segments
            .iter()
            .find(|s| (s.from..s.to).contains(&layer))
            .map(|s| s.device);
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = layer;
            None
        }
    }

    /// The end of the Vulkan segment that runs layer `layer`, when one does.
    pub(crate) fn end_of(&self, layer: usize) -> Option<usize> {
        #[cfg(feature = "vulkan")]
        return self
            .segments
            .iter()
            .find(|s| (s.from..s.to).contains(&layer))
            .map(|s| s.to);
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = layer;
            None
        }
    }

    /// Runs `hidden` (`[1, rows, hidden]`, the first row at `index_pos`) through layers
    /// `[from, to)`, which one segment holds, and returns the rows on the host.
    pub(crate) fn run(
        &mut self,
        hidden: Tensor,
        index_pos: usize,
        from: usize,
        to: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "vulkan")]
        {
            let seg = self
                .segments
                .iter_mut()
                .find(|s| s.from <= from && to <= s.to)
                .ok_or_else(|| {
                    crate::tensor::Error::msg(format!(
                        "no Vulkan segment holds layers {from}..{to}"
                    ))
                })?;
            let shape = hidden.dims().to_vec();
            let rows = hidden
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (local_from, local_to) = (from - seg.from, to - seg.from);
            let run = if local_to == seg.stack.len() && seg.stack.has_head() {
                seg.stack
                    .forward_to_logits(&rows, index_pos, local_from)
                    .map(|(rows, logits)| (rows, Some(logits)))
            } else {
                seg.stack
                    .forward(&rows, index_pos, local_from, local_to)
                    .map(|rows| (rows, None))
            };
            let (out, logits) = run.map_err(|e| crate::tensor::Error::msg(format!("{e:#}")))?;
            self.logits = logits;
            Tensor::from_vec(out, shape, &Device::Cpu)
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (index_pos, from, to);
            Ok(hidden)
        }
    }

    /// The logits the output head on the device gave for the step that just ran, `[1, vocab]`,
    /// taken once.
    pub(crate) fn take_logits(&mut self) -> Option<Tensor> {
        #[cfg(feature = "vulkan")]
        return self.logits.take().and_then(|l| {
            let n = l.len();
            Tensor::from_vec(l, (1, n), &Device::Cpu).ok()
        });
        #[cfg(not(feature = "vulkan"))]
        None
    }

    /// The kernel times of one step of `tokens` rows at `index_pos` on each segment, in
    /// microseconds.
    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn profile(&self, tokens: usize, index_pos: usize) -> Vec<Vec<(&'static str, f64)>> {
        self.segments
            .iter()
            .map(|s| s.stack.profile(tokens, index_pos).unwrap())
            .collect()
    }

    /// The stack of the first Vulkan segment, for a test to drive.
    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn stack_mut(&mut self) -> &mut crate::inference::kernel::vulkan::VulkanStack {
        &mut self.segments[0].stack
    }

    /// Forgets the logits of an earlier step.
    pub(crate) fn clear_logits(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            self.logits = None;
        }
    }

    /// Builds a stack for every run of layers that attend in a Vulkan segment of `plan`.
    /// `layers` are the host builds of the model's layers, the first being layer `first` of the
    /// stack; `read` gives the file's blocks; `cos` and `sin` are the host RoPE tables.
    /// `head_norm` is the final RMS norm's weight when the output head may run on the device of
    /// the segment that ends the model.
    #[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
    pub(super) fn build(
        plan: &HeteroPlan,
        config: &GenericTransformerConfig,
        layers: &[GenericTransformerLayer],
        first: usize,
        read: &RawRead<'_>,
        cos: &Tensor,
        sin: &Tensor,
        head_norm: Option<&Tensor>,
    ) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        {
            use crate::inference::place::layer_executor::DeviceKind;
            let mut segments = Vec::new();
            for seg in &plan.segments {
                let DeviceKind::Vulkan(device) = seg.kind else {
                    continue;
                };
                let attends = |at: usize| layers[at - first].attends();
                let mut start = seg.layer_start;
                while start < seg.layer_end {
                    if !attends(start) {
                        start += 1;
                        continue;
                    }
                    let end = (start..seg.layer_end)
                        .find(|&at| !attends(at))
                        .unwrap_or(seg.layer_end);
                    let (from, to) = (start - first, end - first);
                    let head = head_norm.filter(|_| end == plan.total_layers);
                    let stack = build_stack(
                        device,
                        config,
                        &layers[from..to],
                        start,
                        read,
                        cos,
                        sin,
                        head,
                    )
                    .map_err(|e| {
                        crate::tensor::Error::msg(format!(
                            "vulkan({device}): layers {start}..{end}: {e:#}"
                        ))
                    })?;
                    tracing::info!("vulkan({device}): layers {start}..{end} resident");
                    segments.push(Segment {
                        from,
                        to,
                        device,
                        stack,
                    });
                    start = end;
                }
            }
            Ok(Self {
                segments,
                logits: None,
            })
        }
        #[cfg(not(feature = "vulkan"))]
        Ok(Self::default())
    }
}

/// Whether the experts of a layer need more than a softmax router over experts held as one
/// stack each of gate, up and down, gated by silu or by the clamped SwiGLU, with biases or
/// without.
#[cfg(feature = "vulkan")]
fn unsupported_experts(m: &super::moe_weights::MoeWeights) -> bool {
    m.sigmoid_routing.is_some()
        || m.shared_expert.is_some()
        || m.up_relu2
        || m.gelu
        || m.swiglu_limit.is_some()
        || m.block_scaled.is_some()
        || m.down_exps_scale.is_some()
        || m.local.is_some()
        || !m.router_prenormed
        || !m.shards.is_empty()
        || m.n_experts > 256
}

/// Why the stack cannot run this model's layers, or `None` when it can.
#[cfg(feature = "vulkan")]
fn refusal(
    config: &GenericTransformerConfig,
    layers: &[GenericTransformerLayer],
) -> Option<String> {
    let c = config;
    let f = &c.flags;
    let checks: [(bool, &str); 11] = [
        (f.fused_qkv, "a fused QKV projection"),
        (
            f.fused_ffn_gate_up || f.is_phi2_simple_ffn,
            "a fused or gate-less FFN",
        ),
        (f.layer_norm, "LayerNorm"),
        (f.parallel_attn, "parallel attention"),
        (f.value_rms_norm, "normalised values"),
        (f.has_ffn_bias, "an FFN bias"),
        (c.shared_kv_layers > 0, "shared KV layers"),
        (
            c.hyper.is_some() || c.ngram.is_some(),
            "wide residual streams",
        ),
        (c.attn_output_gate, "a gated attention output"),
        (c.indexer.is_some(), "sparse attention"),
        (c.max_alibi_bias.is_some(), "an ALiBi position bias"),
    ];
    if let Some((_, why)) = checks.iter().find(|(on, _)| *on) {
        return Some(why.to_string());
    }
    for l in layers {
        let shape = [
            (!l.attends(), "a recurrent layer"),
            (
                l.moe.as_ref().is_some_and(unsupported_experts),
                "routed experts of a kind the Vulkan layers do not run",
            ),
            (l.ffn_norm.is_none(), "a layer without an FFN norm"),
            (
                [Some(&l.attn_norm), l.ffn_norm.as_ref()]
                    .into_iter()
                    .flatten()
                    .any(|n| matches!(n, super::config::WeightedNorm::Layer(_))),
                "a norm other than RMS",
            ),
            (
                l.rope_freq_factors.is_some() != layers[0].rope_freq_factors.is_some(),
                "RoPE frequency factors on some layers only",
            ),
            (l.attn_v_norm_ones.is_some(), "normalised values"),
            (
                l.attn_layer_scale.is_some() || l.ffn_layer_scale.is_some(),
                "a scale per channel on a residual branch",
            ),
            (l.flags.erf_gelu, "the error-function GELU"),
        ];
        if let Some((_, why)) = shape.iter().find(|(on, _)| *on) {
            return Some(why.to_string());
        }
    }
    None
}

#[cfg(feature = "vulkan")]
fn build_stack(
    device: usize,
    config: &GenericTransformerConfig,
    layers: &[GenericTransformerLayer],
    first: usize,
    read: &RawRead<'_>,
    cos: &Tensor,
    sin: &Tensor,
    head_norm: Option<&Tensor>,
) -> anyhow::Result<crate::inference::kernel::vulkan::VulkanStack> {
    use crate::inference::kernel::vulkan::stack::{
        runs_format, DenseSpec, FfnTensors, LayerTensors, RawTensor,
    };
    if let Some(why) = refusal(config, layers) {
        anyhow::bail!("the Vulkan layers do not run {why}");
    }
    let host = |t: &Tensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let raw = |name: String| -> anyhow::Result<RawTensor<'static>> {
        let (dtype, rows, cols, bytes) =
            read(&name)?.ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        if !runs_format(dtype) {
            anyhow::bail!(
                "{name} is {}, which the Vulkan mat-vec does not read",
                dtype.gguf_name()
            );
        }
        Ok(RawTensor {
            dtype,
            rows,
            cols,
            bytes: std::borrow::Cow::Owned(bytes),
        })
    };
    let rope_dim = config
        .rope_dim
        .filter(|&d| d > 0 && d < config.head_dim)
        .unwrap_or(config.head_dim);
    let mut spec = DenseSpec {
        hidden: config.embedding_length,
        n_head: config.n_head,
        n_kv: config.n_kv_head,
        head_dim: config.head_dim,
        ffn: config.flags.intermediate_size.max(config.ffn_dim),
        eps: config.rms_norm_eps as f32,
        rope_dim,
        rope_adjacent: config.flags.use_rope_i,
        attn_scale: config
            .attention_scale
            .unwrap_or_else(|| 1.0 / (config.head_dim as f64).sqrt()) as f32,
        ctx: config.effective_max_context(),
        gelu: config.flags.use_gelu,
        experts: layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map_or(0, |m| m.n_experts),
        used: layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map_or(0, |m| m.n_experts_used),
        expert_ffn: 0,
        norm_topk: layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .is_some_and(|m| m.norm_topk),
        residual_scale: config.residual_scale.map(|s| s as f32),
        attn_softcap: config.attn_logit_softcapping.map(|c| c as f32),
        swiglu_clamp: layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .and_then(|m| m.swiglu_clamp)
            .map(|(alpha, limit)| (alpha as f32, limit as f32)),
    };
    let mut tensors = Vec::with_capacity(layers.len());
    // The width of a dense FFN is the rows of its gate, the same in every layer.
    let mut dense_ffn = None;
    for (i, l) in layers.iter().enumerate() {
        let p = format!("blk.{}", first + i);
        let norm_of =
            |n: &Option<crate::tensor::layer::RmsNorm>| -> anyhow::Result<Option<Vec<f32>>> {
                n.as_ref().map(|n| host(n.weight())).transpose()
            };
        let bias = |b: &Option<Tensor>| -> anyhow::Result<Option<Vec<f32>>> {
            b.as_ref().map(host).transpose()
        };
        let ffn = match &l.moe {
            Some(m) => {
                let (dtype, rows, cols, bytes) = read(&format!("{p}.ffn_gate_inp.weight"))?
                    .ok_or_else(|| anyhow::anyhow!("missing the router of {p}"))?;
                if rows != m.n_experts || cols != spec.hidden {
                    anyhow::bail!("{p}: a router of {rows}x{cols}");
                }
                let router = match dtype {
                    GgmlDType::F32 => bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                    GgmlDType::F16 => bytes
                        .chunks_exact(2)
                        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                        .collect(),
                    other => anyhow::bail!("{p}: a router in {}", other.gguf_name()),
                };
                FfnTensors::Experts {
                    router,
                    gate: raw(format!("{p}.ffn_gate_exps.weight"))?,
                    up: raw(format!("{p}.ffn_up_exps.weight"))?,
                    down: raw(format!("{p}.ffn_down_exps.weight"))?,
                    router_bias: bias(&m.gate_inp_bias)?,
                    gate_bias: bias(&m.gate_exps_bias)?,
                    up_bias: bias(&m.up_exps_bias)?,
                    down_bias: bias(&m.down_exps_bias)?,
                }
            }
            None => {
                let gate = raw(format!("{p}.ffn_gate.weight"))?;
                let width = *dense_ffn.get_or_insert(gate.rows);
                if gate.rows != width {
                    anyhow::bail!(
                        "{p}: an FFN of {} rows where another layer has {width}",
                        gate.rows
                    );
                }
                FfnTensors::Dense {
                    gate,
                    up: raw(format!("{p}.ffn_up.weight"))?,
                    down: raw(format!("{p}.ffn_down.weight"))?,
                }
            }
        };
        // A block without a norm before it reads the residual as it is.
        let pre_norm = |n: &super::config::WeightedNorm| -> anyhow::Result<Option<Vec<f32>>> {
            match n {
                super::config::WeightedNorm::Identity(_) => Ok(None),
                n => host(n.weight()).map(Some),
            }
        };
        tensors.push(LayerTensors {
            attn_norm: pre_norm(&l.attn_norm)?,
            q: raw(format!("{p}.attn_q.weight"))?,
            k: raw(format!("{p}.attn_k.weight"))?,
            v: raw(format!("{p}.attn_v.weight"))?,
            q_bias: bias(&l.attn_q_bias)?,
            k_bias: bias(&l.attn_k_bias)?,
            v_bias: bias(&l.attn_v_bias)?,
            q_norm: norm_of(&l.attn_q_norm)?,
            k_norm: norm_of(&l.attn_k_norm)?,
            // The output projection is stored under either stem.
            o: match read(&format!("{p}.attn_output.weight"))? {
                Some(_) => raw(format!("{p}.attn_output.weight"))?,
                None => raw(format!("{p}.attn_out.weight"))?,
            },
            o_bias: bias(&l.attn_output_bias)?,
            sinks: bias(&l.attn_sinks)?,
            ffn_norm: pre_norm(
                l.ffn_norm
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("{p}: no FFN norm"))?,
            )?,
            ffn,
            post_attn_norm: norm_of(&l.post_attn_norm)?,
            post_ffn_norm: norm_of(&l.post_ffn_norm)?,
            window: l.sliding_window.unwrap_or(0),
            prefill_window: l.prefill_mask_window.unwrap_or(0),
            rope: 0,
        });
    }
    if let Some(width) = dense_ffn {
        spec.ffn = width;
    }
    // The key heads are the rows of the key projection, the same in every layer.
    // A file may give a count for each layer, which the config does not read.
    let kv_rows = tensors.first().map_or(0, |t| t.k.rows);
    if tensors.iter().any(|t| t.k.rows != kv_rows) || !kv_rows.is_multiple_of(spec.head_dim) {
        anyhow::bail!("key projections of different heads across the layers");
    }
    spec.n_kv = kv_rows / spec.head_dim;
    // The width of an expert is the columns of its down projection.
    spec.expert_ffn = tensors
        .iter()
        .find_map(|t| match &t.ffn {
            FfnTensors::Experts { down, .. } => Some(down.cols),
            FfnTensors::Dense { .. } => None,
        })
        .unwrap_or(0);
    let _ = sin;
    // The tables the layers rotate by, one for each distinct set of phases.
    let mut tables: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut table_of = Vec::with_capacity(layers.len());
    for l in layers {
        let (c, s) = l.rope_table().rows(0, cos.dim(0)?, &Device::Cpu)?;
        let mut pair = (host(&c.contiguous()?)?, host(&s.contiguous()?)?);
        // A layer without RoPE rotates by the identity: a cosine of 1 and a sine of 0.
        if l.no_rope {
            pair.0.fill(1.0);
            pair.1.fill(0.0);
        }
        let at = match tables.iter().position(|t| *t == pair) {
            Some(i) => i,
            None => {
                tables.push(pair);
                tables.len() - 1
            }
        };
        table_of.push(at);
    }
    for (t, at) in tensors.iter_mut().zip(table_of) {
        t.rope = at;
    }
    let half = cos.dim(1)?;
    let mut stack =
        crate::inference::kernel::vulkan::VulkanStack::new(device, spec, tensors, &tables, half)?;
    // The head goes with the last layer when the file's projection is in a format the
    // mat-vec reads; otherwise the host keeps it.
    if let Some(norm) = head_norm {
        let proj = ["output.weight", "token_embd.weight"]
            .into_iter()
            .find_map(|n| read(n).transpose())
            .transpose()?;
        match proj {
            Some((dtype, rows, cols, bytes)) if runs_format(dtype) => {
                let proj = RawTensor {
                    dtype,
                    rows,
                    cols,
                    bytes: std::borrow::Cow::Owned(bytes),
                };
                stack.attach_head(&host(norm)?, &proj)?;
            }
            _ => tracing::info!("vulkan({device}): the output head stays on the host"),
        }
    }
    Ok(stack)
}
