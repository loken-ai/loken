//! The layers a placement gives an OpenCL device.
//!
//! Each OpenCL segment of the plan holds its layers in a [`OpenClStack`] built from the file's
//! blocks. The host keeps its own build of those layers, which no forward runs; the layer loop
//! hands an OpenCL segment's rows to the stack and takes the rows it returns. The stack's caches
//! are positional, so a forward from `index_pos` overwrites what a cut cache dropped.
//!
//! Without the `opencl` feature the set is always empty.

use super::*;

/// A tensor of the file, as stored: its format, rows, columns and bytes.
#[cfg_attr(not(feature = "opencl"), allow(dead_code))]
pub(super) type RawRead<'a> = dyn Fn(&str) -> Result<Option<(crate::tensor::quantized::GgmlDType, usize, usize, Vec<u8>)>>
    + 'a;

#[cfg(feature = "opencl")]
struct Segment {
    from: usize,
    to: usize,
    stack: crate::inference::kernel::opencl_stack::OpenClStack,
}

/// The model's OpenCL segments, by the layers they run.
#[derive(Default)]
pub(crate) struct OpenClLayers {
    #[cfg(feature = "opencl")]
    segments: Vec<Segment>,
    /// The logits of the last row of the step that just left the last layer, when the segment
    /// that ran it holds the output head.
    #[cfg(feature = "opencl")]
    logits: Option<Vec<f32>>,
}

impl OpenClLayers {
    /// Whether no layer runs through OpenCL.
    pub(crate) fn is_empty(&self) -> bool {
        #[cfg(feature = "opencl")]
        return self.segments.is_empty();
        #[cfg(not(feature = "opencl"))]
        true
    }

    /// The end of the OpenCL segment that runs layer `layer`, when one does.
    pub(crate) fn end_of(&self, layer: usize) -> Option<usize> {
        #[cfg(feature = "opencl")]
        return self
            .segments
            .iter()
            .find(|s| (s.from..s.to).contains(&layer))
            .map(|s| s.to);
        #[cfg(not(feature = "opencl"))]
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
        #[cfg(feature = "opencl")]
        {
            let seg = self
                .segments
                .iter_mut()
                .find(|s| s.from <= from && to <= s.to)
                .ok_or_else(|| {
                    crate::tensor::Error::msg(format!(
                        "no OpenCL segment holds layers {from}..{to}"
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
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (index_pos, from, to);
            Ok(hidden)
        }
    }

    /// The logits the output head on the device gave for the step that just ran, `[1, vocab]`,
    /// taken once.
    pub(crate) fn take_logits(&mut self) -> Option<Tensor> {
        #[cfg(feature = "opencl")]
        return self.logits.take().and_then(|l| {
            let n = l.len();
            Tensor::from_vec(l, (1, n), &Device::Cpu).ok()
        });
        #[cfg(not(feature = "opencl"))]
        None
    }

    /// Chooses whether decode steps run each layer in one launch, on the segments that can.
    #[cfg(all(test, feature = "opencl"))]
    pub(crate) fn set_single_launch(&mut self, on: bool) {
        for s in &mut self.segments {
            s.stack.set_single_launch(on);
        }
    }

    /// The workgroups of the single-launch layer of the first segment, and a way to set them.
    #[cfg(all(test, feature = "opencl"))]
    pub(crate) fn single_launch_groups(&self) -> usize {
        self.segments
            .first()
            .map_or(0, |s| s.stack.single_launch_groups())
    }

    #[cfg(all(test, feature = "opencl"))]
    pub(crate) fn set_single_launch_groups(&mut self, groups: usize) {
        for s in &mut self.segments {
            s.stack.set_single_launch_groups(groups);
        }
    }

    /// Whether every segment runs decode steps with one launch to a layer.
    #[cfg(all(test, feature = "opencl"))]
    pub(crate) fn single_launch(&self) -> bool {
        self.segments.iter().all(|s| s.stack.single_launch())
    }

    /// Forgets the logits of an earlier step.
    pub(crate) fn clear_logits(&mut self) {
        #[cfg(feature = "opencl")]
        {
            self.logits = None;
        }
    }

    /// Builds a stack for every OpenCL segment of `plan`. `layers` are the host builds of the
    /// model's layers, the first being layer `first` of the stack; `read` gives the file's
    /// blocks; `cos` and `sin` are the host RoPE tables. `head_norm` is the final RMS norm's
    /// weight when the output head may run on the device of the segment that ends the model.
    #[cfg_attr(not(feature = "opencl"), allow(unused_variables))]
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
        #[cfg(feature = "opencl")]
        {
            use crate::inference::place::layer_executor::DeviceKind;
            let mut segments = Vec::new();
            for seg in &plan.segments {
                let DeviceKind::OpenCL(device) = seg.kind else {
                    continue;
                };
                let from = seg.layer_start - first;
                let to = seg.layer_end - first;
                let ends = seg.layer_end == plan.total_layers;
                let head = head_norm.filter(|_| ends);
                let stack = build_stack(
                    device,
                    config,
                    &layers[from..to],
                    seg.layer_start,
                    read,
                    cos,
                    sin,
                    head,
                )
                .map_err(|e| {
                    crate::tensor::Error::msg(format!(
                        "opencl({device}): layers {}..{}: {e:#}",
                        seg.layer_start, seg.layer_end
                    ))
                })?;
                tracing::info!(
                    "opencl({device}): layers {}..{} resident",
                    seg.layer_start,
                    seg.layer_end
                );
                segments.push(Segment { from, to, stack });
            }
            Ok(Self {
                segments,
                logits: None,
            })
        }
        #[cfg(not(feature = "opencl"))]
        Ok(Self::default())
    }
}

/// Why the stack cannot run this model's layers, or `None` when it can.
#[cfg(feature = "opencl")]
fn refusal(
    config: &GenericTransformerConfig,
    layers: &[GenericTransformerLayer],
) -> Option<String> {
    let c = config;
    let f = &c.flags;
    let checks: [(bool, &str); 18] = [
        (f.is_moe, "a mixture of experts"),
        (f.fused_qkv, "a fused QKV projection"),
        (
            f.fused_ffn_gate_up || f.is_phi2_simple_ffn,
            "a fused or gate-less FFN",
        ),
        (f.use_gelu, "a GELU FFN"),
        (f.layer_norm, "LayerNorm"),
        (f.has_post_attn_norm || f.has_post_ffn_norm, "post norms"),
        (f.parallel_attn, "parallel attention"),
        (f.has_attn_sinks, "attention sinks"),
        (f.value_rms_norm, "normalised values"),
        (
            f.has_attn_output_bias || f.has_ffn_bias,
            "output or FFN biases",
        ),
        (c.sliding_window.is_some(), "a sliding window"),
        (c.attn_logit_softcapping.is_some(), "attention softcapping"),
        (c.residual_scale.is_some(), "a residual scale"),
        (c.shared_kv_layers > 0, "shared KV layers"),
        (c.ssm.is_some(), "a recurrent mixer"),
        (
            c.hyper.is_some() || c.ngram.is_some(),
            "wide residual streams",
        ),
        (c.attn_output_gate, "a gated attention output"),
        (c.indexer.is_some(), "sparse attention"),
    ];
    if let Some((_, why)) = checks.iter().find(|(on, _)| *on) {
        return Some(why.to_string());
    }
    for l in layers {
        let shape = [
            (!l.attends(), "a recurrent layer"),
            (l.moe.is_some(), "routed experts"),
            (l.ffn_norm.is_none(), "a layer without an FFN norm"),
            (l.rope_freq_factors.is_some(), "RoPE frequency factors"),
            (l.attn_v_norm_ones.is_some(), "normalised values"),
            (
                !matches!(l.attn_norm, super::config::WeightedNorm::Rms(_)),
                "a norm other than RMS",
            ),
        ];
        if let Some((_, why)) = shape.iter().find(|(on, _)| *on) {
            return Some(why.to_string());
        }
    }
    None
}

#[cfg(feature = "opencl")]
fn build_stack(
    device: usize,
    config: &GenericTransformerConfig,
    layers: &[GenericTransformerLayer],
    first: usize,
    read: &RawRead<'_>,
    cos: &Tensor,
    sin: &Tensor,
    head_norm: Option<&Tensor>,
) -> anyhow::Result<crate::inference::kernel::opencl_stack::OpenClStack> {
    use crate::inference::kernel::opencl_stack::{runs_format, DenseSpec, LayerTensors, RawTensor};
    if let Some(why) = refusal(config, layers) {
        anyhow::bail!("the OpenCL layers do not run {why}");
    }
    let host = |t: &Tensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let raw = |name: String| -> anyhow::Result<RawTensor> {
        let (dtype, rows, cols, bytes) =
            read(&name)?.ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        if !runs_format(dtype) {
            anyhow::bail!(
                "{name} is {}, which the OpenCL mat-vec does not read",
                dtype.gguf_name()
            );
        }
        Ok(RawTensor {
            dtype,
            rows,
            cols,
            bytes,
        })
    };
    let rope_dim = config
        .rope_dim
        .filter(|&d| d > 0 && d < config.head_dim)
        .unwrap_or(config.head_dim);
    let spec = DenseSpec {
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
    };
    let mut tensors = Vec::with_capacity(layers.len());
    for (i, l) in layers.iter().enumerate() {
        let p = format!("blk.{}", first + i);
        let norm_of =
            |n: &Option<crate::tensor::layer::RmsNorm>| -> anyhow::Result<Option<Vec<f32>>> {
                n.as_ref().map(|n| host(n.weight())).transpose()
            };
        let bias = |b: &Option<Tensor>| -> anyhow::Result<Option<Vec<f32>>> {
            b.as_ref().map(host).transpose()
        };
        let ffn = raw(format!("{p}.ffn_gate.weight"))?;
        if ffn.rows != spec.ffn {
            anyhow::bail!(
                "{p}: an FFN of {} rows, the config says {}",
                ffn.rows,
                spec.ffn
            );
        }
        tensors.push(LayerTensors {
            attn_norm: host(l.attn_norm.weight())?,
            q: raw(format!("{p}.attn_q.weight"))?,
            k: raw(format!("{p}.attn_k.weight"))?,
            v: raw(format!("{p}.attn_v.weight"))?,
            q_bias: bias(&l.attn_q_bias)?,
            k_bias: bias(&l.attn_k_bias)?,
            v_bias: bias(&l.attn_v_bias)?,
            q_norm: norm_of(&l.attn_q_norm)?,
            k_norm: norm_of(&l.attn_k_norm)?,
            o: raw(format!("{p}.attn_output.weight"))?,
            ffn_norm: host(
                l.ffn_norm
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("{p}: no FFN norm"))?
                    .weight(),
            )?,
            gate: ffn,
            up: raw(format!("{p}.ffn_up.weight"))?,
            down: raw(format!("{p}.ffn_down.weight"))?,
        });
    }
    let half = cos.dim(1)?;
    let mut stack = crate::inference::kernel::opencl_stack::OpenClStack::new(
        device,
        spec,
        tensors,
        &host(cos)?,
        &host(sin)?,
        half,
    )?;
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
                    bytes,
                };
                stack.attach_head(&host(norm)?, &proj)?;
            }
            _ => tracing::info!("opencl({device}): the output head stays on the host"),
        }
    }
    Ok(stack)
}
