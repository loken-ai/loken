//! The layers of a model whose residual mixes through Sinkhorn projections around latent
//! attention, read from its file.
//!
//! A layer's token mixer is latent attention over 1 shared key head.
//! It attends over a window, or over a window and a compressed band where the header gives the
//! layer a compress ratio.
//! Its feed-forward is a sqrt-softplus router over routed experts, beside a shared expert.
//!
//! Each tensor is read under every spelling a converter writes for it.
//! The tensors come from a GGUF file or from a block-scaled checkpoint.
//! A projection is read in place when a host kernel reads its format, else held dense F32.
//! Norms, mixing weights, the router and the pooling and index weights are held dense F32.
//! Every layer and the head run on the host.

use super::super::block_scaled_experts::BlockScaledExpert;
use super::super::compressed_band::{
    BandCell, BandDims, BandIndexerWeights, CompressedBand, CompressorWeights,
};
use super::super::dense::BlockScaledCheckpoint;
use super::super::hyper::sinkhorn::{SinkhornMeta, SinkhornMixer};
use super::super::latent_mqa::{frequencies, LatentMqa, LatentMqaDims, LatentMqaWeights};
use super::super::wide_sparse_config::{LayerKind, WideSparseMeta};
use super::super::wide_sparse_layer::{PreMixCarry, RoutedFfn, SinkhornBlock, WideSparseHead};
use super::super::*;
use super::TensorSource;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::quantized::GgmlDType;

/// Where the builder reads a tensor, by the loader's name.
enum Tensors<'s, 'a> {
    /// A GGUF file through the loader's source.
    /// The routed experts are views of their stacks.
    File {
        content: &'s gguf_file::Content,
        source: &'s TensorSource<'a>,
    },
    /// A block-scaled checkpoint.
    /// The routed experts are views of their own matrices.
    Checkpoint(&'s BlockScaledCheckpoint),
}

impl Tensors<'_, '_> {
    /// Whether the source holds `name`.
    fn has(&self, name: &str) -> bool {
        match self {
            Self::File { content, .. } => content.tensor_infos.contains_key(name),
            Self::Checkpoint(ck) => ck.has(name),
        }
    }

    /// `name` held dense F32 on the host.
    fn dense(&self, name: &str) -> Result<Tensor> {
        match self {
            Self::File { source, .. } => {
                // CONVERSION: norms, mixing weights, the router and the pooling and index weights
                // are held dense F32, the form the layer's host arithmetic reads.
                source
                    .read(name, &Device::Cpu)?
                    .dequantize(&Device::Cpu)?
                    .to_dtype(DType::F32)
            }
            Self::Checkpoint(ck) => ck.dense_f32(name),
        }
    }

    /// `name` in place when a host kernel reads its format, else held dense F32.
    fn projection(&self, name: &str) -> Result<Projection> {
        match self {
            Self::File { source, .. } => {
                let qt = source.read(name, &Device::Cpu)?;
                if crate::tensor::quant_view::native_view_supported(qt.dtype()) {
                    return Ok(Projection::Quant(Arc::new(qt)));
                }
                // CONVERSION: a format no host kernel reads in place is held dense F32.
                Ok(Projection::Dense(
                    qt.dequantize(&Device::Cpu)?.to_dtype(DType::F32)?,
                ))
            }
            Self::Checkpoint(ck) => ck.projection(name),
        }
    }

    /// The stack `name` of a layer's routed experts.
    ///
    /// `None` when the source holds no such stack.
    /// A checkpoint holds every expert as its own matrices.
    fn stack(&self, name: &str) -> Result<Option<Arc<QTensor>>> {
        match self {
            Self::File { source, .. } => source.read(name, &Device::Cpu).map(|q| Some(Arc::new(q))),
            Self::Checkpoint(_) => Ok(None),
        }
    }

    /// The embedding table `name`.
    ///
    /// A file's table is kept in its stored blocks.
    /// A checkpoint's is held dense F32.
    fn embedding(&self, name: &str) -> Result<Embedding> {
        match self {
            Self::File { source, .. } => {
                Embedding::from_stored(Arc::new(source.read(name, &Device::Cpu)?))
            }
            Self::Checkpoint(ck) => Ok(Embedding::new(ck.dense_f32(name)?)),
        }
    }

    /// The load is over.
    fn finish(&self) {
        if let Self::File { source, .. } = self {
            source.count.finish();
        }
    }
}

/// The names block `il` may carry a tensor under: each stem bare, then with the `.weight`
/// suffix.
fn spellings(il: usize, stems: &[&str]) -> Vec<String> {
    stems
        .iter()
        .flat_map(|s| [format!("blk.{il}.{s}"), format!("blk.{il}.{s}.weight")])
        .collect()
}

/// The configuration the generic forward around the layers reads.
///
/// The token mixers, the residual mixers and the feed-forward hold their own widths.
fn generic_config(
    meta: &WideSparseMeta,
    arch: &str,
    kv_quant: crate::inference::engine::llm_engine::KvQuant,
    max_kv_seq_len: Option<usize>,
) -> GenericTransformerConfig {
    let m = &meta.mixture;
    GenericTransformerConfig {
        arch: arch.to_string(),
        n_head: meta.heads,
        // The latent attention shares 1 key row among its heads.
        n_kv_head: 1,
        head_dim: meta.head_dim,
        n_layers: meta.n_layers,
        vocab_size: meta.vocab,
        embedding_length: meta.d_model,
        ffn_dim: m.inter_dim,
        rope_freq_base: meta.rope.theta,
        rope_freq_base_swa: None,
        rope_dim: Some(meta.rope_dim),
        learned_positions: false,
        yarn: None,
        rms_norm_eps: meta.rms_eps,
        sliding_window: None,
        shared_kv_layers: 0,
        embed_scale: None,
        logit_scale: None,
        attention_scale: None,
        residual_scale: None,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        max_alibi_bias: None,
        swa_pattern: None,
        swa_period: None,
        expert_swiglu_clamp: None,
        expert_sigmoid_gating: false,
        expert_sqrt_softplus_gating: Some(m.gate_temp),
        expert_weights_norm: m.norm_topk,
        expert_topk_norm: m.norm_topk,
        expert_weights_scale: m.route_scale as f64,
        ssm: None,
        delta_output_gate_sigmoid: false,
        hyper: None,
        indexer: None,
        ngram: None,
        mla: None,
        attn_output_gate: false,
        ffn_norm_stem: "ffn_norm",
        flags: GenericLayerFlags {
            float_format: None,
            has_qkv_bias: false,
            has_qk_norm: false,
            has_post_attn_norm: false,
            has_post_ffn_norm: false,
            fused_qkv: false,
            fused_ffn_gate_up: false,
            is_phi2_simple_ffn: false,
            use_gelu: false,
            value_rms_norm: false,
            // The rotated lanes of a head pair up adjacent.
            use_rope_i: true,
            intermediate_size: m.inter_dim,
            parallel_attn: false,
            layer_norm_with_bias: false,
            layer_norm: false,
            has_attn_output_bias: false,
            has_ffn_bias: false,
            has_attn_sinks: true,
            is_moe: true,
            n_experts: m.routed,
            n_experts_used: m.used,
            expert_ffn_dim: m.inter_dim,
            has_pre_ffw_norm_2: false,
            has_post_ffw_norm_split: false,
            mixed_attention_keqv: false,
            fused_qkv_norm_eligible: false,
        },
        kv_quant,
        // The layers keep their own windows: no KV cache of the forward reads this.
        max_q8_seq_len: max_kv_seq_len.unwrap_or(meta.window),
    }
}

impl GenericHeteroTransformer {
    /// The model `content` describes, its tensors read from `source`.
    ///
    /// Every layer and the head run on the host.
    /// A plan that gives a layer another device, or places part of the stack, is refused.
    pub(super) fn from_wide_sparse(
        content: &gguf_file::Content,
        source: &TensorSource<'_>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let meta = WideSparseMeta::from_file(content)?;
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_default();
        Self::build_wide_sparse(
            &meta,
            &arch,
            &Tensors::File { content, source },
            plan,
            kv_quant,
            max_kv_seq_len,
        )
    }

    /// The model a block-scaled checkpoint holds, described by its reference arguments.
    ///
    /// The routed experts are views of the checkpoint's shards.
    /// A checkpoint that declares n-gram hash layers is refused.
    /// Every layer and the head run on the host.
    pub fn from_block_scaled(
        checkpoint: &BlockScaledCheckpoint,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let hashed = checkpoint
            .reference
            .get("engram_layer_ids")
            .and_then(|v| v.as_array())
            .is_some_and(|layers| !layers.is_empty());
        if hashed {
            crate::tensor::bail!(
                "the n-gram hash layers of this checkpoint do not run on the generic route"
            );
        }
        let meta = WideSparseMeta::from_reference(&checkpoint.reference, None)?;
        Self::build_wide_sparse(
            &meta,
            crate::inference::model::deepseek_v41::ARCH,
            &Tensors::Checkpoint(checkpoint),
            plan,
            kv_quant,
            max_kv_seq_len,
        )
    }

    /// The model `meta` describes, its tensors read from `tensors`.
    fn build_wide_sparse(
        meta: &WideSparseMeta,
        arch: &str,
        tensors: &Tensors<'_, '_>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let n_layers = meta.n_layers;
        let on_host = plan
            .segments
            .iter()
            .all(|s| matches!(s.kind, DeviceKind::Cpu));
        if !on_host || plan.range() != (0, n_layers) {
            crate::tensor::bail!(
                "a Sinkhorn wide residual around latent attention runs its whole stack on the host"
            );
        }
        if meta.ngram.is_some() {
            crate::tensor::bail!(
                "the n-gram hash layers of this file do not run on the generic route"
            );
        }

        let present =
            |names: &[String]| -> Option<String> { names.iter().find(|n| tensors.has(n)).cloned() };
        let need = |names: &[String]| -> Result<String> {
            present(names).ok_or_else(|| {
                let first = names.first().map_or("", String::as_str);
                crate::tensor::Error::msg(format!("missing tensor: {first}"))
            })
        };
        let dense = |name: &str| tensors.dense(name);
        let projection = |name: &str| tensors.projection(name);
        // A projection field of the layer that its forward never reads.
        let placeholder = || -> Result<QMatMul> {
            let side = GgmlDType::Q8_0.block_size();
            let zeros = Tensor::zeros((side, side), DType::F32)?;
            // CONVERSION: a placeholder for a projection field the layer never runs.
            QMatMul::from_qtensor(QTensor::quantize_onto(
                &zeros,
                GgmlDType::Q8_0,
                &Device::Cpu,
            )?)
        };

        let mixing = SinkhornMeta {
            streams: meta.streams.streams,
            iterations: meta.streams.sinkhorn_iters,
            eps: meta.streams.eps,
        };
        let eps = meta.rms_eps as f32;
        let carry = Arc::new(PreMixCarry::default());
        let cell = Arc::new(BandCell::default());
        let config = Arc::new(generic_config(meta, arch, kv_quant, max_kv_seq_len));
        let flags = Arc::new(config.flags.clone());
        let m = &meta.mixture;

        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let name = |stems: &[&str]| need(&spellings(il, stems));
            let optional = |stems: &[&str]| -> Result<Option<Tensor>> {
                present(&spellings(il, stems))
                    .map(|n| dense(&n))
                    .transpose()
            };

            // A window layer turns at the base theta, unstretched.
            // A band layer turns at the theta of the compressed rows, stretched past the
            // original context.
            let inv_freq = match meta.layer_kind(il) {
                LayerKind::Window => {
                    frequencies(meta.rope_dim, 0, meta.rope.theta, meta.rope.scaling_factor)
                }
                LayerKind::Compressed { .. } => frequencies(
                    meta.rope_dim,
                    meta.rope.original_context,
                    meta.rope.compress_theta,
                    meta.rope.scaling_factor,
                ),
            };
            let window = LatentMqa::new(
                LatentMqaDims::from_meta(meta),
                LatentMqaWeights {
                    wq_a: projection(&name(&["attn_q_a"])?)?,
                    q_norm: dense(&name(&["attn_q_a_norm"])?)?,
                    wq_b: projection(&name(&["attn_q_b"])?)?,
                    wkv: projection(&name(&["attn_kv"])?)?,
                    kv_norm: dense(&name(&["attn_kv_a_norm", "attn_kv_norm"])?)?,
                    sink: dense(&name(&["attn_sinks", "attn_sink"])?)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    wo_a: projection(&name(&["attn_output_a", "attn_o_a"])?)?,
                },
                inv_freq,
            )?;
            // A kv-source holds the pooling weights, an index-source the indexer's.
            let token_mixer = match BandDims::from_meta(meta, il) {
                None => mixer::Mixer::LatentMqa(window),
                Some(band) => {
                    let compressor = match band.kv_source {
                        true => Some(CompressorWeights {
                            wkv: dense(&name(&["attn_compressor_kv"])?)?,
                            wgate: optional(&["attn_compressor_gate"])?,
                            norm: dense(&name(&["attn_compressor_norm"])?)?,
                        }),
                        false => None,
                    };
                    let indexer = match band.index_source {
                        true => Some(BandIndexerWeights {
                            wq_b: projection(&name(&["indexer.attn_q_b", "attn_indexer_q_b"])?)?,
                            weights_proj: dense(&name(&["indexer.proj", "attn_indexer_weights"])?)?,
                            wk: optional(&["indexer.attn_k", "attn_indexer_k"])?,
                            k_norm: optional(&["indexer.k_norm", "attn_indexer_k_norm"])?,
                        }),
                        false => None,
                    };
                    mixer::Mixer::CompressedBand(CompressedBand::new(
                        window,
                        band,
                        compressor,
                        indexer,
                        cell.clone(),
                    )?)
                }
            };

            let residual_mixer = |part: &str| -> Result<SinkhornMixer> {
                let (func, scale, bias) = (
                    format!("hc_{part}_fn"),
                    format!("hc_{part}_scale"),
                    format!("hc_{part}_base"),
                );
                SinkhornMixer::new(
                    &dense(&need(&spellings(il, &[func.as_str()]))?)?,
                    &dense(&need(&spellings(il, &[scale.as_str()]))?)?,
                    &dense(&need(&spellings(il, &[bias.as_str()]))?)?,
                    mixing,
                    meta.d_model,
                    eps,
                )
            };

            // The routed experts: 1 stack per projection, or 3 tensors of its own per expert.
            let stack = |stem: &str| -> Result<Option<Arc<QTensor>>> {
                match present(&spellings(il, &[stem])) {
                    Some(n) => tensors.stack(&n),
                    None => Ok(None),
                }
            };
            let one = |stack: &Arc<QTensor>, e: usize| -> Result<Projection> {
                if !crate::tensor::quant_view::native_view_supported(stack.dtype()) {
                    crate::tensor::bail!(
                        "the experts of layer {il} are {:?}: no host kernel reads them in place",
                        stack.dtype()
                    );
                }
                Ok(Projection::Quant(Arc::new(
                    crate::tensor::quant_view::expert_view(stack, e)?,
                )))
            };
            let stacks = (
                stack("ffn_gate_exps")?,
                stack("ffn_up_exps")?,
                stack("ffn_down_exps")?,
            );
            let experts = match stacks {
                (Some(gate), Some(up), Some(down)) => (0..m.routed)
                    .map(|e| -> Result<BlockScaledExpert> {
                        Ok(BlockScaledExpert {
                            gate: one(&gate, e)?,
                            up: one(&up, e)?,
                            down: one(&down, e)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => (0..m.routed)
                    .map(|e| -> Result<BlockScaledExpert> {
                        Ok(BlockScaledExpert {
                            gate: projection(&format!("blk.{il}.ffn_gate_exps.{e}"))?,
                            up: projection(&format!("blk.{il}.ffn_up_exps.{e}"))?,
                            down: projection(&format!("blk.{il}.ffn_down_exps.{e}"))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let shared = match m.shared {
                0 => None,
                _ => Some(BlockScaledExpert {
                    gate: projection(&name(&["ffn_gate_shexp"])?)?,
                    up: projection(&name(&["ffn_up_shexp"])?)?,
                    down: projection(&name(&["ffn_down_shexp"])?)?,
                }),
            };
            let ffn = RoutedFfn {
                router: Projection::Dense(dense(&name(&["ffn_gate_inp"])?)?),
                bias: dense(&name(&[
                    "exp_probs_b",
                    "exp_probs_b.bias",
                    "ffn_gate_inp.bias",
                ])?)?
                .flatten_all()?
                .to_vec1::<f32>()?,
                experts,
                shared,
                used: m.used,
                temperature: m.gate_temp,
                normalize: m.norm_topk,
                scale: m.route_scale,
                limit: m.swiglu_limits.get(il).copied().unwrap_or(0.0),
            };
            let attn_norm = dense(&name(&["attn_norm"])?)?;
            let ffn_norm = dense(&name(&["ffn_norm"])?)?;
            let block = SinkhornBlock {
                attn_mix: residual_mixer("attn")?,
                ffn_mix: residual_mixer("ffn")?,
                attn_norm: attn_norm.clone(),
                ffn_norm: ffn_norm.clone(),
                eps,
                output: projection(&name(&["attn_output_b", "attn_o_b"])?)?,
                ffn,
                carry: carry.clone(),
            };

            // The forward reads no rotation table of the layer: its token mixer turns its own.
            let unused_table = Tensor::zeros((1usize, 1usize), DType::F32)?;
            layers.push(GenericTransformerLayer {
                tp_shards: Vec::new(),
                tp_comms: None,
                mixer: token_mixer,
                has_ffn: true,
                attn_gate: None,
                attn_q: None,
                attn_k: None,
                attn_v: None,
                attn_qkv: None,
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                attn_qkv_bias: None,
                attn_q_norm: None,
                attn_k_norm: None,
                attn_v_norm_ones: None,
                attn_output: placeholder()?,
                attn_output_bias: None,
                attn_sinks: None,
                attn_sinks_host: None,
                attn_norm: WeightedNorm::Rms(RmsNorm::from_tensor(attn_norm, meta.rms_eps)),
                attn_norm_weight_f32: None,
                post_attn_norm: None,
                ffn_gate: None,
                ffn_up: placeholder()?,
                ffn_up_bias: None,
                ffn_down: placeholder()?,
                ffn_down_bias: None,
                moe: None,
                pre_ffw_norm_2: None,
                post_ffw_norm_1: None,
                post_ffw_norm_2: None,
                embedding_length_for_moe: meta.d_model,
                ffn_norm: Some(WeightedNorm::Rms(RmsNorm::from_tensor(
                    ffn_norm,
                    meta.rms_eps,
                ))),
                ffn_norm_weight: None,
                ffn_norm_eps: eps,
                post_ffn_norm: None,
                ple_inp_gate: None,
                ple_proj: None,
                ple_post_norm: None,
                ple_output_scale: None,
                context: None,
                kv_cache: SpecKvCache::new(config.effective_max_context()),
                cpu_q8_kv: None,
                host_q8_kv: false,
                cpu_f16_kv: None,
                cpu_decode_arena: None,
                cpu_norm_cache: None,
                #[cfg(feature = "gpu")]
                quant_kv: None,
                populate_dual_kv: false,
                kv_f16: false,
                padded_mask: None,
                graph_q_buffer: None,
                graph_rope_cos: None,
                graph_rope_sin: None,
                graph_kv_pos: None,
                graph_pos_i32: None,
                graph_attn_qk_buffer: None,
                graph_attn_out_buffer: None,
                graph_attn_proj_buffer: None,
                graph_ffn_up_buffer: None,
                graph_ffn_activated_buffer: None,
                graph_ffn_down_buffer: None,
                graph_phi2_merge_buffer: None,
                graph_ffn_up_concat_buffer: None,
                graph_post_ffn_norm_buffer: None,
                graph_ple_gate_buffer: None,
                graph_ple_gelu_buffer: None,
                graph_ple_proj_buffer: None,
                graph_ple_final_buffer: None,
                graph_post_attn_norm_buffer: None,
                graph_x_norm_ffn_buffer: None,
                graph_x_norm_attn_buffer: None,
                graph_ple_input_buffer: None,
                graph_ngram_rows: None,
                spec: None,
                spec_record: false,
                kv_only: false,
                graph_kv_cap: None,
                verify_pos_i32: None,
                verify_rope_cos: None,
                verify_rope_sin: None,
                verify_ngram_rows: None,
                verify_keep: None,
                graph_attn_mask_added_buffer: None,
                graph_attn_softmax_buffer: None,
                graph_ffn_gate_out: None,
                graph_x_out_scaled: None,
                graph_attn_scaled_buffer: None,
                graph_alive_tensors: Vec::new(),
                cos: unused_table.clone(),
                sin: unused_table,
                neg_inf: Tensor::new(f32::NEG_INFINITY, &Device::Cpu)?,
                rope_freq_factors: None,
                rope_factored_freqs: None,
                n_head: meta.heads,
                n_kv_head: config.n_kv_head,
                head_dim: meta.head_dim,
                sliding_window: None,
                prefill_mask_window: None,
                rope_dim: meta.rope_dim,
                rope_rows: None,
                rope_base: meta.rope.theta,
                no_rope: false,
                attention_scale: None,
                attn_softcap: None,
                alibi_slopes: None,
                residual_scale: None,
                flags: flags.clone(),
                hyper: None,
                sparse: None,
                ngram: None,
                sinkhorn: Some(block),
            });
        }

        let embed = need(&["token_embd".to_string(), "token_embd.weight".to_string()])?;
        let embeddings = tensors.embedding(&embed)?;
        let norm = dense(&need(&[
            "output_norm".to_string(),
            "output_norm.weight".to_string(),
        ])?)?;
        // The head is its own tensor when the file has one, else the embedding table.
        let head = match present(&["output".to_string(), "output.weight".to_string()]) {
            Some(n) => projection(&n)?,
            None => Projection::Dense(dense(&embed)?),
        };
        let wide_sparse = WideSparseHead {
            carry,
            cell,
            streams: mixing.streams,
            norm: norm.clone(),
            eps,
            head,
        };
        tensors.finish();
        Ok(Self {
            adapters: Vec::new(),
            config,
            embeddings,
            position_embd: None,
            embed_cards: Vec::new(),
            layers,
            first_layer: 0,
            layer_devs: vec![LayerDevice::Cpu; n_layers],
            vulkan: Default::default(),
            opencl: Default::default(),
            output_norm: WeightedNorm::Rms(RmsNorm::from_tensor(norm, meta.rms_eps)),
            output_proj: None,
            output_bias: None,
            capture_feature: false,
            tier_workers: Vec::new(),
            #[cfg(feature = "gpu")]
            hot_locks: None,
            kept_graphs_at: None,
            verify: Default::default(),
            graph_feature: None,
            last_feature: None,
            spec_window: None,
            spec_recording: false,
            output_proj_cuda: None,
            output_norm_cuda_weight: None,
            output_proj_cuda_qmm: None,
            head_share: 0.0,
            output_proj_dev: None,
            cuda_devices: HashMap::new(),
            sliding_window: None,
            mask_cache: HashMap::new(),
            ple_token_embd: None,
            ple_model_proj: None,
            ple_proj_norm: None,
            ple_dim: 0,
            graph_hidden_buffer: None,
            graph_work_buffer: None,
            #[cfg(feature = "gpu")]
            graph_hops: Vec::new(),
            graph_logits_buffer: None,
            graph_alive_tensors_model: Vec::new(),
            graph_proven: 0,
            graph_generation: 0,
            graph_instance: super::super::graph::next_graph_instance(),
            graph_state_groups: None,
            workspace_pre_grown: false,
            graph_pos: 0,
            kv_snapshots: Vec::new(),
            cache_scope: Default::default(),
            kv_snapshot_tick: 0,
            kv_lease: None,
            prefix_boundary: None,
            prefix_covered: 0,
            decode_reserve: 0,
            hybrid_cache: None,
            spliced_vision: None,
            #[cfg(feature = "audio")]
            spliced_audio: None,
            host_experts_hot: None,
            output_hyper: None,
            ngram: None,
            draft: None,
            wide_sparse: Some(wide_sparse),
        })
    }
}
