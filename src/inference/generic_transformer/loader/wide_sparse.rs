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
//! A layer on a CUDA card runs the projections and the attention of its token mixer there.
//! The residual mixing, the feed-forward, the engram and the head run on the host.
//!
//! An engram layer reads its table in place: e4m3 rows beside their scales, or blocks a host
//! kernel reads.
//! A table in any other format is held dense F32.
//!
//! The routed experts of a GGUF file may also be read from further files of its checkpoint,
//! each storing them in its own format.
//!
//! A file whose metadata declares a block draft gives the model its draft head.
//! The head's tensors keep the checkpoint's names, `mtp.{stage}.*`.
//! Its projections and routed experts are read as the layers' are.

use super::super::block_draft::{BlockDraftHead, BlockDraftMeta, DraftStage};
use super::super::block_scaled_experts::BlockScaledExpert;
use super::super::compressed_band::{
    BandCell, BandDims, BandIndexerWeights, CompressedBand, CompressorWeights,
};
use super::super::dense::BlockScaledCheckpoint;
use super::super::hyper::sinkhorn::{SinkhornMeta, SinkhornMixer};
use super::super::latent_mqa::{frequencies, LatentMqa, LatentMqaDims, LatentMqaWeights};
use super::super::ngram::engram::{EngramBlock, EngramMeta, EngramTable};
use super::super::wide_sparse_config::{LayerKind, NgramHashFields, WideSparseMeta};
use super::super::wide_sparse_layer::{
    ExpertVariants, LayerEngram, PreMixCarry, RoutedFfn, SinkhornBlock, TokenCarry, WideSparseHead,
};
use super::super::wide_sparse_names::{ARCH, ENGRAM_TABLE};
use super::super::*;
use super::TensorSource;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::mapped::MappedBytes;
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

    /// The engram table `name`.
    fn engram_table(&self, name: &str) -> Result<EngramTable> {
        match self {
            Self::File { source, .. } => engram_table(source, name),
            Self::Checkpoint(ck) => checkpoint_engram_table(ck, name),
        }
    }

    /// The load is over.
    fn finish(&self) {
        if let Self::File { source, .. } = self {
            source.count.finish();
        }
    }
}

/// The engram table `name` of a file.
///
/// A pair of byte tensors, the rows and the `.scale` beside them, is read in place as e4m3 rows
/// with 1 ue8m0 scale per block.
/// A table in a format a host kernel reads is read in place, block by block.
/// Any other table is held dense F32.
fn engram_table(source: &TensorSource<'_>, name: &str) -> Result<EngramTable> {
    let scale = format!("{}.scale", name.strip_suffix(".weight").unwrap_or(name));
    if let Some(table) = released_fp8_rows(source, name, &scale)? {
        return Ok(table);
    }
    let qt = source.read(name, &Device::Cpu)?;
    if crate::tensor::quant_view::native_view_supported(qt.dtype()) {
        let table = EngramTable::Streamed(Arc::new(qt));
        table.advise_random();
        return Ok(table);
    }
    // CONVERSION: a table no host kernel reads in place is held dense F32.
    Ok(EngramTable::Resident(
        qt.dequantize(&Device::Cpu)?.to_dtype(DType::F32)?,
    ))
}

/// The table `weight` as e4m3 rows beside their ue8m0 scales `scale`, both byte tensors of a
/// mapped file.
///
/// `None` when the file does not hold the pair as byte tensors of a mapping.
/// The block is the row length over the scales per row.
fn released_fp8_rows(
    source: &TensorSource<'_>,
    weight: &str,
    scale: &str,
) -> Result<Option<EngramTable>> {
    let (Some(w), Some(s)) = (source.info(weight), source.info(scale)) else {
        return Ok(None);
    };
    if w.ggml_dtype != GgmlDType::I8 || s.ggml_dtype != GgmlDType::I8 {
        return Ok(None);
    }
    let (&[rows, row_len], &[scale_rows, per_row]) = (w.shape.dims(), s.shape.dims()) else {
        crate::tensor::bail!("{weight}: an engram table is two-dimensional");
    };
    if scale_rows != rows || per_row == 0 || !row_len.is_multiple_of(per_row) {
        crate::tensor::bail!(
            "{weight}: scales {:?} do not tile [{rows}, {row_len}] by rows",
            s.shape.dims()
        );
    }
    let (Some((wm, wo, wl)), Some((sm, so, sl))) =
        (source.mapped_range(weight), source.mapped_range(scale))
    else {
        return Ok(None);
    };
    source.count.read(weight, w);
    source.count.read(scale, s);
    let table = EngramTable::Fp8Rows {
        bytes: MappedBytes::new(wm, wo, wl)?,
        scales: MappedBytes::new(sm, so, sl)?,
        rows,
        row_len,
        block: row_len / per_row,
    };
    table.advise_random();
    Ok(Some(table))
}

/// The engram table `name` of a block-scaled checkpoint.
///
/// An e4m3 table is read in place as its rows beside their ue8m0 scales.
/// Any other table is held dense F32.
fn checkpoint_engram_table(ck: &BlockScaledCheckpoint, name: &str) -> Result<EngramTable> {
    Ok(match ck.table_of(name)? {
        TableRows::Fp8 {
            bytes,
            scales,
            rows,
            row_len,
            block,
        } => {
            let table = EngramTable::Fp8Rows {
                bytes,
                scales,
                rows,
                row_len,
                block,
            };
            table.advise_random();
            table
        }
        TableRows::Dense {
            values,
            rows,
            row_len,
        } => EngramTable::Resident(Tensor::from_vec(values, (rows, row_len), &Device::Cpu)?),
    })
}

/// The hash layout of the engram layers of a block-scaled checkpoint.
///
/// The token map comes from the checkpoint's `tokenizer.json`.
/// `None` when the checkpoint declares no engram layer.
/// `None`, with a warning, when the token map cannot be built or the layout does not derive from
/// the reference arguments and that map: the model then runs without its engram layers.
fn checkpoint_engram(ck: &BlockScaledCheckpoint) -> Option<EngramMeta> {
    let declared = ck
        .reference
        .get("engram_layer_ids")
        .and_then(|v| v.as_array())
        .is_some_and(|layers| !layers.is_empty());
    if !declared {
        return None;
    }
    let path = ck.dir.join("tokenizer.json");
    let engram = match super::super::ngram::token_map::build_from_file(&path) {
        Ok((map, _)) => EngramMeta::derive(&ck.reference, map),
        Err(e) => {
            tracing::warn!("no engram, the token map could not be built: {e}");
            None
        }
    };
    if engram.is_none() {
        tracing::warn!("running without the engram layers");
    }
    engram
}

/// The names block `il` may carry a tensor under: each stem bare, then with the `.weight`
/// suffix.
fn spellings(il: usize, stems: &[&str]) -> Vec<String> {
    stems
        .iter()
        .flat_map(|s| [format!("blk.{il}.{s}"), format!("blk.{il}.{s}.weight")])
        .collect()
}

/// The `routed` experts of block `il`, read from `tensors`.
///
/// A block stores them as 1 stack per projection, or as 3 tensors of its own per expert.
/// An expert of a stack is a view of its rows.
/// A stack in a format no host kernel reads in place is refused.
fn routed_experts(
    tensors: &Tensors<'_, '_>,
    il: usize,
    routed: usize,
) -> Result<Vec<BlockScaledExpert>> {
    let stack = |stem: &str| -> Result<Option<Arc<QTensor>>> {
        match spellings(il, &[stem]).into_iter().find(|n| tensors.has(n)) {
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
    match stacks {
        (Some(gate), Some(up), Some(down)) => (0..routed)
            .map(|e| -> Result<BlockScaledExpert> {
                Ok(BlockScaledExpert {
                    gate: one(&gate, e)?,
                    up: one(&up, e)?,
                    down: one(&down, e)?,
                })
            })
            .collect(),
        _ => (0..routed)
            .map(|e| -> Result<BlockScaledExpert> {
                Ok(BlockScaledExpert {
                    gate: tensors.projection(&format!("blk.{il}.ffn_gate_exps.{e}"))?,
                    up: tensors.projection(&format!("blk.{il}.ffn_up_exps.{e}"))?,
                    down: tensors.projection(&format!("blk.{il}.ffn_down_exps.{e}"))?,
                })
            })
            .collect(),
    }
}

/// Whether every projection of `expert` is read in place from its file's blocks.
fn read_in_place(expert: &BlockScaledExpert) -> bool {
    [&expert.gate, &expert.up, &expert.down]
        .iter()
        .all(|p| matches!(p, Projection::Quant(_)))
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
        rope_linear: None,
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
            erf_gelu: false,
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
    /// A layer on a CUDA card runs its token mixer there, the rest on the host.
    /// A plan that gives a layer another device is refused.
    /// A plan over part of the stack gives a model holding those layers alone, without a draft.
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
        let engram = EngramMeta::from_gguf(content, &arch);
        let tensors = Tensors::File { content, source };
        let mut model = Self::build_wide_sparse(
            &meta,
            &arch,
            &tensors,
            engram,
            plan,
            kv_quant,
            max_kv_seq_len,
        )?;
        // A share takes no draft: the draft reads the layers of every range of the stack.
        if plan.range() != (0, meta.n_layers) {
            return Ok(model);
        }
        // A draft the file declares and that does not load leaves the model without one.
        let draft = BlockDraftMeta::from_metadata(&content.metadata).and_then(|meta| {
            meta.map(|m| build_block_draft(&tensors, m).and_then(|h| model.attach_block_draft(h)))
                .transpose()
        });
        if let Err(e) = draft {
            tracing::warn!("the block draft of the model file is not loaded: {e}");
        }
        Ok(model)
    }

    /// Takes the block draft the first companion file of `model` declares, in place of the one
    /// the model file carried.
    ///
    /// A companion is one whose metadata holds a key under the draft's prefix.
    /// A blob of a store takes it from the manifests that name the blob, a plain file from its
    /// directory.
    /// A file that does not load is passed over for the next.
    /// A model that is not a Sinkhorn wide residual reads no file.
    /// Returns whether a companion file gave the draft.
    pub fn locate_block_draft(&mut self, model: &std::path::Path) -> bool {
        if self.wide_sparse.is_none() {
            return false;
        }
        let prefix = format!("{ARCH}.dspark.");
        for path in super::super::wide_sparse_sibling::find_companions(model, &prefix) {
            let loaded = companion_draft(&path)
                .and_then(|d| d.map(|h| self.attach_block_draft(h)).transpose());
            match loaded {
                Ok(Some(())) => {
                    tracing::info!("block draft loaded from {}", path.display());
                    return true;
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("block draft in {} not loaded: {e}", path.display()),
            }
        }
        false
    }

    /// Reads the routed experts of every layer this model holds from `files` as well: further
    /// files of the checkpoint the model's own file belongs to, each storing them in its own
    /// format.
    ///
    /// Variant 0 is the model's own file, variant `i + 1` is `files[i]`.
    /// Every expert reads variant 0 until `place_expert_variants` chooses.
    /// An expert that is not read in place from its file's blocks, in any file, refuses the
    /// variants and leaves the model as it was.
    /// A model that is not a Sinkhorn wide residual is refused.
    pub fn attach_expert_variants(
        &mut self,
        files: &[crate::tensor::quantized::gguf_source::SplitGguf],
    ) -> Result<()> {
        let Some(ws) = self.wide_sparse.as_ref() else {
            crate::tensor::bail!("expert variants: the model is not a Sinkhorn wide residual");
        };
        let routed = ws.meta.mixture.routed;
        // Each file's tensors, read without counting toward the load's progress.
        let contents: Vec<gguf_file::Content> = files.iter().map(|f| f.merged_content()).collect();
        let sources: Vec<TensorSource<'_>> = files
            .iter()
            .map(|f| {
                let mut source = TensorSource::of_split(f);
                source.count = super::LoadCount::none();
                source
            })
            .collect();
        let mut attached = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let il = self.first_layer + i;
            let Some(block) = layer.sinkhorn.as_ref() else {
                attached.push(None);
                continue;
            };
            if !block.ffn.experts.iter().all(read_in_place) {
                crate::tensor::bail!(
                    "expert variants: the experts of layer {il} have no format read in place"
                );
            }
            let mut stored = vec![block.ffn.experts.clone()];
            for (v, (content, source)) in contents.iter().zip(&sources).enumerate() {
                let tensors = Tensors::File { content, source };
                let experts = routed_experts(&tensors, il, routed)?;
                if !experts.iter().all(read_in_place) {
                    crate::tensor::bail!(
                        "expert variants: the experts of layer {il} in variant {} have no \
                         format read in place",
                        v + 1
                    );
                }
                stored.push(experts);
            }
            attached.push(Some(stored));
        }
        for (layer, stored) in self.layers.iter_mut().zip(attached) {
            if let (Some(block), Some(stored)) = (layer.sinkhorn.as_mut(), stored) {
                block.ffn.variants = ExpertVariants {
                    chosen: vec![0; block.ffn.experts.len()],
                    stored,
                };
            }
        }
        Ok(())
    }

    /// The model a block-scaled checkpoint holds, described by its reference arguments.
    ///
    /// The routed experts are views of the checkpoint's shards.
    /// The engram layers hash with the layout `checkpoint_engram` derives.
    /// A layer on a CUDA card runs its token mixer there, the rest on the host.
    pub fn from_block_scaled(
        checkpoint: &BlockScaledCheckpoint,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let engram = checkpoint_engram(checkpoint);
        let meta = WideSparseMeta::from_reference(
            &checkpoint.reference,
            engram.as_ref().map(NgramHashFields::from),
        )?;
        Self::build_wide_sparse(
            &meta,
            ARCH,
            &Tensors::Checkpoint(checkpoint),
            engram,
            plan,
            kv_quant,
            max_kv_seq_len,
        )
    }

    /// The model `meta` describes, its tensors read from `tensors`.
    ///
    /// `engram` is the hash layout of the engram layers, `None` for a model without one.
    fn build_wide_sparse(
        meta: &WideSparseMeta,
        arch: &str,
        tensors: &Tensors<'_, '_>,
        engram: Option<EngramMeta>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let n_layers = meta.n_layers;
        let hosted = plan
            .segments
            .iter()
            .all(|s| matches!(s.kind, DeviceKind::Cpu | DeviceKind::Cuda(_)));
        let (from, to) = plan.range();
        if !hosted || from >= to || to > n_layers {
            crate::tensor::bail!(
                "a Sinkhorn wide residual around latent attention runs on the host or on CUDA cards, layers within its stack"
            );
        }
        // A share is built on the whole stack, then keeps its own layers.
        // A layer outside the share is built on the host.
        // The card of each layer: the CUDA card its segment names.
        let card_of = |il: usize| {
            plan.segments
                .iter()
                .find(|s| (s.layer_start..s.layer_end).contains(&il))
                .and_then(|s| match s.kind {
                    DeviceKind::Cuda(ordinal) => Some(ordinal),
                    _ => None,
                })
        };
        // The description and the engram blocks read the same hash layout.
        if meta.ngram.is_some() != engram.is_some() {
            crate::tensor::bail!(
                "the n-gram hash layout of this file reads differently for the engram blocks"
            );
        }
        // Every engram layer hashes from the token ids of 1 log the model shares.
        let engram = engram.map(|m| (Arc::new(m), Arc::new(TokenCarry::default())));

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

            let experts = routed_experts(tensors, il, m.routed)?;
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
                lossy: Default::default(),
                variants: Default::default(),
                card: Default::default(),
                watch: Default::default(),
                host: None,
            };
            // An engram layer: its table, under its own name or as the table every layer shares,
            // its fused key and value projection, and its gate.
            // The gate is the file's, else the product of the query and key weights.
            let layer_engram = match engram
                .as_ref()
                .and_then(|(hash, tokens)| hash.which(il).map(|which| (hash, tokens, which)))
            {
                None => None,
                Some((hash, tokens, which)) => {
                    let table = match present(&spellings(il, &["engram_embd"])) {
                        Some(n) => n,
                        None => need(&[ENGRAM_TABLE.to_string()])?,
                    };
                    let gate = match present(&spellings(il, &["engram_gate"])) {
                        Some(n) => dense(&n)?,
                        None => {
                            let q = dense(&name(&["engram_q"])?)?.to_vec2::<f32>()?;
                            let k = dense(&name(&["engram_k"])?)?.to_vec2::<f32>()?;
                            let width = q.first().map_or(0, Vec::len);
                            let product: Vec<f32> = q
                                .iter()
                                .zip(&k)
                                .flat_map(|(qr, kr)| qr.iter().zip(kr).map(|(a, b)| a * b))
                                .collect();
                            Tensor::from_vec(product, (q.len(), width), &Device::Cpu)?
                        }
                    };
                    Some(LayerEngram {
                        block: EngramBlock::new(
                            tensors.engram_table(&table)?,
                            projection(&name(&["engram_wkv", "engram_kv"])?)?,
                            &gate,
                            hash,
                            mixing.streams,
                            meta.d_model,
                            eps,
                        )?,
                        meta: hash.clone(),
                        which,
                        tokens: tokens.clone(),
                    })
                }
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
                engram: layer_engram,
                card_mix: Default::default(),
                capture: None,
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
                attn_layer_scale: None,
                ffn_layer_scale: None,
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
                host_f16_kv: false,
                cpu_f16_kv: None,
                cpu_decode_arena: None,
                cpu_norm_cache: None,
                #[cfg(feature = "gpu")]
                quant_kv: None,
                populate_dual_kv: false,
                key_scaled: false,
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
                donor_rows: None,
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

        // A layer on a card runs its token mixer's projections and attention there.
        // The residual mixing, the feed-forward and the head stay on the host.
        // The block weights of every card layer share the products measured per unit.
        #[cfg(feature = "gpu")]
        let choices = super::super::card_packed::Choices::default();
        for (il, layer) in layers.iter_mut().enumerate() {
            if let Some(ordinal) = card_of(il) {
                #[cfg(feature = "gpu")]
                layer.mixer.on_card(ordinal, &choices)?;
                #[cfg(not(feature = "gpu"))]
                crate::tensor::bail!(
                    "layer {il} is placed on CUDA card {ordinal}: no card feature"
                );
            }
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
            engram: engram.as_ref().map(|(hash, _)| Arc::clone(hash)),
            tokens: engram.map(|(_, tokens)| tokens),
            draft: None,
            meta: Arc::new(meta.clone()),
            prior: meta.mixture.hot.clone(),
            heat: None,
            card_rooms: Vec::new(),
            lent: Vec::new(),
            savings: Vec::new(),
            shares: None,
            card_head: Default::default(),
            head_collapse: Default::default(),
        };
        tensors.finish();
        let mut model = Self {
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
            routing_pending: None,
            #[cfg(feature = "gpu")]
            hot_locks: None,
            kept_graphs_at: None,
            #[cfg(test)]
            failing_checkpoints: Default::default(),
            #[cfg(feature = "gpu")]
            parked_pool: Default::default(),
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
        };
        if (from, to) != (0, n_layers) {
            model.narrow_to_layers(from, to)?;
            model.first_layer = from;
        }
        Ok(model)
    }
}

/// The block draft `meta` describes, its tensors read from `tensors` under the checkpoint's
/// names.
///
/// A stage is `mtp.{stage}`: its attention, its 2 Sinkhorn mixers, its router and experts.
/// The main projection is read under stage 0, the final norm and the Markov head under the last.
/// A stage carries a shared expert only when the source holds its 3 projections.
fn build_block_draft(tensors: &Tensors<'_, '_>, meta: BlockDraftMeta) -> Result<BlockDraftHead> {
    let dense = |name: &str| tensors.dense(name);
    let projection = |name: &str| tensors.projection(name);
    let mixing = SinkhornMeta {
        streams: meta.streams,
        iterations: meta.sinkhorn_iters,
        eps: meta.hc_eps,
    };
    let dims = LatentMqaDims {
        heads: meta.heads,
        head_dim: meta.head_dim,
        rope_dim: meta.rope_dim,
        o_groups: meta.o_groups,
        o_lora_rank: meta.o_lora_rank,
        window: meta.window,
        eps: meta.rms_eps,
    };
    // The stages turn at the base theta, unstretched.
    let inv_freq = frequencies(meta.rope_dim, 0, meta.rope_theta, 1.0);
    let mut stages = Vec::with_capacity(meta.stages);
    for s in 0..meta.stages {
        let name = |rest: &str| format!("mtp.{s}.{rest}");
        let attn = LatentMqa::new(
            dims,
            LatentMqaWeights {
                wq_a: projection(&name("attn.wq_a.weight"))?,
                q_norm: dense(&name("attn.q_norm.weight"))?,
                wq_b: projection(&name("attn.wq_b.weight"))?,
                wkv: projection(&name("attn.wkv.weight"))?,
                kv_norm: dense(&name("attn.kv_norm.weight"))?,
                sink: dense(&name("attn.attn_sink"))?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                wo_a: projection(&name("attn.wo_a.weight"))?,
            },
            inv_freq.clone(),
        )?;
        let mixer = |part: &str| -> Result<SinkhornMixer> {
            SinkhornMixer::new(
                &dense(&name(&format!("hc_{part}_fn")))?,
                &dense(&name(&format!("hc_{part}_scale")))?,
                &dense(&name(&format!("hc_{part}_base")))?,
                mixing,
                meta.dim,
                meta.rms_eps,
            )
        };
        // An expert's gate is `w1`, its up `w3` and its down `w2`.
        let expert = |stem: &str| -> Result<BlockScaledExpert> {
            Ok(BlockScaledExpert {
                gate: projection(&name(&format!("{stem}.w1.weight")))?,
                up: projection(&name(&format!("{stem}.w3.weight")))?,
                down: projection(&name(&format!("{stem}.w2.weight")))?,
            })
        };
        let shared_held = ["w1", "w3", "w2"]
            .iter()
            .all(|w| tensors.has(&name(&format!("ffn.shared_experts.{w}.weight"))));
        let ffn = RoutedFfn {
            router: Projection::Dense(dense(&name("ffn.gate.weight"))?),
            bias: dense(&name("ffn.gate.bias"))?
                .flatten_all()?
                .to_vec1::<f32>()?,
            experts: (0..meta.routed)
                .map(|e| expert(&format!("ffn.experts.{e}")))
                .collect::<Result<Vec<_>>>()?,
            shared: shared_held
                .then(|| expert("ffn.shared_experts"))
                .transpose()?,
            used: meta.used,
            temperature: meta.gate_temp,
            normalize: meta.norm_topk,
            scale: meta.route_scale,
            limit: meta.swiglu_limit,
            lossy: Default::default(),
            variants: Default::default(),
            card: Default::default(),
            watch: Default::default(),
            host: None,
        };
        stages.push(DraftStage {
            attn,
            wo_b: projection(&name("attn.wo_b.weight"))?,
            attn_norm: dense(&name("attn_norm.weight"))?,
            ffn_norm: dense(&name("ffn_norm.weight"))?,
            attn_mix: mixer("attn")?,
            ffn_mix: mixer("ffn")?,
            ffn,
        });
    }
    let last = meta.stages.saturating_sub(1);
    let main_proj = projection("mtp.0.main_proj.weight")?;
    let main_norm = dense("mtp.0.main_norm.weight")?;
    let final_norm = dense(&format!("mtp.{last}.norm.weight"))?;
    let markov_embed = dense(&format!("mtp.{last}.markov_head.embed.weight"))?;
    let markov_head = projection(&format!("mtp.{last}.markov_head.head.weight"))?;
    BlockDraftHead::new(
        meta,
        stages,
        main_proj,
        main_norm,
        final_norm,
        markov_embed,
        markov_head,
    )
}

/// The block draft the file set at `path` declares, `None` when it declares none.
fn companion_draft(path: &std::path::Path) -> Result<Option<BlockDraftHead>> {
    let split = crate::tensor::quantized::gguf_source::SplitGguf::open(path)?;
    let content = split.merged_content();
    let Some(meta) = BlockDraftMeta::from_metadata(&content.metadata)? else {
        return Ok(None);
    };
    let source = TensorSource::of_split(&split);
    let tensors = Tensors::File {
        content: &content,
        source: &source,
    };
    let head = build_block_draft(&tensors, meta)?;
    tensors.finish();
    Ok(Some(head))
}

impl BlockDraftHead {
    /// The block draft a block-scaled checkpoint holds, its shape `meta`.
    pub fn from_checkpoint(
        checkpoint: &BlockScaledCheckpoint,
        meta: BlockDraftMeta,
    ) -> Result<Self> {
        build_block_draft(&Tensors::Checkpoint(checkpoint), meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::wide_sparse_frozen as frozen;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::{open_mapped, Value};

    /// Port of the V4.1 test of the same name: a table the file carries as fp8 rows beside their
    /// ue8m0 scales, both byte tensors, is read in place as those rows.
    /// The geometry and the bytes are the ones the file stores.
    #[test]
    fn engram_rows_are_read_as_released_fp8() {
        let (rows, row_len, per_row) = (7usize, 64usize, 2usize);
        let weight: Vec<u8> = (0..rows * row_len).map(|i| (i % 251) as u8).collect();
        let scales: Vec<u8> = (0..rows * per_row).map(|i| (120 + i % 9) as u8).collect();
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("deepseek4".into()),
        )];
        let entries = vec![
            GgufEntry {
                name: "blk.1.engram_embd.weight".to_string(),
                dims: vec![rows, row_len],
                dtype: GgmlDType::I8,
                data: weight.clone(),
            },
            GgufEntry {
                name: "blk.1.engram_embd.scale".to_string(),
                dims: vec![rows, per_row],
                dtype: GgmlDType::I8,
                data: scales.clone(),
            },
        ];
        let path = std::env::temp_dir().join(format!(
            "loken-generic-engram-fp8-{}.gguf",
            std::process::id()
        ));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let source = TensorSource::one(&g.content, &g.mmap[..]);
        let got = engram_table(&source, "blk.1.engram_embd.weight").unwrap();
        match got {
            EngramTable::Fp8Rows {
                bytes,
                scales: sc,
                rows: r,
                row_len: rl,
                block,
            } => {
                assert_eq!((r, rl, block), (rows, row_len, row_len / per_row));
                assert_eq!(bytes.as_slice(), weight.as_slice());
                assert_eq!(sc.as_slice(), scales.as_slice());
            }
            _ => panic!("the pair of byte tensors was not read as released fp8 rows"),
        }
    }

    /// The engram tables of a block-scaled checkpoint are read in place as e4m3 rows beside their
    /// scales, with the geometry and bytes the V4.1 checkpoint source gave.
    #[test]
    fn checkpoint_engram_rows_are_read_as_the_v41_source_reads_them() {
        use super::super::super::block_scaled_engram_tests::{engram_checkpoint, ENGRAM_LAYERS};
        let dir = engram_checkpoint("tables");
        let ck = BlockScaledCheckpoint::open(&dir).unwrap();
        for l in ENGRAM_LAYERS {
            let got = checkpoint_engram_table(&ck, &format!("blk.{l}.engram_embd")).unwrap();
            match &got {
                EngramTable::Fp8Rows {
                    bytes,
                    scales,
                    rows,
                    row_len,
                    block,
                } => {
                    let read = format!(
                        "{:?} {} {}",
                        (rows, row_len, block),
                        frozen::digest(bytes.as_slice()),
                        frozen::digest(scales.as_slice())
                    );
                    frozen::expect("units", &format!("checkpoint table {l}"), &read);
                }
                _ => panic!("layer {l}: the table is not read as fp8 rows"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The hash layout of a block-scaled checkpoint, its token map built from the shipped
    /// tokenizer, is the layout the V4.1 checkpoint loader derived from the same directory.
    /// A checkpoint whose token map cannot be built keeps no layout, as the V4.1 loader kept
    /// none when its token map builder failed.
    #[test]
    fn the_checkpoint_engram_layout_is_the_v41_layout() {
        use super::super::super::block_scaled_engram_tests::engram_checkpoint;
        use super::super::super::ngram::token_map;
        let dir = engram_checkpoint("layout");
        let ck = BlockScaledCheckpoint::open(&dir).unwrap();
        let got = checkpoint_engram(&ck).expect("a layout");
        let fields = (
            &got.layer_ids,
            got.heads,
            got.head_dim,
            got.max_ngram,
            &got.token_map,
            got.pad_id,
            &got.multipliers,
            &got.primes,
            &got.offsets,
        );
        frozen::expect_debug("units", "checkpoint layout", &fields);
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        assert!(checkpoint_engram(&ck).is_none());
        assert!(token_map::build_from_file(&dir.join("tokenizer.json")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
