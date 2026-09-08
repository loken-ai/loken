//! HeteroZImage: Multi-device Z-Image transformer pipeline
//!
//! Splits the 34 Z-Image transformer blocks (2 noise refiner + 2 context refiner + 30 main)
//! across CUDA, OpenCL (Arc), and CPU devices using HeteroPlan.
//!
//! CUDA/CPU blocks run the substrate ZImageTransformerBlock directly.
//! OpenCL blocks use custom OpenCL kernels (F32 matmul, attention, etc.) via OpenCLZImageBlock.

#![allow(clippy::too_many_arguments)]

use crate::inference::model::zimage::dit::{
    create_coordinate_grid, patch_grid, patchify, stem, unpatchify, Config, FinalLayer,
    RopeEmbedder, TimestepEmbedder, Vb, ZImageTransformerBlock,
};
use crate::tensor::layer::RmsNorm;
use crate::tensor::layer::{linear, rms_norm, Linear};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};
use tracing::info;

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};

#[cfg(feature = "opencl")]
use crate::inference::kernel::opencl::OpenCLPipelines;
#[cfg(feature = "opencl")]
use crate::inference::model::zimage::opencl::{OclZImageScratch, OpenCLZImageBlock};
#[cfg(feature = "opencl")]
use std::sync::Arc;

/// Segment kind for dispatch
enum SegmentKind {
    /// CUDA or CPU - tensor operations on the substrate
    Native,
    /// OpenCL (Arc) - use custom OpenCL kernels
    #[cfg(feature = "opencl")]
    OpenCL,
}

/// Pre-computed segment boundaries for efficient forward dispatch
struct ZImageSegment {
    device: Device,
    dtype: DType,
    kind: SegmentKind,
    start: usize,
    end: usize, // exclusive
}

/// Multi-device Z-Image transformer
pub struct HeteroZImage {
    // Embeddings (on primary device - small)
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: Linear,
    x_embedder: Linear,
    // x_pad_token + cap_pad_token tensors are present in every Z-Image
    // checkpoint but the forward path never references them. Skipping
    // the load saves two F-dtype VarBuilder.get() calls + the on-device
    // alloc per HeteroZImage construction.

    // Blocks with per-device assignment (the substrate path)
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    main_layers: Vec<ZImageTransformerBlock>,

    // OpenCL blocks (indexed by layer number, only populated for OpenCL segments)
    #[cfg(feature = "opencl")]
    ocl_blocks: Vec<Option<OpenCLZImageBlock>>,
    #[cfg(feature = "opencl")]
    ocl_scratch: Option<OclZImageScratch>,
    #[cfg(feature = "opencl")]
    ocl_pipelines: Option<Arc<OpenCLPipelines>>,

    // Segments for main layers (pre-computed boundaries)
    main_segments: Vec<ZImageSegment>,

    // Final layer (on primary device)
    final_layer: FinalLayer,

    // RoPE (pre-computed, device-agnostic - will be moved per-segment)
    rope_embedder: RopeEmbedder,

    // Config
    cfg: Config,
    primary_device: Device,
    primary_dtype: DType,

    pub plan: HeteroPlan,

    // Per-image cache for tensors that don't change across denoise steps.
    // Built fresh on first forward when the (b, f, h, w, text_len) key
    // changes; reused for every subsequent step within one image gen.
    forward_cache: std::sync::Mutex<Option<HeteroZImagePerImageCache>>,
}

/// Cloning an entry shares every tensor's storage rather than copying it, which is what lets a
/// hit be handed out and kept in the same breath.
#[derive(Clone)]
struct HeteroZImagePerImageCache {
    // Includes cap_feats.id() + cap_mask.id() - cap_refined and the
    // attention masks depend on content, not just shape. Without those
    // two prompts with same text_len would corrupt each other's cache.
    key: (
        usize,
        usize,
        usize,
        usize,
        usize,
        crate::tensor::TensorId,
        crate::tensor::TensorId,
    ),
    x_cos: Tensor,
    x_sin: Tensor,
    // Pre-run context_refiner output - deterministic per image because
    // adaln_input is None for context-refiner blocks.
    cap_refined: Tensor,
    x_attn_mask: Tensor,
    unified_cos: Tensor,
    unified_sin: Tensor,
    unified_attn_mask: Tensor,
    /// True iff `cap_mask` has any zeros, i.e. the caption was shorter
    /// than its padded budget. When false, every attention mask in
    /// play is all-ones - the (m-1)*1e9 + broadcast_add pass in
    /// `attention_basic` becomes a no-op, and we save 4 kernel launches
    /// per attention by passing `None` to the block forwards.
    mask_has_padding: bool,
}

impl HeteroZImage {
    /// Build HeteroZImage from safetensors files with per-device block placement.
    ///
    /// The `plan` assigns the 30 main layers across devices. Noise/context refiners
    /// and embeddings always go on the primary device (they're small and called once).
    pub fn from_safetensors(
        safetensors_files: &[&str],
        name_prefix: Option<&str>,
        cfg: &Config,
        plan: &HeteroPlan,
        primary_device: &Device,
        primary_dtype: DType,
        #[cfg(feature = "opencl")] ocl_pipelines: Option<Arc<OpenCLPipelines>>,
    ) -> Result<Self> {
        let total_main = cfg.n_layers;

        // Create one VarBuilder per unique CUDA device that the plan
        // touches (plus the primary device + CPU fallback). Older logic
        // collapsed every CUDA segment to `primary_device`, which meant
        // a plan like {Cuda(0): 0-14, Cuda(1): 15-29} silently loaded
        // ALL 30 blocks onto GPU0 - defeating multi-GPU. The map below
        // keeps a real per-device VarBuilder so each segment lands on
        // the device the planner picked.
        // Ray all-in-one checkpoints scope the S3-DiT under `model.diffusion_model.`;
        // the official shards use bare names. One optional prefix covers both.
        let scoped = |vb: VarBuilder| -> VarBuilder {
            match name_prefix {
                Some(p) => vb.pp(p),
                None => vb,
            }
        };
        let vb_primary = scoped(unsafe {
            VarBuilder::from_files(safetensors_files, primary_dtype, primary_device)?
        });

        let cpu_dtype = DType::F32;
        let vb_cpu = if !primary_device.is_cpu() {
            Some(scoped(unsafe {
                VarBuilder::from_files(safetensors_files, cpu_dtype, &Device::Cpu)?
            }))
        } else {
            None
        };

        // Per-secondary-CUDA-device builders (lazy on first segment that
        // needs them). Keys are CUDA device indices NOT matching
        // primary_device. mmapping is virtual - no extra weight bytes
        // are read from disk per device; only per-tensor copy on demand.
        let primary_cuda_idx: Option<usize> = match primary_device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
            _ => None,
        };
        let mut vb_cuda_by_idx: std::collections::HashMap<usize, (Device, VarBuilder)> =
            std::collections::HashMap::new();
        for seg in &plan.segments {
            if let DeviceKind::Cuda(idx) = seg.kind {
                if Some(idx) == primary_cuda_idx {
                    continue;
                }
                if vb_cuda_by_idx.contains_key(&idx) {
                    continue;
                }
                let dev = Device::new_cuda(idx)?;
                let vb = scoped(unsafe {
                    VarBuilder::from_files(safetensors_files, primary_dtype, &dev)?
                });
                vb_cuda_by_idx.insert(idx, (dev, vb));
            }
        }

        // 1. Build embeddings on primary device
        info!("HeteroZImage: building embeddings on {:?}", primary_device);
        // THE ORDER THIS LOADER READS THE STEM IN IS ITS OWN - the way back out comes after
        // the main stack below, where the single-device loader takes it before the refiners.
        // Each card's ledger, and so the placement chosen against it, was measured from the
        // order its tensors arrive, so the pieces are taken from `stem` one at a time and
        // left exactly where this loader already took them.
        //
        // Wrapping the primary builder once: `Vb` walks a prefix, it does not read a tensor,
        // and the underlying loader is shared rather than reopened.
        let stem_vb = Vb::Dense(vb_primary.clone());
        let t_embedder = stem::timestep_embedder(cfg, &stem_vb)?;

        // The caption embedder is read DENSE here, at the builder's dtype, where the
        // single-device loader stages it host-side into an exact-F32 bias and a transposed
        // weight. Same tensors, different arithmetic - so the two loads stay apart.
        let cap_embedder_norm = rms_norm(
            cfg.cap_feat_dim,
            cfg.norm_eps as f32,
            &vb_primary.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear = linear(
            cfg.cap_feat_dim,
            cfg.dim,
            &vb_primary.pp("cap_embedder").pp("1"),
        )?;

        let patch_dim = cfg.patch_dim();
        let x_embedder = stem::by_dialect(&vb_primary, "x_embedder", |vb| {
            linear(patch_dim, cfg.dim, &vb)
        })?;

        // 2. Build refiners on primary device
        info!("HeteroZImage: building refiners on {:?}", primary_device);
        let (noise_refiner, context_refiner) = stem::refiners(cfg, &stem_vb)?;

        // 3. Build main layers on their assigned devices per plan
        info!(
            "HeteroZImage: building {} main layers across {} segments",
            total_main,
            plan.segments.len()
        );

        let mut main_layers = Vec::with_capacity(total_main);
        let mut main_segments = Vec::new();
        #[cfg(feature = "opencl")]
        let mut ocl_blocks: Vec<Option<OpenCLZImageBlock>> =
            (0..total_main).map(|_| None).collect();
        #[cfg(feature = "opencl")]
        let mut has_ocl_segments = false;

        // For OpenCL segments, we need raw safetensors access
        #[cfg(feature = "opencl")]
        let raw_st_data: Option<Vec<Vec<u8>>> = if ocl_pipelines.is_some() {
            // Read all safetensors files for direct raw loading
            let mut data = Vec::new();
            for f in safetensors_files {
                data.push(
                    std::fs::read(f)
                        .map_err(|e| crate::tensor::Error::msg(format!("Read {}: {}", f, e)))?,
                );
            }
            Some(data)
        } else {
            None
        };

        for seg in &plan.segments {
            let is_opencl = matches!(seg.kind, DeviceKind::OpenCL(_));

            #[cfg(feature = "opencl")]
            if let (true, Some(pipelines)) = (is_opencl, ocl_pipelines.as_ref()) {
                info!(
                    "  Segment OpenCL: layers {}-{} on Arc GPU",
                    seg.layer_start,
                    seg.layer_end - 1
                );

                // Load blocks from raw safetensors into OpenCL buffers
                // Deserialize all shards so we can find tensors that span shard boundaries
                if let Some(ref st_data) = raw_st_data {
                    let head_dim = cfg.head_dim();
                    let hidden_dim = cfg.hidden_dim();
                    let all_shards: Vec<safetensors::SafeTensors<'_>> = st_data
                        .iter()
                        .map(|data| safetensors::SafeTensors::deserialize(data))
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(|e| {
                            crate::tensor::Error::msg(format!("SafeTensors deserialize: {}", e))
                        })?;

                    // i needs to be a literal for format!("layers.{i}") and the
                    // log message; index loop is the natural shape.
                    #[allow(clippy::needless_range_loop)]
                    for i in seg.layer_start..seg.layer_end {
                        let prefix = format!("layers.{}", i);
                        let block = OpenCLZImageBlock::from_safetensors(
                            &all_shards,
                            &prefix,
                            cfg.dim,
                            hidden_dim,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            head_dim,
                            cfg.norm_eps,
                            true, // main layers have modulation
                            cfg.adaln_dim(),
                            pipelines,
                        )
                        .map_err(|e| {
                            crate::tensor::Error::msg(format!("OpenCL block {}: {}", i, e))
                        })?;
                        ocl_blocks[i] = Some(block);
                    }
                }

                // A placeholder block for the main_layers vec (never called for OpenCL segments)
                // We need the vec to be the right size, but OpenCL segments use ocl_blocks instead
                let dummy_vb = vb_cpu.as_ref().unwrap_or(&vb_primary);
                for i in seg.layer_start..seg.layer_end {
                    main_layers.push(ZImageTransformerBlock::new(
                        cfg,
                        true,
                        Vb::Dense(dummy_vb.pp("layers").pp(i)),
                    )?);
                }

                main_segments.push(ZImageSegment {
                    device: Device::Cpu, // placeholder, not used for OpenCL
                    dtype: DType::F32,
                    kind: SegmentKind::OpenCL,
                    start: seg.layer_start,
                    end: seg.layer_end,
                });
                has_ocl_segments = true;
                continue;
            }

            // The substrate path (CUDA or CPU).
            //
            // The per-device VarBuilder/Device resolution lives in the
            // match below - picking from the pre-built vb_cuda_by_idx
            // map so multi-CUDA segments land on the right physical
            // device (3e3208b). The previous standalone seg_device
            // binding was unused after that refactor.

            // Resolve the VarBuilder + actual Device for this segment.
            // Multi-CUDA case: the planner may have assigned blocks to
            // Cuda(1), Cuda(2), etc. - use the per-device VarBuilder
            // built above so each segment actually lands on its planned
            // device, not silently collapsed onto primary_device.
            let (seg_vb, actual_device, seg_dtype): (&VarBuilder, Device, DType) = match seg.kind {
                DeviceKind::Cuda(idx) if Some(idx) == primary_cuda_idx => {
                    (&vb_primary, primary_device.clone(), primary_dtype)
                }
                DeviceKind::Cuda(idx) => {
                    let (dev, vb) = vb_cuda_by_idx
                        .get(&idx)
                        .expect("vb_cuda_by_idx populated for every plan CUDA idx above");
                    (vb, dev.clone(), primary_dtype)
                }
                DeviceKind::Cpu => {
                    let vb = vb_cpu.as_ref().unwrap_or(&vb_primary);
                    (vb, Device::Cpu, cpu_dtype)
                }
                DeviceKind::OpenCL(_) => {
                    // OpenCL fall-back path (no pipelines): use CPU.
                    let vb = vb_cpu.as_ref().unwrap_or(&vb_primary);
                    (vb, Device::Cpu, cpu_dtype)
                }
            };

            info!(
                "  Segment {:?}: layers {}-{} on {:?} ({:?})",
                seg.kind,
                seg.layer_start,
                seg.layer_end - 1,
                actual_device,
                seg_dtype
            );

            main_segments.push(ZImageSegment {
                device: actual_device.clone(),
                dtype: seg_dtype,
                kind: SegmentKind::Native,
                start: seg.layer_start,
                end: seg.layer_end,
            });

            for i in seg.layer_start..seg.layer_end {
                main_layers.push(ZImageTransformerBlock::new(
                    cfg,
                    true,
                    Vb::Dense(seg_vb.pp("layers").pp(i)),
                )?);
            }
        }

        // 4. Build OpenCL scratch buffers if needed
        #[cfg(feature = "opencl")]
        let ocl_scratch = if has_ocl_segments {
            if let Some(ref pipelines) = ocl_pipelines {
                let head_dim = cfg.head_dim();
                let hidden_dim = cfg.hidden_dim();
                // Max sequence length for image generation (1024x1024 -> ~4096 tokens + text)
                let max_seq_len = 8192;
                Some(
                    OclZImageScratch::new(
                        max_seq_len,
                        cfg.dim,
                        hidden_dim,
                        cfg.n_heads,
                        head_dim,
                        cfg.n_kv_heads,
                        pipelines,
                    )
                    .map_err(|e| crate::tensor::Error::msg(format!("OclZImageScratch: {}", e)))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // 5. Build final layer on primary device - read here, after the main stack, which is
        // where this loader has always read it.
        let final_layer = stem::final_layer(cfg, &stem_vb)?;

        // 6. Build RoPE embedder
        let rope_embedder = stem::rope_embedder(cfg)?;

        info!(
            "HeteroZImage: ready ({} noise_refiner + {} context_refiner + {} main layers)",
            noise_refiner.len(),
            context_refiner.len(),
            main_layers.len()
        );

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            noise_refiner,
            context_refiner,
            main_layers,
            #[cfg(feature = "opencl")]
            ocl_blocks,
            #[cfg(feature = "opencl")]
            ocl_scratch,
            #[cfg(feature = "opencl")]
            ocl_pipelines,
            main_segments,
            final_layer,
            rope_embedder,
            cfg: cfg.clone(),
            primary_device: primary_device.clone(),
            primary_dtype,
            plan: plan.clone(),
            forward_cache: std::sync::Mutex::new(None),
        })
    }

    fn build_forward_cache(
        &self,
        b: usize,
        img_seq_len: usize,
        f_tokens: usize,
        h_tokens: usize,
        w_tokens: usize,
        text_len: usize,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<HeteroZImagePerImageCache> {
        // Built host-side, like the rope tables: the grid is indices, the lookup that
        // consumes them bounces through the host on CUDA anyway, and on a split plan a
        // device-resident copy would be one per card for no gain.
        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0))?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        let cap_normed = self.cap_embedder_norm.forward(cap_feats)?;
        let cap_embedded = cap_normed.apply(&self.cap_embedder_linear)?;

        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0))?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, &self.primary_device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        // Check whether the caption mask has any padding zeros. For a
        // prompt that exactly fills its padded budget (or whose
        // pad-multiple is 1) every mask in play is all-ones and the
        // per-attention mask broadcast becomes a no-op - we can pass
        // None to skip it without losing correctness.
        let mask_has_padding = {
            let v: Vec<u8> = cap_attn_mask.flatten_all()?.to_vec1::<u8>()?;
            v.contains(&0)
        };

        // 8 (hoisted). context_refiner is deterministic when adaln_input
        // is None - run once and store fully-refined cap. Skipped on
        // every subsequent denoise step.
        let mut cap_refined = cap_embedded;
        let cap_mask_arg = if mask_has_padding {
            Some(&cap_attn_mask)
        } else {
            None
        };
        for db in &self.context_refiner {
            cap_refined = db.forward(&cap_refined, cap_mask_arg, &cap_cos, &cap_sin, None)?;
        }

        // A rope row is read from its own position id and from nothing else, so the tables for
        // the joined sequence are the two halves' tables joined in the same order the streams
        // are. Taking them that way costs one concatenation instead of a second lookup over the
        // whole sequence, and the id vectors never have to be joined at all.
        let unified_cos = Tensor::cat(&[&x_cos, &cap_cos], 0)?;
        let unified_sin = Tensor::cat(&[&x_sin, &cap_sin], 0)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        Ok(HeteroZImagePerImageCache {
            key: (
                b,
                f_tokens,
                h_tokens,
                w_tokens,
                text_len,
                cap_feats.id(),
                cap_mask.id(),
            ),
            x_cos,
            x_sin,
            cap_refined,
            x_attn_mask,
            unified_cos,
            unified_sin,
            unified_attn_mask,
            mask_has_padding,
        })
    }

    /// Run a stretch of main layers, in order, on the device they were placed on.
    ///
    /// `sync` is the device to wait on after each block, and it is given ONLY when the stack is
    /// split. The blocks form a sequential chain (N+1 consumes N's output), so there is no
    /// inter-block parallelism to lose - but without the barrier the CPU races ahead and the
    /// caching allocator can hand a block's just-"freed" output buffer to the next block's
    /// allocation before the card has finished reading it. That use-after-free surfaces as a
    /// delayed CUDA_ERROR_LAUNCH_FAILED on a later kernel (seen as "affine launch"), and only
    /// without CUDA_LAUNCH_BLOCKING. One cheap sync per block closes the race; a single-segment
    /// stack has no cross-device traffic to race with and pays nothing.
    fn run_main_layers(
        &self,
        mut unified: Tensor,
        span: std::ops::Range<usize>,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        sync: Option<&Device>,
    ) -> Result<Tensor> {
        for i in span {
            let _timer =
                crate::inference::place::layer_perf::LayerTimer::start_for(i, "z-image-turbo");
            unified = self.main_layers[i].forward(&unified, mask, cos, sin, Some(adaln))?;
            if let Some(dev) = sync {
                let _ = dev.synchronize();
            }
        }
        Ok(unified)
    }

    /// Forward pass with multi-device dispatch.
    /// Synchronize the primary device and every unique CUDA segment device.
    /// Used as a per-step barrier to bound the async memory peak (see the
    /// call site in `forward`). Errors are ignored - a sync failure here is
    /// not itself fatal and any real device fault surfaces on the next op.
    fn synchronize_all_devices(&self) {
        let _ = self.primary_device.synchronize();
        for seg in &self.main_segments {
            if !same_device(&seg.device, &self.primary_device) {
                let _ = seg.device.synchronize();
            }
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, _c, f, h, w) = x.dims5()?;
        // The geometry this forward cuts in and must come back as, read once here rather
        // than reached for again at the far end of the stack.
        let (patch_size, f_patch_size, channels) = (
            self.cfg.all_patch_size[0],
            self.cfg.all_f_patch_size[0],
            self.cfg.in_channels,
        );

        // Bound the async memory peak (multi-GPU robustness). Every op in
        // this forward is issued async on per-device streams; without a
        // barrier the caching allocator can hold many in-flight buffers
        // across devices at once, so the peak far exceeds the serialized
        // footprint - which OOMs (cublas workspace ALLOC_FAILED at the
        // first matmul, or a delayed CUDA_ERROR_LAUNCH_FAILED on the next
        // kernel) exactly where CUDA_LAUNCH_BLOCKING=1 succeeds. Syncing
        // every device at the top of each step flushes the previous step's
        // (and, on the first step, the load's) async allocations before
        // this step allocates - matching the serialized footprint at the
        // cost of one barrier per step (4 total), not per kernel.
        self.synchronize_all_devices();

        // Ensure input is on primary device
        let x = x.to_device(&self.primary_device)?;
        let t = t
            .to_device(&self.primary_device)?
            .to_dtype(self.primary_dtype)?;
        let cap_feats = cap_feats
            .to_device(&self.primary_device)?
            .to_dtype(self.primary_dtype)?;
        let cap_mask = cap_mask.to_device(&self.primary_device)?;

        // 1. Timestep embedding (primary device)
        let t_scaled = (&t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        // 2. Patchify and embed image (primary device)
        let (x_patches, orig_size) = patchify(&x, patch_size, f_patch_size)?;
        let mut x_emb = x_patches.apply(&self.x_embedder)?;
        let img_seq_len = x_emb.dim(1)?;

        // 3-6 + 10. Per-image-constant tensors. Same key -> same contents;
        // built fresh only when the cache key changes.
        // The same cut `patchify` above made, asked of the same function, so the coordinates
        // this keys and builds cannot come to describe a different grid than the one cut.
        let (f_tokens, h_tokens, w_tokens) = patch_grid((f, h, w), patch_size, f_patch_size);
        let text_len = cap_feats.dim(1)?;
        let cache_key = (
            b,
            f_tokens,
            h_tokens,
            w_tokens,
            text_len,
            cap_feats.id(),
            cap_mask.id(),
        );

        let cached = {
            let mut guard = self
                .forward_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // The entry is taken by its key, not by whatever was there: a miss REPLACES the
            // stored entry, so the tensors the previous image left behind are dropped here
            // rather than held for the length of the next one.
            let hit = guard.as_ref().filter(|c| c.key == cache_key).cloned();
            match hit {
                Some(c) => c,
                None => {
                    let built = self.build_forward_cache(
                        b,
                        img_seq_len,
                        f_tokens,
                        h_tokens,
                        w_tokens,
                        text_len,
                        &cap_feats,
                        &cap_mask,
                    )?;
                    *guard = Some(built.clone());
                    built
                }
            }
        };
        let x_cos = cached.x_cos;
        let x_sin = cached.x_sin;
        let cap = cached.cap_refined;
        let x_attn_mask = cached.x_attn_mask;
        let unified_cos = cached.unified_cos;
        let unified_sin = cached.unified_sin;
        let unified_attn_mask = cached.unified_attn_mask;
        let mask_has_padding = cached.mask_has_padding;

        // 7. Noise refiner (primary device, 2 blocks). The image stream
        // has no padding tokens - x_attn_mask is always all-ones in
        // build_forward_cache - so the mask add is a no-op (0 added to
        // attn weights). With `use_accelerated_attn=false` set in the
        // Z-Image config, passing None skips the 4 kernel launches per
        // attention spent on (m-1)*1e9 + broadcast_add without
        // triggering the broken flash-attn fallback path.
        let _ = &x_attn_mask;
        for db in &self.noise_refiner {
            x_emb = db.forward(&x_emb, None, &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 8. (cached) Context refiner output already in `cap`.

        // 9. Concatenate image and text
        let mut unified = Tensor::cat(&[&x_emb, &cap], 1)?;
        drop(x_emb);
        drop(cap);

        // 11. Main transformer layers - segment-based processing with device transfers.
        // When `mask_has_padding == false` every unified mask entry is 1, so
        // passing None to the block forwards skips the no-op
        // (m-1)*1e9 + broadcast_add chain in attention_basic (4 launches/attn).
        let unified_mask_arg = if mask_has_padding {
            Some(&unified_attn_mask)
        } else {
            None
        };
        if self.main_segments.len() <= 1
            && matches!(
                self.main_segments.first().map(|s| &s.kind),
                Some(SegmentKind::Native) | None
            )
        {
            // A single substrate segment - no transfers needed
            unified = self.run_main_layers(
                unified,
                0..self.main_layers.len(),
                unified_mask_arg,
                &unified_cos,
                &unified_sin,
                &adaln_input,
                None,
            )?;
        } else {
            // Multi-segment - transfer unified tensor at boundaries
            let mut cached_cos: Option<(Device, Tensor)> = None;
            let mut cached_sin: Option<(Device, Tensor)> = None;
            let mut cached_mask: Option<(Device, Tensor)> = None;
            let mut cached_adaln: Option<(Device, Tensor)> = None;

            for seg in &self.main_segments {
                match seg.kind {
                    SegmentKind::Native => {
                        // Transfer unified tensor to segment device/dtype if needed
                        if !same_device(&unified.device(), &seg.device)
                            || unified.dtype() != seg.dtype
                        {
                            unified = unified.to_device(&seg.device)?.to_dtype(seg.dtype)?;
                        }

                        let seg_cos =
                            get_or_cache(&unified_cos, &seg.device, seg.dtype, &mut cached_cos)?;
                        let seg_sin =
                            get_or_cache(&unified_sin, &seg.device, seg.dtype, &mut cached_sin)?;
                        let seg_mask = if mask_has_padding {
                            Some(get_or_cache(
                                &unified_attn_mask,
                                &seg.device,
                                DType::U8,
                                &mut cached_mask,
                            )?)
                        } else {
                            None
                        };
                        let seg_adaln =
                            get_or_cache(&adaln_input, &seg.device, seg.dtype, &mut cached_adaln)?;

                        unified = self.run_main_layers(
                            unified,
                            seg.start..seg.end,
                            seg_mask.as_ref(),
                            &seg_cos,
                            &seg_sin,
                            &seg_adaln,
                            Some(&seg.device),
                        )?;
                    }
                    #[cfg(feature = "opencl")]
                    SegmentKind::OpenCL => {
                        unified = self.forward_opencl_segment(
                            unified,
                            seg,
                            &unified_cos,
                            &unified_sin,
                            &unified_attn_mask,
                            &adaln_input,
                        )?;
                    }
                }
            }

            // Move back to primary device/dtype for final layer
            if !same_device(&unified.device(), &self.primary_device)
                || unified.dtype() != self.primary_dtype
            {
                unified = unified
                    .to_device(&self.primary_device)?
                    .to_dtype(self.primary_dtype)?;
            }
        }

        // 12. Extract image portion and final layer (primary device)
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;

        // 13. Unpatchify
        unpatchify(&x_out, orig_size, patch_size, f_patch_size, channels)
    }

    /// Forward pass for an OpenCL segment.
    ///
    /// Uses zero-copy buffers (CL_MEM_USE_HOST_PTR) on integrated GPUs with shared memory.
    /// CPU-side `Vec<f32>` is directly accessible by the iGPU without any data transfer.
    #[cfg(feature = "opencl")]
    fn forward_opencl_segment(
        &self,
        unified: Tensor,
        seg: &ZImageSegment,
        unified_cos: &Tensor,
        unified_sin: &Tensor,
        _unified_attn_mask: &Tensor,
        adaln_input: &Tensor,
    ) -> Result<Tensor> {
        use opencl3::memory::{CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR};

        let pipelines = self
            .ocl_pipelines
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("OpenCL pipelines not initialized"))?;
        let scratch = self
            .ocl_scratch
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("OpenCL scratch not initialized"))?;

        let (b, seq_len, dim) = unified.dims3()?;
        if b != 1 {
            return Err(crate::tensor::Error::msg(
                "OpenCL Z-Image only supports batch=1",
            ));
        }

        // Convert tensor to contiguous F32 on CPU
        let unified_f32 = unified.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let mut unified_data = unified_f32.flatten_all()?.to_vec1::<f32>()?;

        let cos_f32 = unified_cos.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let sin_f32 = unified_sin.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let adaln_f32 = adaln_input.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let cos_data = cos_f32.flatten_all()?.to_vec1::<f32>()?;
        let sin_data = sin_f32.flatten_all()?.to_vec1::<f32>()?;
        let adaln_data = adaln_f32.flatten_all()?.to_vec1::<f32>()?;

        // Zero-copy: wrap CPU memory as OpenCL buffers (CL_MEM_USE_HOST_PTR)
        // On integrated GPU (shared memory), this avoids ALL data transfers.
        // On discrete GPU, the driver handles migration transparently.
        let x_buf = unsafe {
            opencl3::memory::Buffer::<u8>::create(
                &pipelines.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                unified_data.len() * 4,
                unified_data.as_mut_ptr() as *mut std::ffi::c_void,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("x zero-copy: {}", e)))?
        };
        let cos_buf = unsafe {
            opencl3::memory::Buffer::<u8>::create(
                &pipelines.context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                cos_data.len() * 4,
                cos_data.as_ptr() as *mut std::ffi::c_void,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("cos zero-copy: {}", e)))?
        };
        let sin_buf = unsafe {
            opencl3::memory::Buffer::<u8>::create(
                &pipelines.context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                sin_data.len() * 4,
                sin_data.as_ptr() as *mut std::ffi::c_void,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("sin zero-copy: {}", e)))?
        };
        let adaln_ocl_buf = unsafe {
            opencl3::memory::Buffer::<u8>::create(
                &pipelines.context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                adaln_data.len() * 4,
                adaln_data.as_ptr() as *mut std::ffi::c_void,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("adaln zero-copy: {}", e)))?
        };

        // Run each OpenCL block - x_buf is read/written in-place via shared memory
        for i in seg.start..seg.end {
            let _timer =
                crate::inference::place::layer_perf::LayerTimer::start_for(i, "z-image-turbo");
            if let Some(ref ocl_block) = self.ocl_blocks[i] {
                ocl_block
                    .forward(
                        &x_buf,
                        seq_len,
                        &cos_buf,
                        &sin_buf,
                        Some(&adaln_ocl_buf),
                        None, // mask not used for main layers with AdaLN
                        scratch,
                        pipelines,
                    )
                    .map_err(|e| {
                        crate::tensor::Error::msg(format!("OpenCL block {} forward: {}", i, e))
                    })?;
            } else {
                return Err(crate::tensor::Error::msg(format!(
                    "OpenCL block {} not loaded but segment expects it",
                    i
                )));
            }
        }

        // Ensure all kernels complete before reading CPU memory
        pipelines
            .queue
            .finish()
            .map_err(|e| crate::tensor::Error::msg(format!("OpenCL finish: {}", e)))?;

        // unified_data is already updated in-place (shared memory) - create tensor directly
        let result = Tensor::from_slice(&unified_data, (1, seq_len, dim), &Device::Cpu)?;

        Ok(result)
    }
}

/// Check if two devices are the same. One declaration, in `hetero_place`, because a
/// stack that answers "did the device change" its own way is a stack that can answer it
/// differently from the others.
use crate::inference::place::plan::same_device;

/// Get a tensor on the target device and dtype, caching the transfer.
fn get_or_cache(
    src: &Tensor,
    target_device: &Device,
    target_dtype: DType,
    cache: &mut Option<(Device, Tensor)>,
) -> Result<Tensor> {
    if same_device(&src.device(), target_device) && src.dtype() == target_dtype {
        return Ok(src.clone());
    }
    if let Some((ref cached_dev, ref cached_tensor)) = cache {
        if same_device(cached_dev, target_device) && cached_tensor.dtype() == target_dtype {
            return Ok(cached_tensor.clone());
        }
    }
    let transferred = src.to_device(target_device)?.to_dtype(target_dtype)?;
    *cache = Some((target_device.clone(), transferred.clone()));
    Ok(transferred)
}
