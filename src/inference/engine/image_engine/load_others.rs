//! Part of `impl ImageEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl ImageEngine {
    /// Load Boogu-Image for text-to-image (Qwen3-VL-8B encoder + 3-stream MMDiT + FLUX VAE).
    /// Placement follows the same rule as the rest of the fleet: the fastest CUDA device is the
    /// primary (via `device_probe`, ranked by throughput), CPU only when no GPU is present. The
    /// DiT + encoder are loaded per generation (they do not co-fit in RAM), so `boogu_engine`
    /// streams them onto the primary in turn. `hf_models_dir` resolves the boogu/ safetensors paths.
    pub async fn load_boogu(
        &self,
        req: ModelRequest,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
        cancel: Option<crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ModelRequest {
            models_dir: hf_models_dir,
            name: requested,
        } = req;
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_state = self.model_state.clone();
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let stage = LoadStage::new("Boogu", progress_tx);
            let send = |msg: &str| stage.say(msg);
            let _counts = crate::inference::serve::progress::scoped::publish(stage.reporter());
            // The loader places the DiT and the encoder through the ladder itself; this
            // names the device the resident model reports.
            let primary =
                crate::inference::place::ladder::fastest().unwrap_or(crate::tensor::Device::Cpu);
            send("Preparing Boogu (FLUX VAE + tokenizer; DiT/encoder stream per generation)...");
            let boogu = crate::inference::engine::boogu_engine::load(
                &hf_models_dir,
                &primary,
                geom,
                cancel.as_ref(),
            )?;
            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load
                .saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name: requested,
                ckpt_id: None,
                model: LoadedImageModel::Boogu(boogu),
                device: primary,
                dtype: DType::F32,
            });
            send("Boogu ready.");
            Ok(())
        })
        .await?;
        result.map_err(std::convert::Into::into)
    }

    /// Refuse a load whose client is already gone, before a single byte is spent on it.
    ///
    /// The checks inside the loaders stop a load in PROGRESS; this one stops a load that
    /// should never have started. It matters because the first things a loader does -
    /// polling free VRAM, trimming the allocator pools, probing every card - are neither
    /// free nor local: they run before any tensor is read, and a request whose stream was
    /// dropped during the queue wait would still pay for them.
    ///
    /// `None` is every non-serving caller (the render binaries), for which nothing ever
    /// cancels.
    pub(super) fn refuse_if_abandoned(
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match cancel {
            Some(c) => Ok(c.bail()?),
            None => Ok(()),
        }
    }

    /// Load an SDXL single-file checkpoint (UNet + both CLIP towers + the SD VAE).
    ///
    /// `checkpoint` is the full path: the SDXL family ships one file per model, so
    /// unlike the other engines there is nothing to resolve inside a repo. Placement
    /// is decided per component inside the pipeline (hot UNet first, one-shot text
    /// towers on what remains).
    pub async fn load_sdxl(
        &self,
        checkpoint: std::path::PathBuf,
        name: String,
        tokenizer_json: std::path::PathBuf,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.load_sdxl_cancellable(checkpoint, name, tokenizer_json, geom, progress_tx, None)
            .await
    }

    /// [`Self::load_sdxl`] with cooperative cancellation, the contract the Flux /
    /// Qwen-Image / Boogu loaders already take: the caller holds a `CancelGuard` over
    /// the load and this stops at its next check when the guard fires.
    ///
    /// The token is PUBLISHED for the load rather than threaded into the three
    /// component loaders (UNet, text towers, VAE), which build their own weight
    /// readers several call levels down. Every safetensors tensor of every component
    /// is read through one function that consults the published token, so the grain is
    /// one tensor - see `cancel::scoped` and `SafeTensorsLoader::load`.
    pub async fn load_sdxl_cancellable(
        &self,
        checkpoint: std::path::PathBuf,
        name: String,
        tokenizer_json: std::path::PathBuf,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
        cancel: Option<crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Self::refuse_if_abandoned(cancel.as_ref())?;
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_state = self.model_state.clone();
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let stage = LoadStage::new("SDXL", progress_tx);
            let send = |msg: &str| stage.say(msg);
            let _counts = crate::inference::serve::progress::scoped::publish(stage.reporter());
            let _cancelling = cancel
                .as_ref()
                .map(crate::inference::serve::cancel::scoped::publish);
            send("Loading SDXL (UNet + CLIP-L + bigG + VAE)...");
            let pipe = crate::inference::model::sdxl::pipeline::SdxlPipeline::load(
                checkpoint
                    .to_str()
                    .ok_or_else(|| anyhow!("bad SDXL path"))?,
                tokenizer_json
                    .to_str()
                    .ok_or_else(|| anyhow!("bad tokenizer path"))?,
                // F32, because the tensor layer has no half-precision convolution:
                // conv2d's device path needs f32 slices for the input AND the kernel,
                // so a BF16 UNet - which is almost entirely convolutions - finds no
                // path at all and fails on the host. Halving it is worth doing, but it
                // is a tensor-layer change, not a dtype flip here.
                crate::tensor::DType::F32,
                geom,
            )
            .map_err(|e| anyhow!("SDXL load: {e}"))?;
            let device = pipe.device().clone();
            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load
                .saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name,
                // The family ships one file per model, so the file IS the identity.
                ckpt_id: checkpoint
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned()),
                model: LoadedImageModel::Sdxl(pipe),
                device,
                dtype: DType::F32,
            });
            send("SDXL ready.");
            Ok(())
        })
        .await?;
        result.map_err(std::convert::Into::into)
    }

    /// Load Qwen-Image for text-to-image generation (Qwen2.5-VL encoder + DiT + Wan VAE).
    /// The DiT lands on the fastest CUDA device; the encoder on a second GPU (see
    /// `qwen_image_engine::load`). `hf_models_dir` resolves the encoder/tokenizer/VAE,
    /// which every checkpoint of the family shares; `local_ckpt` is the DiT this request
    /// resolved to (a drop-in of the family), `None` loading the family's own.
    /// Load FLUX.2 Klein (Qwen3-4B conditioning + flow-match DiT + KL VAE).
    ///
    /// Placement is the engine's own adaptive plan with an OOM-fallback cascade. This family is
    /// SMALL - ~4.1 GB of DiT and ~2.3 GB of encoder resident - so it usually lands whole on one
    /// card and can sit next to another model, which the 20B families cannot.
    pub async fn load_flux2(
        &self,
        req: ModelRequest,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
        cancel: Option<crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ModelRequest {
            models_dir: hf_models_dir,
            name: model_name,
        } = req;
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_state = self.model_state.clone();
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let send = |msg: &str| {
                info!("  [FLUX.2] {}", msg);
                if let Some(ref tx) = progress_tx {
                    let _ = tx.blocking_send(LoadingProgress::Stage(msg.to_string()));
                }
            };
            // `flux2_engine::load` places its components through the ladder itself; this
            // proves a card exists and names the device the resident model reports.
            let primary = crate::inference::place::ladder::fastest()
                .ok_or_else(|| anyhow!("FLUX.2 generation requires a CUDA GPU"))?;
            send("Loading DiT, Qwen3 encoder and VAE...");
            let flux2 = crate::inference::engine::flux2_engine::load(
                &hf_models_dir,
                &primary,
                geom,
                cancel.as_ref(),
            )?;
            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load
                .saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name: model_name.clone(),
                ckpt_id: None,
                model: LoadedImageModel::Flux2(flux2),
                device: primary,
                dtype: DType::F32,
            });
            send("FLUX.2 ready.");
            Ok(())
        })
        .await?;
        result.map_err(std::convert::Into::into)
    }

    pub async fn load_qwen_image(
        &self,
        req: ModelRequest,
        local_ckpt: Option<std::path::PathBuf>,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
        cancel: Option<crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ModelRequest {
            models_dir: hf_models_dir,
            name: model_name,
        } = req;
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_state = self.model_state.clone();
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let stage = LoadStage::new("Qwen-Image", progress_tx);
            let send = |msg: &str| stage.say(msg);
            let _counts = crate::inference::serve::progress::scoped::publish(stage.reporter());
            // `qwen_image_engine::load` places its components through the ladder itself;
            // this proves a card exists and names the device the resident model reports.
            let primary = crate::inference::place::ladder::fastest()
                .ok_or_else(|| anyhow!("Qwen-Image generation requires a CUDA GPU"))?;
            send("Loading encoder (2nd GPU), DiT (primary GPU), VAE (CPU)...");
            let qwen = crate::inference::engine::qwen_image_engine::load(
                &hf_models_dir,
                &model_name,
                local_ckpt.as_deref(),
                &primary,
                geom,
                cancel.as_ref(),
            )?;
            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load
                .saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name: model_name.clone(),
                // The FILE is the identity: the family now has several checkpoints, and
                // recording which one is resident is what stops a request for one being
                // served by another's weights.
                ckpt_id: local_ckpt
                    .as_ref()
                    .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())),
                model: LoadedImageModel::QwenImage(qwen),
                device: primary,
                dtype: DType::F32,
            });
            send("Qwen-Image ready.");
            Ok(())
        })
        .await?;
        result.map_err(std::convert::Into::into)
    }

    /// Z-Image hot-component estimate for the VRAM pressure protocol: the
    /// transformer shards' total on-disk bytes. The hetero loader splits the
    /// DiT across the GPUs, so the fleet as a whole must be near-empty - and
    /// the protocol's fit test is per-GPU, so demanding the TOTAL forces the
    /// reclaim of every idle co-resident (a half-size demand passed the fit
    /// test on one card while the split still starved both). Without a real
    /// figure the image request demanded 0 bytes of headroom, no idle
    /// resident was reclaimed, and the denoise died in a cuBLAS workspace
    /// alloc.
    /// Bytes-per-element of the FIRST tensor in a safetensors file, from its header
    /// alone (no tensor data read). The Z-Image transformer ships F32 and is loaded
    /// as BF16, so its resident footprint is HALF the file - summing raw file sizes
    /// claimed 24.6 GB + reserve = 30 GB, a demand no 16 GB card can ever satisfy,
    /// which made the model unloadable whenever anything else was resident.
    pub(super) fn safetensors_elem_bytes(path: &std::path::Path) -> Option<u64> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).ok()?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).ok()?;
        let n = u64::from_le_bytes(len) as usize;
        // A header is kilobytes; refuse anything absurd rather than allocate it.
        // not-a-vram-size: a bound on a length read out of the file, not a reservation.
        if n == 0 || n > 64 << 20 {
            return None;
        }
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&buf).ok()?;
        let dtype = v
            .as_object()?
            .iter()
            .find(|(k, _)| *k != "__metadata__")
            .and_then(|(_, t)| t.get("dtype"))?
            .as_str()?
            .to_ascii_uppercase();
        Some(match dtype.as_str() {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            _ => 1,
        })
    }

    /// What a Z-Image transformer built from `files` will HOLD on a card.
    ///
    /// Resident footprint, not file size: the loader converts every tensor to BF16, so a
    /// 24.6 GB F32 export goes resident at 12.3 GB. Taken from the files the loader is
    /// about to read, so the figure that decides where the model goes is the figure the
    /// load produces.
    ///
    /// This is the ONE place that answers "what does this model weigh". The loader had
    /// its own answer, counted off the config, and it charged the feed-forward at four
    /// times the model width where this family builds it at eight thirds - three phantom
    /// gigabytes, which is precisely what decides whether a 16 GB card fits the model or
    /// has to split it. It fit at 512^2, where the request's own reserve is small, and
    /// stopped fitting at 1024^2: one size kept working while the other went hetero, then
    /// out of memory, for a model that fits one card at both.
    pub(super) fn zimage_resident_bytes(files: &[std::path::PathBuf]) -> u64 {
        let total: u64 = files
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        if total == 0 {
            return 0;
        }
        // Bytes per element of the dtype the blocks are BUILT at, against the dtype they
        // are stored at. A checkpoint already at or below it is loaded as it stands.
        const RESIDENT_ELEM: u64 = 2;
        match files.first().and_then(|p| Self::safetensors_elem_bytes(p)) {
            Some(bpe) if bpe > RESIDENT_ELEM => total * RESIDENT_ELEM / bpe,
            _ => total,
        }
    }

    /// The official transformer's shards in the HuggingFace cache, in shard order.
    ///
    /// Ordered because the list is a checkpoint identity as well as a set of files: the
    /// memo in [`zimage_dry_forward_bytes`] keys on it, and a directory walk that came
    /// back shuffled would measure the same checkpoint twice under two names.
    pub fn zimage_transformer_shards(hf_models_dir: &str) -> Vec<std::path::PathBuf> {
        let root = std::path::Path::new(hf_models_dir);
        let hub = if root.file_name().is_some_and(|n| n == "hub") {
            root.to_path_buf()
        } else {
            root.join("hub")
        };
        let snaps = hub.join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            return Vec::new();
        };
        for snap in rd.flatten() {
            let tdir = snap.path().join("transformer");
            let Ok(files) = std::fs::read_dir(&tdir) else {
                continue;
            };
            let mut shards: Vec<std::path::PathBuf> = files
                .flatten()
                .map(|f| f.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            if !shards.is_empty() {
                shards.sort();
                return shards;
            }
        }
        Vec::new()
    }

    pub fn zimage_hot_component_bytes(hf_models_dir: &str) -> u64 {
        Self::zimage_resident_bytes(&Self::zimage_transformer_shards(hf_models_dir))
    }

    /// Load Z-Image-Turbo model from HuggingFace Hub (safetensors).
    pub async fn load_z_image_turbo(
        &self,
        req: ModelRequest,
        local_transformer: Option<std::path::PathBuf>,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.load_z_image_turbo_cancellable(req, local_transformer, geom, progress_tx, None)
            .await
    }

    /// [`Self::load_z_image_turbo`] with cooperative cancellation, the contract the Flux /
    /// Qwen-Image / Boogu loaders already take: the caller holds a `CancelGuard` over the
    /// load and this stops at its next check when the guard fires.
    ///
    /// Two grains, because this load has two kinds of long phase. The weights - text
    /// encoder, transformer, both VAEs - are read through one function that consults the
    /// PUBLISHED token (see `cancel::scoped` and `SafeTensorsLoader::load`), so those stop
    /// within one tensor without any model file knowing a request exists. The downloads
    /// have no such choke point, so they are checked per shard, which is the only boundary
    /// hf_hub offers.
    pub async fn load_z_image_turbo_cancellable(
        &self,
        req: ModelRequest,
        local_transformer: Option<std::path::PathBuf>,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
        progress_tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
        cancel: Option<crate::inference::serve::cancel::CancelToken>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ModelRequest {
            models_dir: hf_models_dir,
            name: requested,
        } = req;
        Self::refuse_if_abandoned(cancel.as_ref())?;
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_state = self.model_state.clone();

        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let stage = LoadStage::new("Z-Image", progress_tx);
            let send_progress = |msg: &str| stage.say(msg);
            let _counts = crate::inference::serve::progress::scoped::publish(stage.reporter());
            let _cancelling = cancel.as_ref().map(crate::inference::serve::cancel::scoped::publish);
            // For the placement ladder below, which must not treat an abandoned load as a
            // placement failure.
            let cancelled = || cancel.as_ref().is_some_and(crate::inference::serve::cancel::CancelToken::is_cancelled);
            let bail = || -> AnyResult<()> {
                match &cancel {
                    Some(c) => c.bail().map_err(|e| anyhow!("{e}")),
                    None => Ok(()),
                }
            };

            send_progress("Initializing HuggingFace API...");
            let api = crate::inference::load::huggingface_manager::hf_api(Some(&hf_models_dir))
                .map_err(|e| {
                    error!("Failed to create HF API client: {}", e);
                    e
                })?;
            let repo = api.model("Tongyi-MAI/Z-Image-Turbo".to_string());

            // RESOLVE THE CHECKPOINT BEFORE DECIDING WHERE IT GOES.
            //
            // A Ray drop-in fine-tune arrives as ONE local all-in-one safetensors
            // (bundled layout: the S3-DiT under a `model.diffusion_model.` prefix + a
            // bundled VAE we ignore in favour of the official one); otherwise the
            // official sharded repo files. Either way these are cached, so this costs
            // nothing on a warm machine - and it is the only way the placement below can
            // ask the FILES what the model weighs, and what one forward of it holds,
            // instead of guessing both from the config.
            let tf_prefix: Option<&str> = local_transformer.as_ref().map(|_| "model.diffusion_model");
            let tf_files: Vec<std::path::PathBuf> = match &local_transformer {
                Some(p) => {
                    send_progress("Loading local Z-Image checkpoint...");
                    vec![p.clone()]
                }
                None => {
                    send_progress("Downloading transformer (~12GB, cached)...");
                    (1..=3)
                        .map(|i| -> AnyResult<std::path::PathBuf> {
                            bail()?;
                            repo.get(&format!("transformer/diffusion_pytorch_model-{:05}-of-00003.safetensors", i))
                                .map_err(|e| {
                                    error!("Failed to download transformer: {}", e);
                                    anyhow!("{e}")
                                })
                        })
                        .collect::<AnyResult<Vec<_>>>()?
                }
            };
            let tf_files_str: Vec<&str> = tf_files.iter().map(|p| p.to_str().unwrap()).collect();

            // WHAT THIS REQUEST WILL ACTUALLY ALLOCATE, from the code that will allocate
            // it: one forward of these files at this geometry, walked on a device that
            // counts instead of allocating, times the margin between that count and what
            // a card was sampled to hold. Measured ONCE per checkpoint and geometry, so
            // every later decision in this load reads the same figure for free.
            let headroom =
                zimage_runtime_demand_from_files(&tf_files_str, tf_prefix, geom.width, geom.height);

            // The cards this load may use, fastest first.
            let cuda_free = crate::inference::place::ladder::cards(0);
            for (idx, free) in &cuda_free {
                info!("Z-Image: CUDA #{} free VRAM: {:.1} GB", idx, *free as f64 / 1e9);
            }

            // What the weights weigh, beside what the forward above needs free.
            //
            // Read off the checkpoint that is about to be loaded, scaled to the dtype it
            // is loaded at - the SAME helper the admission path uses, so the gate that
            // decides a render may start and the planner that decides where its blocks go
            // cannot disagree about the size of the model they are both talking about.
            // The config fallback is only reached when the files cannot be stat'd at all,
            // and it uses the width the blocks are actually built at.
            let transformer_size_est = match Self::zimage_resident_bytes(&tf_files) {
                0 => {
                    let cfg = crate::inference::model::zimage::dit::Config::z_image_turbo();
                    let d = cfg.dim as u64;
                    // Per block: the four attention projections, then a gated
                    // feed-forward's three matrices at the width the config gives.
                    let per_layer = 4 * d * d + 3 * d * cfg.hidden_dim() as u64;
                    (cfg.n_layers + cfg.n_refiner_layers) as u64 * per_layer * DENSE_BYTES / 2
                }
                measured => measured,
            };

            // The transformer's own config, needed HERE rather than at the load below,
            // because the placement is measured by building and running this very config.
            let mut tf_config = crate::inference::model::zimage::dit::Config::z_image_turbo();
            // Z-Image flash-attn is disabled in production: the accelerated path has an
            // unresolved numerical issue at these shapes (n_heads=30, head_dim=128, BF16)
            // that renders a black image. The attention path also decides what the score
            // matrix costs, so a placement measured against the other one would not be
            // this render's.
            tf_config.set_use_accelerated_attn(false);
            let total_main_layers = tf_config.n_layers;

            // WHERE THE BLOCKS GO, ASKED OF EACH CANDIDATE PLACEMENT IN TURN.
            //
            // Each candidate is built and run on devices that COUNT instead of allocating,
            // every card's own count goes through the same margin a whole card's count goes
            // through, and the first placement whose cards all fit is kept. What comes back
            // is a reserve PER CARD, which is the only shape that can follow a plan.
            //
            // The caption encoder is part of the arrangement: one card keeps its room while
            // the stack is planned around it, every card is tried as that card, and the
            // arrangement with the fewest segments wins, the faster starting card breaking
            // the tie. If no card can hold both, the planner's estimate below decides.
            let encoder_room = zimage_text_encoder_bytes();
            let measured_placement = if cuda_free.is_empty() {
                None
            } else {
                let mut measure = |plan: &HeteroPlan| {
                    crate::inference::model::zimage::dit::dry_forward_planned(
                        &tf_config,
                        &tf_files_str,
                        tf_prefix,
                        crate::tensor::DType::BF16,
                        crate::tensor::DType::F32,
                        geom.height / FLUX_VAE_STRIDE,
                        geom.width / FLUX_VAE_STRIDE,
                        FLUX_TEXT_TOKENS,
                        plan,
                        zimage_alloc_shape(),
                    )
                    .ok()
                    .map(|d| zimage_placed_reserve(&d.load, geom.width, geom.height))
                };
                crate::inference::place::dry_plan::plan_beside(
                    total_main_layers,
                    &cuda_free,
                    encoder_room,
                    &mut measure,
                )
                .map(|(keeper, s)| {
                    info!(
                        "Z-Image: CUDA #{keeper} keeps {:.2} GB for the caption encoder while the \
                         stack is planned around it ({} segment(s))",
                        encoder_room as f64 / 1e9,
                        s.plan.segments.len(),
                    );
                    s
                })
            };
            match &measured_placement {
                Some(s) => info!(
                    "Z-Image measured placement at {}x{}: {} would hold {}, each card's own \
                     blocks beside its own forward at {} percent ({} dry runs)",
                    geom.width,
                    geom.height,
                    crate::inference::place::dry_plan::describe_plan(&s.plan),
                    s.load.describe(),
                    ZIMAGE_DRY_MARGIN_PERCENT,
                    s.measurements,
                ),
                None => info!(
                    "Z-Image measured placement at {}x{}: no arrangement of these cards holds \
                     the stack within the margin, so the planner's estimate decides",
                    geom.width, geom.height,
                ),
            }

            // 1. Tokenizer
            send_progress("Downloading tokenizer...");
            bail()?;
            let tokenizer_path = repo.get("tokenizer/tokenizer.json").map_err(|e| {
                error!("Failed to download tokenizer: {}", e);
                e
            })?;
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Z-Image tokenizer: {e}"))?;
            send_progress("Tokenizer loaded");

            // 2. Transformer (S3-DiT) - the hot component, placed first. The measured
            // placement is the first rung when there is one, the planner's estimate
            // otherwise; the ladder recovers from a card that runs out. The stem - the
            // embedders, the refiners and the way back out - is built on the card the plan
            // starts on, which is where the measurement and the estimate both charge it.
            let stem_bytes = zimage_primary_overhead(transformer_size_est);
            let demand = crate::inference::place::ladder::Demand {
                layers: total_main_layers,
                block_bytes: transformer_size_est.saturating_sub(stem_bytes),
                runtime_reserve: headroom,
                ..Default::default()
            };
            let first_plan = zimage_first_plan(
                measured_placement.as_ref(),
                &cuda_free,
                total_main_layers,
                transformer_size_est,
                headroom,
            );
            let accelerators = crate::inference::place::ladder::accelerators(0);
            send_progress(&format!(
                "Loading transformer ({})...",
                crate::inference::place::dry_plan::describe_plan(&first_plan)
            ));
            let (transformer, primary_device, hot_cards) = crate::inference::place::ladder::load_planned(
                "Z-Image transformer",
                first_plan,
                &demand,
                &cuda_free,
                &accelerators,
                cancelled,
                |plan| -> AnyResult<(ZImageVariant, Device, Vec<usize>)> {
                    use crate::inference::place::layer_executor::DeviceKind;
                    // The planner caps its host segment at what the host can give; a
                    // segment past it is a load that finishes in swap, so it is refused.
                    for seg in &plan.segments {
                        if !matches!(seg.kind, DeviceKind::Cpu) {
                            continue;
                        }
                        let blocks = seg.num_layers() as u64;
                        let need = blocks.saturating_mul(transformer_size_est / total_main_layers.max(1) as u64);
                        if need > seg.free_memory_bytes {
                            return Err(anyhow!(
                                "Z-Image would place {blocks} blocks ({:.1} GB) on the host and {:.1} GB \
                                 can be given without swapping; a node with a card that holds them is \
                                 needed",
                                need as f64 / 1e9,
                                seg.free_memory_bytes as f64 / 1e9
                            ));
                        }
                    }
                    let primary = crate::inference::place::ladder::open(
                        plan.segments.first().map_or(DeviceKind::Cpu, |s| match s.kind {
                            DeviceKind::OpenCL(_) => DeviceKind::Cpu,
                            kind => kind,
                        }),
                    )?;
                    let on_accelerator = plan.segments.iter().any(|s| matches!(s.kind, DeviceKind::OpenCL(_)));
                    if !on_accelerator {
                        // The port the measurement walked builds every plan of cards and
                        // host, so the reserve describes the arrangement that runs.
                        let model = zimage_native_from_plan(
                            plan,
                            &tf_files_str,
                            tf_prefix,
                            &tf_config,
                            &primary,
                            total_main_layers,
                        )?;
                        return Ok((ZImageVariant::NativeSingle(model), primary, crate::inference::place::ladder::cards_of(plan)));
                    }
                    // A plan naming a host-side accelerator goes to the orchestrator, the
                    // one constructor with a slot for that device.
                    #[cfg(feature = "opencl")]
                    let ocl_pipelines: Option<Arc<crate::inference::kernel::opencl::OpenCLPipelines>> =
                        crate::inference::kernel::opencl::enumerate_opencl_devices()
                            .first()
                            .and_then(|&dev_id| {
                                crate::inference::kernel::opencl::OpenCLPipelines::new(dev_id)
                                    .inspect_err(|e| error!("Failed to create OpenCL pipelines: {e}"))
                                    .ok()
                            })
                            .map(Arc::new);
                    let dtype = if primary.is_cuda() { DType::BF16 } else { DType::F32 };
                    let hetero = HeteroZImage::from_safetensors(
                        &tf_files_str,
                        tf_prefix,
                        &tf_config,
                        plan,
                        &primary,
                        dtype,
                        #[cfg(feature = "opencl")]
                        ocl_pipelines,
                    )
                    .map_err(|e| anyhow!("HeteroZImage: {e}"))?;
                    Ok((ZImageVariant::Hetero(hetero), primary, crate::inference::place::ladder::cards_of(plan)))
                },
            )?;
            let primary_dtype = if primary_device.is_cuda() { DType::BF16 } else { DType::F32 };
            send_progress("Transformer loaded");

            // Initialize layer performance tracker for Z-Image
            match &transformer {
                ZImageVariant::NativeSingle(_) => {
                    let gpu_layers = if primary_device.is_cuda() { total_main_layers } else { 0 };
                    crate::inference::place::layer_perf::global_tracker().initialize_model("z-image-turbo", total_main_layers, gpu_layers);
                }
                ZImageVariant::Hetero(hetero) => {
                    #[cfg(feature = "opencl")]
                    crate::inference::place::layer_perf::global_tracker().initialize_heterogeneous_model("z-image-turbo", &hetero.plan.segments);
                    #[cfg(not(feature = "opencl"))]
                    {
                        let total = hetero.plan.total_layers;
                        let gpu_layers = hetero.plan.segments.iter()
                            .filter(|s| matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::Cuda(_)))
                            .map(|s| s.layer_end - s.layer_start)
                            .sum();
                        crate::inference::place::layer_perf::global_tracker().initialize_model("z-image-turbo", total, gpu_layers);
                    }
                }
            }

            // 3. Text encoder (Qwen3-based). One-shot: it goes to a card the transformer
            // does not touch AT ALL, read after the transformer is resident, else the host.
            // A card the plan runs is not a card with room: what is left there is a
            // remainder, and the error of every term above it lands in it.
            send_progress("Downloading text encoder (~5GB, cached after first download)...");
            let te_files: Vec<std::path::PathBuf> = (1..=3)
                .map(|i| -> AnyResult<std::path::PathBuf> {
                    // Per SHARD: cached, this costs nothing; on a first run it is the only
                    // boundary at which a download of gigabytes can be given up.
                    bail()?;
                    repo.get(&format!("text_encoder/model-{:05}-of-00003.safetensors", i))
                        .map_err(|e| {
                            error!("Failed to download text encoder: {}", e);
                            anyhow!("{e}")
                        })
                })
                .collect::<AnyResult<Vec<_>>>()?;
            let te_est_bytes = zimage_text_encoder_bytes();
            let te_cards = crate::inference::place::ladder::aside(
                &crate::inference::place::ladder::cards(0),
                &hot_cards,
                u64::MAX,
            );
            // The host path holds the encoder in F32, twice its measured BF16 footprint,
            // and is admitted against what the host can give without swapping. Refused
            // here, the request fails in a second; admitted, it ran for an hour in swap.
            let host_need = te_est_bytes.saturating_mul(2);
            let host_budget = crate::inference::place::layer_executor::host_spill_budget_bytes();
            send_progress("Loading text encoder...");
            let te_config = crate::inference::model::zimage::text_encoder::TextEncoderConfig::z_image();
            let te_files_str: Vec<&str> = te_files.iter().map(|p| p.to_str().unwrap()).collect();
            let (text_encoder, te_device) = crate::inference::place::ladder::load_whole_on(
                "Z-Image text encoder",
                te_est_bytes,
                te_est_bytes,
                &te_cards,
                cancel.as_ref(),
                |dev| {
                    if !dev.is_cuda() && host_need > host_budget {
                        return Err(anyhow!(
                            "Z-Image text encoder needs {:.1} GB of host memory on the CPU path and \
                             {:.1} GB can be given without swapping; a node with a card that holds \
                             it, or more free host memory, is needed",
                            host_need as f64 / 1e9,
                            host_budget as f64 / 1e9
                        ));
                    }
                    // bf16 stack on a card, F32 on the host.
                    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
                    let vb = unsafe { crate::tensor::VarBuilder::from_files(&te_files_str, dtype, dev) }
                        .map_err(|e| anyhow!("z-image TE native weights: {e}"))?;
                    crate::inference::model::zimage::text_encoder::ZImageTextEncoder::new(&te_config, vb)
                        .map_err(|e| anyhow!("z-image TE native load: {e}"))
                },
            )?;
            let te_dtype = DType::F32;
            info!("Z-Image text encoder -> {te_device:?}");
            send_progress("Text encoder loaded");

            // 4. VAE, F32 (the tensor-op Z-Image VAE has no F16-promotion path on its
            // mid-block 512-channel attention). On the card holding the last transformer
            // segment when it has room, which saves the transfer of the final latent;
            // else the fastest card with room for a whole decode, else for the smallest
            // tile; else the host.
            send_progress("Downloading VAE...");
            bail()?;
            let vae_path = repo.get("vae/diffusion_pytorch_model.safetensors").map_err(|e| {
                error!("Failed to download VAE: {}", e);
                e
            })?;
            let last_card = hot_cards.last().copied();
            let (vae_need, vae_want) = vae_placement_bytes(&vae_path, geom.width, geom.height);
            let vae_cards = crate::inference::place::ladder::prefer(
                &crate::inference::place::ladder::cards(0),
                last_card,
            );
            let vae_config = crate::inference::model::zimage::vae::VaeConfig::z_image();
            let vae_path_str = vae_path.to_str().ok_or_else(|| anyhow!("bad VAE path"))?;
            let (vae, vae_device) = crate::inference::place::ladder::load_whole_on(
                "Z-Image VAE",
                vae_need,
                vae_want,
                &vae_cards,
                cancel.as_ref(),
                |dev| {
                    let vb = unsafe { crate::tensor::VarBuilder::from_files(&[vae_path_str], DType::F32, dev) }
                        .map_err(|e| anyhow!("{e}"))?;
                    crate::inference::model::zimage::vae::AutoEncoderKL::new(&vae_config, vb)
                        .map_err(|e| anyhow!("{e}"))
                },
            )?;
            let vae_dtype = DType::F32;
            info!("Z-Image VAE -> {vae_device:?}");
            // CPU-fallback VAE for large images that run out on the card's decode.
            // Slower but never runs out.
            let vae_cpu = if vae_device.is_cuda() {
                let vae_vb_cpu = unsafe {
                    crate::tensor::VarBuilder::from_files(
                        &[vae_path.to_str().unwrap()],
                        crate::tensor::DType::F32,
                        &crate::tensor::Device::Cpu,
                    )?
                };
                Some(crate::inference::model::zimage::vae::AutoEncoderKL::new(&vae_config, vae_vb_cpu).map_err(|e| {
                    error!("Failed to load CPU-fallback z-image VAE: {}", e); e
                })?)
            } else { None };
            send_progress("VAE loaded");

            let scheduler_cfg = crate::inference::model::zimage::sampling::SchedulerConfig::z_image_turbo();

            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load.saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name: requested,
                ckpt_id: None,
                model: LoadedImageModel::ZImage(ZImageModelState {
                    transformer,
                    text_encoder,
                    tokenizer,
                    vae,
                    vae_cpu,
                    vae_path: vae_path.clone(),
                    vae_device,
                    vae_dtype,
                    transformer_cfg: tf_config,
                    scheduler_cfg,
                    te_device,
                    te_dtype,
                    text_cache: PromptTextCache::new(PROMPT_TEXT_CACHE_CAP),
                    vae_latent_cache: PromptTextCache::new(IMG2IMG_LATENT_CACHE_CAP),
                }),
                device: primary_device,
                dtype: primary_dtype,
            });

            send_progress("All components loaded - ready to generate!");
            stage.finished();
            info!("Z-Image-Turbo fully loaded");
            Ok(())
        }).await?;

        result.map_err(std::convert::Into::into)
    }
}
