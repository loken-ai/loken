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
            send("Selecting fastest GPU...");
            // FASTEST GPU THAT FITS, the fleet rule: rank by throughput but skip a card
            // that cannot hold this DiT plus generation headroom. Taking the fastest
            // card unconditionally put the stream on a full GPU whenever another engine
            // was resident - the same index-0 class of bug that sent Flux renders to
            // the CPU. Falls back to the fastest card, then CPU when there is no GPU.
            let primary = {
                // Weights plus what THIS request's denoise takes, from the DiT's own
                // width, head count and feed-forward ratio. A fixed reserve stood here
                // and could not know either.
                let cfg = crate::inference::model::boogu::dit::Config::default();
                let tokens = geom.tokens(FLUX_VAE_STRIDE, cfg.patch_size) + FLUX_TEXT_TOKENS;
                let denoise = crate::inference::place::runtime_demand::dit_activation_bytes(
                    tokens,
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.ffn_inner as f64 / cfg.hidden_size.max(1) as f64,
                );
                let want =
                    crate::inference::engine::boogu_engine::hot_component_bytes(&hf_models_dir)
                        + denoise;
                crate::inference::place::vram_manager::pick_device_for("boogu", want)
                    .map(|(_, _, d)| d)
                    .unwrap_or(crate::tensor::Device::Cpu)
            };
            send("Preparing Boogu (FLUX VAE + tokenizer; DiT/encoder stream per generation)...");
            let boogu = crate::inference::engine::boogu_engine::load(
                &hf_models_dir,
                &primary,
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
            // PLACEMENT-EXEMPT: not a placement. `flux2_engine::load` ignores this argument and
            // builds its own HeteroPlan from request-derived reserves; this only proves a device
            // exists and gives the state a nominal one to report.
            let primary = crate::inference::place::vram_manager::probe(0)
                .into_iter()
                .next()
                .map(|(_, _, d)| d)
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
            send("Selecting fastest GPU...");
            // device_probe now ranks by compute throughput, so the first entry is the
            // fastest card - the DiT's home. The encoder is placed on a second GPU inside
            // qwen_image_engine::load.
            // PLACEMENT-EXEMPT: not a placement. qwen_image_engine::load ignores this
            // argument (`_primary`) and builds its own HeteroPlan from request-derived
            // reserves; this only proves a CUDA device exists and gives the state a
            // nominal device to report.
            let primary = crate::inference::place::vram_manager::probe(0)
                .into_iter()
                .next()
                .map(|(_, _, d)| d)
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
            // For the phases that read no tensors (the downloads) and for the placement
            // cascade below, which must not treat an abandoned load as a placement failure.
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

            // Detect available VRAM across ALL CUDA devices for smart placement.
            // Transformer is ~12GB BF16 of static weights + ~3-4GB of peak
            // activation buffers at 1024^2 (per-block scratch in the S3-DiT
            // attention/MLP). Earlier single-GPU placement on a 17 GB card
            // OOMed at 1024^2 because static-weight estimate (12 GB) plus a
            // 2 GB headroom left only ~3 GB free, less than peak activations.
            //
            // Multi-GPU policy: with more than one card, split the transformer
            // blocks across them - the combined VRAM gives headroom AND the
            // step parallelises. With one card this falls back to the CUDA+CPU
            // hetero path.
            //
            // THE FASTEST CARD THAT FITS, through the fleet's own choke point -
            // not device zero, and not merely the fastest. Device zero is not
            // reliably the fastest, is not always present, and may be the one
            // another engine is already sitting on; taking the head of the probe
            // without asking whether the stem and this request's scratch fit
            // there is the other half of the same mistake, and the placement
            // gate rejects it by name.
            let cuda_device = crate::inference::place::vram_manager::pick_device_for(
                "z-image stem",
                zimage_embedder_bytes() + headroom,
            )
            .and_then(|(idx, _, _)| crate::tensor::cuda_ext::new_device_with_stream(idx).ok())
            .map(|d| d.native_device());
            // Per-device free VRAM, FASTEST-FIRST (probe ranks by compute throughput, never by
            // index or free VRAM) - the same ordering every planner in the fleet uses. Empty vec
            // means no CUDA at all (CPU-only path).
            let mut cuda_free: Vec<(usize, u64)> = Vec::new();
            #[cfg(feature = "cuda")]
            {
                if cuda_device.is_some() {
                    for (idx, free, _) in crate::inference::place::vram_manager::probe(0) {
                        info!("Z-Image: CUDA #{} free VRAM: {:.1} GB", idx, free as f64 / 1e9);
                        cuda_free.push((idx, free));
                    }
                    if cuda_free.is_empty() {
                        cuda_free.push((0, 6 * 1024 * 1024 * 1024)); // NVML-less fallback
                    }
                }
            }
            let total_cuda_vram: u64 = cuda_free.iter().map(|(_, m)| *m).sum();

            // What the weights weigh, beside what the forward above needs free.
            //
            // Read off the checkpoint that is about to be loaded, scaled to the dtype it
            // is loaded at - the SAME helper the admission path uses, so the gate that
            // decides a render may start and the planner that decides where its blocks go
            // cannot disagree about the size of the model they are both talking about.
            //
            // They did disagree, by three gigabytes: the loader counted parameters off
            // the config with the feed-forward at four times the model width, where this
            // family builds it at eight thirds. The config fallback below is only reached
            // when the files cannot be stat'd at all, and it now uses the width the
            // blocks are actually built at.
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
            // because the placement decided in the next paragraph is measured by
            // building and running this very config.
            let mut tf_config = crate::inference::model::zimage::dit::Config::z_image_turbo();
            // flash-attn at Z-Image shapes (n_heads=30, head_dim=128, BF16)
            // still produces a black PNG even after zero-padding heads
            // Z-Image flash-attn is permanently disabled in production
            // - bisection of the black-PNG bug at 1024^2 showed an
            // unresolved numerical issue in the accelerated path. Keep
            // attention_basic. The attention path also decides what the score matrix
            // costs, so a placement measured against the other one would not be this
            // render's.
            tf_config.set_use_accelerated_attn(false);

            // WHERE THE BLOCKS GO, ASKED OF EACH CANDIDATE PLACEMENT IN TURN.
            //
            // The rule below plans from a formula: the weights, plus a reserve, against
            // each card's free VRAM. The reserve cannot depend on the placement - it is
            // computed before there is one - and yet what a card holds DOES depend on it,
            // because the stream, its rotary tables, its modulation and its attention
            // scratch live on whichever card is running the block. So a card given nine
            // blocks of thirty was charged the peak of a card given all thirty, and at
            // 1536 square that reserve was seven gigabytes: the plan named both cards,
            // put nothing on the host, and the first card still ran out - it had been
            // filled until only a whole model's forward would have fitted beside it.
            //
            // Asked the other way round the loop closes. Build and run each candidate on
            // devices that COUNT instead of allocating, put every card's own count
            // through the same margin a whole card's count goes through, and keep the
            // first placement whose cards all fit. What comes back is a reserve PER CARD,
            // which is the only shape that can follow a plan: at 1024 square across two
            // cards the walk says 1.61 GB on the one carrying the stem and 1.54 GB on the
            // other, and no single figure can be both.
            //
            // The margin passed to the solver is ZERO, and not because there is none:
            // it is already inside each card's figure, so what the solver holds against a
            // card's free VRAM is what that card must have free.
            // AND THE CAPTION ENCODER IS PART OF THE ARRANGEMENT, not what follows it.
            //
            // The walk above answers for the block stack alone, and the encoder was then
            // given whatever the plan left over. At 1536 square across two cards it left
            // 0.25 GB against the 6.5 GB the encoder needs, so the encoder went on anyway -
            // its own rule said the card was free enough - and the split's first attention
            // GEMM found the card full. Placing the hot stack first is right; deciding
            // where it goes without counting what has to sit beside it is not.
            //
            // So a card is asked to KEEP the encoder's room while the stack is planned
            // around it, and every card is tried as that host, fastest first: on a machine
            // with one card the stack is planned against a card that already owes the
            // encoder, and on a machine with several the encoder settles wherever the
            // stack leaves it a real place rather than a remainder. If no card can hold
            // both, the encoder is not silently squeezed in - the arrangement is simply
            // not found here, and the rules further down plan it as they did before, with
            // the host as the encoder's last resort.
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
                // WHICH card keeps that room is itself a placement decision, and taking the
                // first that works is the wrong answer: reserving on the fastest card is
                // what pushes a stack that fits it whole into a split, and a split costs a
                // transfer at every block boundary on every step - measured here at 80 per
                // cent of the render. So every host is tried and the arrangement with the
                // FEWEST SEGMENTS wins, the fastest card breaking the tie. Undivided on one
                // card beats divided across two, whichever card ends up lending the room.
                let mut best: Option<(usize, crate::inference::place::dry_plan::Solution)> = None;
                for host in 0..cuda_free.len() {
                    let budgets: Vec<(usize, u64)> = cuda_free
                        .iter()
                        .enumerate()
                        .map(|(i, (idx, free))| {
                            (*idx, if i == host { free.saturating_sub(encoder_room) } else { *free })
                        })
                        .collect();
                    let Some(s) =
                        crate::inference::place::dry_plan::solve(tf_config.n_layers, &budgets, 0, &mut measure)
                    else {
                        continue;
                    };
                    // Fewest segments first, then the RANK OF THE CARD THE STACK STARTS ON.
                    // Both halves are load-bearing and each was measured by getting it
                    // wrong: taking the first arrangement that worked split a stack that
                    // fitted one card, and ranking on segments alone then lent the fastest
                    // card to the encoder and ran the stack on the slower one - a whole
                    // placement, on the wrong card, and eighty per cent slower all the
                    // same. The hot stack takes the quickest card that holds it; what runs
                    // once per image goes wherever that leaves room.
                    let rank = |s: &crate::inference::place::dry_plan::Solution| {
                        let first = s.plan.segments.first().map_or(usize::MAX, |g| match g.kind {
                            crate::inference::place::layer_executor::DeviceKind::Cuda(i) => {
                                cuda_free.iter().position(|(idx, _)| *idx == i).unwrap_or(usize::MAX)
                            }
                            _ => usize::MAX,
                        });
                        (s.plan.segments.len(), first)
                    };
                    if best.as_ref().is_none_or(|(_, b)| rank(&s) < rank(b)) {
                        best = Some((host, s));
                    }
                }
                best.map(|(host, s)| {
                    info!(
                        "Z-Image: CUDA #{} keeps {:.2} GB for the caption encoder while the stack \
                         is planned around it ({} segment(s))",
                        cuda_free[host].0,
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
                     the stack within the margin, so the rules below plan it as they did \
                     before the walk existed",
                    geom.width, geom.height,
                ),
            }

            // Placement decision:
            //   - a card that fits the transformer -> that card, UNDIVIDED
            //   - otherwise, hetero across whatever there is (CUDA, then CPU)
            //   - No CUDA -> CPU only
            //
            // The hot component stays whole on one card whenever one holds it. The
            // previous rule made a SECOND GPU disqualify the single-card path
            // (`num_cuda == 1 &&`), so on a two-card box this family was always
            // planned hetero - and a hetero plan can put segments on the CPU. That is
            // how a model which fits one card ended up running partly on the host,
            // with both GPUs idle and 8 denoise steps taking 5.5 minutes. Spreading a
            // model that fits is not free either: it pays a cross-device transfer at
            // every block boundary, on the hot path, every step.
            // Does ANY card hold it whole - not just the fastest one.
            //
            // This asked `usable_vram`, which is the FASTEST card's free VRAM alone. So
            // a chat model occupying that card made the planner conclude the
            // transformer did not fit and spill it to the CPU, while the second card
            // sat empty with 16 GB. Measured: GPU0 7.7 GB used by a chat model, GPU1 at
            // 156 MB, and the render still went hetero-with-CPU and then failed. Which
            // card it lands on is decided below; all this has to answer is whether one
            // of them can take it.
            let transformer_fits_single = zimage_fits_one_card(
                measured_placement.as_ref(),
                &cuda_free,
                transformer_size_est,
                headroom,
            );
            let use_hetero = cuda_device.is_some()
                && !transformer_fits_single
                && total_cuda_vram > headroom;
            let transformer_fits_gpu = transformer_fits_single;

            // Text encoder (Qwen3-based, ~5 GB BF16). The one-time prompt
            // encode takes ~6 s on CPU - the blanket CPU pin was a VRAM guard,
            // but on a multi-GPU box the SECONDARY card is free while the
            // transformer takes the primary, so run it there (~0.2 s). Only
            // move it to a GPU with room (idx != 0, >= TE footprint free) and
            // RESERVE that footprint in `cuda_free` so the transformer
            // placement below accounts for it - GPU0 (the transformer's home)
            // is never touched, so 1-GPU / tight-VRAM boxes keep the CPU path
            // and its no-OOM guarantee. Loads BF16 on GPU (5 GB, fits) via the
            // `te_ndtype` branch below; the rope-dtype fix makes the BF16 GPU
            // stack correct.
            let te_est_bytes = zimage_text_encoder_bytes();
            // Which card will the transformer take? It is the hot component and gets
            // first claim: the fastest card that fits it whole (the same rule its own
            // loader applies below). The encoder then takes the roomiest OTHER card.
            // Excluding "GPU0" instead assumed the transformer always lands on device 0
            // - it does not, and when it does not, that exclusion pushes the encoder
            // onto the transformer's own card and leaves a free one idle.
            let tf_gpu: Option<usize> = zimage_transformer_card(
                measured_placement.as_ref(),
                &cuda_free,
                transformer_size_est,
                headroom,
            );
            // WHAT THE TRANSFORMER'S PLAN LEAVES, not what a card happens to have free -
            // see `zimage_encoder_card` for what that distinction cost.
            let te_gpu: Option<usize> = zimage_encoder_card(
                &cuda_free,
                measured_placement.as_ref().map(|s| &s.load),
                tf_gpu,
                te_est_bytes,
            );
            if let Some(idx) = te_gpu {
                for e in cuda_free.iter_mut() {
                    if e.0 == idx { e.1 = e.1.saturating_sub(te_est_bytes); }
                }
            }
            let te_device = match te_gpu {
                Some(idx) => match crate::tensor::cuda_ext::new_device_with_stream(idx).map(|d| d.native_device()) {
                    Ok(d) => { info!("Z-Image text encoder -> CUDA:{idx} (BF16, ~6s CPU encode avoided)"); d }
                    Err(e) => { warn!("Z-Image TE: could not open GPU{idx} ({e}); using CPU"); Device::Cpu }
                },
                None => Device::Cpu,
            };
            let te_dtype = DType::F32;
            let te_str = if te_device.is_cuda() { "GPU" } else { "CPU" };

            // Primary device for embeddings/final layer. Mutable because the
            // native single-device loader below may retarget it to the
            // roomiest GPU (e.g. when GPU0 holds a co-resident chat model).
            let mut primary_device = if transformer_fits_gpu || use_hetero {
                cuda_device.as_ref().unwrap().clone()
            } else {
                Device::Cpu
            };
            let primary_dtype = if primary_device.is_cuda() { DType::BF16 } else { DType::F32 };

            #[cfg(feature = "opencl")]
            let has_arc = !crate::inference::kernel::opencl::enumerate_opencl_devices().is_empty();
            #[cfg(not(feature = "opencl"))]
            let has_arc = false;
            let mode_str = if transformer_fits_gpu { "single GPU" }
                else if use_hetero && has_arc { "multi-device (CUDA+Arc+CPU)" }
                else if use_hetero { "multi-device (CUDA+CPU)" }
                else { "CPU only" };
            // VAE placement decided dynamically after the transformer
            // loads (we re-poll NVML at that point and call
            // pick_aux_device with prefer=last-transformer-segment so
            // the final latent doesn't need a cross-device transfer).
            // Per-CUDA-device choice surfaces via "VAE placement: CUDA #N
            // has X GB free" + "Z-Image VAE: placing on CUDA:N (...)"
            // INFO lines below.
            info!("Z-Image placement: transformer={}, text_encoder={}, VAE=dynamic (post-transformer NVML poll)", mode_str, te_str);
            send_progress(&format!("Device: transformer={}, text_encoder={}, VAE=dynamic", mode_str, te_str));

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

            // 2. Text Encoder (Qwen3-based, ~5GB) - CPU unless 24GB+ GPU
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
            send_progress(&format!("Loading text encoder ({te_str})..."));
            let te_config = crate::inference::model::zimage::text_encoder::TextEncoderConfig::z_image();
            let te_files_str: Vec<&str> = te_files.iter().map(|p| p.to_str().unwrap()).collect();
            // Z-image text encoder on the NATIVE substrate (bf16 stack on
            // GPU like T5; F32 when placed on CPU).
            let te_ndevice = &te_device.clone();
            let te_ndtype = if te_device.is_cuda() {
                crate::tensor::DType::BF16
            } else {
                crate::tensor::DType::F32
            };
            let te_vb = unsafe {
                crate::tensor::VarBuilder::from_files(
                    &te_files_str,
                    te_ndtype,
                    te_ndevice,
                )
            }
            .map_err(|e| anyhow!("z-image TE native weights: {e}"))?;
            let text_encoder = crate::inference::model::zimage::text_encoder::ZImageTextEncoder::new(&te_config, te_vb).map_err(|e| {
                error!("Failed to load text encoder: {}", e);
                anyhow!("z-image TE native load: {e}")
            })?;
            send_progress("Text encoder loaded");

            // 3. Transformer (S3-DiT, ~12GB). The files were resolved before the
            // placement decision above, which is what let it weigh them, and its config
            // was built there for the same reason.

            // NATIVE single-device attempt FIRST -
            // the flux loader recipe (commit fbc703a, flipped 33cff6a): the
            // native transformer is the default and the facade placements
            // below are the fallback. Gate: the primary CUDA device must fit
            // the ~12 GB BF16 transformer plus an activation margin. The
            // margin is smaller than the facade single-GPU `headroom` (5 GB)
            // because the native attention is query-chunked (score matrix
            // capped at CHUNKxseq), so the 1024^2 peak that forced 5 GB
            // doesn't materialize; the VAE is placed dynamically afterwards
            // (post-transformer NVML poll) and simply lands elsewhere when
            // this device is full.
            // Which single GPU can hold the whole transformer natively? The demand it
            // is held against is no longer computed here: a separate margin stood at
            // this line once, and after that a separate WEIGHT - the estimate from the
            // file sizes, which weighs the model lighter than the allocator holds it.
            // Both were the same question asked a second time, and the second answer is
            // the one that got used. It goes through `zimage_whole_card` now.
            // Prefer GPU0 when it fits (unchanged behaviour); otherwise fall
            // back to the ROOMIEST other GPU that fits. This is the key
            // robustness fix: when GPU0 is occupied by a co-resident LLM
            // (e.g. the /conversation classifier), GPU0 drops just under the
            // native threshold and the OLD code fell into the multi-device
            // hetero split - whose cross-device forward throws
            // CUDA_ERROR_LAUNCH_FAILED (unspecified launch failure) and
            // poisons both GPU contexts. Routing the transformer whole onto a
            // free GPU keeps it on the WORKING native single-device path (and
            // is faster - no per-step GPU->GPU activation transfers).
            // FIRST device in fastest-first order whose free VRAM fits the whole transformer
            // (the hot per-step component belongs undivided on the fastest GPU that fits - never
            // "most free", never a fixed index).
            let native_gpu: Option<usize> = zimage_whole_card(
                measured_placement.as_ref(),
                &cuda_free,
                transformer_size_est,
                headroom,
                primary_device.is_cuda(),
            );
            let native_single: Option<ZImageVariant> = if let Some(native_idx) = native_gpu {
                // Device handle for the chosen GPU (reuse GPU0's existing
                // stream; open a fresh stream for any other index).
                let native_dev = if native_idx == 0 {
                    primary_device.clone()
                } else {
                    match crate::tensor::cuda_ext::new_device_with_stream(native_idx).map(|d| d.native_device()) {
                        Ok(d) => d,
                        Err(e) => {
                            warn!("Z-Image: could not open GPU{} for native single load ({e}); staying on GPU0 path", native_idx);
                            primary_device.clone()
                        }
                    }
                };
                send_progress("Loading transformer (NATIVE, single GPU)...");
                match (|| -> AnyResult<ZImageVariant> {
                    let ndev = &native_dev.clone();
                    let nvb = unsafe {
                        crate::tensor::VarBuilder::from_files(
                            &tf_files_str,
                            crate::tensor::DType::BF16,
                            ndev,
                        )
                    }
                    .map_err(|e| anyhow!("{e}"))?;
                    let nvb = match tf_prefix { Some(p) => nvb.pp(p), None => nvb };
                    let model = crate::inference::model::zimage::dit::ZImageTransformer2DModel::from_varbuilder(
                        &tf_config, &nvb,
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                    Ok(ZImageVariant::NativeSingle(model))
                })() {
                    Ok(v) => {
                        info!("Z-Image: single-device load - NATIVE transformer (zimage_native) on GPU{}", native_idx);
                        // Retarget the primary device so embeddings / layer
                        // tracking / VAE placement all follow the transformer.
                        primary_device = native_dev;
                        Some(v)
                    }
                    Err(e) => {
                        // An ABANDONED load is not a placement failure. Falling through to
                        // the facade path would read the same twelve gigabytes a second time
                        // for a client that has gone, and each cheaper plan would bail again
                        // - the same short-circuit the Qwen-Image and Boogu cascades take.
                        if cancelled() {
                            return Err(e);
                        }
                        warn!(
                            "Z-Image: NATIVE single-device load failed ({}), falling back to facade placement",
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };

            let transformer = if let Some(native) = native_single {
                native
            } else if use_hetero {
                // Multi-device: split 30 main layers across all available
                // CUDA devices (+ CPU overflow). HeteroPlan packs
                // proportionally to free VRAM per device.
                let total_main_layers = tf_config.n_layers; // 30

                // Detect OpenCL (Arc) devices for overflow
                #[cfg(feature = "opencl")]
                let ocl_devs: Vec<(usize, u64)> = {
                    let device_ids = crate::inference::kernel::opencl::enumerate_opencl_devices();
                    device_ids.iter().enumerate().map(|(i, &dev_id)| {
                        let mem = crate::inference::kernel::opencl::get_opencl_device_memory(dev_id);
                        info!("  OpenCL device {}: {:.1} GB", i, mem as f64 / 1e9);
                        (i, mem)
                    }).collect()
                };
                #[cfg(not(feature = "opencl"))]
                let ocl_devs: Vec<(usize, u64)> = vec![];

                // THE MEASURED PLACEMENT DECIDES WHEN THERE IS ONE.
                //
                // Its cards were each asked what they would hold under this very plan, so
                // what it leaves free on a card is this request's forward on THAT card,
                // not a whole model's forward on every card. The planner below cannot
                // express that - it takes one reserve and charges it everywhere - and
                // that is what filled a card past the forward it was about to run.
                //
                // No measurement, no change: a checkpoint the walk cannot open, a layout
                // it cannot follow, or an arrangement none of these cards holds, and the
                // block planner answers exactly as it did before, reserve for reserve.
                let plan = match &measured_placement {
                    Some(s) => s.plan.clone(),
                    None => zimage_block_plan(
                        &cuda_free,
                        &ocl_devs,
                        total_main_layers,
                        transformer_size_est,
                        headroom,
                    ),
                };
                // The stem - the embedders, the refiners and the way back out - is built
                // on the primary device, and the measurement charged it to the card the
                // plan STARTS on. If those are not the same card the budget is a fiction,
                // so the stem follows the plan rather than the other way round.
                if let Some(crate::inference::place::layer_executor::DeviceKind::Cuda(first)) =
                    plan.segments.first().map(|s| s.kind)
                {
                    if gpu_index_of(&primary_device) != Some(first) {
                        match crate::tensor::cuda_ext::new_device_with_stream(first)
                            .map(|d| d.native_device())
                        {
                            Ok(d) => {
                                info!("Z-Image: the stem follows the plan onto CUDA:{first}");
                                primary_device = d;
                            }
                            Err(e) => warn!(
                                "Z-Image: the plan starts on CUDA:{first} and that device would \
                                 not open ({e}); the stem stays where it was"
                            ),
                        }
                    }
                }
                // THE PLACEMENT IS BUILT BY THE PORT THAT WAS MEASURED.
                //
                // The solver measured this plan by walking `zimage_native`, block by
                // block, on the devices the plan names. Building it with a different
                // implementation is what made every reader of the last day disagree:
                // the reserve described one arrangement and another one ran, and the one
                // that ran was also the slower of the two - a 1536-square render took
                // 137 s split against 66 s whole on the same machine, so the memory
                // safety a split buys was being paid for twice over.
                //
                // So a plan of CUDA and host segments is handed to the same constructor
                // the walk used. A plan naming an OpenCL device still goes to the
                // orchestrator below: that device has no slot in this constructor yet,
                // and dropping the path rather than porting it would leave the kernels
                // written and unreachable.
                let ocl_in_plan = plan.segments.iter().any(|s| {
                    matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::OpenCL(_))
                });
                let placed_native: Option<ZImageVariant> = if ocl_in_plan {
                    None
                } else {
                    match zimage_native_from_plan(
                        &plan,
                        &tf_files_str,
                        tf_prefix,
                        &tf_config,
                        &primary_device,
                        total_main_layers,
                    ) {
                        Ok(model) => {
                            info!(
                                "Z-Image: multi-device load - NATIVE transformer \
                                 (zimage_native) across {} segments, the arrangement the \
                                 measurement walked",
                                plan.segments.len(),
                            );
                            Some(ZImageVariant::NativeSingle(model))
                        }
                        Err(e) => {
                            if cancelled() {
                                return Err(e);
                            }
                            warn!(
                                "Z-Image: the measured placement would not build natively \
                                 ({e}); falling back to the orchestrator"
                            );
                            None
                        }
                    }
                };
                match placed_native {
                    Some(v) => v,
                    None => {
                send_progress(&format!("Loading transformer (multi-device, {} segments)...", plan.segments.len()));
                for seg in &plan.segments {
                    info!("  {:?}: layers {}-{}", seg.kind, seg.layer_start, seg.layer_end - 1);
                }
                // Initialize OpenCL pipelines if we have OpenCL segments
                #[cfg(feature = "opencl")]
                let ocl_pipelines_arc: Option<Arc<crate::inference::kernel::opencl::OpenCLPipelines>> = {
                    let has_ocl = plan.segments.iter().any(|s| matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::OpenCL(_)));
                    if has_ocl {
                        let device_ids = crate::inference::kernel::opencl::enumerate_opencl_devices();
                        if let Some(&dev_id) = device_ids.first() {
                            match crate::inference::kernel::opencl::OpenCLPipelines::new(dev_id) {
                                Ok(p) => {
                                    info!("OpenCL pipelines initialized for Z-Image");
                                    Some(Arc::new(p))
                                }
                                Err(e) => {
                                    error!("Failed to create OpenCL pipelines: {}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                let hetero = HeteroZImage::from_safetensors(
                    &tf_files_str,
                    tf_prefix,
                    &tf_config,
                    &plan,
                    &primary_device,
                    primary_dtype,
                    #[cfg(feature = "opencl")]
                    ocl_pipelines_arc,
                ).map_err(|e| {
                    error!("HeteroZImage load failed: {}", e);
                    anyhow!("HeteroZImage: {}", e)
                })?;
                ZImageVariant::Hetero(hetero)
                    }
                }
            } else if transformer_fits_gpu {
                // Single GPU
                send_progress("Loading transformer (GPU)...");
                let tf_vb = unsafe { crate::tensor::VarBuilder::from_files(&tf_files_str, primary_dtype, &primary_device)? };
                let tf_vb = match tf_prefix { Some(p) => tf_vb.pp(p), None => tf_vb };
                let model = ztf::ZImageTransformer2DModel::from_varbuilder(&tf_config, &tf_vb)
                    .map_err(|e| {
                        error!("Failed to load transformer: {}", e);
                        e
                    })?;
                ZImageVariant::Single(model)
            } else {
                // CPU fallback
                send_progress("Loading transformer (CPU)...");
                let tf_vb = unsafe { crate::tensor::VarBuilder::from_files(&tf_files_str, DType::F32, &Device::Cpu)? };
                let tf_vb = match tf_prefix { Some(p) => tf_vb.pp(p), None => tf_vb };
                let model = ztf::ZImageTransformer2DModel::from_varbuilder(&tf_config, &tf_vb)
                    .map_err(|e| {
                        error!("Failed to load transformer: {}", e);
                        e
                    })?;
                ZImageVariant::Single(model)
            };
            send_progress("Transformer loaded");

            // Initialize layer performance tracker for Z-Image
            match &transformer {
                ZImageVariant::Single(_) | ZImageVariant::NativeSingle(_) => {
                    let total = 30; // 30 main transformer layers
                    let gpu_layers = if primary_device.is_cuda() { total } else { 0 };
                    crate::inference::place::layer_perf::global_tracker().initialize_model("z-image-turbo", total, gpu_layers);
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

            // 4. VAE - prefer GPU at F32 (the tensor-op Z-Image VAE has no
            // F16-promotion path on its mid-block 512-channel attention,
            // so we cannot safely run it at F16; F32 conv2d on cuDNN
            // still beats CPU by 10-20x for a 1024x1024 decode).
            send_progress("Downloading VAE...");
            bail()?;
            let vae_path = repo.get("vae/diffusion_pytorch_model.safetensors").map_err(|e| {
                error!("Failed to download VAE: {}", e);
                e
            })?;
            // VAE placement: poll LIVE per-device free VRAM and pick
            // the best CUDA device via pick_aux_device. Formerly
            // this only checked the PRIMARY device (GPU0) and fell back
            // to CPU if GPU0 was full - which is exactly what happened
            // when the transformer packed GPU0 leaving GPU1 idle: VAE
            // either OOMed on GPU0 or moved to slow CPU. Now:
            //   1. Identify the device holding the LAST transformer
            //      segment (preferred -> saves a PCIe transfer of the
            //      final latent).
            //   2. Poll NVML per CUDA device for current free VRAM.
            //   3. pick_aux_device routes to preferred if it fits,
            //      otherwise to the device with the MOST free VRAM.
            //   4. CPU only if no GPU has >=1.5 GB free.
            // Decode peak DERIVED from what drives it instead of a fixed constant: the decoder's
            // full-resolution stage keeps ~8 conv buffers live (input + output + residual skip +
            // upsample double-buffer + cuDNN workspace + allocator slack) over its widest (128)
            // channel width, in F32, at the family's native side. Buffer count anchored by
            // measurement: a 1024^2 decode OOMed with 2.8 GB free AND with 4.09 GB free
            // (6-buffer estimate = 3.2 GB was still short); 8 buffers = 4.3 GB clears it.
            const NATIVE_SIDE: u64 = 1024;
            const VAE_WIDEST_CHANNELS: u64 = 128;
            let vae_peak_bytes: u64 = NATIVE_SIDE * NATIVE_SIDE * VAE_WIDEST_CHANNELS * 8 * 4;
            let preferred_vae_gpu: Option<usize> = match &transformer {
                ZImageVariant::Hetero(h) => h.plan.segments.iter()
                    .rev()
                    .find_map(|s| match s.kind {
                        crate::inference::place::layer_executor::DeviceKind::Cuda(idx) => Some(idx),
                        _ => None,
                    }),
                ZImageVariant::Single(_) => {
                    if let crate::tensor::DeviceLocation::Cuda { gpu_id } = primary_device.location() {
                        Some(gpu_id)
                    } else {
                        None
                    }
                }
                // NATIVE single-device: the 12 GB BF16 transformer fills the
                // primary GPU to ~4 GB free at load, and the denoise-time
                // activation retention eats most of that - a 1024^2 VAE
                // decode on the same device OOMs (observed) and falls back
                // to the slow CPU VAE. No preference -> pick_aux_device
                // takes the fastest device with room for the decode peak
                // (the idle second GPU when present). The cross-device
                // latent copy this costs is ~1 MB - noise next to a
                // 1.5 GB decode peak.
                ZImageVariant::NativeSingle(_) => None,
            };
            // Fastest-first live free VRAM (probe order), so pick_aux_device's first-fit takes
            // the fastest card that has room.
            let live_free: Vec<(usize, u64)> =
                crate::inference::place::vram_manager::probe(0)
                    .into_iter()
                    .map(|(i, f, _)| (i, f))
                    .collect();
            for (idx, free) in &live_free {
                info!(
                    "  VAE placement: CUDA #{} has {:.2} GB free (post-transformer)",
                    idx, *free as f64 / 1e9,
                );
            }
            // Two tiers, because the whole-image peak is a PREFERENCE, not a requirement.
            // Charging the full peak at load time is what sent the VAE to the CPU with
            // 4.16 GB free on an idle card: it missed a 4.29 GB demand by ~130 MB, and a
            // CPU decode costs minutes where the GPU costs seconds. The decode degrades
            // by tiling now, so the floor is what the weights plus the smallest tile's
            // working set need - anything above that is a faster decode, not a possible one.
            let vae_weight_bytes =
                std::fs::metadata(&vae_path).map(|m| m.len()).unwrap_or(0).saturating_mul(2);
            const SMALLEST_TILE_SIDE: u64 = (32 + 2 * 8) * 8;
            let vae_floor_bytes = vae_weight_bytes
                + SMALLEST_TILE_SIDE * SMALLEST_TILE_SIDE * VAE_WIDEST_CHANNELS * 8 * 4;
            let vae_target_gpu = pick_aux_device(&live_free, vae_peak_bytes, preferred_vae_gpu)
                .or_else(|| {
                    let fallback = pick_aux_device(&live_free, vae_floor_bytes, preferred_vae_gpu);
                    if fallback.is_some() {
                        info!(
                            "Z-Image VAE: no device holds the {:.1} GB whole-image peak; placing                              against the {:.1} GB tiled floor instead",
                            vae_peak_bytes as f64 / 1e9,
                            vae_floor_bytes as f64 / 1e9,
                        );
                    }
                    fallback
                });
            let (vae_device, vae_dtype) = match vae_target_gpu {
                Some(gpu_id) => {
                    let dev = crate::tensor::Device::new_cuda(gpu_id)
                        .map_err(|e| anyhow!("Z-Image VAE: failed to bind to CUDA:{gpu_id}: {e}"))?;
                    let preferred_note = if Some(gpu_id) == preferred_vae_gpu {
                        " (last transformer segment -> no PCIe transfer)"
                    } else {
                        " (fastest device with room; last transformer segment was full)"
                    };
                    info!("Z-Image VAE: placing on CUDA:{gpu_id}{preferred_note}");
                    send_progress(&format!("Loading VAE (GPU:{gpu_id} F32)..."));
                    (dev, DType::F32)
                }
                None => {
                    warn!(
                        "Z-Image VAE: no CUDA device has {} MB free, falling back to CPU",
                        // not-a-vram-size: bytes-to-megabytes for the message.
                        vae_peak_bytes / 1_048_576,
                    );
                    send_progress("Loading VAE (CPU)...");
                    (Device::Cpu, DType::F32)
                }
            };
            let vae_config = crate::inference::model::zimage::vae::VaeConfig::z_image();
            let n_vae_device = &vae_device.clone();
            let vae_vb = unsafe {
                crate::tensor::VarBuilder::from_files(
                    &[vae_path.to_str().unwrap()],
                    crate::tensor::DType::F32,
                    n_vae_device,
                )?
            };
            let vae = crate::inference::model::zimage::vae::AutoEncoderKL::new(&vae_config, vae_vb).map_err(|e| {
                error!("Failed to load VAE: {}", e);
                e
            })?;
            // CPU-fallback VAE for large images (>=768^2) that OOM the GPU decode
            // (16384^2 mid-block attention at 1024^2). Slower but never OOMs.
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
