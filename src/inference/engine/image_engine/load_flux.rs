//! Part of `impl ImageEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl ImageEngine {
    /// Create OpenCL pipelines for Flux if the plan has OpenCL segments.
    #[cfg(feature = "opencl")]
    pub(super) fn create_flux_ocl_pipelines(
        plan: &HeteroPlan,
    ) -> Option<std::sync::Arc<crate::inference::kernel::opencl::OpenCLPipelines>> {
        use crate::inference::place::layer_executor::DeviceKind;
        let has_ocl = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::OpenCL(_)));
        if !has_ocl {
            return None;
        }
        let device_ids = crate::inference::kernel::opencl::enumerate_opencl_devices();
        if let Some(&dev_id) = device_ids.first() {
            match crate::inference::kernel::opencl::OpenCLPipelines::new(dev_id) {
                Ok(p) => {
                    info!("OpenCL pipelines initialized for Flux");
                    Some(std::sync::Arc::new(p))
                }
                Err(e) => {
                    error!("Failed to create OpenCL pipelines for Flux: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Load Flux Schnell quantized model with progress reporting.
    /// Uses local GGUF file for the Flux transformer if provided,
    /// otherwise downloads from HuggingFace Hub.
    /// T5, CLIP, and VAE are always fetched via hf_hub (cached after first download).
    pub async fn load_flux_schnell(
        &self,
        req: ModelRequest,
        local_flux_gguf: Option<std::path::PathBuf>,
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
            let stage = LoadStage::new("Flux", progress_tx);
            let send_progress = |msg: &str| stage.say(msg);
            // Published for the whole load, so every weight read below - the GGUF transformer,
            // T5, CLIP, the VAE - counts into the stage line without any of them knowing this
            // engine exists. Dropped with this closure, which is what unpublishes it.
            let _counts = crate::inference::serve::progress::scoped::publish(stage.reporter());

            send_progress("Initializing HuggingFace API...");
            let api = crate::inference::load::huggingface_manager::hf_api(Some(&hf_models_dir))
                .map_err(|e| {
                    error!("Failed to create HF API client: {}", e);
                    e
                })?;
            // The cards this load may use. Whether there is one decides the compute dtype;
            // where each component goes is decided below, once its size is known.
            let cards = crate::inference::place::ladder::cards(0);

            // Enable TF32/reduced-precision for TensorCore acceleration (~10-15% matmul speedup).
            // F32 (TF32) and BF16 reduced precision are safe here - wide enough exponent to
            // accumulate 512 products. F16 is left at COMPUTE_32F by the switch itself.
            if !cards.is_empty() {
                crate::tensor::cuda_ext::set_gemm_reduced_precision(true);
            }

            let gpu_dtype = if cards.is_empty() { DType::F32 } else { DType::BF16 };
            info!("Flux: {} card(s), gpu_dtype: {:?}", cards.len(), gpu_dtype);

            // HOT-FIRST (the role-aware contract): the transformer - run per denoise step - is
            // loaded BEFORE the one-shot T5/CLIP encoders so it gets first claim on the fastest
            // card. The aux encoders then place on whatever remains (their GPU->CPU fallbacks
            // already handle a full card). Loading aux first starved the 11+ GB fp8 transformer
            // of the primary GPU and OOMed the native path.
            // 3. Flux transformer (quantized GGUF) - use local file if available
            let flux_model_file = if let Some(ref local_path) = local_flux_gguf {
                if local_path.exists() {
                    send_progress(&format!("Loading Flux transformer from local: {}", local_path.file_name().unwrap_or_default().to_string_lossy()));
                    local_path.clone()
                } else {
                    send_progress("Local GGUF not found, downloading from HF Hub (~6GB)...");
                    let flux_repo = api.repo(hf_hub::Repo::model("lmz/candle-flux".to_string()));
                    flux_repo.get("flux1-schnell.gguf").map_err(|e| {
                        error!("Failed to download Flux GGUF: {}", e);
                        e
                    })?
                }
            } else {
                send_progress("Downloading Flux Schnell transformer (~6GB)...");
                let flux_repo = api.repo(hf_hub::Repo::model("lmz/candle-flux".to_string()));
                flux_repo.get("flux1-schnell.gguf").map_err(|e| {
                    error!("Failed to download Flux GGUF: {}", e);
                    e
                })?
            };
            // Decide single-device vs multi-device based on model size and available VRAM
            // Kontext is a guidance-distilled (dev-family) checkpoint -> dev config
            // (guidance_embed on); plain schnell otherwise. A safetensors checkpoint states its
            // own family: a `guidance_in` tensor in the header = dev (e.g. the Rayflux fp8
            // fine-tunes) - probe the header instead of trusting the filename.
            // Stable identity of the resident checkpoint: its file name. Two model
            // tags resolving to the same file legitimately share a resident model;
            // different files must never be treated as interchangeable.
            let flux_ckpt_id: Option<String> = flux_model_file
                .file_name()
                .map(|f| f.to_string_lossy().to_string());
            let is_kontext = flux_model_file.to_string_lossy().to_lowercase().contains("kontext");
            let is_safetensors_ckpt = flux_model_file
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
            let is_dev = is_kontext
                || (is_safetensors_ckpt
                    && crate::inference::load::fp8_scaled::header_contains(&flux_model_file, "guidance_in"));
            let flux_cfg = if is_dev {
                crate::inference::model::flux::common::Config::dev()
            } else {
                crate::inference::model::flux::common::Config::schnell()
            };
            let total_blocks = flux_cfg.depth + flux_cfg.depth_single_blocks; // 57
            // WHAT THIS REQUEST NEEDS TO DENOISE, beyond the weights. Derived from the
            // model's own width and head count and from the geometry being rendered -
            // never a fixed byte count. A constant is wrong in both directions at once:
            // too large at small sizes, refusing placements that would have run, and
            // too small at large ones, where it admits a load whose weights fit and
            // whose denoise then cannot allocate. The second is the failure that has no
            // way back, because by then the model is resident.
            let runtime_demand =
                flux_runtime_demand_for(geom.width, geom.height, Some(&flux_model_file));
            // The dense embedders and the final layer stay resident in F32 on whichever
            // card holds the stem. Counted from the config's own widths rather than
            // rounded off: img_in and txt_in project the latent and the text context to
            // the model width, the three conditioning MLPs are two square layers each,
            // and the final layer projects back out.
            let embedding_overhead = {
                let h = flux_cfg.hidden_size as u64;
                let projections = (flux_cfg.in_channels + flux_cfg.context_in_dim
                    + FLUX_TIME_FREQ_DIM
                    + flux_cfg.vec_in_dim) as u64
                    * h;
                let conditioning_mlps = 3 * h * h;
                let out = h * flux_cfg.in_channels as u64;
                (projections + conditioning_mlps + out) * DENSE_BYTES
            };
            info!(
                "Flux demand for {}x{}: {:.2} GB denoise scratch + {:.2} GB resident embedders",
                geom.width,
                geom.height,
                runtime_demand as f64 / 1e9,
                embedding_overhead as f64 / 1e9,
            );
            // The checkpoint was just resolved, so this stat succeeds; if it somehow
            // does not, fall back to what the config's own parameter count implies
            // rather than to a number picked by hand.
            let file_size = std::fs::metadata(&flux_model_file).map(|m| m.len()).unwrap_or_else(
                |_| {
                    let h = flux_cfg.hidden_size as u64;
                    // A dual-stream block carries both streams' attention and
                    // feed-forward; a single-stream block carries one of each.
                    let per_dual = 2 * (4 + 2 * flux_cfg.mlp_ratio as u64) * h * h;
                    let per_single = (4 + 2 * flux_cfg.mlp_ratio as u64) * h * h;
                    (flux_cfg.depth as u64 * per_dual
                        + flux_cfg.depth_single_blocks as u64 * per_single)
                        * DENSE_BYTES
                        / 2
                },
            );
            // Resident GPU footprint is the FILE SIZE, whatever the container.
            //
            // This used to apply a 0.55 ratio to GGUF checkpoints, on the theory that
            // metadata and alignment padding accounted for the rest. Measured against
            // two real checkpoints it under-estimated by ~1.9x in both cases: an f16
            // GGUF resident 1.00x its file, a Q4_K_M GGUF 1.04x. That is the expected
            // result - weights stay quantized on the device, so what is on disk is what
            // is in VRAM - and it explains a render that passed placement with a
            // comfortable margin and then hit `cuda OOM in kernel out` mid-denoise.
            //
            // The small overshoot on the Q4 side (tensors that do not stay quantized)
            // is left to the runtime reserve rather than padded in here: padding the
            // estimate pushes a checkpoint that genuinely fits one card into a
            // cross-GPU split, which serialises the cards and is the worse failure.
            let estimated_gpu_mem = file_size;

            // What the transformer asks of the cards: its blocks, the dense stem on the card
            // the first block runs on, and what this render needs free beside them. Whole on
            // one card when a card holds it at the floor of that need; spread otherwise; the
            // ladder recovers from a card that runs out. The cards are read again here,
            // after the downloads, so the plan sees what they hold now.
            let cards = crate::inference::place::ladder::cards(0);
            let demand = crate::inference::place::ladder::Demand {
                layers: total_blocks,
                block_bytes: estimated_gpu_mem,
                runtime_reserve: runtime_demand,
                ..Default::default()
            };
            let first_plan =
                demand.plan_with_stem(&cards, embedding_overhead, runtime_floor(runtime_demand));
            let accelerators = crate::inference::place::ladder::accelerators(runtime_demand);
            info!(
                "HeteroFlux: file={:.1}GB, {:.1}GB to denoise {}x{} ({:.1}GB floor) -> {}",
                file_size as f64 / 1e9,
                runtime_demand as f64 / 1e9,
                geom.width,
                geom.height,
                runtime_floor(runtime_demand) as f64 / 1e9,
                crate::inference::place::dry_plan::describe_plan(&first_plan),
            );
            send_progress("Building Flux transformer...");
            let is_safetensors_ckpt = flux_model_file
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
            let cancelled =
                || cancel.as_ref().is_some_and(crate::inference::serve::cancel::CancelToken::is_cancelled);
            let (flux_variant, device) = crate::inference::place::ladder::load_planned(
                "Flux transformer",
                first_plan,
                &demand,
                &cards,
                &accelerators,
                cancelled,
                |plan| -> anyhow::Result<(FluxVariant, Device)> {
                    use crate::inference::place::layer_executor::DeviceKind;
                    // One card: the whole transformer on it, read straight onto the card.
                    // fp8-scaled safetensors (Rayflux / Rayflux-Krea) load through the fp8
                    // bridge; GGUF keeps from_gguf. Q8_0 for the 2-D projections keeps full
                    // CPU+GPU kernel coverage (matches the GGUF path).
                    if let [seg] = plan.segments.as_slice() {
                        if let DeviceKind::Cuda(_) = seg.kind {
                            let ndev = crate::inference::place::ladder::open(seg.kind)?;
                            let nvb = if is_safetensors_ckpt {
                                unsafe {
                                    crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                                        &[&flux_model_file],
                                        crate::tensor::quantized::GgmlDType::Q8_0,
                                        &ndev,
                                        cancel.as_ref(),
                                    )
                                }
                                .map_err(|e| anyhow::anyhow!("{e}"))?
                            } else {
                                crate::inference::cache::qvb::from_gguf_cached(&flux_model_file, &ndev)
                                    .map_err(|e| anyhow::anyhow!("{e}"))?
                            };
                            let nflux = HeteroFlux::whole(&flux_cfg, nvb)
                                .map_err(|e| anyhow::anyhow!("{e}"))?;
                            return Ok((FluxVariant::Whole(nflux), ndev));
                        }
                    }
                    // Several devices, or the host: each block is uploaded to the device the
                    // plan names, through the card's stream-bound handle.
                    let cards_used = crate::inference::place::ladder::cards_of(plan);
                    let mut dev_map: std::collections::HashMap<usize, Device> =
                        std::collections::HashMap::new();
                    for idx in &cards_used {
                        dev_map.insert(*idx, crate::inference::place::ladder::open(DeviceKind::Cuda(*idx))?);
                    }
                    let stem = cards_used
                        .first()
                        .and_then(|idx| dev_map.get(idx).cloned())
                        .unwrap_or(Device::Cpu);
                    let hetero = HeteroFlux::from_gguf(
                        &flux_model_file,
                        &flux_cfg,
                        plan,
                        &dev_map,
                        #[cfg(feature = "opencl")]
                        Self::create_flux_ocl_pipelines(plan),
                    )?;
                    Ok((FluxVariant::Hetero(hetero), stem))
                },
            )
            .inspect_err(|e| error!("Flux transformer load failed: {e}"))?;
            // What the whole placement holds on its card against what the plan charged it,
            // reported with the next render.
            if let FluxVariant::Whole(_) = &flux_variant {
                let before = crate::inference::place::plan::ordinal(&device)
                    .and_then(|idx| cards.iter().find(|(i, _)| *i == idx))
                    .map_or(0, |(_, free)| *free);
                let after = crate::inference::place::ladder::free_on_card(&device).unwrap_or(0);
                let actual = before.saturating_sub(after);
                info!(
                    "Flux resident: predicted {:.2} GB, actual {:.2} GB ({:+.2} GB), {:.2} GB \
                     left against a {:.2} GB reserve{}",
                    estimated_gpu_mem as f64 / 1e9,
                    actual as f64 / 1e9,
                    (actual as f64 - estimated_gpu_mem as f64) / 1e9,
                    after as f64 / 1e9,
                    runtime_demand as f64 / 1e9,
                    if after < runtime_demand { " - SHORT, the generation can starve" } else { "" },
                );
            }
            // The cards the denoiser runs on. The one-shot components below go elsewhere
            // while another card has room, and share these only while this render's own
            // need stays free on them.
            let hot_cards: Vec<usize> = match &flux_variant {
                FluxVariant::Whole(_) => crate::inference::place::plan::ordinal(&device).into_iter().collect(),
                FluxVariant::Hetero(h) => crate::inference::place::ladder::cards_of(&h.plan),
            };
            send_progress("Flux transformer loaded");


            // 1. T5 encoder (always from HF Hub, cached locally)
            send_progress("Downloading T5-v1.1-XXL encoder (~5GB, cached after first download)...");
            let t5_repo = api.repo(hf_hub::Repo::with_revision(
                "google/t5-v1_1-xxl".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/2".to_string(),
            ));
            let t5_model_file = t5_repo.get("model.safetensors").map_err(|e| {
                error!("Failed to download T5 model: {}", e);
                e
            })?;
            let t5_config_file = t5_repo.get("config.json").map_err(|e| {
                error!("Failed to download T5 config: {}", e);
                e
            })?;
            let t5_config: crate::inference::model::t5::encoder::Config = serde_json::from_str(&std::fs::read_to_string(&t5_config_file)?)?;
            // T5-XXL encode is the single largest per-request cost, so it wants a card with
            // room for its weights AND its encode activations. It is a ONE-SHOT component -
            // it runs once per prompt and its result is cached - so it goes to a card the
            // denoiser does not use while one has room, and shares the denoiser's cards only
            // while this render's own need stays free there. The host otherwise: slower for
            // one encode, and correct. The cards are read after the transformer is resident,
            // so what they report is the truth.
            let t5_bytes = t5_encoder_bytes(&t5_config, 2 /* bf16 */) + t5_encode_scratch_bytes(&t5_config);
            let aux_cards = crate::inference::place::ladder::aside(
                &crate::inference::place::ladder::cards(0),
                &hot_cards,
                runtime_demand,
            );
            send_progress("Loading T5 encoder...");
            // T5 runs on the NATIVE substrate: bf16 on a card (bf16 required - T5's FF
            // activations overflow f16), F32 on the host.
            let (t5_model, t5_device) = crate::inference::place::ladder::load_whole_on(
                "Flux T5 encoder",
                t5_bytes,
                t5_bytes,
                &aux_cards,
                cancel.as_ref(),
                |dev| {
                    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
                    let vb = unsafe { crate::tensor::VarBuilder::from_files(&[&t5_model_file], dtype, dev) }
                        .map_err(|e| anyhow!("T5 native weights: {e}"))?;
                    crate::inference::model::t5::encoder::T5EncoderModel::load(vb, &t5_config)
                        .map_err(|e| anyhow!("T5 native load: {e}"))
                },
            )?;
            info!("Flux T5 encoder device: {:?} (wanted {:.1} GB)", t5_device, t5_bytes as f64 / 1e9);
            let t5_tokenizer_file = api
                .model("lmz/mt5-tokenizers".to_string())
                .get("t5-v1_1-xxl.tokenizer.json")?;
            let t5_tokenizer = Tokenizer::from_file(t5_tokenizer_file)
                .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
            send_progress(&format!(
                "T5 encoder loaded ({})",
                if t5_device.is_cuda() { "GPU" } else { "CPU" }
            ));

            // 2. CLIP text encoder (always from HF Hub, cached locally)
            send_progress("Downloading CLIP-ViT-Large (~400MB, cached)...");
            let clip_repo = api.repo(hf_hub::Repo::model("openai/clip-vit-large-patch14".to_string()));
            let clip_model_file = clip_repo.get("model.safetensors").map_err(|e| {
                error!("Failed to download CLIP model: {}", e);
                e
            })?;
            send_progress("Loading CLIP encoder...");
            // CLIP runs in F32 wherever it lands, a one-shot component placed like T5. The
            // file also holds the vision tower, so its size bounds the text tower's.
            let clip_config = crate::inference::model::clip::text::Config::vit_l_14();
            let clip_bytes = std::fs::metadata(&clip_model_file).map_or(0, |m| m.len());
            let aux_cards = crate::inference::place::ladder::aside(
                &crate::inference::place::ladder::cards(0),
                &hot_cards,
                runtime_demand,
            );
            let (clip_model, clip_device) = crate::inference::place::ladder::load_whole_on(
                "Flux CLIP encoder",
                clip_bytes,
                clip_bytes,
                &aux_cards,
                cancel.as_ref(),
                |dev| {
                    let vb = unsafe {
                        crate::tensor::VarBuilder::from_files(&[&clip_model_file], DType::F32, dev)
                    }
                    .map_err(|e| anyhow!("CLIP native weights: {e}"))?;
                    crate::inference::model::clip::text::Transformer::new(vb.pp("text_model"), &clip_config)
                        .map_err(|e| anyhow!("CLIP native load: {e}"))
                },
            )?;
            let clip_tokenizer_file = clip_repo.get("tokenizer.json")?;
            let clip_tokenizer = Tokenizer::from_file(clip_tokenizer_file)
                .map_err(|e| anyhow!("CLIP tokenizer: {e}"))?;
            send_progress("CLIP encoder loaded");


            // Initialize layer performance tracker for image model
            match &flux_variant {
                FluxVariant::Whole(_) => {
                    let total = total_blocks;
                    let gpu_layers = if device.is_cuda() { total } else { 0 };
                    crate::inference::place::layer_perf::global_tracker().initialize_model("flux-schnell", total, gpu_layers);
                }
                FluxVariant::Hetero(hetero) => {
                    #[cfg(feature = "opencl")]
                    crate::inference::place::layer_perf::global_tracker().initialize_heterogeneous_model("flux-schnell", &hetero.plan.segments);
                    #[cfg(not(feature = "opencl"))]
                    {
                        let total = hetero.plan.total_layers;
                        let gpu_layers = hetero.plan.segments.iter()
                            .filter(|s| matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::Cuda(_)))
                            .map(|s| s.layer_end - s.layer_start)
                            .sum();
                        crate::inference::place::layer_perf::global_tracker().initialize_model("flux-schnell", total, gpu_layers);
                    }
                }
            }

            // 4. VAE AutoEncoder - check local file first, then HF Hub
            // Look for ae.safetensors next to the Flux GGUF file
            let ae_model_file = {
                let local_ae = local_flux_gguf.as_ref()
                    .and_then(|p| p.parent())
                    .map(|dir| dir.join("ae.safetensors"))
                    .filter(|p| p.exists());
                if let Some(local_path) = local_ae {
                    send_progress(&format!("Loading VAE from local: {}", local_path.display()));
                    local_path
                } else {
                    // The canonical Flux VAE lives in the GATED repo
                    // black-forest-labs/FLUX.1-schnell. Requires:
                    //   1. Accept the license at
                    //      https://huggingface.co/black-forest-labs/FLUX.1-schnell
                    //   2. `huggingface-cli login` (or set HF_TOKEN)
                    // We DON'T fall back to ostris/Flex.1-alpha's VAE
                    // even though it's the same model family - that
                    // copy follows the diffusers VAE naming convention
                    // (encoder.down.0.block.0.norm1.weight, etc.) which
                    // the reference flux::autoencoder loader rejects with
                    // "cannot find tensor encoder.down.0.block.0.norm1.weight"
                    // because the BFL-native naming differs. Translating
                    // names would be a multi-hundred-LOC mapping pass -
                    // out of scope for the fallback path. Surface the
                    // clear auth-required error and tell the user how
                    // to fix it.
                    send_progress("Downloading VAE AutoEncoder (~200MB, cached)...");
                    let bf_repo = api.repo(hf_hub::Repo::model("black-forest-labs/FLUX.1-schnell".to_string()));
                    bf_repo.get("ae.safetensors").map_err(|e| {
                        let s = e.to_string();
                        let symlink_failure = s.contains("Operation not permitted")
                            || s.contains("symlink")
                            || s.contains("os error 1");
                        if s.contains("401") {
                            error!("VAE download failed (401 Unauthorized): \
                                The FLUX.1-schnell model is gated.");
                            anyhow!(
                                "Flux VAE requires HuggingFace authentication. \
                                 1) Accept the license at \
                                 https://huggingface.co/black-forest-labs/FLUX.1-schnell, \
                                 2) `huggingface-cli login` or `export HF_TOKEN=hf_...`, \
                                 3) restart the server."
                            )
                        } else if symlink_failure {
                            error!("VAE download failed at symlink step (exFAT HF cache).");
                            anyhow!(
                                "Flux VAE download succeeded but hf-hub could not \
                                 create the snapshot symlink on this filesystem \
                                 (exFAT). Either (a) move HF_HOME to a Linux-native \
                                 FS (ext4/xfs/btrfs), or (b) manually place \
                                 ae.safetensors next to the flux1-schnell.gguf in \
                                 the snapshot dir."
                            )
                        } else {
                            error!("Failed to download Flux VAE: {e}");
                            anyhow!("Failed to download Flux VAE: {e}")
                        }
                    })?
                }
            };
            // VAE on the NATIVE substrate, F32 end-to-end (GPU convs via im2col + cuBLAS
            // sgemm; GroupNorm/upsample as dedicated kernels). A one-shot component placed
            // like the encoders, so its decode does not compete with the transformer's
            // denoise: on the first card with room for a whole decode at this geometry,
            // else for the smallest tile, else on the host.
            send_progress("Loading VAE...");
            let (vae_need, vae_want) = vae_placement_bytes(&ae_model_file, geom.width, geom.height);
            let aux_cards = crate::inference::place::ladder::aside(
                &crate::inference::place::ladder::cards(0),
                &hot_cards,
                runtime_demand,
            );
            let ae_cfg = crate::inference::model::flux::vae::Config::schnell();
            let (ae, vae_device) = crate::inference::place::ladder::load_whole_on(
                "Flux VAE",
                vae_need,
                vae_want,
                &aux_cards,
                cancel.as_ref(),
                |dev| {
                    let vb = unsafe {
                        crate::tensor::VarBuilder::from_files(
                            std::slice::from_ref(&ae_model_file),
                            DType::F32,
                            dev,
                        )
                    }
                    .map_err(|e| anyhow!("VAE weights: {e}"))?;
                    crate::inference::model::flux::vae::AutoEncoder::new(&ae_cfg, vb)
                        .map_err(|e| anyhow!("VAE load: {e}"))
                },
            )?;
            send_progress(&format!("VAE loaded ({})", if vae_device.is_cuda() { "GPU F32" } else { "CPU" }));
            // CPU VAE fallback for large images (>=768^2): the GPU VAE decode peaks
            // at several GB of conv workspace + feature maps, which doesn't fit
            // alongside the resident Flux transformer (GPU0) / T5 (GPU1) at 1024^2.
            // The CPU decode is slower but never OOMs. Only loaded when the primary
            // VAE is on GPU (~330 MB RAM); decode dispatch is in generate_flux_image.
            let ae_cpu = if vae_device.is_cuda() {
                let vb_cpu = unsafe {
                    crate::tensor::VarBuilder::from_files(
                        &[ae_model_file],
                        crate::tensor::DType::F32,
                        &crate::tensor::Device::Cpu,
                    )?
                };
                Some(crate::inference::model::flux::vae::AutoEncoder::new(&ae_cfg, vb_cpu).map_err(|e| {
                    error!("Failed to load CPU-fallback VAE: {}", e); e
                })?)
            } else { None };

            let mut guard = model_state.blocking_lock();
            let resident_bytes = free_before_load.saturating_sub(crate::inference::place::vram_manager::free_total());
            *guard = Some(LoadedImageModelState {
                placed_for: geom,
                resident_bytes,
                name: requested,
                // A fixed family name made every non-Ray Flux variant alias every
                // other one, so a request for FLUX.1-schnell was silently served by
                // whatever Flux checkpoint happened to be loaded (measured: schnell
                // and candle-flux both rendered with a resident Rayflux fine-tune,
                // in 2 s, with no load in the log). The FILE decides reuse.
                ckpt_id: flux_ckpt_id.clone(),
                model: LoadedImageModel::Flux(FluxModelState {
                    attached_loras: Vec::new(),
                    flux: flux_variant,
                    t5_model: Some(t5_model),
                    t5_source: Some((t5_model_file.clone(), t5_config.clone())),
                    t5_device,
                    t5_tokenizer,
                    clip_model,
                    clip_device,
                    clip_tokenizer,
                    ae,
                    vae_device,
                    ae_cpu,
                    text_cache: PromptTextCache::new(PROMPT_TEXT_CACHE_CAP),
                    vae_latent_cache: PromptTextCache::new(IMG2IMG_LATENT_CACHE_CAP),
                    schedule_shift: is_dev.then_some((FLUX_DEV_BASE_SHIFT, FLUX_DEV_MAX_SHIFT)),
                }),
                device,
                dtype: gpu_dtype,
            });

            send_progress("All 4 components loaded - ready to generate!");
            stage.finished();
            info!("Flux Schnell fully loaded (4 components)");
            Ok(())
        }).await?;

        result.map_err(std::convert::Into::into)
    }
}
