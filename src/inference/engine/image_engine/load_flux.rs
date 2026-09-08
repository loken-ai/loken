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
            // PLACEMENT-EXEMPT: bootstrap only. Nothing is loaded onto this device - it
            // decides the compute dtype and serves as the no-GPU fallback. The real Flux
            // placement happens below, once the checkpoint size is known, via
            // vram_manager::pick_device_for.
            let device = crate::inference::place::vram_manager::probe(0)
                .first()
                .and_then(|(idx, _, _)| crate::tensor::cuda_ext::new_device_with_stream(*idx).ok())
                .map(|d| d.native_device())
                .unwrap_or(Device::Cpu);

            // Enable TF32/reduced-precision for TensorCore acceleration (~10-15% matmul speedup).
            // F16 stays at COMPUTE_32F: enabling F16 accumulation makes the VAE mid AttnBlock
            // (4096x512 x 512x4096 matmul) overflow -> NaN -> all-black image. F32 (TF32) and
            // BF16 reduced precision are safe - wide enough exponent to accumulate 512 products.
            // The reference also promotes F16 attention to F32 in the VAE attention
            // path; this flag stays off so cuBLAS itself doesn't fall back to 16F accumulation.
            if device.is_cuda() {
                crate::tensor::cuda_ext::set_gemm_reduced_precision(true);
            }

            let gpu_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
            let cpu_dtype = DType::F32; // CPU doesn't support BF16 matmul
            info!("Flux device: {:?}, gpu_dtype: {:?}", device, gpu_dtype);

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

            // Detect available CUDA devices for multi-device placement.
            // Walk EVERY NVML index, not just 0 - same single-GPU
            // enumeration bug that prevented Z-Image from using GPU1
            // (fixed in 3e3208b). On a 2x 16 GB box, querying only
            // device 0 meant Flux's hetero-fallback could split CUDA0+CPU
            // but never CUDA0+CUDA1, defeating the second GPU entirely.
            let mut cuda_devices: Vec<(usize, u64)> = Vec::new();
            let mut raw_free_vram: u64 = 0;
            #[cfg(feature = "cuda")]
            {
                if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                    let count = nvml.device_count().unwrap_or(1);
                    for idx in 0..count {
                        if let Ok(gpu) = nvml.device_by_index(idx) {
                            if let Ok(mem) = gpu.memory_info() {
                                if idx == 0 {
                                    raw_free_vram = mem.free;
                                }
                                let _per_dev_free = mem.free;
                                info!(
                                    "HeteroFlux: CUDA #{} - {:.1} GB free",
                                    idx, mem.free as f64 / 1e9,
                                );
                                // Activations everywhere, plus the resident embedders
                                // on the card that holds the stem. Both derived above.
                                let mut usable = mem.free.saturating_sub(runtime_demand);
                                if idx == 0 {
                                    usable = usable.saturating_sub(embedding_overhead);
                                }
                                cuda_devices.push((idx as usize, usable));
                            }
                        }
                    }
                }
            }
            // FASTEST GPU THAT FITS - the fleet placement rule. Picking merely the
            // fastest card (what the primary selection above does, before the
            // checkpoint size is even known) meant Flux only ever considered GPU0:
            // with GPU0 at 10.3 GB free and GPU1 at 16.5 GB, a 12.1 GB checkpoint
            // "did not fit single-device", split, failed again, and landed all 57
            // blocks on the CPU - a render that never finishes - while a card sat
            // free. Re-pick here, where the resident estimate exists, and take the
            // chosen card's free VRAM as the single-device budget.
            let (device, mut raw_free_vram) = {
                // TWO figures, and they are not the same question. `want` is what a
                // render is COMFORTABLE with; `floor` is what it genuinely cannot go
                // under. I collapsed them into one on the grounds that neither was
                // trustworthy, and that is what made a 1024x1024 render split across
                // cards where it had run whole on one: its spare cleared the floor by a
                // wide margin and missed the comfortable figure by 400 MB.
                //
                // Splitting a model that fits is not a safe default. It costs a
                // transfer per block per step, and it is chosen here on a prediction -
                // while the measured peak of that very render was well under the floor.
                let want = estimated_gpu_mem + runtime_demand;
                let floor = estimated_gpu_mem + runtime_floor(runtime_demand);
                match crate::inference::place::vram_manager::pick_device_tiered("flux", want, floor) {
                    // Flux runs on a non-default stream with graph capture, so it needs
                    // the stream-bound handle for the chosen index rather than the plain
                    // device the picker returns.
                    Some((idx, free, _)) => {
                        let d = crate::tensor::cuda_ext::new_device_with_stream(idx)
                            .map(|h| h.native_device())
                            .unwrap_or(device.clone());
                        (d, free)
                    }
                    None => (device.clone(), raw_free_vram),
                }
            };
            if raw_free_vram == 0 && device.is_cuda() {
                // Fallback: assume 6GB free if NVML unavailable but CUDA works
                raw_free_vram = 6 * 1024 * 1024 * 1024;
                if cuda_devices.is_empty() {
                    // Same derived demand as every other path; there is no reason
                    // for the no-NVML fallback to reserve a different amount than
                    // the one this request actually needs.
                    cuda_devices.push((0, raw_free_vram.saturating_sub(runtime_demand)));
                }
            }

            // Activation/scratch headroom for the single-device fit check + the OCL
            // device sizing below. Per-device CUDA deductions (including the
            // primary-only embedding overhead) already happened inside the NVML walk
            // above. Same derived figure throughout - one request, one demand.
            let headroom = runtime_demand;

            // Detect Arc/OpenCL devices
            #[cfg(feature = "opencl")]
            let ocl_devs: Vec<(usize, u64)> = {
                let device_ids = crate::inference::kernel::opencl::enumerate_opencl_devices();
                device_ids.iter().enumerate().map(|(i, &dev_id)| {
                    let mem = crate::inference::kernel::opencl::get_opencl_device_memory(dev_id);
                    let usable = mem.saturating_sub(headroom);
                    info!("Flux: OpenCL device {} - {:.1} GB total, {:.1} GB usable", i, mem as f64 / 1e9, usable as f64 / 1e9);
                    (i, usable)
                }).collect()
            };
            #[cfg(not(feature = "opencl"))]
            let ocl_devs: Vec<(usize, u64)> = vec![];

            // THE DECISION, not a report. This used to print the spare against a floor
            // and then load single-device regardless, because the only fallback was
            // triggered by a load-time allocation failure. That is the wrong trigger:
            // the weights fit whenever the spare is merely thin, so the load SUCCEEDS
            // and the starvation lands on the first denoise - by which time the model
            // is resident, the card is full, and there is nothing left to fall back
            // to. Observed as a run of identical HTTP 500s, one per request, on a host
            // whose other card sat mostly free the whole time.
            //
            // So the spare decides. If it does not cover what this request needs to
            // denoise, the single-device path is not attempted at all and the split
            // runs up front.
            let spare = raw_free_vram.saturating_sub(estimated_gpu_mem);
            // Against the FLOOR, not the comfortable figure. Above the floor the render
            // runs; between the floor and the comfortable figure it runs with less room
            // than one would choose, which is still better than paying a cross-card
            // transfer on every block of every step.
            //
            // THAT RISK IS ONLY DEFENSIBLE ON THE FIRST ATTEMPT. Once a render has
            // actually run out on this card, taking the same risk again produces the
            // same failure - which is exactly what a run of identical 500s was: each
            // re-plan printed "-> single-device" after the previous one starved, because
            // this decision never looked at the pressure the retry loop had been raising
            // on its behalf. So the pressure tightens the test, and past the second notch
            // the single-device path is not offered at all: the request has proved that
            // this card cannot hold it, and a split that runs is worth more than a fourth
            // identical plan.
            let pressure = crate::inference::place::vram_manager::vram_degrade_level();
            let need = if pressure == 0 { runtime_floor(runtime_demand) } else { runtime_demand };
            let single_device_fits = pressure < 2 && spare >= need;
            info!(
                "HeteroFlux: file={:.1}GB, est_gpu={:.1}GB, free={:.1}GB -> {:.1}GB spare \
                 against {:.1}GB required (of {:.1}GB wanted) to denoise {}x{} at pressure \
                 {pressure} -> {}",
                file_size as f64 / 1e9,
                estimated_gpu_mem as f64 / 1e9,
                raw_free_vram as f64 / 1e9,
                spare as f64 / 1e9,
                need as f64 / 1e9,
                runtime_demand as f64 / 1e9,
                geom.width,
                geom.height,
                if single_device_fits { "single-device" } else { "SPLIT up front" },
            );

            // Strategy:
            // 1. CUDA present AND the card has room to denoise -> single-device.
            // 2. Otherwise, or on a load failure -> split across CUDA+CPU+Arc.
            // 3. No CUDA -> CPU + Arc only.
            //
            // The second case used to be reachable ONLY through a load failure, which
            // is why a thin card loaded happily and starved afterwards.
            let flux_variant = if raw_free_vram > 0 {
                send_progress(if single_device_fits {
                    "Building Flux model (single GPU)..."
                } else {
                    "Not enough room to denoise on one card - splitting..."
                });
                // Single-device strategy: the whole transformer on the chosen card, which is
                // the same blocks the split placement runs and differs only in that nothing
                // competes with them for what is left of the card. When it does not fit, the
                // Err channel below leads to the split.
                match (|| -> anyhow::Result<FluxVariant> {
                    // Decline the single-device path BEFORE loading anything when the
                    // card cannot also hold this request's denoise. The split below is
                    // reached through this same Err channel, so declining here and
                    // failing there take the identical route - the difference is that
                    // this one costs no load and cannot strand a resident model.
                    if !single_device_fits {
                        anyhow::bail!(
                            "{:.1} GB spare on the chosen card, {:.1} GB needed to denoise {}x{}",
                            spare as f64 / 1e9,
                            runtime_demand as f64 / 1e9,
                            geom.width,
                            geom.height,
                        );
                    }
                    let native = (|| -> anyhow::Result<FluxVariant> {
                        let ndev = &device.clone();
                        // Ray fp8-scaled safetensors (Rayflux / Rayflux-Krea) load
                        // through the fp8 bridge; GGUF Schnell/Dev keeps from_gguf.
                        // Same transformer either way -- only the builder source
                        // differs. Q8_0 for the 2-D projections keeps full CPU+GPU
                        // kernel coverage (matches the GGUF path).
                        let is_safetensors = flux_model_file
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
                        let nvb = if is_safetensors {
                            unsafe {
                                crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                                    &[&flux_model_file],
                                    crate::tensor::quantized::GgmlDType::Q8_0,
                                    ndev,
                                    cancel.as_ref(),
                                )
                            }
                            .map_err(|e| anyhow::anyhow!("{e}"))?
                        } else {
                            crate::inference::cache::qvb::from_gguf_cached(
                                &flux_model_file, ndev,
                            ).map_err(|e| anyhow::anyhow!("{e}"))?
                        };
                        let nflux = HeteroFlux::whole(&flux_cfg, nvb)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        Ok(FluxVariant::Whole(nflux))
                    })();
                    match native {
                        Ok(variant) => {
                            info!("Flux: single-device load - the whole transformer on one card");
                            return Ok(variant);
                        }
                        Err(ne) => {
                            // The split fallback below reopens the checkpoint per card, so an
                            // fp8-scaled safetensors reaches it only through a full requantised
                            // GGUF sidecar written to disk. A partial upload from the failed
                            // attempt is the usual cause here, so retry the whole placement
                            // ONCE after returning pooled VRAM to the driver, then surface the
                            // real error rather than pay for that sidecar.
                            let is_safetensors_ckpt = flux_model_file
                                .extension()
                                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
                            if is_safetensors_ckpt {
                                warn!("Flux: fp8 whole-placement load failed ({ne}); trimming pools and retrying once");
                                crate::inference::engine::llm_engine::trim_cuda_pools();
                                let ndev = &device.clone();
                                let nvb = unsafe {
                                    crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                                        &[&flux_model_file],
                                        crate::tensor::quantized::GgmlDType::Q8_0,
                                        ndev,
                                        cancel.as_ref(),
                                    )
                                }
                                .map_err(|e| anyhow::anyhow!("fp8 retry: {e}"))?;
                                let nflux = HeteroFlux::whole(&flux_cfg, nvb)
                                    .map_err(|e| anyhow::anyhow!("fp8 retry: {e}"))?;
                                info!("Flux: fp8 whole-placement load succeeded on retry");
                                return Ok(FluxVariant::Whole(nflux));
                            }
                            warn!("Flux: single-device load failed ({ne}), splitting instead");
                        }
                    }
                    anyhow::bail!(
                        "Flux: the single-device load did not produce a transformer."
                    )
                })() {
                    Ok(variant) => {
                        info!("Flux: single-device CUDA load succeeded");
                        // ESTIMATE vs REALITY, reported where a normal render will show
                        // it. The placement reserves FLUX_RUNTIME_RESERVE beyond the
                        // predicted resident size, but a render was observed leaving only
                        // 2.7 GB free on the transformer's card and then hitting
                        // `cuda OOM in kernel out` mid-denoise - so the prediction, not
                        // the reserve, was short. Nothing here changes placement; it just
                        // stops the question needing a dedicated experiment to answer,
                        // because the answer arrives with the next render either way.
                        {
                            let after: u64 = crate::inference::place::vram_manager::probe(0)
                                .into_iter()
                                .find(|(_, _, d)| {
                                    crate::tensor::Device::same_device(d, &device)
                                })
                                .map(|(_, free, _)| free)
                                .unwrap_or(0);
                            let predicted = estimated_gpu_mem;
                            let actual = raw_free_vram.saturating_sub(after);
                            info!(
                                "Flux resident: predicted {:.2} GB, actual {:.2} GB \
                                 ({:+.2} GB), {:.2} GB left against a {:.2} GB reserve{}",
                                predicted as f64 / 1e9,
                                actual as f64 / 1e9,
                                (actual as f64 - predicted as f64) / 1e9,
                                after as f64 / 1e9,
                                runtime_demand as f64 / 1e9,
                                if after < runtime_demand {
                                    " - SHORT, the generation can starve"
                                } else {
                                    ""
                                },
                            );
                        }
                        variant
                    }
                    Err(e) => {
                        // Two ways in: the planner declined single-device up front
                        // because the card has no room to denoise, or a load actually
                        // failed. Say which - "failed" on a deliberate choice reads as
                        // a defect, and an unexplained split is exactly the kind of
                        // silent degradation that hid this whole class of bug.
                        if single_device_fits {
                            warn!("Flux: single-device CUDA failed ({e}), splitting across CUDA+CPU+Arc");
                        } else {
                            info!("Flux: {e} - splitting across CUDA+CPU+Arc up front");
                        }
                        send_progress("Splitting Flux across GPU + CPU + Arc...");
                        // PLAN OVER EVERY CARD. This used to plan over the chosen card
                        // alone, because the loader built one CUDA VarBuilder and put
                        // every GPU-assigned block on it - so planning for two cards
                        // would have promised capacity it could not honour. The loader
                        // now uploads each block to the card the plan named, so that
                        // restriction is gone, and keeping it would be the real defect:
                        // it spilled blocks to the HOST while a second card sat empty.
                        let plan_cudas: Vec<(usize, u64)> = cuda_devices.clone();
                        // Handles for every card the plan may use. Flux runs on a
                        // non-default stream, so each card is taken through the
                        // stream-bound handle rather than a plain device.
                        let dev_map: std::collections::HashMap<usize, Device> = cuda_devices
                            .iter()
                            .filter_map(|(idx, _)| {
                                crate::tensor::cuda_ext::new_device_with_stream(*idx)
                                    .ok()
                                    .map(|h| (*idx, h.native_device()))
                            })
                            .collect();
                        // RE-PLAN, don't give up. The plan is computed from free VRAM,
                        // and between that reading and the upload another engine can
                        // take the card - the loader then reports "the GPU-assigned
                        // blocks no longer fit" and the whole render dies. Its own
                        // comment names the right answer ("re-plan against what is
                        // actually free") and nobody had written it.
                        //
                        // Each attempt RE-PROBES rather than reusing a stale number, and
                        // hands the GPU a smaller share, so the split walks toward the
                        // CPU one step at a time instead of jumping there. Putting every
                        // block on the host is NOT the fallback: a 12 GB transformer on
                        // the CPU cannot finish inside any request timeout, which is the
                        // 600 s hang that produced nothing and is why this used to fail
                        // fast instead.
                        // The share scales the FREE VRAM we admit to, not the model
                        // size: `calculate`'s last argument is a dequant expansion and
                        // sits in the DENOMINATOR, so shrinking it would put MORE blocks
                        // on the card - the opposite of backing off.
                        // Start where the CURRENT pressure says, not always at the top:
                        // a render that already exhausted a card raised the pressure, and
                        // beginning this walk from the full budget again just repeats the
                        // placement that failed.
                        const REPLAN_SHARES: [f64; 4] = [1.0, 0.7, 0.45, 0.25];
                        let pressure_share =
                            crate::inference::place::runtime_demand::weight_budget_share();
                        let mut hetero = None;
                        let mut last_err = None;
                        for (attempt, share) in REPLAN_SHARES.iter().enumerate() {
                            let fresh: Vec<(usize, u64)> = if attempt == 0 {
                                plan_cudas.clone()
                            } else {
                                crate::inference::place::vram_manager::probe(runtime_demand)
                                    .into_iter()
                                    .map(|(idx, free, _)| (idx, free))
                                    .collect()
                            };
                            let fresh: Vec<(usize, u64)> = fresh
                                .into_iter()
                                .map(|(i, free)| (i, (free as f64 * share * pressure_share) as u64))
                                .collect();
                            let plan = HeteroPlan::calculate(
                                total_blocks, estimated_gpu_mem, &fresh, &ocl_devs, 1.0,
                            );
                            match HeteroFlux::from_gguf(
                                &flux_model_file, &flux_cfg, &plan, &dev_map,
                                #[cfg(feature = "opencl")]
                                Self::create_flux_ocl_pipelines(&plan),
                            ) {
                                Ok(h) => {
                                    if attempt > 0 {
                                        info!(
                                            "HeteroFlux: re-planned at {:.0}% of the card after \
                                             {attempt} attempt(s); the split now fits",
                                            share * 100.0
                                        );
                                    }
                                    hetero = Some(h);
                                    break;
                                }
                                Err(e) => {
                                    warn!(
                                        "HeteroFlux: attempt {} at {:.0}% of the card failed \
                                         ({e}); re-probing and re-planning",
                                        attempt + 1,
                                        share * 100.0
                                    );
                                    last_err = Some(e);
                                }
                            }
                        }
                        let hetero = match hetero {
                            Some(h) => h,
                            None => {
                                let e = last_err.unwrap_or_else(|| {
                                    anyhow!("HeteroFlux: no plan was attempted")
                                });
                                error!("HeteroFlux CUDA+CPU+Arc fallback failed: {}", e);
                                return Err(e);
                            }
                        };
                        FluxVariant::Hetero(hetero)
                    }
                }
            } else {
                // No GPU available
                send_progress("Building Flux model (CPU + Arc)...");
                let plan = HeteroPlan::calculate(
                    total_blocks, estimated_gpu_mem, &[], &ocl_devs, 1.0,
                );
                let hetero = HeteroFlux::from_gguf(
                    &flux_model_file, &flux_cfg, &plan, &std::collections::HashMap::new(),
                    #[cfg(feature = "opencl")]
                    Self::create_flux_ocl_pipelines(&plan),
                ).map_err(|e| {
                    error!("HeteroFlux CPU load failed: {}", e);
                    e
                })?;
                FluxVariant::Hetero(hetero)
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
            // T5-XXL encode is the single largest per-request cost (~4.7s on CPU
            // = ~79% of a 6s/image generation), so it wants a GPU - but only one
            // with room for its weights AND its encode activations. Hardcoding
            // "the second CUDA device" assumed the transformer always sits on
            // device 0; once placement became fastest-that-fits, the transformer
            // could take that same card and T5 loaded into the sliver left over,
            // then bounced every attention matmul to the host at encode time -
            // far slower than simply running the whole encoder on the CPU.
            // The probe here runs AFTER the transformer is resident, so its free
            // VRAM is the truth, and no card fitting means CPU.
            let t5_bytes = t5_encoder_bytes(&t5_config, 2 /* bf16 */) + t5_encode_scratch_bytes(&t5_config);
            // ANYWHERE BUT THE DENOISER'S CARD, while another card can take it. This
            // is a ONE-SHOT component - it runs once per prompt and its result is
            // cached - so it must not compete with the hot DiT, which runs every step.
            // Probing free VRAM after the transformer is resident is not enough on its
            // own: that reading counts the DiT's WEIGHTS and not the scratch its
            // denoise will need, so the encoder could take 9.7 GB of the 11.8 GB left
            // on the DiT's own card and leave 2 GB for a render needing 9.8. Observed
            // exactly that, as a 1536x1536 request that OOM'd, re-planned itself onto
            // the host and then ran past the request timeout.
            //
            // Sharing the DiT's card is still allowed when nothing else will have it -
            // but only if what remains after the encoder still covers the denoise.
            // Otherwise the host runs it, which is slower for one encode and correct.
            let t5_pick = crate::inference::place::vram_manager::pick_device_elsewhere(
                "flux t5 encoder",
                t5_bytes,
                &device,
            )
            .or_else(|| {
                crate::inference::place::vram_manager::pick_device_for(
                    "flux t5 encoder (sharing the denoiser's card)",
                    t5_bytes + runtime_demand,
                )
            });
            let (mut t5_device, _t5_dtype) = match t5_pick
                .filter(|(_, free, _)| *free >= t5_bytes)
                .and_then(|(idx, _, _)| crate::tensor::cuda_ext::new_device_with_stream(idx).ok())
            {
                Some(d) => (d.native_device(), DType::BF16),
                None => (Device::Cpu, cpu_dtype),
            };
            send_progress(&format!("Loading T5 encoder ({})...",
                if t5_device.is_cuda() { "CUDA, BF16" } else { "CPU, F32" }));
            info!(
                "Flux T5 encoder device: {:?} (wanted {:.1} GB)",
                t5_device,
                t5_bytes as f64 / 1e9
            );
            // T5 runs on the NATIVE substrate: bf16 on GPU (bf16 required - T5's FF activations
            // overflow f16), F32 on CPU. Load on the chosen device; on ANY failure (e.g. OOM when
            // another image model is still resident on GPU1) FALL BACK TO CPU rather than hard-error
            // - the load must never OOM. The GPU success path is unchanged (only the error path adds
            // the CPU retry).
            let mut t5_ndevice = t5_device.clone();
            let mut t5_ndtype = if t5_device.is_cuda() {
                crate::tensor::DType::BF16
            } else {
                crate::tensor::DType::F32
            };
            let t5_model = loop {
                let attempt = (|| -> anyhow::Result<crate::inference::model::t5::encoder::T5EncoderModel> {
                    let vb = unsafe {
                        crate::tensor::VarBuilder::from_files(
                            &[&t5_model_file], t5_ndtype, &t5_ndevice,
                        )
                    }
                    .map_err(|e| anyhow!("T5 native weights: {e}"))?;
                    crate::inference::model::t5::encoder::T5EncoderModel::load(vb, &t5_config)
                        .map_err(|e| anyhow!("T5 native load: {e}"))
                })();
                match attempt {
                    Ok(m) => break m,
                    Err(e) if t5_ndevice.is_cuda() => {
                        warn!("T5 GPU load failed ({e}); falling back to CPU (never OOM)");
                        send_progress("T5 GPU tight - loading on CPU...");
                        t5_device = Device::Cpu;
                        t5_ndevice = crate::tensor::Device::Cpu;
                        t5_ndtype = crate::tensor::DType::F32;
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            };
            let t5_tokenizer_file = api
                .model("lmz/mt5-tokenizers".to_string())
                .get("t5-v1_1-xxl.tokenizer.json")?;
            let t5_tokenizer = Tokenizer::from_file(t5_tokenizer_file)
                .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
            // Say where it ACTUALLY landed. This was the literal string
            // "T5 encoder loaded (CPU)" regardless of device, so a T5 sitting on a GPU
            // reported itself on the CPU - which sent me chasing a CPU-fallback theory
            // for a black-image bug, on evidence the log had invented.
            send_progress(&format!(
                "T5 encoder loaded ({})",
                if t5_ndevice.is_cuda() { "GPU" } else { "CPU" }
            ));

            // 2. CLIP text encoder (always from HF Hub, cached locally)
            send_progress("Downloading CLIP-ViT-Large (~400MB, cached)...");
            let clip_repo = api.repo(hf_hub::Repo::model("openai/clip-vit-large-patch14".to_string()));
            let clip_model_file = clip_repo.get("model.safetensors").map_err(|e| {
                error!("Failed to download CLIP model: {}", e);
                e
            })?;
            send_progress("Loading CLIP encoder...");
            // CLIP runs on the NATIVE substrate (CPU F32 - ~120M params,
            // one forward per image-gen; host compute matches the reference
            // numerics exactly and frees the GPU dtype/device concerns).
            let clip_vb = unsafe {
                crate::tensor::VarBuilder::from_files(
                    &[clip_model_file],
                    crate::tensor::DType::F32,
                    &crate::tensor::Device::Cpu,
                )
            }
            .map_err(|e| anyhow!("CLIP native weights: {e}"))?;
            let clip_config = crate::inference::model::clip::text::Config {
                vocab_size: 49408,
                projection_dim: 768,
                activation: crate::tensor::ops::Activation::QuickGelu,
                intermediate_size: 3072,
                embed_dim: 768,
                max_position_embeddings: 77,
                pad_with: None,
                num_hidden_layers: 12,
                num_attention_heads: 12,
            };
            let clip_model = crate::inference::model::clip::text::Transformer::new(clip_vb.pp("text_model"), &clip_config).map_err(|e| {
                error!("Failed to load CLIP model: {}", e);
                anyhow!("CLIP native load: {e}")
            })?;
            let clip_tokenizer_file = clip_repo.get("tokenizer.json")?;
            let clip_tokenizer = Tokenizer::from_file(clip_tokenizer_file)
                .map_err(|e| anyhow!("CLIP tokenizer: {e}"))?;
            send_progress("CLIP encoder loaded");


            // Initialize layer performance tracker for image model
            match &flux_variant {
                FluxVariant::Whole(_) => {
                    let total = 57;
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
            // VAE on the NATIVE substrate, F32 end-to-end (GPU convs via
            // im2col + cuBLAS sgemm; GroupNorm/upsample as dedicated kernels).
            // Prefer the SECOND CUDA device (where T5 lives, idle by VAE-decode
            // time) so the VAE upsampling activations don't compete with the
            // Flux transformer that fills GPU0 during denoise (was an OOM).
            let vae_device = if t5_device.is_cuda() {
                send_progress("Loading VAE (GPU1)...");
                t5_device.clone()
            } else if device.is_cuda() {
                send_progress("Loading VAE (GPU)...");
                device.clone()
            } else {
                send_progress("Loading VAE (CPU)...");
                Device::Cpu
            };
            let n_vae_device = &vae_device.clone();
            let ae_vb = unsafe {
                crate::tensor::VarBuilder::from_files(
                    std::slice::from_ref(&ae_model_file),
                    crate::tensor::DType::F32,
                    n_vae_device,
                )?
            };
            let ae_cfg = crate::inference::model::flux::vae::Config::schnell();
            let ae = crate::inference::model::flux::vae::AutoEncoder::new(&ae_cfg, ae_vb).map_err(|e| {
                error!("Failed to load VAE: {}", e);
                e
            })?;
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
                    clip_tokenizer,
                    ae,
                    vae_device,
                    ae_cpu,
                    text_cache: PromptTextCache::new(PROMPT_TEXT_CACHE_CAP),
                    vae_latent_cache: PromptTextCache::new(IMG2IMG_LATENT_CACHE_CAP),
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
