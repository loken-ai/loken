//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// If `arch` is phi2-class AND a projector blob was located for this
/// model, load the CLIP vision tower and return a `GenericHeteroVision`
/// variant. Otherwise (text-only model, no projector, or non-phi2 arch)
/// return the plain `GenericHetero(text)`. A CLIP load failure is
/// non-fatal: it falls back to text-only so a corrupt projector doesn't
/// take down the entire LM load.
/// Last-resort CPU placement for a generic-arch GGUF. Re-parses the
/// mmap (the prior `Content` was consumed by the failed GPU attempt),
/// builds an all-CPU plan, and forces kv_quant off (Q8/Q4 KV require
/// CUDA). Used by the load-time OOM-recovery ladder when neither the
/// optimistic single-GPU placement nor a forced CUDA split fits.
pub(super) fn load_generic_on_cpu(
    mmap_bytes: &[u8],
    // What backs `mmap_bytes`. Taken alongside them so the pair cannot come apart: this is
    // the last-resort placement, reached only by models too large to fit anywhere else, and
    // it is precisely where copying every tensor into owned host memory costs the most.
    mmap_owner: &std::sync::Arc<memmap2::Mmap>,
    arch: &str,
    num_layers: usize,
    file_size: u64,
    projector_blob: Option<&std::path::Path>,
) -> AnyResult<BoxedModelBackend> {
    let content_fb =
        gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap_owner.clone())
            .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
    let cpu_plan = crate::inference::place::layer_executor::HeteroPlan::calculate(
        num_layers,
        file_size,
        &[],
        &[],
        1.0,
    );
    let empty_devs = std::collections::HashMap::new();
    match GenericHeteroTransformer::from_gguf_with_kv_quant(
        content_fb,
        mmap_bytes,
        &empty_devs,
        &cpu_plan,
        KvQuant::Off,
        None,
    ) {
        Ok(m) => {
            info!("✅ {} loaded on CPU (fallback, kv_quant forced off)", arch);
            try_wrap_with_vision(m, arch, projector_blob)
        }
        Err(e2) => Err(anyhow!("Failed to load {arch}: {e2}")),
    }
}

/// GPU+CPU hybrid OOM-recovery. Reached when the optimistic and forced
/// even-GPU-split plans both OOM at load (the flat weights+KV per-layer
/// accounting under-reserves for the activation peak / CUDA scratch, so
/// `calculate_with_kv` over-committed the GPUs). Instead of collapsing to
/// pure CPU - which throws away ALL GPU acceleration - we retry with
/// progressively tighter GPU budgets. A smaller budget makes
/// `calculate_with_kv` place fewer layers on the GPUs and spill the
/// remainder to CPU, yielding a genuine GPU0(+GPU1)+CPU hybrid that fits.
/// Only the all-CPU plan is the true last resort.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_generic_hybrid_or_cpu(
    mmap_bytes: &[u8],
    // Threaded alongside the bytes for the same reason as in `load_generic_on_cpu`: a
    // re-parse that loses it turns every weight into committed host memory, silently.
    mmap_owner: &std::sync::Arc<memmap2::Mmap>,
    arch: &str,
    num_layers: usize,
    file_size: u64,
    cuda_devices: &[(usize, u64)],
    hetero_cuda_devs: &std::collections::HashMap<usize, Device>,
    kv_bytes_per_layer: u64,
    kv_quant: KvQuant,
    effective_ctx: usize,
    projector_blob: Option<&std::path::Path>,
) -> AnyResult<BoxedModelBackend> {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
    // Tighten the GPU budget in steps so the planner leaves headroom for the
    // under-counted runtime memory and overflows the rest to CPU. Each step
    // keeps more layers on the (much faster) GPUs than the next.
    for frac_pct in [70u64, 50, 35, 20] {
        // The budgets in `cuda_devices` were measured BEFORE the failed
        // attempts that landed us here - whose dropped allocations still sit
        // in the stream-ordered pools and would per-layer-OOM this tier's
        // load into a silent CPU spill (a campaign: gemma4:31b
        // "56/60 layers on GPU" plan executed as ~5 GB on GPU -> 1.2 tok/s).
        // Trim first so the retry actually has the memory its plan assumes.
        #[cfg(feature = "cuda")]
        release_cuda_pools();
        let conservative: Vec<(usize, u64)> = cuda_devices
            .iter()
            .map(|(idx, mem)| (*idx, mem.saturating_mul(frac_pct) / 100))
            .filter(|(_, m)| *m > 0)
            .collect();
        if conservative.is_empty() {
            break;
        }
        let plan = HeteroPlan::calculate_with_kv(
            num_layers,
            file_size,
            &conservative,
            &[],
            1.0,
            kv_bytes_per_layer,
        );
        let gpu_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .map(|s| s.layer_end - s.layer_start)
            .sum();
        let has_cpu = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Cpu));
        // Only worth loading when it's an actual GPU+CPU hybrid: some layers
        // still on GPU (else it's no better than the all-CPU fallback) AND a
        // CPU spill (else it's the same over-committed GPU plan that just
        // OOM'd). Skip to a tighter budget otherwise.
        if gpu_layers == 0 || !has_cpu {
            continue;
        }
        let content_fb =
            gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap_owner.clone())
                .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
        match GenericHeteroTransformer::from_gguf_with_kv_quant(
            content_fb,
            mmap_bytes,
            hetero_cuda_devs,
            &plan,
            kv_quant,
            Some(effective_ctx),
        ) {
            Ok(m) => {
                info!(
                    "✅ {} recovered via GPU+CPU HYBRID ({}% GPU budget): {}/{} layers on GPU, {} on CPU ({} segments)",
                    arch, frac_pct, gpu_layers, num_layers, num_layers - gpu_layers, plan.segments.len()
                );
                return try_wrap_with_vision(m, arch, projector_blob);
            }
            Err(e) => warn!(
                "GPU+CPU hybrid at {}% GPU budget also OOM'd ({e}); trying a tighter split",
                frac_pct
            ),
        }
    }
    warn!("All GPU+CPU hybrid splits failed; loading pure CPU (last resort)");
    load_generic_on_cpu(
        mmap_bytes,
        mmap_owner,
        arch,
        num_layers,
        file_size,
        projector_blob,
    )
}

/// Resolve a model reference to an AWQ checkpoint directory. Accepts
/// either a direct snapshot dir (`<dir>/config.json` with quant_method==awq)
/// or an HF hub dir (`<dir>/snapshots/<hash>/config.json`). Returns None for
/// anything else - so the normal GGUF/Ollama resolution is untouched.
pub(super) fn resolve_awq_dir(candidate: &str) -> Option<String> {
    use crate::inference::load::awq_loader::find_awq_model_dir;
    if candidate.is_empty() {
        return None;
    }
    if find_awq_model_dir(candidate).is_some() {
        return Some(candidate.to_string());
    }
    let snaps = format!("{candidate}/snapshots");
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        // Pick the first snapshot whose config.json is AWQ (HF caches usually
        // hold one snapshot per revision).
        let mut dirs: Vec<String> = rd
            .flatten()
            .map(|e| e.path().to_string_lossy().to_string())
            .collect();
        dirs.sort();
        for p in dirs {
            if find_awq_model_dir(&p).is_some() {
                return Some(p);
            }
        }
    }
    None
}

/// Resolve an HF repo id (`org/name[:tag]`) to an AWQ hub directory by mapping
/// it to the standard HF cache layout `models--org--name/snapshots/<hash>` under
/// each candidate root. Lets a request use the natural model name
/// `Quickpanda/deepcoder-14b-preview-awq` (the validator allows single slashes).
pub(super) fn resolve_awq_repo(model_id: &str, roots: &[PathBuf]) -> Option<String> {
    let base = model_id.split(':').next().unwrap_or(model_id);
    let (org, name) = base.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    let hub = format!("models--{org}--{name}");
    for root in roots {
        let cand = root.join(&hub);
        if let Some(d) = resolve_awq_dir(&cand.to_string_lossy()) {
            return Some(d);
        }
    }
    None
}

/// Read the EOS token id(s) for an AWQ checkpoint from generation_config.json
/// (falling back to config.json). `eos_token_id` may be a scalar or an array.
#[cfg(feature = "cuda")]
pub(super) fn awq_eos_tokens(dir: &str) -> (u32, Vec<u32>) {
    let read = |f: &str| {
        std::fs::read_to_string(format!("{dir}/{f}"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    };
    let mut ids: Vec<u32> = Vec::new();
    if let Some(v) = read("generation_config.json").or_else(|| read("config.json")) {
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(i) = n.as_u64() {
                    ids.push(i as u32);
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for x in a {
                    if let Some(i) = x.as_u64() {
                        ids.push(i as u32);
                    }
                }
            }
            _ => {}
        }
    }
    if ids.is_empty() {
        ids.push(0); // safe fallback; sampler also honors tokenizer-side stops
    }
    (ids[0], ids[1..].to_vec())
}

/// Build a `LoadedModelState` from an AWQ (HF safetensors) checkpoint.
/// Self-contained - bypasses the GGUF machinery entirely. Single-GPU placement
/// (cuda:0); the 2-GPU split for deepseek-r1:32b is deferred.
#[cfg(feature = "cuda")]
pub(super) fn load_awq_model_state(
    awq_dir: &str,
    model_id: &str,
    user_context_length: usize,
) -> AnyResult<LoadedModelState> {
    let cfg = crate::inference::load::awq_loader::parse_config_json(awq_dir)?;

    // Collect the safetensors shards (any count).
    let mut shards: Vec<String> = std::fs::read_dir(awq_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(anyhow!("AWQ: no *.safetensors shards in {awq_dir}"));
    }
    let file_size: u64 = shards
        .iter()
        .filter_map(|s| std::fs::metadata(s).ok().map(|m| m.len()))
        .sum();

    // Single-GPU placement on the FASTEST probed card (the AWQ loader is single-device; it
    // takes the lowest-keyed map entry, so hand it exactly one - never a fixed index 0).
    // Fall back to the fastest card WITHOUT the demand subtraction rather than to index
    // 0: when another subsystem's declared demand covers every card, the demand-aware
    // probe returns nothing, and `unwrap_or(0)` then names a device by position - the
    // one thing placement here must never do. It is the same ranking either way, only
    // without the deduction that left no candidate.
    let probed = {
        let with_demand = crate::inference::place::device_probe::probe_cuda_devices_for("llm", 0);
        if with_demand.is_empty() {
            crate::inference::place::device_probe::probe_cuda_devices(0)
        } else {
            with_demand
        }
    };
    let fastest = probed.first().map(|(i, _, _)| *i).unwrap_or(0);
    let mut cuda = std::collections::HashMap::new();
    cuda.insert(fastest, Device::new_cuda(fastest)?);
    let device = cuda.get(&fastest).unwrap().clone();

    let context_length = if user_context_length > 0 {
        user_context_length
    } else {
        4096
    };

    info!(
        "🔄 Loading AWQ checkpoint {} (arch={}, {} layers, {} shards, ctx={})",
        awq_dir,
        cfg.gguf_arch,
        cfg.n_layers,
        shards.len(),
        context_length
    );
    let t0 = std::time::Instant::now();
    let model = GenericHeteroTransformer::from_awq_safetensors(
        awq_dir,
        &shards,
        &cuda,
        Some(context_length),
    )?;
    info!(
        "✅ AWQ model loaded in {:.1}s ({:.2} GB)",
        t0.elapsed().as_secs_f64(),
        file_size as f64 / 1e9
    );

    let tokenizer = Tokenizer::from_file(format!("{awq_dir}/tokenizer.json"))
        .map_err(|e| anyhow!("AWQ tokenizer load: {e}"))?;
    let (eos_token_id, eos_token_ids_extra) = awq_eos_tokens(awq_dir);

    Ok(LoadedModelState {
        name: model_id.to_string(),
        num_layers: cfg.n_layers,
        hidden_size: cfg.hidden_size,
        num_heads: cfg.n_head,
        vocab_size: cfg.vocab_size,
        context_length,
        eos_token_id,
        moondream_graph: None,
        moondream_decode_count: 0,
        eos_token_ids_extra,
        file_size,
        model: Box::new(GenericBackend(model)),
        tokenizer,
        device,
        image_embeds: None,
        qwen35_image: None,
        image_embed_cache: None,
        grammar_factory: std::sync::OnceLock::new(),
    })
}

pub(super) fn try_wrap_with_vision(
    text: GenericHeteroTransformer,
    arch: &str,
    projector_blob: Option<&std::path::Path>,
) -> AnyResult<BoxedModelBackend> {
    // Phi-class arches are the ones that ship as Ollama dual-blob today
    // (moondream -> phi2). Llama-arch + a pixtral mmproj = Pixtral-12B.
    let phi_class = matches!(arch, "phi" | "phi2" | "phi3" | "phi4");
    let llama_class = matches!(arch, "llama" | "mistral");
    let blob = match (phi_class || llama_class, projector_blob) {
        (true, Some(p)) => p,
        _ => return Ok(Box::new(GenericBackend(text))),
    };
    // Sniff the projector type from the mmproj GGUF metadata so the right
    // tower loads (clip.projector_type: "pixtral" vs moondream's CLIP).
    let projector_type = sniff_projector_type(blob).unwrap_or_default();
    if llama_class && projector_type != "pixtral" {
        return Ok(Box::new(GenericBackend(text))); // llama text-only
    }
    // Pick a CUDA device for the CLIP vision tower. Prefer a GPU the
    // text model does NOT have LAYERS on (its pool isn't competing
    // with the text-model KV cache + activations). Fix:
    // was using `cuda_device_ordinals()` which returns ALL detected
    // GPUs in the model's device map (not just placed-on ones)  -
    // for moondream's text-model-on-GPU-0 + GPU-1-detected case, the
    // "find different GPU" search returned None and the fallback
    // landed CLIP back on GPU 0, OOMing on first request. The
    // `layer_cuda_ordinals()` accessor returns only GPUs that
    // actually hold text-model layers, so the search finds GPU 1.
    let busy_text_gpus = text.layer_cuda_ordinals();
    let nvml = nvml_wrapper::Nvml::init().ok();
    let total_gpus = nvml
        .as_ref()
        .and_then(|n| n.device_count().ok())
        .unwrap_or(0) as usize;
    let mut chosen: Option<usize> = (0..total_gpus).find(|i| !busy_text_gpus.contains(i));
    if chosen.is_none() {
        chosen = busy_text_gpus.iter().next().copied();
    }
    let device = chosen
        .and_then(|idx| crate::tensor::Device::new_cuda(idx).ok())
        .unwrap_or(crate::tensor::Device::Cpu);
    info!(
        "📷 Loading {} vision tower from {} onto {:?}",
        if projector_type == "pixtral" {
            "Pixtral"
        } else {
            "CLIP"
        },
        blob.display(),
        device.location(),
    );
    let t0 = std::time::Instant::now();
    let vision = if projector_type == "pixtral" {
        // Pixtral-ViT lives on the native substrate; mirror the chosen device.
        let ndev = match device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => {
                crate::tensor::Device::new_cuda(gpu_id).unwrap_or(crate::tensor::Device::Cpu)
            }
            _ => crate::tensor::Device::Cpu,
        };
        match crate::inference::model::pixtral::PixtralVision::load_mmproj(
            &blob.to_string_lossy(),
            &ndev,
        ) {
            Ok(v) => VisionTower::Pixtral(v),
            Err(e) => {
                warn!(
                    "⚠️  Pixtral vision load failed ({}); model will work for text only",
                    e
                );
                return Ok(Box::new(GenericBackend(text)));
            }
        }
    } else {
        match crate::inference::model::moondream::vision::MoondreamVisionEncoder::from_clip_gguf(
            blob, &device,
        ) {
            Ok(v) => VisionTower::Clip(v),
            Err(e) => {
                warn!(
                    "⚠️  CLIP load failed ({}); model will work for text only",
                    e
                );
                return Ok(Box::new(GenericBackend(text)));
            }
        }
    };
    info!("✅ Vision tower loaded in {} ms", t0.elapsed().as_millis());
    Ok(Box::new(GenericVisionBackend { text, vision }))
}

/// Read `clip.projector_type` from an mmproj GGUF (cheap: metadata only).
pub(super) fn sniff_projector_type(path: &std::path::Path) -> Option<String> {
    let content = crate::tensor::quantized::gguf_file::open_header(path).ok()?;
    content
        .metadata
        .iter()
        .find(|(k, _)| *k == "clip.projector_type")
        .and_then(|(_, v)| v.to_string().ok().cloned())
}

/// Try to move a loaded `cb_eligible` GPU `GenericHetero` model into a
/// `ContinuousServer` worker for batched concurrent serving. Sizes the paged KV
/// pool from free VRAM (single-digit-MB-per-block decode KV). Returns the model
/// back unchanged on any failure so the caller keeps the serial path.
#[cfg(feature = "cuda")]
pub(super) fn try_spawn_continuous(
    model: GenericHeteroTransformer,
    eos: u32,
    context_length: usize,
) -> std::result::Result<
    crate::inference::serve::continuous_serve::ContinuousServer,
    GenericHeteroTransformer,
> {
    use crate::inference::serve::continuous_serve::ContinuousServer;
    let (n_kv, hd, n_layers, _vocab) = model.paged_geometry();
    let block_size = 16usize;
    let bytes_per_block = 2 * n_kv * hd * 2 * n_layers * block_size; // K+V, F16, all layers
    let dev = model.compute_device();
    let free = dev
        .as_cuda_device()
        .ok()
        .and_then(|cd| cd.cuda_stream().context().mem_get_info().ok())
        .map(|(f, _)| f as usize)
        .unwrap_or(0);
    // Reserve for activations + cuBLAS workspace (~1.5 GB) AND the shared CUDA-
    // graph capture arena (~2 GB) so the KV pool doesn't starve the arena (an
    // undersized arena -> overflow -> permanent eager fallback -> throughput loss).
    let reserve = (1536usize + 2048) << 20;
    let usable = free.saturating_sub(reserve);
    let max_running = 32usize; // concurrent decode batch-width cap
    let cap_ctx = context_length.min(8192);
    let want_blocks = (max_running * cap_ctx).div_ceil(block_size);
    let fit_blocks = if bytes_per_block > 0 {
        usable / bytes_per_block
    } else {
        0
    };
    let num_blocks = want_blocks.min(fit_blocks);
    let min_blocks = (max_running * 256).div_ceil(block_size); // ~256 tokens/seq floor
    if num_blocks < min_blocks {
        warn!("CB serve: only {num_blocks} KV blocks fit ({} MB usable) < min {min_blocks} - staying serial",
            usable >> 20);
        return Err(model);
    }
    info!(
        "CB serve: ContinuousServer - {num_blocks} blocks x {block_size} = {} KV slots, \
           max_running {max_running}, pool {} MB",
        num_blocks * block_size,
        (num_blocks * bytes_per_block) >> 20
    );
    let arc = std::sync::Arc::new(model);
    match ContinuousServer::spawn(
        std::sync::Arc::clone(&arc),
        eos,
        num_blocks,
        block_size,
        max_running,
        1 << 20,
    ) {
        Ok(s) => Ok(s),
        Err(e) => {
            warn!("CB serve: spawn failed ({e}) - staying serial");
            // spawn drops its model ref on every error path before the worker
            // thread takes ownership, so the original Arc is now unique.
            Err(std::sync::Arc::try_unwrap(arc)
                .unwrap_or_else(|_| unreachable!("spawn failed yet worker holds the model")))
        }
    }
}

/// If `state` holds a `cb_eligible` GPU `GenericHetero`, swap it for a `Continuous`
/// worker (batched paged decode - beats serial 2.4-3.3x under concurrent load,
/// with automatic prefix caching).
///
/// DISABLED IN CODE: the paged decode path returns INCORRECT output
/// (garbage / immediate EOS) for any prompt that prefills >256 tokens (>16 KV
/// blocks) - reproduced on mistral-nemo long-context, while the serial path is
/// coherent and beats ollama. Slot mapping is consistent, so the fault is subtler
/// in the paged prefill / first decode and needs tensor-dump instrumentation.
/// Correctness wins over the unvalidated throughput gain, so the
/// worker stays off (no knob) until the paged path is fixed - then delete this
/// early return to restore it.
pub(super) fn cb_maybe_wrap(state: &mut LoadedModelState) {
    #[cfg(feature = "cuda")]
    {
        // Opt-in via config.toml `[inference] continuous_batching = true`.
        // Default OFF: single-stream decode is faster on the serial path;
        // CB is the concurrent-throughput (vLLM) regime. The paged-rope
        // >256-prefill bug that originally forced this off was fixed.
        let cb_enabled = crate::config::Config::load_default()
            .ok()
            .and_then(|c| c.inference.continuous_batching)
            .unwrap_or(false);
        if cb_enabled {
            if !state.model.cb_eligible_gpu() {
                return;
            }
            let (eos, ctx) = (state.eos_token_id, state.context_length);
            let taken: BoxedModelBackend =
                std::mem::replace(&mut state.model, Box::new(TakenBackend));
            let m = match taken.take_generic() {
                Ok(m) => m,
                Err(other) => {
                    state.model = other;
                    return;
                }
            };
            state.model = match try_spawn_continuous(m, eos, ctx) {
                Ok(s) => Box::new(ContinuousBackend(std::sync::Arc::new(s))),
                Err(m) => Box::new(GenericBackend(m)),
            };
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = state;
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for MoondreamGraphState {}
#[cfg(feature = "cuda")]
unsafe impl Sync for MoondreamGraphState {}

/// Full state when a model is loaded
/// CUDA graph capture state for QuantizedMoondream.
/// Captured once after warmup; replayed each decode token to collapse
/// per-launch overhead from ~720 launches/token down to 1 graph launch.
/// Falls back to plain forward on any replay failure (cleared on
/// failure; recaptured on next eligible warmup).
#[cfg(feature = "cuda")]
pub(super) struct MoondreamGraphState {
    pub(super) graph: crate::tensor::cuda_ext::CudaGraph,
}

pub(super) struct LoadedModelState {
    pub(super) name: String,
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) num_heads: usize,
    pub(super) vocab_size: usize,
    pub(super) context_length: usize,
    pub(super) eos_token_id: u32,
    /// Lazily captured CUDA graph for QuantizedMoondream decode (Step 5).
    #[cfg(feature = "cuda")]
    pub(super) moondream_graph: Option<MoondreamGraphState>,
    /// Decode tokens seen since model load. Drives moondream graph
    /// capture warmup gating (capture fires once after N warmup tokens).
    pub(super) moondream_decode_count: u32,
    /// Additional EOS-equivalent token IDs (e.g. Gemma4's
    /// `tokenizer.ggml.eos_token_ids` = [1, 106, 50] where 106 and 50
    /// are turn-end markers used in the chat template). Single
    /// `eos_token_id` is the primary one; this set captures all that
    /// should also terminate generation.
    pub(super) eos_token_ids_extra: Vec<u32>,
    pub(super) file_size: u64,
    pub(super) model: BoxedModelBackend,
    pub(super) tokenizer: Tokenizer,
    pub(super) device: Device,
    /// Pre-encoded image embeddings for the current request (vision models only)
    pub(super) image_embeds: Option<Tensor>,
    /// qwen35moe (Qwen3-VL) raw preprocessed image for the current request:
    /// `(pixel_values [n_patches,1536], patch grid (gh,gw))`. Unlike moondream
    /// (pre-encoded + prepended), qwen35 keeps the raw patches and runs the ViT
    /// + splice inside `forward_with_image` at prefill. None = no image.
    pub(super) qwen35_image: Option<(Tensor, (usize, usize))>,
    /// One-slot cache for the most-recent vision encode: (image_hash,
    /// projector_output). Same-image requests (bench, chat with shared
    /// image, multi-turn captioning) skip the CLIP forward and reuse the
    /// stored projector output - matches Ollama's prompt-cache semantics
    /// for vision. Single-entry to keep memory bounded; a bigger LRU
    /// could ship later if the use case shows up.
    pub(super) image_embed_cache: Option<(u64, Tensor)>,
    /// Lazily-built llguidance parser factory. Constructed on first
    /// grammar-constrained request from the live tokenizer; cached for reuse
    /// across requests on the same loaded model.
    pub(super) grammar_factory: std::sync::OnceLock<std::sync::Arc<llguidance::ParserFactory>>,
}

/// Snapshot of one session's cached conversation state. The KV cache itself
/// lives inside `LoadedModelState.model` (a single shared cache) - this
/// struct only records which token sequence the cache currently contains, so
/// the next request can find a common prefix and skip redundant prefill.
#[derive(Clone)]
pub(crate) struct SessionState {
    /// Full token sequence currently represented in the KV cache
    /// (prompt + generated). For vision sessions, this records ONLY the
    /// post-image text tokens (the image embeds occupy positions
    /// [0, image_prefix_len) but aren't in the token stream).
    pub(super) tokens: Vec<u32>,
    /// Model name that owned the cache. If the engine swaps models the
    /// cache is invalidated.
    pub(super) model_name: String,
    /// Hash of the image whose embeddings are in cache positions
    /// [0, image_prefix_len). None for text-only sessions. Set by the
    /// vision prefill path to enable cross-call prefix-KV reuse when the
    /// same image is sent again.
    pub(super) image_hash: Option<u64>,
    /// Number of cache positions occupied by the image prefix
    /// (typically 1 BOS + 729 image embeds = 730 for moondream).
    /// Tokens in `tokens` are at positions [image_prefix_len, ...).
    pub(super) image_prefix_len: usize,
    /// KV rows the cache ACTUALLY holds.
    ///
    /// NOT `tokens.len()`, which is what every reuse decision used to assume. A
    /// sampled token is recorded the moment it is produced but only written to the
    /// cache when it is FORWARDED, so the last one of a generation is always named by
    /// `tokens` and absent from the KV; a rejected speculative draft trims rows the
    /// list still names, widening the gap further. Reusing a prefix longer than the
    /// cache holds builds the attention mask for positions that do not exist, and the
    /// next request dies on the shape - `cannot broadcast [191, 620] to
    /// [1, 16, 191, 617]`, the two lengths in plain sight.
    pub(super) kv_len: usize,
}

/// Longest prefix of `prompt` the resident cache can actually serve.
///
/// Two things must agree and did not: the token list names what was PRODUCED, `kv_len`
/// counts what was WRITTEN. Taking the common prefix of the lists alone yields a start
/// position past the end of the cache, and the mask built for it names columns the
/// attention tensor does not have.
/// How many leading tokens of `prompt_tokens` the resident KV can be reused for.
///
/// ONE implementation, called by both the streaming and non-streaming paths. They each
/// carried a copy of a four-branch version of this and the copies had drifted: the
/// streaming one returned 0 without trimming when there was nothing to reuse, leaving the
/// previous request's KV in place while the new prompt prefilled from position 0.
///
/// The rule, stated once: cut the KV to exactly what is being reused, and always leave at
/// least one token for the forward to run on. Every case follows - no entry gives 0, a
/// full match gives len-1, a partial or extending match gives the shared prefix.
pub(crate) fn kv_reuse_start(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    sessions: &std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, SessionState>>>,
    model_name: &str,
    prompt_tokens: &[u32],
    disabled: bool,
    shift: bool,
    disk: Option<(&crate::inference::cache::kv_disk::KvDiskStore, usize)>,
) -> usize {
    let mut reuse = if disabled || !model.supports_trim_kv() {
        None
    } else {
        let g = sessions.blocking_lock();
        match g.get(GLOBAL_PROMPT_CACHE_KEY) {
            Some(s) if s.model_name == model_name && s.image_hash.is_none() => {
                Some(reusable_prefix(&s.tokens, prompt_tokens, s.kv_len))
            }
            _ => None,
        }
    };
    // Reuse is only sound on a cold-run chunk boundary. A cold run prefills in fixed
    // chunks from position 0; a warm run re-prefills the tail from `keep`. Off a boundary
    // the tail runs GEMM shapes a cold run never used at those positions, and the
    // reassociated arithmetic flips greedy argmax wherever the top two candidates are
    // close. Rounding down makes the warm tail exactly cold's tail chunks - a fixed
    // remainder was measured insufficient, since any shape difference reassociates.
    let chunk = crate::inference::engine::model_backend::adaptive_prefill_chunk(model.widest_ffn());
    // A snapshot of another conversation that shares more of this prompt than the
    // resident KV does becomes the resident: the resident sequence itself was copied
    // aside when its request ended, so nothing is lost by the swap.
    if !disabled && model.supports_trim_kv() {
        let cur_keep = reuse
            .map(|c| chunk_aligned_keep(c, prompt_tokens.len(), chunk))
            .unwrap_or(0);
        if let Some((index, snap_common)) = model.best_kv_snapshot(prompt_tokens) {
            if chunk_aligned_keep(snap_common, prompt_tokens.len(), chunk) > cur_keep {
                match model.restore_kv_snapshot(index) {
                    Ok((tokens, kv_len)) => {
                        let mut g = sessions.blocking_lock();
                        g.insert(
                            GLOBAL_PROMPT_CACHE_KEY.to_string(),
                            SessionState {
                                tokens,
                                model_name: model_name.to_string(),
                                image_hash: None,
                                image_prefix_len: 0,
                                kv_len,
                            },
                        );
                        tracing::info!(
                            "kv snapshot restored: {snap_common} tokens shared with the prompt, {kv_len} resident"
                        );
                        reuse = Some(snap_common);
                    }
                    Err(e) => tracing::warn!("kv snapshot restore failed: {e}; resident KV kept"),
                }
            }
        }
    }
    // Then the disk: a sequence written after an earlier request, or an earlier run,
    // that covers more of the prompt than anything in memory. Its blocks come back as
    // a snapshot, which is restored like the others.
    if let (Some((store, cap)), false) = (disk, disabled) {
        if let Some((n_layers, _, _)) = model.kv_layout() {
            let cur_keep = reuse
                .map(|c| chunk_aligned_keep(c, prompt_tokens.len(), chunk))
                .unwrap_or(0);
            let layout = kv_layout_id(model);
            if let Some((mi, covered)) = store.best(model_name, layout, prompt_tokens) {
                if chunk_aligned_keep(covered, prompt_tokens.len(), chunk) > cur_keep {
                    let blocks = covered / store.block_tokens();
                    let loaded = store
                        .load(mi, blocks, n_layers)
                        .map_err(|e| crate::tensor::Error::msg(e.to_string()))
                        .and_then(|rows| {
                            let dt = store.manifest_kv_dtype(mi);
                            model.import_kv_snapshot(prompt_tokens[..covered].to_vec(), covered, rows, cap, dt)
                        })
                        .and_then(|()| {
                            let (index, _) = model
                                .best_kv_snapshot(prompt_tokens)
                                .ok_or_else(|| crate::tensor::Error::msg("kv disk: imported snapshot not found".to_string()))?;
                            model.restore_kv_snapshot(index)
                        });
                    match loaded {
                        Ok((tokens, kv_len)) => {
                            let mut g = sessions.blocking_lock();
                            g.insert(
                                GLOBAL_PROMPT_CACHE_KEY.to_string(),
                                SessionState {
                                    tokens,
                                    model_name: model_name.to_string(),
                                    image_hash: None,
                                    image_prefix_len: 0,
                                    kv_len,
                                },
                            );
                            tracing::info!("kv disk: {covered} tokens of the prompt restored from disk");
                            reuse = Some(covered);
                        }
                        Err(e) => tracing::warn!("kv disk restore failed: {e}; resident KV kept"),
                    }
                }
            }
        }
    }
    // A prompt clamped to the window keeps its first token and its tail; once a
    // conversation outgrows the window the common prefix with the resident sequence
    // is that first token alone, and every turn re-prefills the whole window. The tail
    // is still resident, at positions `discard` further on: shift it down and prefill
    // only what is new. The shifted keys are re-phased, not recomputed, so the result
    // is not the cold run's bit for bit; the option says the caller accepts that, and
    // with it the chunk grid below: a shifted tail is kept whole, since rounding it to
    // a grid that can exceed the window would keep nothing.
    let mut shifted: Option<usize> = None;
    if shift && model.supports_shift_kv() && reuse.is_some_and(|c| c < chunk) {
        let mut g = sessions.blocking_lock();
        if let Some(s) = g.get_mut(GLOBAL_PROMPT_CACHE_KEY) {
            if let Some((discard, reused)) = shift_reuse_plan(
                &s.tokens,
                prompt_tokens,
                s.kv_len,
                SHIFT_KEEP,
                SHIFT_PROBE,
                SHIFT_PROBE,
            ) {
                match model.shift_kv_tail(SHIFT_KEEP + discard, discard) {
                    Ok(()) => {
                        s.tokens.drain(SHIFT_KEEP..SHIFT_KEEP + discard);
                        s.kv_len = s.kv_len.saturating_sub(discard);
                        tracing::info!(
                            "context shift: dropped {discard} tokens after the first {SHIFT_KEEP}, {reused} reused"
                        );
                        shifted = Some(SHIFT_KEEP + reused);
                    }
                    Err(e) => {
                        tracing::warn!("context shift refused: {e}; cold prefill");
                        model.trim_kv(0);
                        s.tokens.clear();
                        s.kv_len = 0;
                        reuse = None;
                    }
                }
            }
        }
    }
    let keep = match shifted {
        Some(kept) => kept.min(prompt_tokens.len().saturating_sub(1)),
        None => reuse
            .map(|common| chunk_aligned_keep(common, prompt_tokens.len(), chunk))
            .unwrap_or(0),
    };
    if model.supports_trim_kv() {
        model.trim_kv(keep);
    }
    keep
}

/// What identifies a KV layout on disk: the row geometry. Two builds of one model with
/// the same geometry read each other's blocks; a different quantisation of the weights
/// changes the values, not the layout, and the model name keeps those apart.
pub(crate) fn kv_layout_id(model: &dyn crate::inference::engine::model_backend::ModelBackend) -> u64 {
    match model.kv_layout() {
        Some((layers, n_kv, hd)) => ((layers as u64) << 40) | ((n_kv as u64) << 20) | hd as u64,
        None => 0,
    }
}

/// Writes the snapshot just taken for `tokens` to the disk tier, block by block.
pub(crate) fn persist_kv_snapshot(
    store: &crate::inference::cache::kv_disk::KvDiskStore,
    model: &dyn crate::inference::engine::model_backend::ModelBackend,
    model_name: &str,
    tokens: &[u32],
) {
    let Some((index, common)) = model.best_kv_snapshot(tokens) else {
        return;
    };
    let Some((_, kv_len)) = model.kv_snapshot_ref(index) else {
        return;
    };
    if common < kv_len.min(tokens.len()) {
        return; // the table holds another sequence; nothing of this one to write
    }
    // A windowed (SWA) model keeps only a recent window in some layers, not the whole
    // prefix, so its snapshot cannot be laid out as a token-prefix of uniform-length
    // blocks. Persist only when every layer holds the same [0, kv_len] prefix.
    if model.kv_snapshot_uniform_len(index) != Some(kv_len) {
        tracing::debug!("kv disk: snapshot is not prefix-uniform (windowed layers); not persisted");
        return;
    }
    let layout = kv_layout_id(model);
    let kv_dtype = model.kv_snapshot_dtype(index).unwrap_or(0);
    if let Err(e) = store.persist(model_name, layout, tokens, kv_len, kv_dtype, |from, to| {
        model
            .export_kv_rows(index, from, to)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }) {
        tracing::warn!("kv disk persist failed: {e}");
    }
}

/// Head a clamped prompt keeps, the same one `clamp_prompt_to_window` keeps.
const SHIFT_KEEP: usize = 1;
/// Tokens compared to locate the prompt's tail in the resident sequence before the
/// match is extended; long enough that chat text does not repeat it by chance.
const SHIFT_PROBE: usize = 64;

/// Where a clamped prompt's tail sits in the resident sequence: `(discard, reused)` such
/// that `cached[n_keep + discard..][..reused] == prompt[n_keep..][..reused]`, found by
/// matching a `probe`-token window then extending it. `None` when the heads differ, the
/// window occurs nowhere after the head, or the match is shorter than `min_reuse`.
pub(crate) fn shift_reuse_plan(
    cached: &[u32],
    prompt: &[u32],
    kv_len: usize,
    n_keep: usize,
    probe: usize,
    min_reuse: usize,
) -> Option<(usize, usize)> {
    let cached = &cached[..kv_len.min(cached.len())];
    if cached.len() <= n_keep || prompt.len() <= n_keep || cached[..n_keep] != prompt[..n_keep] {
        return None;
    }
    let head = &prompt[n_keep..];
    let body = &cached[n_keep..];
    let probe = probe.min(head.len());
    if probe == 0 || body.len() < probe + 1 {
        return None;
    }
    for discard in 1..=body.len() - probe {
        if body[discard..discard + probe] == head[..probe] {
            let reused = body[discard..]
                .iter()
                .zip(head)
                .take_while(|(a, b)| a == b)
                .count();
            return (reused >= min_reuse).then_some((discard, reused));
        }
    }
    None
}

/// The largest reusable prefix that sits on a cold-run chunk boundary, always leaving at
/// least the final token to re-prefill so the call produces logits.
pub(crate) fn chunk_aligned_keep(common: usize, prompt_len: usize, chunk: usize) -> usize {
    let capped = common.min(prompt_len.saturating_sub(1));
    capped - capped % chunk.max(1)
}

pub(crate) fn reusable_prefix(cached: &[u32], prompt: &[u32], kv_len: usize) -> usize {
    cached
        .iter()
        .zip(prompt.iter())
        .take_while(|(a, b)| a == b)
        .count()
        .min(kv_len)
}

#[cfg(test)]
mod prompt_cache_reuse_tests {
    use super::shift_reuse_plan;

    #[test]
    fn shift_plan_finds_the_tail_of_a_grown_conversation() {
        // Resident: BOS then 1..=100. The next turn was clamped to BOS + the last 60
        // of the old window + 10 new tokens: 40 tokens fell out after the head.
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let prompt: Vec<u32> = std::iter::once(0).chain(41..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &prompt, cached.len(), 1, 8, 16),
            Some((40, 60))
        );
    }

    #[test]
    fn shift_plan_refuses_a_different_head_or_an_absent_tail() {
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let other_head: Vec<u32> = std::iter::once(7).chain(41..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &other_head, cached.len(), 1, 8, 16),
            None
        );
        let elsewhere: Vec<u32> = std::iter::once(0).chain(500..=560).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &elsewhere, cached.len(), 1, 8, 16),
            None
        );
    }

    #[test]
    fn shift_plan_wants_a_match_worth_a_chunk_and_honours_kv_len() {
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let short: Vec<u32> = std::iter::once(0)
            .chain(91..=100)
            .chain(200..=230)
            .collect();
        assert_eq!(
            shift_reuse_plan(&cached, &short, cached.len(), 1, 8, 16),
            None
        );
        // Only 50 tokens are in the KV: the match cannot extend past them.
        let prompt: Vec<u32> = std::iter::once(0).chain(21..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &prompt, 51, 1, 8, 16),
            Some((20, 30))
        );
    }

    use super::{chunk_aligned_keep, reusable_prefix};

    /// Reuse snaps DOWN to the chunk grid of a cold prefill: the warm tail must be the
    /// cold run's exact tail chunks, or the boundary arithmetic differs and greedy flips.
    #[test]
    fn reuse_snaps_down_to_a_cold_chunk_boundary() {
        assert_eq!(chunk_aligned_keep(1200, 1300, 512), 1024);
        assert_eq!(chunk_aligned_keep(512, 1300, 512), 512);
        assert_eq!(
            chunk_aligned_keep(511, 1300, 512),
            0,
            "below one chunk nothing is safe"
        );
    }

    /// A fully-cached prompt still re-prefills its final chunk: the last token must go
    /// through the forward to produce logits, from the same boundary a cold run used.
    #[test]
    fn a_fully_cached_prompt_reprefills_from_the_last_boundary() {
        assert_eq!(chunk_aligned_keep(1024, 1024, 512), 512);
        assert_eq!(chunk_aligned_keep(513, 513, 512), 512);
        assert_eq!(chunk_aligned_keep(512, 512, 512), 0);
    }

    /// The failure this exists for: a generation leaves its last sampled token in the
    /// list and out of the cache, so a repeat of the same prompt asked to resume from a
    /// position the cache never held.
    #[test]
    fn never_reuses_more_than_the_cache_holds() {
        let cached = [1u32, 2, 3, 4, 5];
        // Four rows written, five tokens named - the classic commit-delayed tail.
        assert_eq!(reusable_prefix(&cached, &[1, 2, 3, 4, 5], 4), 4);
        // A rejected speculative draft can trim several rows at once.
        assert_eq!(reusable_prefix(&cached, &[1, 2, 3, 4, 5], 2), 2);
    }

    #[test]
    fn a_shorter_divergence_still_wins() {
        // Divergence before the cache end is the binding limit, not kv_len.
        assert_eq!(reusable_prefix(&[1, 2, 9], &[1, 2, 3, 4], 3), 2);
    }

    #[test]
    fn an_empty_cache_reuses_nothing() {
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3], 0), 0);
    }
}
