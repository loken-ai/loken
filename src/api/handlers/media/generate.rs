//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Did every one of these failures come from the CLIENT stopping the render?
///
/// A cancellation is not a server error. It arrived as one - HTTP 500, logged at ERROR
/// - so a user pressing Cancel, or a client that reissues a request, saw "Internal
/// Server Error" and the log recorded a fault that never happened. It also buried the
/// real failures: eight cancellations in a row look exactly like eight crashes.
pub(super) fn all_cancelled(failures: &[String]) -> bool {
    !failures.is_empty()
        && failures.iter().all(|f| {
            let l = f.to_lowercase();
            l.contains("cancel") || l.contains("aborted by the client")
        })
}

/// What a stopped render should answer: the client closed the request, so nothing was
/// produced and nothing went wrong.
pub(super) const CLIENT_CLOSED_REQUEST: u16 = 499;

/// Did this render die for want of VRAM?
///
/// The substrate tags allocation failures with a stable marker precisely so callers
/// can branch on memory pressure, but nothing in the image path ever asked - so an
/// exhausted card produced the same opaque failure as a corrupt checkpoint, and the
/// one recovery that would have worked was never attempted.
pub(super) fn is_vram_exhaustion(e: &(dyn std::error::Error + Send + Sync)) -> bool {
    let s = e.to_string();
    s.contains("[oom]") || s.contains("out of memory") || s.contains("out_of_memory")
}

/// Render, and if the render runs out of VRAM, RE-PLAN and try once more.
///
/// Placement reserves what the request needs, but the card is shared: another engine,
/// another process, or a concurrent request can take the room between the decision
/// and the denoise. The model is resident by then, so the request has nowhere to go -
/// and since nothing about the resident model changes on its own, every request after
/// it fails the same way. That is what a warm server returning one identical failure
/// per request looks like from the outside.
///
/// Dropping the resident returns its card, and the reload plans again against what is
/// actually free - a split or a spill, which is slow but finishes. One retry only: if
/// the second attempt also runs out, the pressure is not transient and the error is
/// the honest answer.
/// Whether this node has the weights `model_name` resolves to.
pub(crate) fn image_served_here(state: &APIServer, model_name: &str) -> bool {
    requested_checkpoint(&state.huggingface_models_dir, model_name).is_some()
}

/// Whether one card of this node holds the family's hot component and this request's
/// scratch whole: the same figure the admission below asks the pressure protocol for.
/// A model that only fits split against the host is one a peer with a larger card
/// should take.
pub(crate) fn image_fits_a_card(
    state: &APIServer,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
) -> bool {
    let Ok(loader) = image_family_loader(image_family(model_name)) else {
        return true;
    };
    let hot = family_hot_bytes(&state.huggingface_models_dir, loader, model_name)
        + family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom);
    let largest = crate::inference::place::device_probe::probe_cuda_gpus(
        state.default_inference_config.max_gpu_memory_fraction,
    )
    .iter()
    .map(|g| g.available)
    .max()
    .unwrap_or(0);
    largest >= hot
}

/// Whether a peer's catalogue holds a checkpoint of the same image family as
/// `model_name`. Requests name a family by alias as often as by checkpoint, and the
/// catalogue lists checkpoints, so the family is what the two have in common.
pub(crate) fn serves_image_family(
    peer: &crate::distributed::membership::NodeState,
    model_name: &str,
) -> bool {
    let family = image_family(model_name);
    peer.serves.as_ref().is_some_and(|catalogue| {
        catalogue
            .iter()
            .any(|m| is_image_gen_model(m) && image_family(m) == family)
    })
}

pub(super) async fn generate_image_resilient(
    state: &APIServer,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
    prompt: &str,
    params: crate::inference::engine::image_engine::ImageGenParams,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Enough attempts to reach the guaranteed floor. Each exhaustion raises the
    // pressure one notch, and the top notch places the model on the host, which always
    // fits - so the sequence terminates in something that runs rather than in an error.
    // A single retry was not enough: at the first notch the model may still be planned
    // mostly on the card that just ran out.
    const ATTEMPTS: usize = 5;
    // START FROM NO PRESSURE, for the same reason the LLM load does: the degrade level is
    // this request's escalation, not a standing property of the machine. Left over from an
    // earlier request it shrinks every card by 4 GB a notch and sends work to the host
    // while VRAM sits free.
    crate::inference::place::vram_manager::vram_degrade_reset();
    // Make this render's activation peak visible to every OTHER subsystem for as long as
    // it runs. NVML reports what is ALLOCATED, and a denoise allocates its scratch as it
    // goes, so a model loading alongside a render reads the card as far emptier than it
    // is about to be and places on top of memory this request has already committed to.
    // The weights are deliberately NOT declared: they are resident and already visible,
    // and counting them twice would push the other subsystem to the host for nothing.
    // The family has to resolve to a loader before it can be reserved for: a family with
    // no budget of its own must be refused by name here, never declared at another
    // family's figure.
    let loader = image_family_loader(image_family(model_name))?;
    let _demand = crate::inference::place::vram_manager::declare_demand(
        "media",
        family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom),
    );
    for attempt in 0..ATTEMPTS {
        match state
            .image_engine
            .generate_image(prompt, params.clone())
            .await
        {
            Ok(v) => {
                // The render finished, so whatever pressure it raised is over. Leaving
                // it raised would keep spilling later requests to the host for as long
                // as the process lives.
                crate::inference::place::vram_manager::vram_degrade_reset();
                return Ok(v);
            }
            Err(e) if is_vram_exhaustion(e.as_ref()) && attempt + 1 < ATTEMPTS => {
                // NOTHING IS RECORDED HERE.
                //
                // The store behind this holds what a render was measured to HOLD. The free
                // memory of the resident card at the moment an attempt died is a different
                // quantity: one is a demand, the other is how much room there happened to be
                // where it stopped. A render that exhausts has not measured its peak - it
                // has measured where it was interrupted.
                //
                // And the store only rises, so a single failure became the demand for
                // every later request of that shape, for as long as the process lives.
                // Observed: a 1536 square attempt died with 9.53 GB free, that figure
                // became the demand, and the re-plan it produced asked 16.04 GB of a
                // 16.4 GB card - the failure had pushed the next placement towards the
                // edge instead of away from it.
                //
                // The escalation this was standing in for is the pressure counter below,
                // which exists for exactly this and is bounded. Two mechanisms for one
                // decision, and the one that decided was the one measuring the wrong
                // thing.
                // NO REFUSAL. I put one here - "this machine needs X GB and has Y,
                // about NxN fits" - on the reasoning that a mostly-host placement
                // cannot finish inside a request, so an immediate answer beats a
                // timeout. That is not the contract: the system degrades, it does not
                // decline. A render that takes the slow path is a render; a 500 is
                // nothing, and it arrived on a machine that had just done the same
                // size moments earlier, because the figure it compared against was
                // free VRAM at that instant - which still held the model this very
                // request was about to unload.
                //
                // The escalation below is the fallback, and it ends somewhere that
                // always fits.
                // Raise the pressure BEFORE re-planning. Re-planning against unchanged
                // numbers hands back a byte-identical placement that fails identically -
                // observed directly, which is what this counter fixes.
                let level = crate::inference::place::vram_manager::vram_degrade();
                tracing::warn!(
                    "image: {model_name} ran out of VRAM rendering {}x{} ({e}) - pressure now \
                     {level}, dropping it and re-planning further out rather than failing",
                    geom.width,
                    geom.height,
                );
                state.image_engine.unload().await;
                ensure_image_model_loaded(state, model_name, geom)
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            }
            Err(e) => {
                crate::inference::place::vram_manager::vram_degrade_reset();
                return Err(e);
            }
        }
    }
    unreachable!("the loop returns on the last attempt")
}

/// A spawned task that must not outlive the request that wanted it.
///
/// A load driven from inside an SSE generator runs in a task of its own so its progress
/// can be forwarded while it happens. Nothing links that task to the connection, so a
/// client that stops reading leaves gigabytes loading for nobody - and the engine
/// answers the NEXT request with "every GPU is busy" on behalf of a request that has
/// gone. Aborting drops the loader's future, which fires the cancel guards it already
/// carries.
pub(crate) struct AbortOnDrop<T>(pub(crate) tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// `geom` is the geometry the caller is about to generate at. It is REQUIRED, not
/// optional with a default: placement reserves scratch for this request, and a
/// stand-in resolution is exactly the fixed reserve this replaced.
pub(crate) async fn ensure_image_model_loaded(
    state: &APIServer,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
) -> Result<(), String> {
    ensure_image_model_loaded_reporting(state, model_name, geom, None).await
}

/// [`ensure_image_model_loaded`], with the loader's stage lines and weight counts sent
/// to `report` as they happen.
///
/// The channel is what a STREAMING route forwards to its client. Without one the loaders
/// build a `LoadStage` with no sink, so every stage line and every tensor count is
/// discarded on the floor - which is why `/v1/images/generations` streamed the denoise
/// alone and nothing at all for the minute before it.
pub(crate) async fn ensure_image_model_loaded_reporting(
    state: &APIServer,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
    report: Option<tokio::sync::mpsc::Sender<ImageEngineLoadingProgress>>,
) -> Result<(), String> {
    // Resolve the loader BEFORE anything is unloaded or reserved: a family nobody wired
    // must fail by name here, never fall through to another family's loader - and the
    // VRAM budgets below are keyed on it too, so it also decides what this load reserves.
    let loader = image_family_loader(image_family(model_name))?;
    let current = state.image_engine.model_name().await;
    // Same FAMILY is not enough: a Ray drop-in and its base share the family but are different
    // checkpoints - the resident must be swapped whenever the requested variant differs, or a
    // request for one silently generates with the other.
    // When the resident tracks its checkpoint FILE (Flux), compare files: two tags
    // over one file share the resident model, two different checkpoints never alias
    // (a family+ray-flag comparison silently served FLUX.1-schnell requests from a
    // resident Rayflux fine-tune). Otherwise fall back to family identity.
    let resident_ckpt = state.image_engine.resident_ckpt_id().await;
    // File comparison only applies when the REQUEST resolves to a local checkpoint.
    // A tag served from the HF cache (FLUX.1-schnell, candle-flux) has no local file,
    // so requiring equality marked it different from itself and reloaded the whole
    // model on EVERY request - a silent ~14 s tax per render. Those tags fall back to
    // family + Ray identity, which still prevents the aliasing this guards against:
    // a Ray fine-tune can never satisfy a base-model request, or the reverse.
    let requested_local = requested_checkpoint(&state.huggingface_models_dir, model_name)
        .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()));
    let kind_matches = resident_serves_request(
        current.as_deref(),
        resident_ckpt.as_deref(),
        model_name,
        requested_local.as_deref(),
    );
    // GPU REPATRIATION: a model that loaded while VRAM was tight stays demoted (split
    // across cards, or spilled to CPU) for as long as it is resident, paying transfers
    // on every step long after the pressure is gone. When the resident IS demoted and
    // a single card can now hold its hot component plus this family's runtime reserve,
    // drop it so the load below re-plans it whole. Only on a request for that same
    // model, so the reload cost buys a better placement for the request that pays it.
    let mut repatriate = false;
    if kind_matches && state.image_engine.resident_is_demoted().await {
        // Ask for what the placement will ACTUALLY need: the weights AND the runtime
        // scratch that shares the card. `family_hot_bytes` is the weights;
        // `family_runtime_bytes` is the scratch, and adding it ONCE is the demand.
        // Adding it twice - which this did - asks for a card that does not exist and
        // silently made repatriation unreachable for every large family.
        let reserve = family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom);
        let hot = family_hot_bytes(&state.huggingface_models_dir, loader, model_name) + reserve;
        let free: Vec<u64> = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .map(|(_, f, _)| f)
            .collect();
        let fits_whole = free.iter().any(|f| *f >= hot);
        // A split ACROSS THE GPUS is a normal placement; a spill to the CPU is not.
        // Testing only "does one card hold it whole" left a model whose weights exceed
        // any single card running partly on the host forever - which is what the user
        // sees as a generation that pins the CPU with both GPUs idle. If the cards can
        // hold it between them, re-planning is worth the reload whatever one card can
        // do alone.
        let fits_across = free.iter().sum::<u64>() >= hot;
        if fits_whole || fits_across {
            tracing::info!(
                "image: resident {model_name} is split/spilled and the GPUs can now take it \
                 ({:.1} GB incl. {:.1} GB reserve; {} whole, {} across) - re-planning",
                hot as f64 / 1e9,
                reserve as f64 / 1e9,
                if fits_whole { "fits" } else { "does not fit" },
                if fits_across { "fits" } else { "does not fit" },
            );
            repatriate = true;
        }
    }
    // THE RESIDENT WAS PLACED FOR A SMALLER REQUEST.
    //
    // A placement reserves denoise scratch for the geometry that triggered it, and a
    // later request for the same model can ask for four times the tokens - so the
    // resident has to be re-planned when the new request needs more than the placed one
    // did. That is the ONLY thing this asks.
    //
    // It used to ask whether the card still had the whole demand FREE, which is a
    // different question with a wrong answer: after a load, free VRAM sits near the
    // reserve by design, because the resident's own weights are what consumed it. So the
    // check fired on every request, unloaded twelve gigabytes and reloaded them between
    // two variations of the same picture - each one slower than the last until one
    // failed outright. A guard that trips in the nominal case is worse than no guard.
    let mut restretch = false;
    if kind_matches && !repatriate {
        if let Some(placed) = state.image_engine.resident_placed_for().await {
            if placed != geom {
                let need =
                    family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom);
                let had =
                    family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, placed);
                if need > had {
                    tracing::info!(
                        "image: resident {model_name} was placed for {}x{} ({:.1} GB) and this \
                         request needs {:.1} GB for {}x{} - re-planning",
                        placed.width,
                        placed.height,
                        had as f64 / 1e9,
                        need as f64 / 1e9,
                        geom.width,
                        geom.height,
                    );
                    restretch = true;
                }
            }
        }
    }
    if state.image_engine.is_loaded().await && (!kind_matches || repatriate || restretch) {
        state.image_engine.unload().await;
    }
    if !state.image_engine.is_loaded().await {
        // Pressure protocol (vram_manager): make sure SOME GPU can host the family's hot
        // component. The probe inside trims the mempools first (freed tensors from prior work
        // otherwise read as used); idle residents of other engines are reclaimed ONLY when the
        // hot component would otherwise land on CPU - hetero placement around them is always
        // preferred.
        // The pressure protocol must ask for what the placement will ACTUALLY need:
        // weights AND the runtime scratch. Asking for weights alone let a card that
        // fits the model exactly win, and the generation then OOM'd on its first
        // workspace allocation (observed: 'mmq workspace' after a chat model had
        // taken the other card).
        let hot = family_hot_bytes(&state.huggingface_models_dir, loader, model_name)
            + family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom);
        // WAIT for the gap instead of spilling. A resident is only reclaimable while
        // idle, and a chat model under steady load is busy almost continuously - one
        // pass then reports "no room", the loader puts every DiT block on the CPU,
        // and a 1024^2 render runs past the request timeout without ever producing
        // an image (observed: 90 chat requests served while the single image request
        // sat on the CPU path for 600 s). The VRAM is not gone, it is busy.
        use crate::inference::place::vram_manager::Headroom;
        match crate::inference::place::vram_manager::ensure_gpu_headroom_within(
            "image",
            hot,
            0,
            std::time::Duration::from_secs(IMAGE_HEADROOM_WAIT_SECS),
        )
        .await
        {
            // No CUDA at all: the CPU path is the intended one here.
            Headroom::Ready | Headroom::NoGpu => {}
            Headroom::Busy => {
                return Err(format!(
                    "every GPU is busy with an in-flight generation and {model_name} needs                      {:.1} GB; retry in a moment (loading it on the CPU instead would not                      finish within the request timeout)",
                    hot as f64 / 1e9
                ));
            }
        }
        // A headroom check is not a RESERVATION: between the check above and the
        // upload below, another engine can take the card - and it does, under
        // concurrent chat/audio load. The loaders now refuse to silently put a hot
        // transformer on the CPU (that produced a 600 s hang that returned nothing),
        // so a lost race surfaces as a load error. Wait for the next gap and try
        // again a bounded number of times before telling the caller to retry.
        type LoadErr = Box<dyn std::error::Error + Send + Sync>;
        let mut load_result: Result<(), LoadErr> =
            Err("image load not attempted".to_string().into());
        // Serialise against any OTHER subsystem's load. The retry loop below exists
        // because a load can lose the card to a concurrent one; this stops the race from
        // happening rather than recovering from it afterwards. Taken AFTER the headroom
        // protocol on purpose - that step can wait on a busy resident, and holding the
        // admission lock through the wait would block the very loader it is waiting for.
        let _admission = crate::inference::place::vram_manager::load_admission().await;
        for attempt in 1..=IMAGE_LOAD_ATTEMPTS {
            // Load-scoped cancellation: if the client disconnects while the (long) load runs,
            // this future is dropped, the guard fires, and the blocking loader bails at its
            // next per-tensor/per-block check instead of decoding gigabytes for nobody.
            let load_cancel = crate::inference::serve::cancel::CancelToken::new();
            let load_guard = crate::inference::serve::cancel::CancelGuard::new(load_cancel.clone());
            // Cloned per attempt: a retry builds a new `LoadStage`, and the caller's
            // receiver has to outlive all of them.
            let report = report.clone();
            load_result = load_image_family(
                &state.image_engine,
                &state.huggingface_models_dir,
                loader,
                model_name,
                geom,
                None,
                report,
                Some(load_cancel.clone()),
            )
            .await;
            load_guard.disarm();
            if load_result.is_ok() || attempt == IMAGE_LOAD_ATTEMPTS {
                break;
            }
            tracing::warn!(
                "image: load attempt {attempt}/{IMAGE_LOAD_ATTEMPTS} lost the card to another \
                 engine ({}); waiting for a gap and retrying",
                load_result
                    .as_ref()
                    .err()
                    .map(|e| e.to_string())
                    .unwrap_or_default()
            );
            // Drop whatever partial state the failed load left, then wait for room
            // again - the resident that took the card is usually idle by now.
            state.image_engine.unload().await;
            if crate::inference::place::vram_manager::ensure_gpu_headroom_within(
                "image",
                hot,
                0,
                std::time::Duration::from_secs(IMAGE_HEADROOM_WAIT_SECS),
            )
            .await
                == Headroom::Busy
            {
                break;
            }
        }
        load_result.map_err(|e| format!("load image model: {e}"))?;
    } else {
        // ALREADY RESIDENT: the pressure protocol above only runs on the load path,
        // so a generation against a warm image model never checked whether some
        // OTHER engine had meanwhile filled the card. That is the "edit -> enhance
        // -> edit" failure: the enhancer left a chat model resident and the second
        // edit died allocating its MMQ workspace. Ask for the family's runtime
        // headroom here too - reclaiming idle residents of other engines when, and
        // only when, the card would otherwise be too tight.
        // Runtime headroom for ONE generation of this family. Derived where the
        // engine exposes a derivation, else a family floor measured from the
        // engines' own peaks (T5/CLIP encode + DiT activations + VAE decode).
        // Covering only one family here is what let the failure come back on
        // /v1/images/generations with a Flux model after an enhance left a chat
        // model resident.
        let need = family_runtime_bytes(&state.huggingface_models_dir, loader, model_name, geom);
        if need > 0 {
            crate::inference::place::vram_manager::ensure_gpu_headroom("image", need, 0).await;
        }
    }
    crate::inference::place::vram_manager::touch("image");
    Ok(())
}

/// Handle image generation requests (Flux or Z-Image models)
pub(crate) async fn handle_image_generation(
    state: &APIServer,
    model_name: &str,
    request: &OllamaGenerateRequest,
) -> Result<Response, ApiError> {
    // One media job at a time: two diffusion engines cannot share these cards,
    // and letting them try is what produced the OOM storm (see `media_gate`).
    let _media_guard = state.media_lock().await;

    // Extract image generation params from options
    let options = request.options.as_ref();
    let input_image = request
        .images
        .as_ref()
        .and_then(|imgs| imgs.first().cloned());
    // Apply the same per-image size cap the /v1/images/* multipart paths
    // use. The DefaultBodyLimit gates the WHOLE request body, so
    // without this check a single Ollama-shape POST could ship an outsized
    // base64 image through `images[0]` - the engine would then try to
    // decode + VAE-encode it and either OOM or take minutes. Estimate
    // decoded bytes from the b64 string length (base64 expands raw bytes
    // by ~4/3) so we reject before paying the decode cost.
    if let Some(ref b64) = input_image {
        let estimated_bytes = b64.len().saturating_mul(3) / 4;
        if let Err(e) = validate_image_input_size(estimated_bytes) {
            return Err(ApiError::Validation(e));
        }
    }
    let defaults = image_model_defaults(model_name).map_err(ApiError::Validation)?;
    let default_steps: u64 = defaults.steps as u64;
    let default_guidance: f64 = defaults.guidance;
    let default_size: u64 = defaults.size as u64;
    // Resolve seed: caller-provided wins, else pick + echo a random
    // one so the user can reproduce a generation they liked. Matches
    // the /v1/images/* behaviour from f8456d0.
    let resolved_seed = options
        .and_then(|o| o.get("seed").and_then(serde_json::Value::as_u64))
        .unwrap_or_else(rand_u64);
    let requested_steps_u64 = options
        .and_then(|o| o.get("num_steps").and_then(serde_json::Value::as_u64))
        .unwrap_or(default_steps);
    // Boundary check the caller-supplied num_steps before it reaches
    // the engine: a value like 100000 would pin the GPU on a single
    // /api/chat request. The check is symmetric with the /v1/images/*
    // endpoints' validation so chat-client behaviour matches the
    // OpenAI-shaped path.
    let requested_steps_u32 = u32::try_from(requested_steps_u64).unwrap_or(u32::MAX);
    if let Err(e) = validate_image_num_steps(requested_steps_u32) {
        return Err(ApiError::Validation(e));
    }
    let requested_height_u64 = options
        .and_then(|o| o.get("height").and_then(serde_json::Value::as_u64))
        .unwrap_or(default_size);
    let requested_width_u64 = options
        .and_then(|o| o.get("width").and_then(serde_json::Value::as_u64))
        .unwrap_or(default_size);
    // Bound-check caller-supplied dimensions before they reach
    // ImageGenParams. Without this an options.width=100000 would
    // try to allocate ~40 GB of latents (OOM/hang). Same cap as the
    // /v1/images/* parser so chat and OpenAI surfaces agree.
    let height_usize = usize::try_from(requested_height_u64).unwrap_or(usize::MAX);
    let width_usize = usize::try_from(requested_width_u64).unwrap_or(usize::MAX);
    // What the placement below must reserve denoise scratch for. Copy, so the
    // detached loader task can take it without cloning anything.
    let geom =
        crate::inference::place::runtime_demand::RequestGeometry::new(width_usize, height_usize);
    if let Err(e) = validate_image_dimensions(width_usize, height_usize) {
        return Err(ApiError::Validation(e));
    }
    // Clamp guidance + strength to the same ranges /v1/images/* uses.
    // clamp_finite_f64 also catches NaN / Inf coming in via the JSON
    // body - NaN strength feeds NaN into the noise mix and produces a
    // black image (or crashes the downstream kernel); Inf guidance
    // propagates NaN through Flux's embedded-guidance tensor. Either
    // value is nonsense the server shouldn't pass through.
    let raw_guidance = options
        .and_then(|o| o.get("guidance").and_then(serde_json::Value::as_f64))
        .unwrap_or(default_guidance);
    let raw_strength = options
        .and_then(|o| o.get("strength").and_then(serde_json::Value::as_f64))
        .unwrap_or(0.75);
    // Solver and sigma curve, both optional: the scheduler decides WHICH noise levels
    // the run visits, the sampler how it moves between two of them.
    let sampler_name = options
        .and_then(|o| o.get("sampler").and_then(serde_json::Value::as_str))
        .map(str::to_string);
    let lora_list =
        parse_loras(options.and_then(|o| o.get("loras"))).map_err(super::ApiError::Validation)?;
    let scheduler_name = options
        .and_then(|o| o.get("scheduler").and_then(serde_json::Value::as_str))
        .map(str::to_string);
    let params = ImageGenParams {
        step_reuse: step_reuse_from(options),
        negative_prompt: options
            .and_then(|o| o.get("negative_prompt"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        regions: parse_regions(options.and_then(|o| o.get("regions"))),
        mask: None,
        control_image: None,
        control_scale: 1.0,
        cancel: crate::inference::serve::cancel::CancelToken::new(),
        height: height_usize,
        width: width_usize,
        num_steps: requested_steps_u64 as usize,
        guidance: clamp_finite_f64(raw_guidance, 0.0, 30.0, default_guidance),
        seed: Some(resolved_seed),
        input_image,
        strength: clamp_finite_f64(raw_strength, 0.0, 1.0, 0.75),
        kontext: false,
        sampler: sampler_name.clone(),
        scheduler: scheduler_name.clone(),
        loras: lora_list.clone(),
    };

    let start = std::time::Instant::now();

    if request.stream {
        // Streaming mode: start SSE immediately and show loading + generation progress.
        //
        // Getting the model up is `ensure_image_model_loaded_reporting`'s job, exactly as
        // on the non-streaming branch below. This route used to carry its OWN copy of the
        // family-to-loader dispatch so it could stream the loader's stage lines, and the
        // copy fell behind: it never grew an SDXL arm, so its `_ =>` default handed every
        // SDXL request to the Flux loader. The reporting channel is what removes the
        // reason to duplicate the table at all.
        let image_engine = state.image_engine.clone();
        let model_name_owned = model_name.to_string();
        let prompt = request.prompt.clone();
        // For re-planning a render that runs out of VRAM: this route names no size,
        // so it renders at the family default and reserves for that.
        let stream_state = state.clone();
        let stream_geom = crate::inference::place::runtime_demand::RequestGeometry::new(
            defaults.size,
            defaults.size,
        );

        let stream = async_stream::stream! {
            // Phase 1: get the model up, REPORTING - the single family-aware entry point,
            // which also owns the unload-on-family-switch, the headroom protocol and the
            // load retry cascade this route never had.
            {
                let (load_tx, mut load_rx) =
                    tokio::sync::mpsc::channel::<ImageEngineLoadingProgress>(32);
                let load_state = stream_state.clone();
                let load_model = model_name_owned.clone();
                // The load runs beside this generator so its stage lines can be forwarded
                // while it happens - and is ABORTED when the generator is dropped, so a
                // client that stops reading does not leave gigabytes loading for nobody.
                let mut load_task = AbortOnDrop(tokio::spawn(async move {
                    ensure_image_model_loaded_reporting(&load_state, &load_model, geom, Some(load_tx))
                        .await
                }));
                // Drained to CHANNEL CLOSE, not to the loader's own "done": a load that
                // lost its card retries, and each attempt announces a completion of its own.
                let mut load_failed: Option<String> = None;
                while let Some(ev) = load_rx.recv().await {
                    match ev {
                        ImageEngineLoadingProgress::Stage(msg) => {
                            let chunk = serde_json::json!({
                                "model": model_name_owned,
                                "created_at": chrono::Utc::now().to_rfc3339(),
                                "response": msg,
                                "done": false,
                            });
                            yield Ok::<_, axum::Error>(Event::default().data(chunk.to_string()));
                        }
                        ImageEngineLoadingProgress::Error(msg) => load_failed = Some(msg),
                        ImageEngineLoadingProgress::Done => {}
                    }
                }
                let outcome = match (&mut load_task.0).await {
                    Ok(r) => r,
                    Err(e) => Err(format!("image load task failed: {e}")),
                };
                if let Err(e) = outcome.map_err(|e| load_failed.unwrap_or(e)) {
                    error!("Failed to load image model: {e}");
                    let chunk = serde_json::json!({
                        "model": model_name_owned,
                        "created_at": chrono::Utc::now().to_rfc3339(),
                        "response": format!("Failed to load image model: {e}"),
                        "done": true,
                        "done_reason": "error",
                        "error": e,
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                    return;
                }
            }

            // Phase 2: Generate image with progress.
            //
            // An exhausted card is a placement that stopped being true, not a broken
            // render: drop the resident so its memory returns, plan again against
            // what is free, and try once. Reachable from BOTH the failed call and the
            // mid-render event, because the denoise reports exhaustion as an event.
            let mut replanned = false;
            'render: loop {
            let attempt_params = params.clone();
            let _cg = crate::inference::serve::cancel::CancelGuard::new(attempt_params.cancel.clone());
            let _res = image_engine.generate_image_stream(&prompt, attempt_params).await;
            _cg.disarm();
            if let Err(e) = &_res {
                if !replanned && is_vram_exhaustion(e.as_ref()) {
                    replanned = true;
                    let level = crate::inference::place::vram_manager::vram_degrade();
                    tracing::warn!("image stream: {model_name_owned} out of VRAM ({e}) - re-planning at pressure {level}");
                    image_engine.unload().await;
                    let _ = ensure_image_model_loaded(&stream_state, &model_name_owned, stream_geom).await;
                    continue 'render;
                }
            }
            match _res {
                Ok(mut rx) => {
                    while let Some(event) = rx.recv().await {
                        if let ImageStreamEvent::Error(msg) = &event {
                            if !replanned
                                && (msg.contains("[oom]")
                                    || msg.contains("out of memory")
                                    || msg.contains("out_of_memory"))
                            {
                                replanned = true;
                                let level = crate::inference::place::vram_manager::vram_degrade();
                                tracing::warn!("image stream: {model_name_owned} out of VRAM mid-render ({msg}) - re-planning at pressure {level}");
                                image_engine.unload().await;
                                let _ = ensure_image_model_loaded(&stream_state, &model_name_owned, stream_geom).await;
                                continue 'render;
                            }
                        }
                        match event {
                            ImageStreamEvent::Progress { completed, total } => {
                                let chunk = serde_json::json!({
                                    "model": model_name_owned,
                                    "created_at": chrono::Utc::now().to_rfc3339(),
                                    "response": format_image_step_progress(completed, total),
                                    "done": false,
                                    "completed": completed,
                                    "total": total,
                                });
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                            ImageStreamEvent::Complete { image_base64 } => {
                                let chunk = serde_json::json!({
                                    "model": model_name_owned,
                                    "created_at": chrono::Utc::now().to_rfc3339(),
                                    "response": format_seed_echo(resolved_seed),
                                    "done": true,
                                    "done_reason": "stop",
                                    "images": [image_base64],
                                });
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                            ImageStreamEvent::Error(e) => {
                                // Engine-layer log already emitted via
                                // log_engine_error; only the wire-format
                                // shaping happens here.
                                let chunk = serde_json::json!({
                                    "error": format!("Image generation error: {}", e),
                                });
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                        }
                    }
                }
                Err(e) => {
                    crate::inference::engine::log_engine_error("image", "stream-init", &e);
                    let chunk = serde_json::json!({
                        "error": format!("Error starting image generation: {e:#}"),
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
            }
            break;
            }
        };
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming: route the family-aware ensure() so a request that
        // switches between Flux and Z-Image unloads the previous model first
        // (otherwise the second load races against the first's still-resident
        // weights and produces CUDA_ERROR_OUT_OF_MEMORY on tight 16 GB GPUs).
        // No explicit geometry on this route: reserve for what the family will
        // actually render, which is its own default size.
        // The family's default is a square side, and it is what this route renders.
        let side = defaults.size;
        let geom = crate::inference::place::runtime_demand::RequestGeometry::new(side, side);
        if let Err(e) = ensure_image_model_loaded(state, model_name, geom).await {
            error!("Failed to load image model: {}", e);
            let mut response = OllamaGenerateResponse::new(
                model_name.to_string(),
                format!("Failed to load image model: {}", e),
            );
            response.done = true;
            response.done_reason = Some("error".to_string());
            return Ok(Json(response).into_response());
        }

        let _cg = crate::inference::serve::cancel::CancelGuard::new(params.cancel.clone());
        let _res = generate_image_resilient(state, model_name, geom, &request.prompt, params).await;
        _cg.disarm();
        match _res {
            Ok(image_base64) => {
                let total_duration = start.elapsed().as_nanos() as u64;
                // Echo the seed in the assistant message body so the
                // GUI surfaces it to the user verbatim. They can
                // copy-paste this back into options.seed on the next
                // request to re-roll the same generation. (The
                // `images` field already carries the bytes; this
                // text annotation is the reproducibility hook.)
                let mut response = OllamaGenerateResponse::new(
                    model_name.to_string(),
                    format_seed_echo(resolved_seed),
                );
                response.done = true;
                response.done_reason = Some("stop".to_string());
                response.total_duration = Some(total_duration);
                response.images = Some(vec![image_base64]);
                Ok(Json(response).into_response())
            }
            Err(e) => {
                error!("Image generation error: {}", e);
                let mut response = OllamaGenerateResponse::new(
                    model_name.to_string(),
                    format!("Image generation error: {}", e),
                );
                response.done = true;
                response.done_reason = Some("error".to_string());
                Ok(Json(response).into_response())
            }
        }
    }
}

// ------------
// OpenAI-compatible image generation
// ------------
