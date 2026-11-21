//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

/// Largest window the runtime reserve prices the KV cache at: the caches start smaller and
/// grow, and a longer window is admitted by the context planner, not by this reserve.
const RESERVE_KV_WINDOW: u64 = 8192;

/// Bytes a KV cache of `elems` elements occupies: a quantised cache by its block geometry
/// (a block of values and its scale), an unquantised one at `native_bytes` per element.
pub(crate) fn kv_cache_bytes(kv_quant: KvQuant, native_bytes: u64, elems: u64) -> u64 {
    use crate::tensor::quantized::GgmlDType;
    let blocks = |d: GgmlDType| elems.div_ceil(d.block_size() as u64) * d.type_size() as u64;
    match kv_quant {
        KvQuant::Off => elems * native_bytes,
        KvQuant::Q8 => blocks(GgmlDType::Q8_0),
        KvQuant::Q4 => blocks(GgmlDType::Q4_0),
    }
}

/// The cache format a model is served with. A quantised format the kernels do not serve for
/// the head shape falls back to the native width. `Off` is raised to Q8 where they do serve it
/// and the level quantises the cache: half the bytes per token read at each decode step, and
/// the device-position kernels a captured decode graph needs. The `reference` level keeps the
/// full-width cache the published model computes with.
pub(crate) fn kv_cache_format(asked: KvQuant, kernels_serve: bool, fidelity: Fidelity) -> KvQuant {
    match asked {
        KvQuant::Off if kernels_serve && fidelity.quantizes_cache() => KvQuant::Q8,
        KvQuant::Off => KvQuant::Off,
        _ if !kernels_serve => KvQuant::Off,
        quantised => quantised,
    }
}

/// The dimensions of a language model a placement is priced from, read from its GGUF header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelDims {
    pub(crate) num_layers: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// The widest feed-forward of a layer, dense or of one expert; 0 when the header names none.
    pub(crate) widest_ffn: usize,
    /// For a model whose residual is wide: what its prefill keeps live on a card per token and the
    /// chunk it wants (see `generic_transformer::hyper::prefill_footprint`).
    pub(crate) wide_prefill: Option<crate::inference::generic_transformer::hyper::PrefillFootprint>,
}

impl ModelDims {
    pub(crate) fn read(content: &gguf_file::Content, arch: &str) -> Self {
        let num_layers = get_gguf_u32(content, &format!("{arch}.num_layers"))
            .or_else(|| get_gguf_u32(content, &format!("{arch}.num_hidden_layers")))
            .or_else(|| get_gguf_u32(content, &format!("{arch}.block_count")))
            .or_else(|| get_gguf_u32(content, "model.num_layers"))
            .or_else(|| get_gguf_u32(content, "model.num_hidden_layers"))
            .or_else(|| get_gguf_u32(content, "num_layers"))
            .or_else(|| get_gguf_u32(content, "num_hidden_layers"))
            .or_else(|| get_gguf_u32(content, "block_count"))
            .unwrap_or(32) as usize;
        let hidden_size = get_gguf_u32(content, &format!("{arch}.hidden_size"))
            .or_else(|| get_gguf_u32(content, &format!("{arch}.embedding_length")))
            .or_else(|| get_gguf_u32(content, "embedding_length"))
            .unwrap_or(4096) as usize;
        let num_heads = get_gguf_u32(content, &format!("{arch}.num_attention_heads"))
            .or_else(|| get_gguf_u32(content, &format!("{arch}.attention.head_count")))
            .unwrap_or(32) as usize;
        let num_kv_heads = get_gguf_u32(content, &format!("{arch}.attention.head_count_kv"))
            .or_else(|| get_gguf_u32(content, &format!("{arch}.num_key_value_heads")))
            .unwrap_or(num_heads as u32) as usize;
        let head_dim = get_gguf_u32(content, &format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if num_heads > 0 {
                    hidden_size / num_heads
                } else {
                    128
                }
            });
        let widest_ffn = ["feed_forward_length", "expert_feed_forward_length"]
            .iter()
            .filter_map(|key| get_gguf_u32(content, &format!("{arch}.{key}")))
            .max()
            .unwrap_or(0) as usize;
        let streams =
            get_gguf_u32(content, &format!("{arch}.hyper_connection.count")).unwrap_or(0) as usize;
        let wide_prefill = (streams > 1).then(|| {
            let experts = get_gguf_u32(content, &format!("{arch}.expert_count"))
                .zip(get_gguf_u32(content, &format!("{arch}.expert_used_count")))
                .map(|(n, k)| (n as usize, k as usize));
            let expert_ffn = get_gguf_u32(content, &format!("{arch}.expert_feed_forward_length"))
                .unwrap_or(0) as usize;
            crate::inference::generic_transformer::hyper::prefill_footprint(
                streams,
                hidden_size,
                experts,
                expert_ffn,
            )
        });
        Self {
            num_layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            widest_ffn,
            wide_prefill,
        }
    }

    /// Card memory a model needs at run time beyond its weights, on each card it uses: its
    /// KV cache at the reserve window (one byte per element, the quantised cache), the cuBLAS
    /// workspace and the chunked-prefill activation peak.
    pub(crate) fn runtime_reserve(&self, context_length: usize) -> u64 {
        let kv_ctx = (context_length as u64).min(RESERVE_KV_WINDOW);
        let kv = kv_cache_bytes(
            KvQuant::Q8,
            0,
            2u64.saturating_mul(self.num_kv_heads as u64)
                .saturating_mul(self.head_dim as u64)
                .saturating_mul(kv_ctx)
                .saturating_mul(self.num_layers as u64),
        );
        kv.saturating_add(self.work_reserve())
    }

    /// What a card keeps free beside the weights and the cache: the cuBLAS workspace and
    /// the activations of one prefill chunk.
    pub(crate) fn work_reserve(&self) -> u64 {
        if let Some(fp) = self.wide_prefill.filter(|w| w.target_chunk > 0) {
            // A wide residual's prefill takes the whole chunk it wants, counted from the buffers
            // it keeps live, beside the mixers' block temporaries and the cuBLAS workspace.
            return crate::inference::place::runtime_demand::CUBLAS_WORKSPACE_BYTES
                .saturating_add((fp.target_chunk as u64).saturating_mul(fp.bytes_per_token as u64))
                .saturating_add(fp.fixed_bytes as u64);
        }
        crate::inference::place::runtime_demand::llm_work_reserve(
            self.hidden_size.max(self.num_kv_heads * self.head_dim),
            PREFILL_CHUNK_TOKENS,
        )
    }

    /// What a card keeps free so that the prefill's chunk sizing, measured against the card's
    /// free memory, runs a whole chunk of `PREFILL_CHUNK_TOKENS`: the cuBLAS workspace and the
    /// chunk's activations over the share of free memory a chunk may take.
    ///
    /// `measured` is what an earlier load's prefill forwards kept live per row on a card, 0 when
    /// none was measured.
    /// A measured row counts every live buffer, as the model's own prefill shape then does.
    pub(crate) fn full_chunk_reserve(&self, measured: u64) -> u64 {
        if self.wide_prefill.is_some() || (self.widest_ffn == 0 && measured == 0) {
            return self.work_reserve();
        }
        let declared =
            crate::inference::engine::model_backend::PrefillShape::from_widest_ffn(self.widest_ffn);
        let shape = match measured as usize {
            m if m > declared.bytes_per_token => {
                crate::inference::engine::model_backend::PrefillShape {
                    bytes_per_token: m,
                    share: declared
                        .share
                        .min(crate::inference::generic_transformer::hetero::COUNTED_SHARE),
                    ..declared
                }
            }
            _ => declared,
        };
        let chunk = (shape.share * shape.bytes_per_token * PREFILL_CHUNK_TOKENS) as u64;
        crate::inference::place::runtime_demand::CUBLAS_WORKSPACE_BYTES
            .saturating_add(chunk)
            .max(self.work_reserve())
    }
}

/// Card memory a load of the GGUF at `path` keeps free on each card it uses, beyond the
/// weights: the runtime reserve at the model's declared window plus the staging the load
/// of that checkpoint churns through.
pub(crate) fn per_card_reserve(path: &std::path::Path) -> Option<u64> {
    reserve_of(&gguf_file::open_header(path).ok()?)
}

/// [`per_card_reserve`] of a header already read.
fn reserve_of(content: &gguf_file::Content) -> Option<u64> {
    let arch = get_gguf_string(content, "general.architecture")?;
    let context_length = trained_context(content, &arch)
        .unwrap_or(crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
    let staging = load_staging(content, &arch);
    Some(ModelDims::read(content, &arch).runtime_reserve(context_length) + staging)
}

/// What the GGUF at `path` asks of the cards, each card it uses keeping [`per_card_reserve`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct CardNeeds {
    /// Every block and the output head, for the model to run alone on one card.
    pub(crate) whole: crate::inference::place::room::CardNeed,
    /// What the cards hold for none of the model's layers to run on the host.
    /// A wide residual's layers are its token mixers.
    /// A mixture's layers leave out its routed experts: the host keeps them beside a layer on a card.
    pub(crate) layers: crate::inference::place::room::CardNeed,
}

/// [`CardNeeds`] of the GGUF at `path`, read from its header.
pub(crate) fn card_needs(path: &std::path::Path) -> Option<CardNeeds> {
    use crate::inference::place::room::CardNeed;
    let content = gguf_file::open_header(path).ok()?;
    let per_card = reserve_of(&content)?;
    let (mut blocks, mut routed) = (0u64, 0u64);
    for (name, t) in &content.tensor_infos {
        let Some((_, tail)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.')) else {
            continue;
        };
        let bytes = t.size_in_bytes() as u64;
        blocks += bytes;
        if super::load::is_routed_expert_tensor(tail) {
            routed += bytes;
        }
    }
    let tensor_bytes = |name: &str| {
        content
            .tensor_infos
            .get(name)
            .map_or(0, |t| t.size_in_bytes() as u64)
    };
    // An untied model uploads its output projection, a tied one its embedding table.
    let head = match tensor_bytes("output.weight") {
        0 => tensor_bytes("token_embd.weight"),
        b => b,
    };
    let whole = if blocks > 0 {
        blocks + head
    } else {
        std::fs::metadata(path).ok()?.len()
    };
    let layers =
        crate::inference::generic_transformer::wide_sparse_plan::card_mixer_bytes(&content)
            .map_or(whole.saturating_sub(routed), |mixers| mixers.total());
    Some(CardNeeds {
        whole: CardNeed {
            bytes: whole,
            per_card,
        },
        layers: CardNeed {
            bytes: layers,
            per_card,
        },
    })
}

/// Card memory a load keeps free on each card it uses for the staging of one upload.
///
/// A wide residual uploads its token mixers' projections one at a time.
/// Any other file uploads its largest placed tensor.
pub(super) fn load_staging(content: &gguf_file::Content, arch: &str) -> u64 {
    let largest =
        crate::inference::generic_transformer::wide_sparse_plan::card_mixer_bytes(content)
            .map_or_else(
                || {
                    crate::inference::generic_transformer::ngram::largest_placed_tensor(
                        content, arch,
                    )
                },
                |mixers| mixers.largest_upload,
            );
    crate::inference::place::runtime_demand::tensor_load_staging(largest)
}

/// Size the rayon pool the host kernels run on, once per process.
///
/// Called from `main` before anything else can touch rayon. It used to sit at the top of the
/// first model load, and by then something had already built the global pool - so
/// `build_global` failed on every start since at least 2026-08-27, the warning was the only
/// trace, and every host kernel ran on rayon's default of one thread per LOGICAL core. On a
/// non-hybrid SMT part that is two siblings contending for one core's execution units and L1,
/// which is what the count below exists to avoid.
pub fn configure_thread_pool() {
    RAYON_INIT.call_once(|| {
        // Detect hybrid CPU topology - if P/E cores are
        // distinguishable, pin worker threads to P-cores via
        // start_handler so E-cores don't steal cycles during
        // prefill. Falls back to all-cores on non-hybrid
        // hardware or when detection fails.
        let topo = crate::cpu::detect_cpu_topology();
        let p_cores: Vec<usize> = topo.get_inference_cores();
        // Hybrid: use P-core count, not physical-core total
        // (otherwise the worker pool spills onto E-cores).
        // Non-hybrid: physical cores (skip HT siblings).
        let n = if topo.is_hybrid && !p_cores.is_empty() {
            p_cores.len()
        } else {
            num_cpus::get_physical()
        };
        let p_cores_for_pin = p_cores.clone();
        // Pin to P-cores only on HYBRID CPUs (to keep workers off the
        // slow E-cores). On non-hybrid SMT, hard-pinning measured worse
        // package energy than leaving the scheduler to spread the
        // physical-core-count pool itself (it idles memory-stalled
        // workers so cores drop P-state); count is the lever, not pin.
        let pin_workers = topo.is_hybrid && !p_cores.is_empty();
        let mut builder = crate::cpu::worker_pool_builder().num_threads(n);
        if pin_workers {
            // Cycle assignments across the P-core list so each
            // rayon worker thread gets pinned to a distinct P-core
            // (modulo when n > p_cores.len(), but that's an
            // explicit cpu_threads override).
            builder = builder.start_handler(move |worker_id| {
                let core_idx = p_cores_for_pin[worker_id % p_cores_for_pin.len()];
                let _ = crate::cpu::set_thread_affinity(&[core_idx]);
            });
        }
        if let Err(e) = builder.build_global() {
            warn!("⚠️  Failed to configure rayon thread pool: {}", e);
        } else if pin_workers {
            info!("🔧 Rayon thread pool: {} workers pinned to P-cores", n);
        } else {
            debug!("🔧 Rayon thread pool initialized: {} threads", n);
        }
    });
}

impl LlmEngine {
    /// Counts of the next loads go to `report`.
    pub fn report_loads_to(&self, report: crate::inference::serve::progress::SharedProgressFn) {
        *self.load_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(report);
    }

    fn load_report(&self) -> Option<crate::inference::serve::progress::SharedProgressFn> {
        self.load_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Loads the configured model. A model that will spill to the host may take a drafter
    /// in the room its placement leaves, once placed (see `plan_drafter`).
    pub async fn load_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.load_model_warming(Warmup::Ahead).await
    }

    /// Loads the configured model for a request that waits on the load.
    /// The load does not warm the prefill path: the request's prompt is the first to run it.
    pub async fn load_model_for_request(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.load_model_warming(Warmup::ByRequest).await
    }

    async fn load_model_warming(&self, warmup: Warmup) -> Result<(), Box<dyn std::error::Error>> {
        let planned = self
            .find_model_file(&self.config.model_id)
            .and_then(|target| {
                let size = std::fs::metadata(&target).ok()?.len();
                self.plan_drafter(&target, size)
            });
        let loaded = self
            .load_model_only(warmup)
            .await
            .map_err(|e| e.to_string());
        if loaded.is_ok() {
            self.load_planned_drafter(planned).await;
            self.calibrate_between_requests().await;
        }
        // The cards the placement opened and the model did not take.
        release_idle_cards();
        Ok(loaded?)
    }

    /// Loads the configured model alone.
    /// `warmup` says whether the load warms the prefill path before it returns.
    pub(super) async fn load_model_only(
        &self,
        warmup: Warmup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model_id = self.config.model_id.clone();
        // The draft head's experts stay on its card only when the drafts run: otherwise
        // their room serves the layers.
        let draft_resident = self.config.recorded_drafts;
        let context_parallel = self.config.context_parallel;
        let kv_lease_secs = self.config.kv_lease_secs;
        crate::inference::generic_transformer::set_probe_dir(self.config.probe_dump_dir.clone());
        let load_start = std::time::Instant::now();
        crate::inference::place::device_probe::debug_vram_by_card("load-entry");
        info!("🔄 Loading model: {}", model_id);
        // Placement below reads free VRAM to decide where the layers go. That reading is
        // only true while nobody else is allocating: a media engine loading at the same
        // moment sees the same free card, plans onto it too, and one of the two runs out
        // mid-upload. Waiting for the other loader costs seconds; planning against its
        // half-taken card costs the request.
        let _admission = crate::inference::place::vram_manager::load_admission().await;
        // The models resident on the cards keep their prompt work declared while this one loads.
        let _arrival = crate::inference::place::reclaim::arrive(&model_id);
        // A load produces an EMPTY KV, so anything the prompt cache remembers about the
        // previous one is a length with nothing behind it. `unload` clears these too;
        // this covers the paths that replace a model without going through it.
        self.sessions.lock().await.clear();
        debug!(
            "📂 Models directory configured: {:?}",
            self.config.models_dir
        );
        // A new model may have more (or less) per-GPU headroom than the last;
        // forget the previous model's adapted prefill chunk so this one starts
        // at the default and re-discovers its own fit.
        reset_adaptive_prefill_chunk();

        // Try to find the model file
        let model_file = match self.find_model_file(&model_id) {
            Some(f) => {
                info!("✅ Model file found at: {}", f.display());
                f
            }
            None => {
                error!(
                    "❌ Model file not found. Searched in: {:?}",
                    self.config.models_dir
                );
                let err_msg = format!("Model file not found for: {}", model_id);
                // `load_model` is an async fn driving on a tokio worker
                // thread - `blocking_lock()` here panicked the worker
                // with 'Cannot block the current thread from within a
                // runtime', taking down the in-flight image-gen request
                // alongside it. Use the async `lock().await` instead.
                let mut last_err = self.last_error.lock().await;
                *last_err = Some(err_msg.clone());
                return Err(err_msg.into());
            }
        };

        // A checkpoint another route serves is refused with the route that serves it.
        if let Some(why) =
            crate::inference::load::awq_loader::text_route_refusal(&model_file.to_string_lossy())
        {
            error!("❌ {why}");
            *self.last_error.lock().await = Some(why.clone());
            return Err(why.into());
        }

        debug!("📂 Found model file: {}", model_file.display());

        let metadata = std::fs::metadata(&model_file)?;
        let file_size = metadata.len();
        debug!(
            "📊 Model file size: {:.2} MB",
            file_size as f64 / 1_000_000.0
        );

        let disable_arc_layers = self.config.disable_arc_layers;
        let kv_quant = self.config.kv_quant;
        let expert_profile = self.config.expert_profile.clone();
        let layer_share = self.config.layer_share;
        let quality = self.config.quality;
        let fidelity = self.config.fidelity;
        #[cfg(test)]
        let level = crate::inference::generic_transformer::LEVEL_LOCK
            .read()
            .unwrap_or_else(|e| e.into_inner());
        crate::inference::generic_transformer::set_float_weights(fidelity.float_weights());
        crate::inference::generic_transformer::set_narrows(fidelity.narrows());
        #[cfg(test)]
        drop(level);
        // What the placement reads of the configuration, as the placement preview reads it.
        let ask = self.placement_ask();
        let state_cache = self.state_cache.clone();
        let user_context_length = self.config.context_length;
        let model_state = self.model_state.clone();
        let cached_model_size = self.cached_model_size.clone();
        let listing = self.listing.clone();
        let prefill_row_bytes = self
            .prefill_row_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        let configured_window = self.config.context_length;
        let planned_kv = self.planned_kv.clone();
        planned_kv.store(0, std::sync::atomic::Ordering::Relaxed);
        // The load sets the decode rate its plan predicts, when the plan gives one.
        let pace = self.pace.clone();
        pace.loaded(&model_id, None);
        let model_id_clone = model_id.clone(); // Clone for the closure
                                               // Resolve the projector blob (vision tower) for dual-blob Ollama
                                               // packages like moondream. None for text-only models. Moved into
                                               // the spawn_blocking closure so the CLIP loader can pick it up
                                               // after the LM has loaded as GenericHetero.
        let projector_blob = self.find_projector_blob_for_model(&model_id_clone);
        // Further files of the checkpoint the manifest names, each holding the routed experts
        // in another format.
        let variant_blobs = self.find_variant_blobs_for_model(&model_id_clone);
        let expert_variants = self.config.expert_variants;
        let generation_defaults = self.generation_defaults_for(&model_id_clone, &model_file);
        // A speech tower that names its text model: the weights and the tokenizer are the text
        // model's; the tower is read from the checkpoint the request named (`projector_blob`).
        let model_file =
            discover::composite_text_dir(&model_file, &self.hub_roots()).unwrap_or(model_file);

        // A text-image pair is served on the embeddings routes, beside no language model.
        if crate::inference::model::clip::pair::claims_checkpoint_dir(&model_file) {
            let dir = model_file.clone();
            let pair = tokio::task::spawn_blocking(move || {
                crate::inference::model::clip::pair::DualEncoder::load(&dir)
            })
            .await??;
            info!(
                "Text-image pair {model_id} loaded: {}-wide vectors in {:.1}s",
                pair.dim(),
                load_start.elapsed().as_secs_f64()
            );
            *self.dual_encoder.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(pair));
            *self.cached_model_size.lock().await =
                crate::inference::model::clip::pair::weight_bytes(&model_file);
            return Ok(());
        }

        // Load model in blocking thread since GGUF parsing and weight loading are CPU-intensive
        let load_report = self.load_report();
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            // The reporter belongs to this thread, the one that reads the weights, and is
            // withdrawn when the closure returns, on an error as on success.
            let _published = load_report.map(crate::inference::serve::progress::scoped::publish);
            crate::inference::place::device_probe::debug_vram_by_card("closure-start");

            // --- 0a. A checkpoint directory of a wide-residual family ---
            if crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::claims_checkpoint(&model_file) {
                let state = load_wide_checkpoint(
                    &model_file,
                    &model_id_clone,
                    user_context_length,
                    quality,
                    kv_quant,
                    state_cache.clone(),
                    generation_defaults.clone(),
                )?;
                let fsz = state.file_size;
                let mut guard = model_state.blocking_lock();
                crate::inference::place::vram_manager::residency_changed();
                *guard = Some(state);
                *listing.lock().unwrap_or_else(|e| e.into_inner()) =
                    super::query::ModelListing::of(guard.as_ref(), configured_window);
                drop(guard);
                let mut size_guard = cached_model_size.blocking_lock();
                *size_guard = fsz;
                return Ok(());
            }

            let placement::Opened {
                mmap,
                content,
                awq,
                split,
                file_size,
            } = placement::open(&model_file, file_size)?;
            // The placeholders a spliced vision projector may ask the vocabulary for.
            let vision_vocab: Vec<(&'static str, u32)> =
                crate::inference::model_facts::vision_placeholders()
                    .filter_map(|p| super::gguf::gguf_token_id(&content, p).map(|id| (p, id)))
                    .collect();
            let generic_weights = placement::generic_weights(&mmap, &awq, &split);
            // Captured now: `content` is consumed by one of the loader arms below, and the
            // page advice can only be given once every layer has a device. The advice is per
            // mapping, since a set's offsets are relative to its part; AWQ shards are not mapped.
            let page_advice: Vec<MappedAdvice> = match (&awq, &split) {
                (Some(_), _) => Vec::new(),
                (None, Some(s)) => s
                    .parts()
                    .iter()
                    .map(|p| MappedAdvice {
                        mmap: p.mmap.clone(),
                        ranges: page_ranges_of(&p.content),
                    })
                    .collect(),
                (None, None) => vec![MappedAdvice {
                    mmap: mmap.clone(),
                    ranges: page_ranges_of(&content),
                }],
            };
            // Set by the generic route when its plan keeps routed experts in host memory.
            let mut host_experts_from: Option<usize> = None;

            // --- 2. Extract metadata ---
            // Empty when the file names none: the generic route then refuses it for the facts
            // it cannot look up.
            let arch = get_gguf_string(&content, "general.architecture").unwrap_or_default();
            let facts = crate::inference::model_facts::facts_or_default(&arch);
            // A file may carry neither the key nor a vocabulary; the embedding's row count is
            // the vocabulary whatever the converter wrote. In a split model the embedding may
            // sit in another part, so the parts are consulted as one.
            let vocab_size = get_gguf_u32(&content, &format!("{arch}.vocab_size"))
                .or_else(|| count_gguf_tokens(&content))
                .or_else(|| {
                    use crate::tensor::quantized::gguf_source::{GgufMeta, SplitGguf};
                    let names = ["token_embd", "token_embd.weight"];
                    let local = names
                        .iter()
                        .find_map(|n| content.tensor_infos.get(*n))
                        .and_then(|i| i.shape.dims().first().copied());
                    local
                        .or_else(|| {
                            let split = SplitGguf::open(&model_file).ok()?;
                            names
                                .iter()
                                .find_map(|n| split.info(n))
                                .and_then(|i| i.shape.dims().first().copied())
                        })
                        .map(|v| v as u32)
                })
                .unwrap_or(32000) as usize;
            // A file without tokenizer metadata may still ship its tokenizer files alongside;
            // they name the end-of-sequence token and carry its id.
            let eos_token_id = get_gguf_u32(&content, "tokenizer.ggml.eos_token_id")
                .or_else(|| super::gguf::sidecar_token_id(&model_file, "eos_token"))
                .unwrap_or_else(|| {
                    // A guess, and one that generation runs past: said out loud rather than
                    // discovered from answers that never stop.
                    warn!(
                        "the file names no end-of-sequence token (tokenizer.ggml.eos_token_id) \
                         and no tokenizer_config.json sits beside it; assuming id 2"
                    );
                    2
                });
            // Some files list several end-of-sequence ids; every one stops generation.
            let mut eos_token_ids_extra: Vec<u32> = match content.metadata.get("tokenizer.ggml.eos_token_ids") {
                Some(gguf_file::Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|v| match v {
                        gguf_file::Value::U32(n) => Some(*n),
                        gguf_file::Value::I32(n) => Some(*n as u32),
                        _ => None,
                    })
                    .filter(|t| *t != eos_token_id)
                    .collect(),
                _ => Vec::new(),
            };
            // A file without the eos_token_ids array stops on the pieces its architecture ends
            // a turn with, resolved by their text in this file's vocabulary.
            if eos_token_ids_extra.is_empty() {
                for text in facts.turn_stop_pieces {
                    match super::gguf::gguf_token_id(&content, text) {
                        Some(tok) if tok != eos_token_id => eos_token_ids_extra.push(tok),
                        Some(_) => {}
                        None => warn!("the vocabulary has no {text} piece; not stopping on it"),
                    }
                }
            }
            if !eos_token_ids_extra.is_empty() {
                info!("📋 EOS token IDs: canonical={} extras={:?}", eos_token_id, eos_token_ids_extra);
            }

            // Extract actual layer count from GGUF (not hardcoded!)
            // Try multiple key variations for different architectures
            let dims = ModelDims::read(&content, &arch);
            let num_layers = dims.num_layers;
            // The context the model serves and the format of its cache.
            let window = placement::window(&content, &arch, &dims, &ask);
            let placement::Window {
                context_length,
                kv_quant,
                ..
            } = window;

            // Log metadata extraction for debugging
            info!("🔍 Architecture detected: {}", arch);
            debug!("📊 Extracted num_layers: {}", num_layers);

            // Debug: log available GGUF keys if num_layers wasn't found
            if num_layers == 32 {
                let keys: Vec<_> = content.metadata.keys().filter(|k| k.contains("layer") || k.contains("hidden")).collect();
                if !keys.is_empty() {
                    warn!("⚠️  Found layer-related keys: {:?}", keys);
                } else {
                    warn!("⚠️  No layer/hidden keys found in GGUF metadata");
                }
                // Show all keys for debugging (first 20)
                let all_keys: Vec<_> = content.metadata.keys().take(20).collect();
                debug!("📋 Available GGUF keys (first 20): {:?}", all_keys);
            }
            // Try multiple key variations for hidden_size (Mistral3 uses embedding_length)
            let ModelDims { hidden_size, num_heads, num_kv_heads, head_dim, .. } = dims;

            info!("📋 Model info: arch={}, layers={}, hidden_size={}, vocab_size={}, context_length={}, eos_token_id={}, n_kv_heads={}, head_dim={}",
                arch, num_layers, hidden_size, vocab_size, context_length, eos_token_id,
                num_kv_heads, head_dim);

            // Compile the shape's attention kernels now rather than on the first decoded token.
            #[cfg(feature = "gpu")]
            if window.kv_quant_supported {
                crate::inference::quantized_cuda::prewarm_shape(head_dim, window.n_q_per_kv);
            }
            // --- 3. Build tokenizer ---
            debug!("🔍 Looking for tokenizer.json...");
            let mut tokenizer: Tokenizer = {
                let tokenizer_path = model_file.parent()
                    .map(|p| p.join("tokenizer.json"))
                    .filter(|p| p.exists());

                // A file converted here carries its checkpoint's own tokenizer, whole: nothing to
                // rebuild, and no sidecar to find.
                if let Some(json) = get_gguf_string(&content, super::gguf::EMBEDDED_TOKENIZER_KEY) {
                    debug!("📖 Loading the tokenizer the file carries");
                    Tokenizer::from_bytes(json.as_bytes())
                        .map_err(|e| anyhow!("Failed to load the file's tokenizer: {e}"))?
                } else if let Some(path) = tokenizer_path {
                    debug!("📖 Loading tokenizer from: {}", path.display());
                    Tokenizer::from_file(&path)
                        .map_err(|e| anyhow!("Failed to load tokenizer from {}: {e}", path.display()))?
                } else {
                    warn!("⚠️  tokenizer.json not found for {}, building from GGUF metadata", &model_id_clone);
                    build_tokenizer_from_gguf(&content)
                        .map_err(|e| { error!("❌ {e}"); e })?
                }
            };

            debug!("✓ Tokenizer loaded");

            // The other tokens the files declare as ending a turn or a sequence end it too.
            // A checkpoint directory declares its chat template beside its weights.
            let template = get_gguf_string(&content, "tokenizer.chat_template").or_else(|| {
                let (dir, _) = awq.as_ref()?;
                crate::api::handlers::path_chat_format(std::path::Path::new(dir)).template
            });
            let eos_token_ids_extra = super::end_tokens::end_ids(
                eos_token_id,
                eos_token_ids_extra,
                Some(&content),
                &tokenizer,
                template.as_deref(),
                &generation_defaults.stop,
            );

            // --- 4. The devices the placement offers this model, and its plan on them ---
            //
            // The placement preview runs the same plan.
            let placed = placement::plan(
                &placement::Source {
                    content: &content,
                    arch: &arch,
                    dims: &dims,
                    weights: &generic_weights,
                    file_size,
                    awq_dir: awq.as_ref().map(|(dir, _)| dir.as_str()),
                    split: split.is_some(),
                },
                &window,
                &ask,
                true,
            )?;
            let runtime_reserve_bytes = placed.runtime_reserve;
            let offer = placed.offer.clone();
            let cuda_devices: Vec<(usize, u64)> = placed.cards.clone();
            // The kernels the first prompt runs compile beside the load.
            #[cfg(feature = "gpu")]
            crate::inference::quantized_cuda::compile_ahead(
                cuda_devices.iter().map(|(card, _)| *card).collect(),
            );
            let model_fits_single_gpu = file_size > 0 && offer.whole_on.is_some();
            let gpu_idx = offer.primary().unwrap_or_default();

            // --- 4.5 The Vulkan and OpenCL devices the plan may use, each by its backend ---
            //
            // The OpenCL enumeration is behind a deadline, because it is a driver call: a card
            // removed while the system ran leaves a device node that blocks every OpenCL client.
            // See `opencl_probe`.
            if !disable_arc_layers {
                for (kind, free) in crate::inference::place::ladder::layer_accelerators(0) {
                    let name = crate::inference::place::ladder::accelerator_facts(kind)
                        .map(|(name, _)| name)
                        .unwrap_or_default();
                    info!("🖥️  {kind}: {name} ({:.1} GB free)", free as f64 / 1e9);
                }
            }

            let mut device = match offer.whole_on.filter(|_| model_fits_single_gpu) {
                Some(idx) => {
                    info!("📊 Model ({:.1} GB) fits in single GPU memory, loading on GPU {}",
                        file_size as f64 / 1e9, idx);
                    // Default stream with event tracking disabled: capturable in
                    // cudaStreamCaptureModeRelaxed, which the engine's graph capture uses.
                    #[cfg_attr(not(feature = "gpu"), allow(clippy::map_identity))]
                    let dev_result: crate::tensor::Result<Device> = crate::tensor::Device::new_cuda(idx)
                        .map(|d| {
                            #[cfg(feature = "gpu")]
                            if let Device::Cuda(ref cd) = d {
                                unsafe { cd.context().disable_event_tracking(); }
                            }
                            d
                        });
                    dev_result.unwrap_or_else(|e| {
                        warn!("CUDA device {} unavailable ({e}), falling back to CPU", idx);
                        Device::Cpu
                    })
                }
                None => {
                    if cuda_devices.is_empty() {
                        debug!("No CUDA GPU memory offered, loading on CPU");
                    } else {
                        info!("📊 Model ({:.1} GB) does not sit whole on one card; the plan spreads it over {} card(s)",
                            file_size as f64 / 1e9, cuda_devices.len());
                    }
                    Device::Cpu
                }
            };

            debug!("✓ Using device: {:?}", device);

            // --- 5. Load model weights based on architecture ---
            debug!("⚖️  Architecture: {}, Device: {:?}", arch, device);
            let weights_t = std::time::Instant::now();
            // The routing counts a run wrote, read once.
            // The generic arm ranks a wide residual's experts by them.
            // The expert caches are filled from them once the model is loaded.
            let profile = expert_profile.as_deref().and_then(|path| {
                use crate::inference::generic_transformer::expert_profile::ExpertProfile;
                match ExpertProfile::read_if_present(std::path::Path::new(path)) {
                    Ok(Some(profile)) => Some(profile),
                    Ok(None) => {
                        debug!("no expert profile at {path} yet");
                        None
                    }
                    Err(e) => {
                        warn!("expert profile not read: {e}");
                        None
                    }
                }
            });
            // The further files of the checkpoint the routed experts are read from.
            let mut variant_files: Vec<crate::tensor::quantized::gguf_source::SplitGguf> = Vec::new();
            // A special loader claims a file by what it contains; every other file takes the
            // generic hetero route, which builds what the file describes or says what it cannot.
            let model: BoxedModelBackend = match placed.layers.as_ref() {
                // Moondream legacy path: a file with the original PyTorch state-dict
                // naming, whatever its architecture key says. A file with llama.cpp
                // naming (token_embd, blk.{i}.attn_qkv) takes the generic route.
                // The legacy model runs whole on one device, placed by the ladder.
                None => {
                    debug!("Loading Moondream vision model from GGUF...");
                    let mut built_on = Device::Cpu;
                    let model = crate::inference::place::ladder::load_whole(
                        "moondream",
                        file_size,
                        runtime_reserve_bytes,
                        |on| {
                            let vb = crate::tensor::quantized::QVarBuilder::from_gguf(&model_file, on)
                                .map_err(|e| anyhow!("Failed to load Moondream GGUF: {e}"))?;
                            let config = crate::inference::model::moondream::quantized::Config::from_checkpoint(&vb)
                                .map_err(|e| anyhow!("Moondream shape: {e}"))?;
                            let model = moondream::Model::new(&config, vb)
                                .map_err(|e| anyhow!("Failed to load Moondream weights: {e}"))?;
                            built_on = on.clone();
                            Ok::<_, anyhow::Error>(model)
                        },
                    )
                    .inspect_err(|e| error!("{e}"))?;
                    info!("Vision model loaded (Moondream v2) on {:?}", built_on.location());
                    device = built_on;
                    Box::new(MoondreamBackend(model))
                }
                Some(layers) => {
                    debug!("⬇️  Loading '{}' as generic hetero transformer (CUDA+CPU)...", arch);
                    // The further files of the checkpoint the manifest names, each checked to be
                    // of it before any weight is placed.
                    // A wide residual reads its routed experts from them as well.
                    if !variant_blobs.is_empty() {
                        if crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::claims(&content) {
                            variant_files = crate::inference::generic_transformer::wide_sparse_variants::open_checked(
                                &model_file,
                                &variant_blobs,
                                &model_id_clone,
                            )
                            .map_err(|e| anyhow!("{e}"))?;
                        } else {
                            warn!(
                                "{} expert variant file(s) of {model_id_clone} left unread: the experts of a wide residual alone are read from them",
                                variant_blobs.len()
                            );
                        }
                    }
                    let demand = layers.demand;
                    let routed = layers.routed;
                    let experts_in_blocks = layers.experts_in_blocks;
                    // The caches' growth the plan counts, declared once the model is loaded.
                    if matches!(layers.kv_quant, KvQuant::Q4 | KvQuant::Q8) {
                        planned_kv.store(
                            layers.planned_ctx as usize,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    if let Some(group) = &layers.first_plan.tp_cards {
                        info!("⬇️  Loading {arch} with every layer split across GPU {group:?}");
                    }
                    {
                            let (first_plan, stem_fits) = (layers.first_plan.clone(), layers.stem_fits);
                            // A handle per card a plan uses, made on first use, with per-slice
                            // event tracking off before any tensor loads: a device owns one
                            // working stream, so the events would order a stream against itself,
                            // and a decode capture that meets them is invalidated. Fresh handles
                            // rather than the engine's `device`, whose stream state is for
                            // capture.
                            let mut handles: std::collections::HashMap<usize, Device> =
                                std::collections::HashMap::new();
                            let model = crate::inference::place::ladder::load_planned(
                                &arch,
                                first_plan.clone(),
                                &demand,
                                &cuda_devices,
                                &layers.accelerators,
                                || false,
                                |plan| {
                                    for idx in crate::inference::place::ladder::cards_of(plan) {
                                        if let std::collections::hash_map::Entry::Vacant(slot) = handles.entry(idx) {
                                            let d = Device::new_cuda(idx)?;
                                            #[cfg(feature = "gpu")]
                                            if let Device::Cuda(ref cd) = d {
                                                unsafe { cd.context().disable_event_tracking(); }
                                            }
                                            slot.insert(d);
                                        }
                                    }
                                    let on_plan: std::collections::HashMap<usize, Device> =
                                        crate::inference::place::ladder::cards_of(plan)
                                            .into_iter()
                                            .filter_map(|idx| handles.get(&idx).map(|d| (idx, d.clone())))
                                            .collect();
                                    // The host alone keeps an unquantised cache that grows to the whole window.
                                    let (kv, ctx) = layers.cache_for(plan);
                                    let ctx = Some(ctx);
                                    let stem_on_card = stem_fits
                                        && crate::inference::place::dry_plan::same_plan(plan, &first_plan);
                                    // The ledger of this load: every conversion a loader declares
                                    // lands in it, and the card growth is compared with what the
                                    // file and those conversions explain.
                                    crate::tensor::conversion::open();
                                    let cards: Vec<usize> = on_plan.keys().copied().collect();
                                    let free_before: Vec<Option<u64>> = cards
                                        .iter()
                                        .map(|&i| crate::inference::place::vram_manager::free_on(i))
                                        .collect();
                                    // The plan is over the layers of the share; the model reads
                                    // them by their place in the stack.
                                    let placed = match layer_share {
                                        Some((from, _)) => plan.shifted(from, num_layers),
                                        None => plan.clone(),
                                    };
                                    let mut m = generic_weights.build(
                                        &on_plan,
                                        &placed,
                                        kv,
                                        ctx,
                                        stem_on_card,
                                        draft_resident,
                                    )?;
                                    // A wide residual's routed experts may take what the cards have
                                    // left beside the plan's cache and a request's work.
                                    // A card the load was offered outside the plan keeps routed
                                    // experts only, unless it takes a range of the cache below.
                                    let offered: Vec<usize> = match context_parallel {
                                        true => Vec::new(),
                                        false => cuda_devices.iter().map(|&(i, _)| i).collect(),
                                    };
                                    m.note_wide_card_rooms(
                                        &placed,
                                        &offered,
                                        demand.kv_bytes_per_layer,
                                        demand.runtime_reserve,
                                    );
                                    // Every card the machine shows, the plan's first, takes a
                                    // range of each attention layer's cache.
                                    if context_parallel {
                                        let mut cards: Vec<Device> = on_plan.values().cloned().collect();
                                        for (i, _) in &cuda_devices {
                                            if !on_plan.contains_key(i) {
                                                if let Ok(d) = Device::new_cuda(*i) {
                                                    cards.push(d);
                                                }
                                            }
                                        }
                                        m.attach_context_shards(&cards);
                                    }
                                    let grown: Option<u64> = cards
                                        .iter()
                                        .zip(&free_before)
                                        .map(|(&i, before)| {
                                            let after = crate::inference::place::vram_manager::free_on(i)?;
                                            Some((*before)?.saturating_sub(after))
                                        })
                                        .sum::<Option<u64>>()
                                        .filter(|_| !cards.is_empty());
                                    let reserve = per_card_reserve(&model_file)
                                        .unwrap_or(0)
                                        .saturating_mul(cards.len() as u64);
                                    crate::tensor::conversion::close(
                                        demand.whole(),
                                        grown.map(|g| g.saturating_sub(reserve)),
                                    );
                                    host_experts_from = plan.host_experts_from;
                                    let placement = plan
                                        .segments
                                        .iter()
                                        .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    // The routed experts the build put in host memory beside a
                                    // layer that runs on a card.
                                    let host_experts = m
                                        .host_expert_runs()
                                        .iter()
                                        .map(|(a, b)| format!("{a}-{b}"))
                                        .collect::<Vec<_>>();
                                    let host_experts = match host_experts.is_empty() {
                                        true => String::new(),
                                        false => format!(", routed experts of layers {} in host memory", host_experts.join(", ")),
                                    };
                                    info!("✅ {arch} loaded: layers {placement}{host_experts} (kv_quant={kv:?}, kv_ctx={ctx:?})");
                                    // The decode rate the plan predicts, which the first answer
                                    // after the load sets against the rate it measures.
                                    // The block bytes of a wide residual's plan hold no routed expert.
                                    let sets = routed.map(|r| {
                                        crate::inference::place::decode_time::RoutedSets::on(plan, r, experts_in_blocks, None)
                                    });
                                    let predicted = demand
                                        .decode_secs(
                                            plan,
                                            &crate::inference::place::ladder::card_bandwidth,
                                            &crate::inference::place::host_read::host_bandwidth,
                                            sets.as_ref(),
                                        )
                                        .filter(|secs| *secs > 0.0)
                                        .map(|secs| 1.0 / secs);
                                    match predicted {
                                        Some(rate) => info!("{arch} decode predicted at {rate:.1} tok/s on this plan"),
                                        None => info!("{arch} decode not predicted on this plan: a device's bandwidth, a measure of the routed experts it runs on the host, or the cost of a split across cards is not known"),
                                    }
                                    pace.loaded(&model_id_clone, predicted);
                                    #[cfg(feature = "gpu")]
                                    for card in crate::inference::place::ladder::cards_of(plan) {
                                        if let Some((used, reserved)) = crate::inference::place::device_probe::pool_bytes(card) {
                                            info!(
                                                "card {card} after the load: pool holds {:.2} GB of live buffers and {:.2} GB cached",
                                                used as f64 / 1e9,
                                                reserved.saturating_sub(used) as f64 / 1e9
                                            );
                                        }
                                    }
                                    Ok::<_, crate::tensor::Error>(m)
                                },
                            )
                            .map_err(|e| anyhow!("Failed to load {arch}: {e}"))?;
                            let mut model = model;
                            // A wide residual whose file carries no routing prior takes the one a
                            // companion file beside it declares.
                            // The profile then ranks its experts, the prior ordering the uncounted.
                            model.locate_routing_prior(&model_file);
                            if let (Some(path), Some(profile)) = (expert_profile.as_deref(), profile.as_ref()) {
                                if let Err(e) = model.prime_routing_heat(profile) {
                                    warn!("expert profile {path} not used: {e}");
                                }
                            }
                            // Each routed expert is read from the file the policy chooses, in the
                            // order of the heat, and the cards that pay keep the hottest.
                            if !variant_files.is_empty() {
                                model
                                    .attach_expert_variants(&variant_files)
                                    .map_err(|e| anyhow!("{model_id_clone}: {e}"))?;
                            }
                            model.place_wide_experts(expert_variants);
                            model.set_expert_quality(quality);
                            if quality != QualityLevel::Exact && layer_share.is_none() {
                                let bos = get_gguf_u32(&content, "tokenizer.ggml.bos_token_id");
                                let file_len = std::fs::metadata(&model_file).map_or(0, |m| m.len());
                                let name = model_file
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                if let Some(ids) =
                                    crate::inference::generic_transformer::routing_calibration::calibration_ids(&tokenizer, bos)
                                {
                                    if let Err(e) =
                                        crate::inference::generic_transformer::routing_calibration::calibrate_loaded(
                                            &mut model,
                                            quality,
                                            &format!("{name}:{file_len}"),
                                            &ids,
                                        )
                                    {
                                        warn!("quality calibration failed, routing stays exact: {e}");
                                    }
                                }
                            }
                            let model = with_companion_draft(model, &model_file, layer_share);
                            try_wrap_with_vision(
                                model,
                                projector_blob.as_deref().filter(|_| layer_share.is_none()),
                                demand.whole().saturating_sub(demand.block_bytes + demand.head_bytes),
                                &vision_vocab,
                            )?
                        }

                }
            };
            debug!("✓ Weights loaded from mmap in {:.1}ms", weights_t.elapsed().as_secs_f64() * 1000.0);

            debug!("✓ Model weights loaded");
            // A model with a recurrent state keeps the states of its prompts on disk when the
            // operator opened a store, under the file and the numerics they were computed with.
            let mut model = model;
            model.set_kv_lease(
                (kv_lease_secs > 0).then(|| std::time::Duration::from_secs(kv_lease_secs)),
            );
            if let Some(store) = state_cache.clone() {
                let mut files = crate::inference::cache::state_disk::FilesDigest::new();
                let head = &mmap[..mmap.len().min(crate::inference::cache::state_disk::DIGEST_HEAD_BYTES)];
                // Every part of a set names the states: a state is valid for the files it ran on.
                let digested = match &split {
                    Some(set) => set.parts().iter().try_for_each(|part| {
                        let part_head = &part.mmap[..part.mmap.len().min(crate::inference::cache::state_disk::DIGEST_HEAD_BYTES)];
                        part.file.metadata().map(|meta| files.file(&meta, part_head))
                    }),
                    None => std::fs::metadata(&model_file).map(|meta| files.file(&meta, head)),
                };
                // The files the routed experts are read from name the states too, with the policy
                // that chose between them.
                let digested = digested.and_then(|()| {
                    variant_files.iter().flat_map(|v| v.parts()).try_for_each(|part| {
                        let part_head = &part.mmap[..part.mmap.len().min(crate::inference::cache::state_disk::DIGEST_HEAD_BYTES)];
                        part.file.metadata().map(|meta| files.file(&meta, part_head))
                    })
                });
                let policy = format!("{expert_variants:?}");
                match digested {
                    Ok(()) => {
                        let digest = files.finish();
                        let kv = format!("{kv_quant:?}");
                        let mut parts: Vec<&[u8]> = vec![&digest[..], kv.as_bytes()];
                        if !variant_files.is_empty() {
                            parts.push(policy.as_bytes());
                        }
                        let identity = crate::inference::cache::state_disk::identity(&parts);
                        if model.attach_state_cache(store, identity) {
                            info!("state cache on for {arch}");
                        }
                    }
                    Err(e) => warn!("state cache off: {e}"),
                }
            }
            // The advice names layers by their place in the stack; a share counts its own.
            advise_pages_by_placement(
                if layer_share.is_some() { &[] } else { &page_advice },
                &model.device_layer_distribution(),
                host_experts_from,
            );

            // --- 5.5 Initialize layer performance tracking ---
            use crate::inference::place::layer_perf;
            let tracker = layer_perf::global_tracker();

            // Initialize tracker with actual device placement. Multi-GPU
            // placement (GenericHetero) reports its own per-layer timing; this
            // coarse GPU/CPU split is the fallback for single-device loads.
            {
                let actual_gpu_layers = match &model {
                    _ if device.is_cuda() => num_layers, // all on GPU
                    _ => 0, // all on CPU
                };
                tracker.initialize_with_device(num_layers, actual_gpu_layers, gpu_idx);
                debug!("📊 Layer tracker: {} layers ({} GPU, {} CPU)",
                    num_layers, actual_gpu_layers, num_layers - actual_gpu_layers);
            }

            // The expert caches filled from a profile, before any prompt routes.
            let mut model = model;
            if let (Some(path), Some(profile)) = (expert_profile.as_deref(), profile.as_ref()) {
                match model.prime_expert_caches(profile) {
                    Ok(n) => info!("expert caches filled from {path}: {n} experts copied"),
                    Err(e) => warn!("expert caches not filled from {path}: {e}"),
                }
            }

            // --- 6. Store state ---
            let new_state = LoadedModelState {
                name: model_id_clone.clone(),
                num_layers,
                hidden_size,
                num_heads,
                vocab_size,
                // The window a request may open: what the cache was allocated for, which
                // the placement plan may have set below the configured context.
                context_length: model
                    .kv_capacity()
                    .map_or(context_length, |c| c.min(context_length)),
                eos_token_id,
                #[cfg(feature = "gpu")]
                moondream_graph: None,
                moondream_decode_count: 0,
                #[cfg(feature = "gpu")]
                kept_graphs: Default::default(),
                #[cfg(feature = "gpu")]
                draft_graph: Default::default(),
                #[cfg(feature = "gpu")]
                draft_step: Default::default(),
                draft_primed: false,
                draft_mark: None,
                exact_prefill: None,
                eos_token_ids_extra,
                file_size,
                model,
                tokenizer: std::sync::Arc::new(tokenizer),
                device,
                image_embeds: None,
                audio: None,
                qwen35_vision: Vec::new(),
                image_embed_cache: super::image_cache::ImageEmbedCache::new(),
                grammar_factory: std::sync::OnceLock::new(),
                // Whether the loaded GGUF declares `{architecture}.pooling_type`: the model is an
                // embedding model and reranks by late interaction over token vectors.
                pools: crate::inference::engine::llm_engine::load::declares_pooling(
                    content.metadata.keys().map(|k| k.as_str()),
                    &arch,
                ),
                generation_defaults: generation_defaults.clone(),
            };
            let mut guard = model_state.blocking_lock();
            // Free VRAM just changed by the size of a model.
            crate::inference::place::vram_manager::residency_changed();
            new_state.model.seed_prefill_row_bytes(prefill_row_bytes);
            *guard = Some(new_state);
            // --- Ready the model for its first request ---
            // It runs after the residency epoch moved, so the chunk sizing the warmup memoises
            // is the one the first request reads.
            if let Some(st) = guard.as_mut() {
                ready_for_requests(st.model.as_mut(), &st.device, warmup);
            }
            *listing.lock().unwrap_or_else(|e| e.into_inner()) =
                super::query::ModelListing::of(guard.as_ref(), configured_window);
            drop(guard);  // Release lock early

            // Cache file_size for non-blocking stats queries
            let mut size_guard = cached_model_size.blocking_lock();
            *size_guard = file_size;

            Ok(())
        }).await;

        match result {
            Ok(Ok(())) => {
                // Clear any previous error on success
                self.clear_last_error().await;
                // Loading churns large transients (dequant/requant, GGUF read
                // buffers, weight-concat fusions) that glibc arenas RETAIN
                // after free, leaving resident memory at a multiple of the
                // model size. Return the freed heap to the OS now that the
                // load transients are dropped (same rationale as the
                // unload-path trim).
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::malloc_trim(0);
                }
            }
            Ok(Err(e)) => {
                let err_msg = e.to_string();
                let mut last_err = self.last_error.lock().await;
                *last_err = Some(err_msg.clone());
                return Err(e.into());
            }
            Err(e) => {
                let err_msg = format!("spawn_blocking join error: {e}");
                let mut last_err = self.last_error.lock().await;
                *last_err = Some(err_msg.clone());
                return Err(err_msg.into());
            }
        }

        crate::inference::place::device_probe::debug_vram_by_card("after-construction");
        let load_time = load_start.elapsed();
        let load_time_secs = load_time.as_secs_f64();
        info!(
            "✅ Model {} loaded successfully in {:.2}ms",
            model_id,
            load_time_secs * 1000.0
        );
        // What a hand-over is weighed against: a peer already holding the weights is worth
        // the difference between this and nothing. Measured here rather than configured,
        // because it depends on the disk, the model and the placement, none of which a
        // constant can know.
        crate::distributed::rate_meter::record_load(&model_id, load_time_secs);
        // Hardware-derived decode ceiling: a resident model streams its bytes at the
        // card's memory bandwidth at most. Every node derives this the same way, so the
        // cluster can compare two nodes before either has measured anything.
        if file_size > 0 {
            let bw = crate::inference::place::room::fastest_bandwidth_gbs();
            if bw > 0.0 {
                crate::distributed::rate_meter::record_prior(
                    &model_id,
                    bw * 1e9 / file_size as f64,
                );
            }
        }
        debug!(
            "📈 Load throughput: {:.2} MB/s",
            (file_size as f64 / 1_000_000.0) / load_time_secs
        );

        // Pre-warm custom fused-kernel PTX module on each CUDA device.
        // Forces NVRTC compilation + cuModuleLoad outside graph capture,
        // so subsequent capture-time launches hit the cached path with
        // no driver-level allocations.
        #[cfg(feature = "gpu")]
        {
            let state = self.model_state.lock().await;
            if let Some(model) = state.as_ref() {
                let dev_set: std::collections::HashSet<usize> = model.model.cuda_device_ordinals();
                for ord in dev_set {
                    // Best-effort: use a fresh CudaDevice handle via
                    // the reference device cache (already populated by load).
                    if let Ok(dev) = crate::tensor::Device::cuda_if_available(ord) {
                        if let Ok(cd) = dev.as_cuda_device() {
                            if let Err(e) =
                                crate::inference::kernel::fused::prewarm_fused_kernels(&cd)
                            {
                                tracing::warn!("prewarm_fused_kernels(gpu={ord}) non-fatal: {e}");
                            }
                            if let Err(e) = crate::inference::quantized_cuda::prewarm(&cd) {
                                tracing::warn!("quantized_cuda::prewarm(gpu={ord}) non-fatal: {e}");
                            }
                            // Known-answer test: an arch-mismatched kernel returns garbage
                            // without an error, so each family answers a fixed question on
                            // each card before it is allowed to serve real tokens.
                            for (probe, ok) in
                                crate::tensor::quantized::kernel_known_answer_test(&dev)
                            {
                                if ok {
                                    tracing::debug!("kernel KAT gpu={ord}: {probe} ok");
                                } else {
                                    tracing::error!(
                                        "kernel KAT gpu={ord}: {probe} FAILED - family gated to \
                                         the dequant path for this device"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Another component on the model's cards sees what it keeps for later requests.
        self.list_kept_memory().await;
        Ok(())
    }
}

/// Tell the kernel which of the file's pages still matter, now that every layer has a device.
///
/// Every weight was read once to be placed. A layer that landed on a card is finished with
/// the file - its bytes live in VRAM and the page cache copy is dead weight that the kernel
/// keeps hot anyway, because a mapped page counts as in use. A layer that stayed on the host
/// is the opposite: it is read in full on every token, and a page that is not resident by
/// then is a fault in the middle of a decode step.
///
/// So: DontNeed on the card-resident layers, WillNeed on the host-resident ones. Both are
/// advice about residency, not about data - a DontNeed page that is touched again simply
/// comes back from the file - so a wrong guess costs a refault, never a wrong weight. Only
/// `blk.N.*` tensors are covered: embeddings and the output head are placed elsewhere and
/// are small next to the layers.
/// One mapping's share of the page advice: its byte ranges, in the mapping's own offsets.
struct MappedAdvice {
    mmap: std::sync::Arc<memmap2::Mmap>,
    ranges: Vec<(usize, usize, usize, bool)>,
}

/// Each block tensor's `(layer, start, len, is a routed expert)`, in `content`'s own offsets.
fn page_ranges_of(content: &gguf_file::Content) -> Vec<(usize, usize, usize, bool)> {
    content
        .tensor_infos
        .iter()
        .filter_map(|(name, info)| {
            let rest = name.strip_prefix("blk.")?;
            let dot = rest.find('.')?;
            let layer = rest[..dot].parse::<usize>().ok()?;
            Some((
                layer,
                (content.tensor_data_offset + info.offset) as usize,
                info.size_in_bytes(),
                super::load::is_routed_expert_tensor(&rest[dot + 1..]),
            ))
        })
        .collect()
}

fn advise_pages_by_placement(
    advice: &[MappedAdvice],
    dist: &[(String, usize, u32, u32)],
    host_experts_from: Option<usize>,
) {
    if dist.is_empty() {
        return; // this backend did not say where its layers are; leave the cache alone
    }
    let on_host = |layer: usize| -> Option<bool> {
        dist.iter()
            .find(|(_, _, a, b)| (*a as usize) <= layer && layer <= (*b as usize))
            .map(|(kind, _, _, _)| kind == "CPU")
    };
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    let (mut kept, mut dropped, mut n_kept, mut n_dropped) = (0u64, 0u64, 0usize, 0usize);
    for part in advice {
        let mmap = &part.mmap;
        let total = mmap.len();
        for &(layer, start, len, routed_expert) in part.ranges.iter() {
            let Some(host) = on_host(layer) else {
                continue;
            };
            // Routed experts the plan keeps in host memory are read from these pages.
            let host =
                host || (routed_expert && host_experts_from.is_some_and(|from| layer >= from));
            // Page-aligned outward: a partial page shared with a neighbour is only ever
            // advised, so widening the range costs at most one refault of that neighbour.
            let a = start / page * page;
            let b = (start + len).div_ceil(page) * page;
            let b = b.min(total);
            if a >= b {
                continue;
            }
            if host {
                let _ = mmap.advise_range(memmap2::Advice::WillNeed, a, b - a);
                kept += len as u64;
                n_kept += 1;
            } else {
                // DontNeed is the "unchecked" advice because it discards a private or anonymous
                // mapping's contents. This one is a read-only shared file mapping: the page
                // comes back from the file, unchanged, if it is ever read again.
                let _ = unsafe {
                    mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, a, b - a)
                };
                dropped += len as u64;
                n_dropped += 1;
            }
        }
    }
    info!(
        "📄 page cache after placement: {:.1} GB released ({n_dropped} tensors on the cards), {:.1} GB kept ({n_kept} on the host)",
        dropped as f64 / 1e9,
        kept as f64 / 1e9
    );
}

#[cfg(test)]
mod kv_bytes_tests {
    use super::*;

    #[test]
    fn a_quantised_cache_is_priced_by_its_blocks() {
        assert_eq!(kv_cache_bytes(KvQuant::Q8, 0, 64), 2 * 34);
        assert_eq!(kv_cache_bytes(KvQuant::Q4, 0, 64), 2 * 18);
        assert_eq!(kv_cache_bytes(KvQuant::Q8, 0, 33), 2 * 34);
    }

    #[test]
    fn an_unquantised_cache_is_priced_at_its_native_width() {
        assert_eq!(kv_cache_bytes(KvQuant::Off, 2, 100), 200);
        assert_eq!(kv_cache_bytes(KvQuant::Off, 4, 100), 400);
    }
}

#[cfg(test)]
mod cache_format_tests {
    use super::*;

    /// A model the quantised kernels serve takes a Q8 cache at `balanced` and `fast`, and
    /// keeps its native width at `reference`.
    #[test]
    fn the_level_decides_the_cache_of_a_served_head_shape() {
        assert_eq!(
            kv_cache_format(KvQuant::Off, true, Fidelity::Reference),
            KvQuant::Off
        );
        for level in [Fidelity::Balanced, Fidelity::Fast] {
            assert_eq!(kv_cache_format(KvQuant::Off, true, level), KvQuant::Q8);
        }
    }

    /// A head shape the kernels do not serve keeps the native width at every level, and a
    /// quantised format asked for it falls back to the native width.
    #[test]
    fn an_unserved_head_shape_keeps_the_native_width() {
        for level in [Fidelity::Reference, Fidelity::Balanced, Fidelity::Fast] {
            assert_eq!(kv_cache_format(KvQuant::Off, false, level), KvQuant::Off);
            assert_eq!(kv_cache_format(KvQuant::Q8, false, level), KvQuant::Off);
            assert_eq!(kv_cache_format(KvQuant::Q4, false, level), KvQuant::Off);
        }
    }

    /// A quantised format asked for a served head shape is kept, whatever the level.
    #[test]
    fn a_quantised_format_asked_for_is_kept_where_served() {
        for level in [Fidelity::Reference, Fidelity::Balanced, Fidelity::Fast] {
            assert_eq!(kv_cache_format(KvQuant::Q8, true, level), KvQuant::Q8);
            assert_eq!(kv_cache_format(KvQuant::Q4, true, level), KvQuant::Q4);
        }
    }
}

/// Whether a load warms the prefill path before it returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Warmup {
    /// No request waits on the load: the load warms the path for the first one.
    Ahead,
    /// A request waits on the load: its own prompt is the first to run the path.
    ByRequest,
}

/// The offers of `cards` with the room of the vision tower of `tower` bytes taken from the first:
/// the tower loads on the card of the first layer, beside the blocks the plan places there.
pub(super) fn beside_the_tower(
    mut cards: Vec<(usize, u64)>,
    tower: Option<u64>,
) -> Vec<(usize, u64)> {
    if let (Some(bytes), Some(first)) = (tower.filter(|&b| b > 0), cards.first_mut()) {
        first.1 = first.1.saturating_sub(bytes);
        info!(
            "card {} keeps {:.2} GB for the vision tower beside its layers",
            first.0,
            bytes as f64 / 1e9
        );
    }
    cards
}

/// Readies a loaded model for its first request.
/// The prefill path is warmed first when `warmup` asks for it ahead of the requests: the first
/// prompt of a fresh server otherwise pays for the code objects, the device buffers, the card
/// probes and the clocks it is the first to touch.
/// The host pages of the hot experts a profile ranked are registered next, in both cases: a
/// registration holds the driver until it ends, and a request run beside it waits for it.
fn ready_for_requests(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    device: &Device,
    warmup: Warmup,
) {
    let warm_started = std::time::Instant::now();
    if warmup == Warmup::Ahead && warm_prefill(model, device) {
        info!(
            "prefill warmed in {:.0}ms",
            warm_started.elapsed().as_secs_f64() * 1e3
        );
    }
    let settle_started = std::time::Instant::now();
    if model.settle_hot_experts() {
        info!(
            "hot experts' host pages registered, {:.0}ms waited",
            settle_started.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// Runs prompts of a few lengths through a loaded text model and cuts the KV back to empty, so
/// the first request finds the kernels loaded, the buffers sized and the card clocked up.
/// On a card whose float products offer variants, a first 1-token prompt runs them as a decode
/// step does, and each weight shape there measures its variants, so no request pays for that.
/// Returns whether the warmup ran. A model that cannot cut its KV back, or that fails a pass,
/// is left as loaded.
fn warm_prefill(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    device: &Device,
) -> bool {
    const LENGTHS: [usize; 5] = [8, 24, 64, 160, 400];
    if !model.capabilities().trim_kv || model.is_vision_model() {
        return false;
    }
    let room = model.kv_capacity().unwrap_or(0) / 2;
    let pass = |model: &mut dyn crate::inference::engine::model_backend::ModelBackend, n: usize| {
        let ids: Vec<u32> = (0..n as u32).map(|i| 10 + i % 1000).collect();
        let pass = Tensor::new(&ids[..], device)
            .and_then(|t| t.unsqueeze(0))
            .and_then(|x| model.forward_prefill_chunked(&x, 0))
            .and_then(|l| l.to_device(&Device::Cpu));
        model.trim_kv(0);
        pass
    };
    // The 1-token prompt runs only on a card whose float products offer variants to measure.
    // It leaves what the warmup returns to the prompts below.
    if room >= 1 && crate::inference::moe_cuda::measures_float_variants(device) {
        // A request in flight on the server shares the card's stream with the timing: its
        // choices are forgotten, and the next warm-up measures again.
        let idle = std::sync::atomic::AtomicUsize::new(0);
        let contended = || super::batching::busy(&idle);
        let choose = crate::inference::moe_cuda::ChooseFloatVariants::enter();
        let before = contended();
        if let Err(e) = pass(model, 1) {
            debug!("float product variants unmeasured: the 1-token warmup failed: {e}");
        }
        if before || contended() {
            debug!("float product variants measured beside another request: not kept");
            choose.forget();
        }
    }
    let mut ran = false;
    for n in LENGTHS.into_iter().filter(|n| *n <= room) {
        match pass(model, n) {
            Ok(_) => ran = true,
            Err(e) => {
                debug!("prefill warmup stopped at {n} tokens: {e}");
                break;
            }
        }
    }
    model.invalidate_graph_state();
    ran
}

#[cfg(test)]
mod tower_offer_tests {
    /// The card of the first layer offers the blocks its room less the vision tower's, the other
    /// cards their whole room; a model without a tower changes no offer.
    #[test]
    fn the_first_card_offers_its_room_less_the_tower() {
        let cards = vec![(0, 15_000), (1, 16_000)];
        assert_eq!(
            super::beside_the_tower(cards.clone(), Some(1_500)),
            vec![(0, 13_500), (1, 16_000)]
        );
        assert_eq!(super::beside_the_tower(cards.clone(), None), cards);
        assert_eq!(super::beside_the_tower(Vec::new(), Some(1_500)), Vec::new());
    }
}

#[cfg(test)]
mod ready_tests {
    use super::*;
    use crate::inference::engine::model_backend::{BoxedModelBackend, Capabilities, ModelBackend};
    use crate::inference::generic_transformer::GenericHeteroTransformer;
    use std::sync::{Arc, Mutex};

    /// A backend that records the calls the load makes of it, in order.
    struct ReadyToy {
        calls: Arc<Mutex<Vec<&'static str>>>,
        trims: bool,
    }

    impl ReadyToy {
        fn note(&self, call: &'static str) {
            self.calls.lock().unwrap().push(call);
        }
    }

    impl ModelBackend for ReadyToy {
        fn forward(&mut self, _: &Tensor, _: usize) -> crate::tensor::Result<Tensor> {
            Tensor::zeros((1, 4), crate::tensor::DType::F32)
        }

        fn forward_prefill_chunked(
            &mut self,
            _: &Tensor,
            _: usize,
        ) -> crate::tensor::Result<Tensor> {
            self.note("prefill");
            Tensor::zeros((1, 4), crate::tensor::DType::F32)
        }

        fn trim_kv(&mut self, _: usize) {
            self.note("trim");
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                trim_kv: self.trims,
                ..Capabilities::default()
            }
        }

        fn kv_capacity(&self) -> Option<usize> {
            Some(4096)
        }

        fn settle_hot_experts(&self) -> bool {
            self.note("settle");
            true
        }

        fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
            Err(self)
        }

        fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
            Err(self)
        }
    }

    /// The calls, in order, that a load with `warmup` makes of a model that can cut its cache
    /// back when `trims` holds.
    fn calls_of(trims: bool, warmup: Warmup) -> Vec<&'static str> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut model = ReadyToy {
            calls: calls.clone(),
            trims,
        };
        ready_for_requests(&mut model, &Device::Cpu, warmup);
        let made = calls.lock().unwrap().clone();
        made
    }

    /// A load no request waits on warms the prefill path where the model can cut its cache
    /// back, then waits for the registration of the hot experts' pages once, before it hands
    /// the model to a request.
    #[test]
    fn a_load_ahead_of_the_requests_warms_then_settles_the_registration() {
        for trims in [true, false] {
            let calls = calls_of(trims, Warmup::Ahead);
            assert_eq!(
                calls.last(),
                Some(&"settle"),
                "the load ended with {calls:?}"
            );
            assert_eq!(
                calls.iter().filter(|c| **c == "settle").count(),
                1,
                "{calls:?}"
            );
            assert_eq!(
                calls.contains(&"prefill"),
                trims,
                "a warmup ran where it could not, or none where it could: {calls:?}"
            );
        }
    }

    /// A load a request waits on runs no prompt before that request, and still waits for the
    /// registration once.
    #[test]
    fn a_load_for_a_waiting_request_settles_the_registration_without_a_warmup() {
        for trims in [true, false] {
            assert_eq!(calls_of(trims, Warmup::ByRequest), vec!["settle"]);
        }
    }
}
