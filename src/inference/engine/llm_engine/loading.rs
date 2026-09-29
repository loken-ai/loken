//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

/// Largest window the runtime reserve prices the KV cache at: the caches start smaller and
/// grow, and a longer window is admitted by the context planner, not by this reserve.
const RESERVE_KV_WINDOW: u64 = 8192;

/// The dimensions of a language model a placement is priced from, read from its GGUF header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelDims {
    pub(crate) num_layers: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
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
        Self {
            num_layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Card memory a model needs at run time beyond its weights, on each card it uses: its
    /// KV cache at the reserve window (one byte per element, the quantised cache), the cuBLAS
    /// workspace and the chunked-prefill activation peak.
    pub(crate) fn runtime_reserve(&self, context_length: usize) -> u64 {
        let kv_ctx = (context_length as u64).min(RESERVE_KV_WINDOW);
        let kv = 2u64
            .saturating_mul(self.num_kv_heads as u64)
            .saturating_mul(self.head_dim as u64)
            .saturating_mul(kv_ctx)
            .saturating_mul(self.num_layers as u64);
        kv.saturating_add(self.work_reserve())
    }

    /// What a card keeps free beside the weights and the cache: the cuBLAS workspace and
    /// the activations of one prefill chunk.
    pub(crate) fn work_reserve(&self) -> u64 {
        crate::inference::place::runtime_demand::llm_work_reserve(
            self.hidden_size.max(self.num_kv_heads * self.head_dim),
            PREFILL_CHUNK_TOKENS,
        )
    }
}

/// Card memory a load of the GGUF at `path` keeps free on each card it uses, beyond the
/// weights: the runtime reserve at the model's declared window plus the staging the load
/// of that checkpoint churns through.
pub(crate) fn per_card_reserve(path: &std::path::Path) -> Option<u64> {
    let content = gguf_file::open_header(path).ok()?;
    let arch = get_gguf_string(&content, "general.architecture")?;
    let context_length = get_gguf_u32(&content, &format!("{arch}.context_length"))
        .map_or(crate::inference::cache::KV_WORKING_WINDOW_TOKENS, |c| {
            c as usize
        });
    let staging = crate::inference::place::runtime_demand::load_runtime_floor(
        std::fs::metadata(path).ok()?.len(),
    );
    Some(ModelDims::read(&content, &arch).runtime_reserve(context_length) + staging)
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
        let mut builder = rayon::ThreadPoolBuilder::new().num_threads(n);
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
    /// Loads the configured model. A model that will spill to the host has room held for a
    /// drafter while it is placed, and the drafter loads there only if the placed model can
    /// use it (see `plan_drafter`).
    pub async fn load_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        let planned = self
            .find_model_file(&self.config.model_id)
            .and_then(|target| {
                let size = std::fs::metadata(&target).ok()?.len();
                self.plan_drafter(&target, size)
            });
        self.load_model_only().await?;
        self.load_planned_drafter(planned).await;
        Ok(())
    }

    /// Loads the configured model alone.
    pub(super) async fn load_model_only(&self) -> Result<(), Box<dyn std::error::Error>> {
        let model_id = self.config.model_id.clone();
        let load_start = std::time::Instant::now();
        crate::inference::place::device_probe::debug_vram_by_card("load-entry");
        info!("🔄 Loading model: {}", model_id);
        // Placement below reads free VRAM to decide where the layers go. That reading is
        // only true while nobody else is allocating: a media engine loading at the same
        // moment sees the same free card, plans onto it too, and one of the two runs out
        // mid-upload. Waiting for the other loader costs seconds; planning against its
        // half-taken card costs the request.
        let _admission = crate::inference::place::vram_manager::load_admission().await;
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

        debug!("📂 Found model file: {}", model_file.display());

        let metadata = std::fs::metadata(&model_file)?;
        let file_size = metadata.len();
        debug!(
            "📊 Model file size: {:.2} MB",
            file_size as f64 / 1_000_000.0
        );

        let device_index = self.config.device_index;
        let disable_arc_layers = self.config.disable_arc_layers;
        // `--cpu` serve flag forces the same all-CPU placement as the
        // disable_cuda config (empty cuda_devices -> every layer on CPU).
        let disable_cuda = self.config.disable_cuda || crate::gpu::force_cpu();
        let kv_quant = self.config.kv_quant;
        let user_context_length = self.config.context_length;
        let model_state = self.model_state.clone();
        let cached_model_size = self.cached_model_size.clone();
        let model_id_clone = model_id.clone(); // Clone for the closure
                                               // Resolve the projector blob (vision tower) for dual-blob Ollama
                                               // packages like moondream. None for text-only models. Moved into
                                               // the spawn_blocking closure so the CLIP loader can pick it up
                                               // after the LM has loaded as GenericHetero.
        let projector_blob = self.find_projector_blob_for_model(&model_id_clone);
        let generation_defaults = self.generation_defaults_for(&model_id_clone, &model_file);

        // Load model in blocking thread since GGUF parsing and weight loading are CPU-intensive
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            crate::inference::place::device_probe::debug_vram_by_card("closure-start");

            // --- 0a. A checkpoint directory the deepseek_v41 source serves ---
            if crate::inference::model::deepseek_v41::claims_checkpoint_dir(&model_file) {
                let mut state =
                    load_deepseek_v41_checkpoint(&model_file, &model_id_clone, user_context_length)?;
                state.generation_defaults = generation_defaults.clone();
                let fsz = state.file_size;
                let mut guard = model_state.blocking_lock();
                crate::inference::place::vram_manager::residency_changed();
                *guard = Some(state);
                drop(guard);
                let mut size_guard = cached_model_size.blocking_lock();
                *size_guard = fsz;
                return Ok(());
            }

            // --- 0. AWQ (HF safetensors) ---
            // Its layers come from the shards; its header is synthesised from the checkpoint's
            // config, and from there it takes the same route as a GGUF file.
            let awq = resolve_awq_dir(&model_file.to_string_lossy())
                .map(|dir| awq_checkpoint(&dir).map(|(content, shards)| (dir, content, shards)))
                .transpose()?;

            // --- 1. Parse GGUF header via mmap ---
            let (mmap, content, awq) = match awq {
                Some((dir, content, shards)) => {
                    // No GGUF file backs this model: a one-byte anonymous map stands in for
                    // the bytes the GGUF-only readers would take.
                    let map = memmap2::MmapOptions::new().len(1).map_anon()?.make_read_only()?;
                    (std::sync::Arc::new(map), content, Some((dir, shards)))
                }
                None => {
                    debug!("📂 Opening GGUF file: {}", model_file.display());
                    let file = std::fs::File::open(&model_file)
                        .map_err(|e| {
                            let err_msg = format!("Failed to open GGUF file at {}: {e}", model_file.display());
                            error!("❌ {}", err_msg);
                            anyhow!(err_msg)
                        })?;

                    // Memory-map the file for efficient random access and OS read-ahead
                    debug!("🔗 Memory-mapping GGUF file...");
                    let mmap = unsafe { Mmap::map(&file) }
                        .map_err(|e| {
                            let err_msg = format!("Failed to mmap GGUF file: {e}");
                            error!("❌ {}", err_msg);
                            anyhow!(err_msg)
                        })?;
                    // Arc so loaders can hand out zero-copy file views: a view
                    // holds an Arc clone, keeping the mapping alive past this scope.
                    // No blanket WillNeed here. It prefetched the WHOLE file into the page cache
                    // before anything was placed, so a model that spills had its GPU-resident 28 GB
                    // cached for nothing next to the host layers it actually reads - on a 64 GB box
                    // that put 16 GB of the process into swap during the very decode being timed.
                    // The advice is given per layer once placement is known, see
                    // `advise_pages_by_placement` below: WillNeed for what the host will read every
                    // token, DontNeed for what the cards already hold.
                    let mmap = std::sync::Arc::new(mmap);
                    crate::inference::place::device_probe::debug_vram_by_card("after-mmap-parse");
                    debug!("✓ GGUF file mmap'd: {} bytes", mmap.len());

                    debug!("📖 Parsing GGUF header...");
                    let mmap_parse_t = std::time::Instant::now();
                    let mut content = gguf_file::Content::read_mapped(&mut Cursor::new(&mmap[..]), mmap.clone())
                        .map_err(|e| {
                            let err_msg = format!("GGUF parse error: {e}");
                            error!("❌ {}", err_msg);
                            anyhow!(err_msg)
                        })?;
                    // Back the tensor data with the Arc<Mmap> so the loader reads weights as
                    // zero-copy views (no host copy, the Arc pins the mmap) instead of copying
                    // each tensor into owned host bytes - faster load + less committed RAM.
                    content.mmap_owner = Some(mmap.clone() as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                    debug!("✓ GGUF header parsed in {:.1}ms", mmap_parse_t.elapsed().as_secs_f64() * 1000.0);
                    (mmap, content, None)
                }
            };
            let mmap_bytes: &[u8] = &mmap[..];
            // What the cards will hold: the shards of an AWQ checkpoint, the file otherwise.
            let file_size = match &awq {
                Some((_, shards)) => shards
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                    .sum(),
                None => file_size,
            };
            let generic_weights = match &awq {
                Some((dir, shards)) => GenericWeights::Awq { dir, shards },
                None => GenericWeights::Gguf { mmap_bytes, mmap: &mmap },
            };
            // Captured now: `content` is consumed by one of the loader arms below, and the
            // page advice can only be given once every layer has a device.
            // Page advice applies to a GGUF file's own bytes; the shards are read, not mapped.
            let layer_page_ranges: Vec<(usize, usize, usize)> = content
                .tensor_infos
                .iter()
                .filter(|_| awq.is_none())
                .filter_map(|(name, info)| {
                    let rest = name.strip_prefix("blk.")?;
                    let dot = rest.find('.')?;
                    let layer = rest[..dot].parse::<usize>().ok()?;
                    Some((
                        layer,
                        (content.tensor_data_offset + info.offset) as usize,
                        info.size_in_bytes(),
                    ))
                })
                .collect();

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
            let context_length = get_gguf_u32(&content, &format!("{arch}.context_length"))
                .map_or(crate::inference::cache::KV_WORKING_WINDOW_TOKENS, |c| c as usize);
            // No window asked for: the model's own.
            let user_context_length = if user_context_length == 0 {
                context_length
            } else {
                user_context_length
            };
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

            // The Q8/Q4 attention kernels are instantiated for this model's
            // (head_dim, n_q_per_kv); a shape outside what they can hold
            // downgrades to kv_quant=Off below.
            let n_q_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
            // An architecture whose output a quantised cache degrades keeps it unquantised.
            let quantized_kv_degrades = facts.quantized_kv_degrades;
            // The quantised attention kernels do not cap scores: a model that caps them
            // (Gemma2) keeps the unfused attention that does.
            let caps_attention = content
                .metadata
                .contains_key(&format!("{arch}.attn_logit_softcapping"));
            // Nor do they fold attention sinks into the softmax: a model with sinks keeps the
            // F-dtype cache its sink attention reads.
            let has_attn_sinks = content.tensor_infos.contains_key("blk.0.attn_sinks")
                || content.tensor_infos.contains_key("blk.0.attn_sinks.weight");
            let kv_quant_supported =
                crate::inference::cache::kv_core::kv_shape_supported(head_dim, n_q_per_kv)
                    && !quantized_kv_degrades
                    && !caps_attention
                    && !has_attn_sinks;
            // Compile the shape's attention kernels now rather than on the first decoded token.
            #[cfg(feature = "cuda")]
            if kv_quant_supported {
                crate::inference::quantized_cuda::prewarm_shape(head_dim, n_q_per_kv);
            }
            // The KV precision follows what the quantised kernels can serve.
            let kv_quant = if !kv_quant_supported && !matches!(kv_quant, KvQuant::Off) {
                warn!(
                    "⚠️  Q8/Q4 KV cache kernels don't support head_dim={} x n_q_per_kv={}. Auto-downgrading kv_quant to Off for {}.",
                    head_dim, n_q_per_kv, &model_id_clone
                );
                KvQuant::Off
            } else if kv_quant_supported && matches!(kv_quant, KvQuant::Off) {
                // Every model the quantised kernels serve takes a Q8 cache: half the bytes
                // per token read at each decode step, reference-grade accuracy (Q8_0 KV
                // perplexity sits within the F16 error bars), and the device-position
                // kernels a captured decode graph needs. The models kept off it are decided
                // above by what they are - a head shape the kernels do not cover, capped
                // attention scores, attention sinks, a cache precision the architecture's
                // facts rule out - not by a list of names.
                info!("Auto-promoting kv_quant Off -> Q8 for arch={arch}");
                KvQuant::Q8
            } else {
                kv_quant
            };

            // --- 3. Build tokenizer ---
            debug!("🔍 Looking for tokenizer.json...");
            let tokenizer: Tokenizer = {
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

            // --- 4. The cards the placement offers this model ---
            //
            // Each card offers its free memory under the fleet's pressure, less the staging
            // a load of this checkpoint churns through; `place::llm::offer` makes every
            // listed card's context before it measures it, so a card reads the same on every
            // load. What one card must hold for the model to live on it alone is the weights
            // and the runtime reserve: the cache at the reserve window, the cuBLAS workspace
            // and a prefill chunk's activations, sized from the model's own dimensions.
            let runtime_reserve_bytes = dims.runtime_reserve(context_length);
            let staging = crate::inference::place::runtime_demand::load_runtime_floor(file_size);
            let offer = if disable_cuda {
                debug!("CUDA disabled by config (disable_cuda=true)");
                crate::inference::place::llm::Offer::default()
            } else {
                crate::inference::place::llm::offer(
                    device_index,
                    file_size.saturating_add(runtime_reserve_bytes),
                    staging,
                )
            };
            crate::inference::place::device_probe::debug_vram_by_card("after-offer");
            let cuda_devices: Vec<(usize, u64)> = offer.cards.clone();
            let model_fits_single_gpu = file_size > 0 && offer.whole_on.is_some();
            let gpu_idx = offer.primary().unwrap_or_default();

            // --- 4.5 The non-CUDA GPUs, if any answer ---
            //
            // Behind a deadline, because this is a driver call: a card removed while the system
            // ran leaves a device node that blocks every OpenCL client, and enumerating here
            // once wedged a whole node mid-load. See `opencl_probe`.
            if !disable_arc_layers {
                for (idx, (name, vram)) in crate::inference::kernel::opencl_probe::non_cuda_devices()
                    .into_iter()
                    .enumerate()
                {
                    debug!("🖥️  OpenCL device {}: {} ({:.1} GB)", idx, name, vram as f64 / 1e9);
                }
            }

            let mut device = match offer.whole_on.filter(|_| model_fits_single_gpu) {
                Some(idx) => {
                    info!("📊 Model ({:.1} GB) fits in single GPU memory, loading on GPU {}",
                        file_size as f64 / 1e9, idx);
                    // Default stream with event tracking disabled: capturable in
                    // cudaStreamCaptureModeRelaxed, which the engine's graph capture uses.
                    #[cfg_attr(not(feature = "cuda"), allow(clippy::map_identity))]
                    let dev_result: crate::tensor::Result<Device> = crate::tensor::Device::new_cuda(idx)
                        .map(|d| {
                            #[cfg(feature = "cuda")]
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

            // Enable reduced precision GEMM on CUDA for better throughput
            #[cfg(feature = "cuda")]
            if matches!(device, Device::Cuda(_)) {
                crate::tensor::cuda_ext::set_gemm_reduced_precision(true);
                debug!("✓ Enabled reduced precision GEMM (TF32/FP16/BF16) for quantized inference");
            }

            // --- 5. Load model weights based on architecture ---
            debug!("⚖️  Architecture: {}, Device: {:?}", arch, device);
            let weights_t = std::time::Instant::now();
            // A special loader claims a file by what it contains; every other file takes the
            // generic hetero route, which builds what the file describes or says what it cannot.
            let model: BoxedModelBackend = match () {
                // Moondream legacy path: a file with the original PyTorch state-dict
                // naming, whatever its architecture key says. A file with llama.cpp
                // naming (token_embd, blk.{i}.attn_qkv) takes the generic route.
                // The legacy model runs whole on one device, placed by the ladder.
                _ if content.tensor_infos.contains_key("text_model.transformer.embd.wte.weight") => {
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
                    #[cfg(feature = "cuda")]
                    if built_on.is_cuda() {
                        crate::tensor::cuda_ext::set_gemm_reduced_precision(true);
                    }
                    device = built_on;
                    Box::new(MoondreamBackend(model))
                }
                // Residual hyper-connections mark the DeepSeek V4.1 checkpoint, whatever its
                // converter named the architecture.
                _ if crate::inference::model::deepseek_v41::claims_gguf(&content) => {
                    // The CPU port: latent MQA, CSA2 bands, hyper-connections, and a MoE whose
                    // routed experts are read in place from the mapping by the quantised dot
                    // engine. The model may ship as one file or a split set; every part is
                    // mapped and stays the tensor source, so only the always-resident path is
                    // committed and the page cache holds whichever experts the prompts route to.
                    use crate::inference::model::deepseek_v41::{
                        model::DeepseekV41Model, DeepseekV41Config,
                    };
                    let source = crate::tensor::quantized::gguf_source::SplitGguf::open(&model_file)
                        .map_err(|e| anyhow!("deepseek_v41: open parts: {e}"))?;
                    let cfg = DeepseekV41Config::from_meta(&source)?;
                    let model = DeepseekV41Model::load(&source, &cfg)
                        .map_err(|e| anyhow!("deepseek_v41 load failed: {e}"))?;
                    // Without a kept tier the page cache is the only cache, and a prompt's
                    // prefill sweeps every expert through it: the decode that follows starts
                    // with the early layers evicted. The tier keeps each layer's most routed
                    // experts within what the host memory allows.
                    let slots = model
                        .size_expert_cache_to_memory(&cfg)
                        .map_err(|e| anyhow!("deepseek_v41: {e}"))?;
                    info!(
                        "✅ deepseek_v41 loaded on CPU from {} part(s) ({} layers, {} routed + {} \
                         shared experts, {} kept per layer)",
                        source.parts().len(),
                        cfg.n_layers,
                        cfg.n_routed_experts,
                        cfg.n_shared_experts,
                        slots,
                    );
                    // The draft the file carries first, the hub checkpoint's when it has none.
                    let draft =
                        crate::inference::model::deepseek_v41::dspark::Dspark::locate(&source);
                    Box::new(
                        crate::inference::engine::model_backend::DeepseekV41Backend::with_draft(
                            model,
                            cfg.moe_inter_dim,
                            user_context_length,
                            draft,
                        ),
                    )
                }
                _ => {
                    debug!("⬇️  Loading '{}' as generic hetero transformer (CUDA+CPU)...", arch);
                    // A config needs the RoPE pairing the file does not carry: an architecture
                    // the facts table does not know is refused here, before any weight is placed.
                    crate::inference::model_facts::rope_pairing(&arch).map_err(|e| {
                        let err = format!("cannot build {arch:?} through the generic transformer: {e}");
                        error!("❌ {err}");
                        anyhow!(err)
                    })?;
                    // For generic models: CUDA + CPU only (no OpenCL, those need per-model kernels).
                    // Per-layer KV cache memory cost. Used by HeteroPlan to
                    // reserve headroom on each GPU so weights + KV both fit.
                    // Cost: 2 x n_kv_heads x head_dim x max_seq x kv_dtype_bytes
                    // (x2 for K and V). Also adds a small activations
                    // headroom (~4 MB/layer) since at long ctx the
                    // intermediate score tensors aren't free either.
                    let kv_dtype_bytes: u64 = match kv_quant {
                        KvQuant::Off => 2,                 // F16
                        KvQuant::Q8 => 1,                  // 1 byte/elem (Q8_0 + scale negligible)
                        KvQuant::Q4 => 1,                  // ~0.5 byte/elem nibble + scale; round up to 1 for safety
                    };
                    // For models with sliding-window attention (gemma3/4),
                    // most layers cap their KV at sliding_window. With a
                    // pattern like [SWA,SWA,SWA,SWA,SWA,Global,...] the
                    // average KV cost per layer is far below the
                    // worst-case (full ctx). Use the average to avoid
                    // overestimating the budget and unnecessarily pushing
                    // layers to CPU.
                    let sliding_window = get_gguf_u32(&content, &format!("{arch}.attention.sliding_window"))
                        .unwrap_or(0) as u64;
                    let pattern: Vec<bool> = match content.metadata
                        .get(&format!("{arch}.attention.sliding_window_pattern"))
                    {
                        Some(gguf_file::Value::Array(arr)) => arr.iter().map(|v| match v {
                            gguf_file::Value::U8(n) => *n != 0,
                            gguf_file::Value::Bool(b) => *b,
                            gguf_file::Value::U32(n) => *n != 0,
                            _ => true,
                        }).collect(),
                        _ => Vec::new(),
                    };
                    // A quantised cache (Q4KvCache / Q8KvCache) starts at the working window
                    // and grows on demand up to the configured context, so it is planned at the
                    // size it starts at; a cache that grows past what its card holds is
                    // recovered by the ladder. An F-dtype cache pre-allocates in full.
                    let lazy_kv_initial_cap =
                        crate::inference::cache::KV_WORKING_WINDOW_TOKENS as u64;
                    // What one layer's cache costs at a given context, sliding windows
                    // included: a windowed layer never holds more than its window.
                    // Only the attending layers keep a cache; a recurrent mixer keeps a state
                    // of fixed size instead, so the cache is spread over the attending share.
                    let attn_layers = crate::inference::generic_transformer::mixer::attention_layer_count(
                        &content, num_layers,
                    ) as u64;
                    let over_attending = |bytes: u64| -> u64 {
                        if attn_layers < num_layers as u64 {
                            bytes.saturating_mul(attn_layers) / num_layers.max(1) as u64
                        } else {
                            bytes
                        }
                    };
                    let kv_bytes_at = |ctx: u64| -> u64 {
                        let avg = if !pattern.is_empty() && sliding_window > 0 {
                            let n_swa = pattern.iter().filter(|&&is_swa| is_swa).count() as u64;
                            let n_global = pattern.len() as u64 - n_swa;
                            (n_swa.saturating_mul(ctx.min(sliding_window))
                                + n_global.saturating_mul(ctx))
                                / pattern.len() as u64
                        } else {
                            ctx
                        };
                        over_attending(2u64 * num_kv_heads as u64 * head_dim as u64 * avg * kv_dtype_bytes)
                    };
                    // What the model asks of the cards; the ladder in `place` turns it into a
                    // plan, whole on one card when it fits there, and recovers from running out
                    // of card memory. The cache is part of what has to fit: a placement sized
                    // on the weights alone runs out once the context fills.
                    let (block_bytes, head_bytes) = generic_weights
                        .block_and_head_bytes(&content)
                        .unwrap_or((file_size, 0));
                    let demand_at = |ctx: u64| crate::inference::place::ladder::Demand {
                        layers: num_layers,
                        block_bytes,
                        head_bytes,
                        kv_bytes_per_layer: kv_bytes_at(ctx),
                        runtime_reserve: dims.work_reserve(),
                    };

                    // Choose the context that yields the best PLAN, not the largest number.
                    // An F-dtype cache honouring a context the cards cannot hold does not fail -
                    // it pushes layers onto the host, and a host layer costs far more than a
                    // context window nobody asked for. So the placement takes the largest
                    // context whose plan keeps every layer on the cards, halving down to the
                    // working window.
                    let ctx_for_planning: u64 = if matches!(kv_quant, KvQuant::Q4 | KvQuant::Q8) {
                        lazy_kv_initial_cap.min(user_context_length as u64)
                    } else {
                        let full = user_context_length as u64;
                        crate::inference::place::llm::context_the_cards_hold(
                            full,
                            lazy_kv_initial_cap,
                            &cuda_devices,
                            demand_at,
                        )
                    };
                    let demand = demand_at(ctx_for_planning);
                    // Cap user_context_length at the model's GGUF context: the model cannot
                    // generate beyond its trained context. The plan chose its context because
                    // the cards can hold it; the cache must be allocated at that size or the
                    // plan promised room the allocation then takes back. A quantised cache was
                    // planned at its initial cap and grows on demand, so its ceiling is the
                    // configured context; an F-dtype cache is allocated in full at the context
                    // the plan chose.
                    let effective_ctx = if matches!(kv_quant, KvQuant::Q4 | KvQuant::Q8) {
                        user_context_length.min(context_length)
                    } else {
                        user_context_length
                            .min(context_length)
                            .min(ctx_for_planning as usize)
                    };

                    // A model the placement runs tensor-parallel runs every layer on each of its
                    // cards, joined by an all-reduce, instead of one card per token with the
                    // others idle. Which files the decoder computes correctly is read from their
                    // tensors and metadata. A tensor-parallel load that fails leaves the model to
                    // the ladder below.
                    #[cfg(feature = "cuda")]
                    let tensor_parallel: Option<BoxedModelBackend> = {
                        use crate::inference::serve::tp_support::{tensor_parallel_support, TP_RANKS};
                        let tp = (awq.is_none()
                            && tensor_parallel_support(&content, &arch)
                                .inspect_err(|why| info!("{arch} stays off tensor parallelism: {why}"))
                                .is_ok())
                        .then(|| crate::inference::place::llm::tensor_parallel(&demand, &cuda_devices, TP_RANKS))
                        .flatten();
                        tp.and_then(|tp| {
                            info!("⬇️  Loading {arch} tensor-parallel across GPU {:?}, feed-forward share {}/{}",
                                tp.cards, tp.ffn_share.0, tp.ffn_share.1);
                            let built = tp
                                .cards
                                .iter()
                                .map(|&idx| Device::new_cuda(idx).map_err(|e| anyhow!("cuda:{idx}: {e}")))
                                .collect::<AnyResult<Vec<Device>>>()
                                .and_then(|cards| {
                                    crate::inference::serve::tp_model::TpQwen2::from_gguf(
                                        mmap_bytes, &mmap, &arch, &cards, tp.ffn_share,
                                    )
                                });
                            match built {
                                Ok(m) => {
                                    info!("✅ {arch} loaded as TpQwen2 (tensor-parallel across {} GPUs)", tp.cards.len());
                                    Some(Box::new(TpQwen2Backend(m)) as BoxedModelBackend)
                                }
                                Err(e) => {
                                    warn!("{arch}: tensor-parallel load failed ({e}); placing it layer by layer");
                                    crate::inference::engine::llm_engine::release_cuda_pools();
                                    None
                                }
                            }
                        })
                    };
                    #[cfg(not(feature = "cuda"))]
                    let tensor_parallel: Option<BoxedModelBackend> = None;

                    match tensor_parallel {
                        Some(m) => m,
                        None => {
                            // The embedding table joins the first layer's card when the plan can
                            // hold it there without moving a layer.
                            let stem_bytes = generic_weights.stem_bytes(&content);
                            let (first_plan, stem_fits) =
                                demand.plan_stem_if_free(&cuda_devices, stem_bytes);
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
                                &[],
                                || false,
                                |plan| {
                                    for idx in crate::inference::place::ladder::cards_of(plan) {
                                        if let std::collections::hash_map::Entry::Vacant(slot) = handles.entry(idx) {
                                            let d = Device::new_cuda(idx)?;
                                            #[cfg(feature = "cuda")]
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
                                    // The host keeps an unquantised cache sized by the model.
                                    let host_only = on_plan.is_empty();
                                    let (kv, ctx) = if host_only {
                                        (KvQuant::Off, None)
                                    } else {
                                        (kv_quant, Some(effective_ctx))
                                    };
                                    let stem_on_card = stem_fits
                                        && crate::inference::place::dry_plan::same_plan(plan, &first_plan);
                                    let m = generic_weights.build(&on_plan, plan, kv, ctx, stem_on_card)?;
                                    let placement = plan
                                        .segments
                                        .iter()
                                        .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    info!("✅ {arch} loaded: layers {placement} (kv_quant={kv:?}, kv_ctx={ctx:?})");
                                    Ok::<_, crate::tensor::Error>(m)
                                },
                            )
                            .map_err(|e| anyhow!("Failed to load {arch}: {e}"))?;
                            try_wrap_with_vision(
                                model,
                                projector_blob.as_deref(),
                                demand.whole().saturating_sub(demand.block_bytes + demand.head_bytes),
                            )?
                        }
                    }
                }
            };
            debug!("✓ Weights loaded from mmap in {:.1}ms", weights_t.elapsed().as_secs_f64() * 1000.0);

            debug!("✓ Model weights loaded");
            advise_pages_by_placement(&mmap, &layer_page_ranges, &model.device_layer_distribution());

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
                #[cfg(feature = "cuda")]
                moondream_graph: None,
                moondream_decode_count: 0,
                eos_token_ids_extra,
                file_size,
                model,
                tokenizer,
                device,
                image_embeds: None,
                qwen35_image: None,
                image_embed_cache: None,
                grammar_factory: std::sync::OnceLock::new(),
                generation_defaults: generation_defaults.clone(),
            };
            let mut guard = model_state.blocking_lock();
            // Free VRAM just changed by the size of a model.
            crate::inference::place::vram_manager::residency_changed();
            *guard = Some(new_state);
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
        #[cfg(feature = "cuda")]
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
fn advise_pages_by_placement(
    mmap: &std::sync::Arc<memmap2::Mmap>,
    ranges: &[(usize, usize, usize)],
    dist: &[(String, usize, u32, u32)],
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
    let total = mmap.len();
    let (mut kept, mut dropped, mut n_kept, mut n_dropped) = (0u64, 0u64, 0usize, 0usize);
    for &(layer, start, len) in ranges {
        let Some(host) = on_host(layer) else {
            continue;
        };
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
    info!(
        "📄 page cache after placement: {:.1} GB released ({n_dropped} tensors on the cards), {:.1} GB kept ({n_kept} on the host)",
        dropped as f64 / 1e9,
        kept as f64 / 1e9
    );
}
