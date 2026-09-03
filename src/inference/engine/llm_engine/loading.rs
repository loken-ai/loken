//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

/// Size the rayon pool the host kernels run on, once per process.
///
/// Called from `main` before anything else can touch rayon. It used to sit at the top of the
/// first model load, and by then something had already built the global pool - so
/// `build_global` failed on every start since at least 2026-08-27, the warning was the only
/// trace, and every host kernel ran on rayon's default of one thread per LOGICAL core. On a
/// non-hybrid SMT part that is two siblings contending for one core's execution units and L1,
/// which is what the count below exists to avoid.
pub fn configure_thread_pool(cpu_threads: usize) {
    RAYON_INIT.call_once(|| {
        // Detect hybrid CPU topology - if P/E cores are
        // distinguishable, pin worker threads to P-cores via
        // start_handler so E-cores don't steal cycles during
        // prefill. Falls back to all-cores on non-hybrid
        // hardware or when detection fails.
        let topo = crate::cpu::detect_cpu_topology();
        let p_cores: Vec<usize> = topo.get_inference_cores();
        let n = if cpu_threads == 0 {
            // Hybrid: use P-core count, not physical-core total
            // (otherwise the worker pool spills onto E-cores).
            // Non-hybrid: physical cores (skip HT siblings).
            if topo.is_hybrid && !p_cores.is_empty() {
                p_cores.len()
            } else {
                num_cpus::get_physical()
            }
        } else {
            cpu_threads
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
    /// Load a model from disk with real GGUF parsing and weight loading
    pub async fn load_model(&self) -> Result<(), Box<dyn std::error::Error>> {
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
        let max_gpu_memory_fraction = self.config.max_gpu_memory_fraction;
        let cpu_threads = self.config.cpu_threads;
        let disable_arc_layers = self.config.disable_arc_layers;
        // `--cpu` serve flag forces the same all-CPU placement as the
        // disable_cuda config (empty cuda_devices -> every layer on CPU).
        let disable_cuda = self.config.disable_cuda || crate::gpu::force_cpu();
        let force_gpu_layers = self.config.force_gpu_layers;
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

        // Load model in blocking thread since GGUF parsing and weight loading are CPU-intensive
        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            crate::inference::place::device_probe::debug_vram_by_card("closure-start");

            // --- 0. AWQ (HF safetensors) early path ---
            // If the resolved path is an AWQ checkpoint dir, load it via the
            // dedicated loader and store the state directly - bypassing the
            // entire GGUF parse/placement machinery below.
            #[cfg(feature = "cuda")]
            if let Some(awq_dir) =
                resolve_awq_dir(&model_file.to_string_lossy())
            {
                let mut state = load_awq_model_state(&awq_dir, &model_id_clone, user_context_length)?;
                let fsz = state.file_size;
                cb_maybe_wrap(&mut state);
                let mut guard = model_state.blocking_lock();
                // Free VRAM just changed by the size of a model.
                crate::inference::place::vram_manager::residency_changed();
                *guard = Some(state);
                drop(guard);
                let mut size_guard = cached_model_size.blocking_lock();
                *size_guard = fsz;
                info!("✅ AWQ model '{}' ready (state stored)", model_id_clone);
                return Ok(());
            }

            // --- 1. Parse GGUF header via mmap ---
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
            let mmap_bytes: &[u8] = &mmap[..];
            debug!("✓ GGUF file mmap'd: {} bytes", mmap_bytes.len());

            debug!("📖 Parsing GGUF header...");
            let mmap_parse_t = std::time::Instant::now();
            let mut content = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                .map_err(|e| {
                    let err_msg = format!("GGUF parse error: {e}");
                    error!("❌ {}", err_msg);
                    anyhow!(err_msg)
                })?;
            // Back the tensor data with the Arc<Mmap> so the loader reads weights as
            // zero-copy views (no host copy, the Arc pins the mmap) instead of copying
            // each tensor into owned host bytes - faster load + less committed RAM.
            content.mmap_owner = Some(mmap.clone() as std::sync::Arc<dyn std::any::Any + Send + Sync>);
            // Captured now: `content` is consumed by one of the loader arms below, and the
            // page advice can only be given once every layer has a device.
            let layer_page_ranges: Vec<(usize, usize, usize)> = content
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
                    ))
                })
                .collect();
            debug!("✓ GGUF header parsed in {:.1}ms", mmap_parse_t.elapsed().as_secs_f64() * 1000.0);
            debug!("✅ GGUF header parsed successfully");

            // --- 2. Extract metadata ---
            let arch = get_gguf_string(&content, "general.architecture")
                .unwrap_or_else(|| "llama".to_string());
            let vocab_size = get_gguf_u32(&content, &format!("{arch}.vocab_size"))
                .or_else(|| count_gguf_tokens(&content))
                .unwrap_or(32000) as usize;
            let context_length = get_gguf_u32(&content, &format!("{arch}.context_length"))
                .unwrap_or(4096) as usize;
            let eos_token_id = get_gguf_u32(&content, "tokenizer.ggml.eos_token_id")
                .unwrap_or(2);
            // Some models advertise multiple EOS-equivalent token IDs.
            // Gemma4: [1, 106, 50] - 1 is the canonical EOS, 106 & 50
            // are chat turn-end markers. Without honoring all three the
            // model can emit a turn-end and we'd keep generating.
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
            // Arch-specific fallback: when GGUF doesn't ship the eos_token_ids
            // array, fall back to known chat-template terminators. Without
            // these, gemma4's `<end_of_turn>` (token 106) and `<start_of_turn>`
            // (token 105) leak into the response and never stop generation.
            // verified gemma4 GGUF didn't have the array key.
            if eos_token_ids_extra.is_empty() && arch == "gemma4" {
                for tok in [105u32, 106, 50] {
                    if tok != eos_token_id { eos_token_ids_extra.push(tok); }
                }
            }
            if !eos_token_ids_extra.is_empty() {
                info!("📋 EOS token IDs: canonical={} extras={:?}", eos_token_id, eos_token_ids_extra);
            }

            // Extract actual layer count from GGUF (not hardcoded!)
            // Try multiple key variations for different architectures
            let num_layers = get_gguf_u32(&content, &format!("{arch}.num_layers"))
                .or_else(|| get_gguf_u32(&content, &format!("{arch}.num_hidden_layers")))
                .or_else(|| get_gguf_u32(&content, &format!("{arch}.block_count")))  // Mistral3 uses block_count
                .or_else(|| get_gguf_u32(&content, "model.num_layers"))        // Fallback: generic key
                .or_else(|| get_gguf_u32(&content, "model.num_hidden_layers"))  // Another generic key
                .or_else(|| get_gguf_u32(&content, "num_layers"))              // Bare key (some GGUF files)
                .or_else(|| get_gguf_u32(&content, "num_hidden_layers"))       // Bare key variant
                .or_else(|| get_gguf_u32(&content, "block_count"))             // Bare block_count
                .unwrap_or(32) as usize;

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
            let hidden_size = get_gguf_u32(&content, &format!("{arch}.hidden_size"))
                .or_else(|| get_gguf_u32(&content, &format!("{arch}.embedding_length")))
                .or_else(|| get_gguf_u32(&content, "embedding_length"))
                .unwrap_or(4096) as usize;
            let num_heads = get_gguf_u32(&content, &format!("{arch}.num_attention_heads"))
                .or_else(|| get_gguf_u32(&content, &format!("{arch}.attention.head_count")))
                .unwrap_or(32) as usize;
            let num_kv_heads = get_gguf_u32(&content, &format!("{arch}.attention.head_count_kv"))
                .or_else(|| get_gguf_u32(&content, &format!("{arch}.num_key_value_heads")))
                .unwrap_or(num_heads as u32) as usize;
            let head_dim = get_gguf_u32(&content, &format!("{arch}.attention.key_length"))
                .map(|v| v as usize)
                .unwrap_or_else(|| if num_heads > 0 { hidden_size / num_heads } else { 128 });

            info!("📋 Model info: arch={}, layers={}, hidden_size={}, vocab_size={}, context_length={}, eos_token_id={}, n_kv_heads={}, head_dim={}",
                arch, num_layers, hidden_size, vocab_size, context_length, eos_token_id,
                num_kv_heads, head_dim);

            // The Q8/Q4 fused attention kernels only support specific
            // (head_dim, n_q_per_kv) combinations. If the loaded config
            // doesn't match, the per-step decode fast path errors out
            // with "unsupported combo" - observed live on deepseek-r1
            // (40 q-heads / 8 kv-heads = n_q_per_kv=5; kernel wants
            // {1, 4, 8}). Auto-downgrade to kv_quant=Off so the model
            // is usable.
            let n_q_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
            // Phi2-style models (parallel attention + LayerNorm with bias)
            // hit a kernel launch failure on the Q8 attn_output path
            // ("too many resources requested for launch") even though
            // their (head_dim, n_q_per_kv) is in the supported set  - 
            // probably because of the partial RoPE / LayerNorm path
            // upstream that isn't covered by the Q8 fast path. Force the
            // KV cache to Off for these models.
            let _phi2_style = !content.tensor_infos.contains_key("blk.0.ffn_norm.weight")
                && content.tensor_infos.contains_key("blk.0.attn_norm.bias");
            // Gemma4 26B-A4B (30-layer MoE variant) has per-layer
            // head_dim variation (256 SWA + 512 Global). The per-layer
            // Q8 KV construction in inference/generic_transformer/ gates HD=512
            // layers off and uses n_kv_head_layer so VRAM stays in
            // budget. Allow kv_quant through here for it specifically.
            // gemma4:8b (42 layers, n_kv=2) has its own SWA layers but
            // the Q8 attention path produces incoherent output on its
            // shape - keep it on the auto-downgrade path.
            let kv_arch_check = content.metadata.get("general.architecture")
                .and_then(|v| match v {
                    crate::tensor::quantized::gguf_file::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            let _is_gemma4_moe = kv_arch_check == "gemma4" && num_layers == 30;
            // gemma4 stays OFF a quantized KV cache, and the probe that put it back on is
            // reverted here.
            //
            // The probe below reasoned that the old "incoherent" verdict was a missing
            // HD=512 kernel rather than a precision wall, and re-enabled Q8 for the whole
            // family. That was already answered. On 2026-06-19 the degeneration was traced
            // to the cache itself: gemma's k_norm is a uniform ~0.127 scalar, and Q8's
            // per-block scale amplifies its quantization noise until the output collapses
            // into repetition - coherent below 1k of context, garbage by 2.5k. The blanket
            // F16 answer OOM'd the tight 12b/26b/31b, so the shipped fix bounded the F16
            // SWA cache to its window instead: VRAM-cheap and mathematically exact, since
            // those layers attend only within it. gemma4:31b then benched at +94-103%.
            //
            // Re-enabling Q8 brought the repetition back across the family. The bandwidth
            // it was reaching for is real but it is not free, and it is not paid for by a
            // kernel - it needs a K-norm-aware quantization that does not yet exist.
            let is_gemma4_q8_ok = false;
            // HD=512 deliberately excluded - Q8 KV at HD=512 produces
            // degenerate output for gemma4 due to tiny K-norm weights
            // amplifying quantization noise. Verified with
            // gemma4:latest. Re-include only when K-norm-aware quant
            // is implemented.
            // verified the original gate is correct. Lifting
            // `!phi2_style` and benching moondream surfaced
            // `Q8 attn_output: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. The
            // non-dev_pos kernel `attn_output_q8_0_f32_hd64_nq1` uses
            // 32-warp x 32-thread = 1024-thread blocks; at HD=64 with
            // moondream's register usage it tips past the 64-reg/thread
            // budget per the 64K-reg/SM limit. The dev_pos kernel (16
            // warps, 512 threads) fits but only engages in graph mode,
            // and phi2 graph capture is itself blocked on the
            // ILLEGAL_ADDRESS class of bug.
            //
            // gate lifted. `try_q8_graph_decode` in
            // `inference/generic_transformer/` now lazy-primes
            // cur_pos_dev and runs without the `use_graph_ops` gate,
            // so phi2 routes through the dev_pos kernels (which DO fit)
            // even when graph capture is unavailable. Q8KvCache's
            // `grow_to` gate switched from `cur_pos_dev.is_some()` ->
            // `graph_captured` flag so the non-graph dev_pos path can
            // still grow its buffers. phi2 unblocked.
            let kv_quant_supported = (matches!(head_dim, 64 | 128 | 256)
                && matches!(n_q_per_kv, 1 | 2 | 4 | 5 | 8))
                && kv_arch_check != "gemma4"
                || is_gemma4_q8_ok;
            // The KV precision follows the architecture: what the quantised kernels can
            // serve, then what each family was measured to prefer.
            let kv_quant = if !kv_quant_supported && !matches!(kv_quant, KvQuant::Off) {
                warn!(
                    "⚠️  Q8/Q4 KV cache kernels don't support head_dim={} x n_q_per_kv={} (want {{64,128,256}} x {{1,2,4,5,8}}). Auto-downgrading kv_quant to Off for {}.",
                    head_dim, n_q_per_kv, &model_id_clone
                );
                KvQuant::Off
            } else if kv_quant_supported && matches!(kv_quant, KvQuant::Off)
                && (arch == "qwen3moe" || arch == "qwen3") {
                // Auto-promote qwen3/qwen3moe to Q4 KV.
                // qwen3moe (qwen3-coder), measured: Q4 outperforms
                // Q8 here - decode +17% WIN. Likely benefits from Q4's lower
                // KV memory bandwidth on the MoE expert routing path.
                // qwen3 (dense), measured: F-dtype KV decode loses
                // -12.9% to Ollama at 512 tokens (-0.4% at 128 tokens) because
                // the attention scan grows with seq_len at full F-dtype
                // bandwidth. Q4 KIVI's device-pos kernels stay graph-safe
                // and cut the per-token memory traffic in half.
                info!("Auto-promoting kv_quant Off -> Q4 for arch={arch}");
                KvQuant::Q4
            } else if kv_quant_supported && matches!(kv_quant, KvQuant::Off)
                && matches!(arch.as_str(), "llama" | "qwen2" | "mistral3"
                                         | "granite" | "granitemoe" | "olmoe") {
                // granite and granitemoe joined on their own A/B against the
                // F-dtype cache they were left on: both decode materially faster,
                // and the dense one additionally starts capturing a decode graph
                // that the F-dtype cache had refused. The routed one gains WITHOUT
                // capture, so its win is the cache's memory traffic rather than
                // submission cost.
                //
                // olmoe is held out, and it is the interesting case: it decodes
                // faster here AND returns full answers, right up until a request
                // REUSES a cached prefix - from the second request onward it stops
                // a fraction of the way through on a malformed token. Measured
                // three ways on the same cell: quantized + reuse truncates,
                // F-dtype + reuse is correct, quantized with reuse disabled is
                // correct and still faster than F-dtype. So neither the cache nor
                // the reuse is wrong alone; the pair is, and until that is
                // understood the model keeps the cache that answers correctly.
                // Suppressing reuse for it would be the better trade, but only
                // once the engine can decide that from a property rather than
                // from this model's name.
                //
                // Auto-promote llama/qwen2/mistral3-arch models (devstral,
                // devstral-small-2, deepcoder, mistral, qwen2-family) to Q8 KV.
                // mistral3: rerouted from the legacy MultiDeviceMistral3
                // path (which had F-dtype KV only) into GenericHetero - Q8 KV is one
                // of the optimizations it now inherits. It also fixes a first-forward
                // OOM: F-dtype KV reserved the full 32K context up-front and the
                // fill-first planner over-packed GPU0; Q8 lazy-grows from a 4096 cap.
                // Engages the GH_APQ
                // fast-decode path. Q4 was tried for qwen2 (deepcoder)
                // REGRESSED decode -28% vs Q8 under the
                // 2-GPU split. Re-tested post lazy-KV single-GPU
                // (commit 0ca111c): still REGRESSED (-4 to -5 % across
                // short/medium/long). Q8 wins regardless of placement.
                info!("Auto-promoting kv_quant Off -> Q8 for arch={arch} (enables GH_APQ fast-decode)");
                KvQuant::Q8
            } else if matches!(arch.as_str(), "phi2")
                && matches!(kv_quant, KvQuant::Off)
            {
                // Auto-promote phi2 (moondream) KV to Q8. Re-verified
                // With all session optimizations: F-dtype 260
                // tok/s vs Q8 347 tok/s -> Q8 +33% faster.
                info!("Auto-promoting kv_quant Off -> Q8 for arch=phi2 (GH_APQ partial-rope fast-decode)");
                KvQuant::Q8
            } else {
                // gemma4 (dense + 26B-MoE): stay on Off -> F16 SpecKvCache with
                // bounded sliding-window attention. gemma4's SWA layers attend only within a
                // 512-token window, so the F16 cache is bounded to window+chunk
                // keys (set_window in inference/generic_transformer/) - VRAM-cheap AND
                // exact. This replaces the old Off->Q8/Q4 promotions, whose Q8/Q4
                // KV degenerated gemma4's output at long context (tiny K-norm
                // ~0.127 amplifies quant noise past the window). The full-F16
                // global layers (1 in 6) keep correct long-range attention.
                kv_quant
            };

            // --- 3. Build tokenizer ---
            debug!("🔍 Looking for tokenizer.json...");
            let tokenizer: Tokenizer = {
                let tokenizer_path = model_file.parent()
                    .map(|p| p.join("tokenizer.json"))
                    .filter(|p| p.exists());

                if let Some(path) = tokenizer_path {
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

            // --- 4. Select device (size-aware: only use GPU if model fits) ---
            let gpu_idx = device_index.unwrap_or(0);

            // Enumerate all CUDA GPUs via NVML, sorted by performance (fastest first).
            // Shared, process-aware probe (also serves the ACE-Step native stack):
            // see `inference::place::device_probe`. `available` = stable_free x fraction.
            // Under pressure every card offers LESS, so a retry after an out-of-memory
            // load plans differently instead of handing back the plan that just failed.
            // Without this the escalation is a counter nobody reads: the image path had
            // exactly that bug, three identical re-plans and three identical failures.
            // Create every CUDA context, and force its cuBLAS workspace, BEFORE
            // reading free memory. On a fresh process none of that exists yet, so
            // the FIRST probe reports about half a gigabyte more than any later
            // one - memory this very load is about to spend. On a model that
            // spills to CPU the budget IS the split point, so that phantom half
            // gigabyte moved one layer across the boundary and greedy decoding
            // at temperature zero answered something else. Once, per process:
            // afterwards every load sees the same world as the first.
            // Warm ONLY the fastest card here. The warm's cuBLAS init latches the
            // card's primary context for the life of the process (the proven cost:
            // ~150 MB and idle watts, forever), and that latch is exactly what makes
            // budget probes stable across loads. So each card pays it the first time
            // a plan can actually reach it: the fastest card always can; the others
            // only when the model does not fit one GPU - see the multi-GPU branch.
            #[cfg(feature = "cuda")]
            warm_card(gpu_idx);

            // Hand the caching allocator's retained blocks back before probing.
            // They belong to no live tensor, but the driver still reports them as
            // used, so the budget depends on which model happened to run before.
            // For a model that must spill to CPU that budget IS the split point:
            // two loads of the SAME model saw 31.6 GB then 31.1 GB, moved one
            // layer across the boundary, and greedy decoding at temperature 0
            // answered two different things - a user-visible irreproducibility,
            // not a benchmark artefact.
            crate::inference::engine::llm_engine::release_cuda_pools();
            crate::inference::place::device_probe::debug_vram_by_card("after-pool-release");
            let boost = crate::inference::place::vram_manager::vram_reserve_boost();
            let all_cuda_gpus: Vec<(usize, u64, u64)> = // (index, stable_free, available)
                crate::inference::place::device_probe::probe_cuda_gpus_for("llm", max_gpu_memory_fraction)
                    .into_iter()
                    .map(|g| {
                        let free = g.stable_free.saturating_sub(boost);
                        (g.index, free, (free as f64 * max_gpu_memory_fraction) as u64)
                    })
                    .collect();

            // Primary GPU memory (for single-GPU decisions)
            let (_free_gpu_memory, available_gpu_memory) = all_cuda_gpus.iter()
                .find(|(idx, _, _)| *idx == gpu_idx)
                .map(|(_, free, avail)| (*free, *avail))
                .unwrap_or((0, 0));

            // Total available VRAM across all GPUs
            let total_available_gpu_memory: u64 = all_cuda_gpus.iter().map(|(_, _, avail)| avail).sum();

            // --- 4.5 The non-CUDA GPUs, if any answer ---
            //
            // Behind a deadline, because this is a driver call: a card removed while the system
            // ran leaves a device node that blocks every OpenCL client, and enumerating here
            // once wedged a whole node mid-load. See `opencl_probe`.
            let opencl_devices: Vec<(usize, u64)> = crate::inference::kernel::opencl_probe::non_cuda_devices()
                .into_iter()
                .enumerate()
                .map(|(idx, (name, vram))| {
                    debug!("🖥️  OpenCL device {}: {} ({:.1} GB)", idx, name, vram as f64 / 1e9);
                    (idx, vram)
                })
                .collect();

            // Compute device plan for multi-device support
            // CUDA: expansion 1.0 (quantized weights on GPU)
            // OpenCL: expansion 1.0 (keeps raw Q4 bytes, dequantizes on-the-fly in kernels)
            use crate::inference::place::layer_executor::HeteroPlan;

            // Prepare heterogeneous devices for potential hetero planning
            // Apply device filters (for benchmarking different configurations).
            //
            // Reserve a per-GPU loader overhead beyond the user's
            // `max_gpu_memory_fraction` budget - the second loader pass
            // performs dequant of small tensors (e.g. output projection),
            // KV cache dynamic staging, and CUDA scratch buffers that aren't
            // in the static `file_size` weights number. Without this
            // reserve, plans that fit weights+KV exactly within budget
            // OOM during that pass and the loader falls back to CPU.
            const LOADER_OVERHEAD_PER_GPU: u64 = 768 * 1024 * 1024; // 768 MB
            let cuda_devices: Vec<(usize, u64)> = if disable_cuda {
                debug!("CUDA disabled by config (disable_cuda=true)");
                vec![]
            } else {
                all_cuda_gpus.iter()
                    .map(|(idx, _, avail)| (*idx, avail.saturating_sub(LOADER_OVERHEAD_PER_GPU)))
                    .filter(|(_, avail)| *avail > 0)
                    .collect()
            };
            let opencl_devices = if disable_arc_layers {
                debug!("OpenCL/Arc disabled by config (disable_arc_layers=true)");
                vec![]
            } else {
                opencl_devices
            };
            let available_gpu_memory = if disable_cuda { 0 } else { available_gpu_memory };

            // Determine loading strategy:
            //   1. Model fits in GPU -> single-device GPU (CUDA)
            //   2. Model > GPU but CUDA available + mistral3 -> multi-CUDA GPU+CPU
            //   3. CUDA + OpenCL available + mistral3 -> heterogeneous (CUDA+Arc+CPU)
            //   4. Only OpenCL available + mistral3 -> OpenCL only (Arc+CPU)
            //   5. No GPU or unsupported arch -> single-device CPU
            // Single-GPU fit needs WEIGHTS + a runtime reserve, not just weights.
            // A model whose weights fit one card (e.g. devstral 14.3 GB on a
            // 15.6 GB budget) but leaves <1 GB free loads fine, then OOMs at the
            // FIRST request when cuBLAS initialises its handle/workspace, the KV
            // cache grows, and the prefill activation allocates - none of which
            // are in `file_size`. That runtime OOM happens AFTER the layers are
            // committed to one GPU, so the load-time split ladder never fires and
            // the idle second GPU goes unused. Reserve for it up front so the
            // planner spills onto the second GPU instead. The reserve scales with
            // the loaded context (KV cache, Q8: 2xK/V x kv_heads x head_dim x
            // ctx x n_layers) plus a fixed cuBLAS-workspace + prefill-activation
            // margin. Sized from the model's own dimensions - no per-machine
            // constants - so it adapts across architectures.
            let runtime_reserve_bytes: u64 = {
                // KV-cache reserve at the loaded context (Q8 ≈ 1 byte/elem; round
                // up for the F16 scale). The lazy caches start smaller and grow,
                // but reserving the steady footprint keeps the single-GPU choice
                // honest about where the cache will end up.
                // (Tried sizing this reserve to the request num_ctx instead of 8192
                // to fit borderline 24B single-GPU - NEUTRAL: mistral-small3.2 is
                // genuinely >15.6 GB so it doesn't fit 1 GPU at any reserve, and the
                // models that DO fit (devstral 14.3 GB) already fit at 8192. Kept the
                // conservative 8192 reserve for OOM safety margin.)
                let kv_ctx = (context_length as u64).min(8192);
                let kv = 2u64
                    .saturating_mul(num_kv_heads as u64)
                    .saturating_mul(head_dim as u64)
                    .saturating_mul(kv_ctx)
                    .saturating_mul(num_layers as u64);
                // cuBLAS handle/workspace + the chunked-prefill activation peak
                // (PREFILL_CHUNK_TOKENS x widest_ffn x f32 x ~2 live buffers).
                const CUBLAS_WORKSPACE_BYTES: u64 = 512 * 1024 * 1024; // ~512 MB
                let widest = hidden_size.max(num_kv_heads * head_dim) as u64;
                let act = (PREFILL_CHUNK_TOKENS as u64)
                    .saturating_mul(widest)
                    .saturating_mul(4)
                    .saturating_mul(2);
                kv.saturating_add(CUBLAS_WORKSPACE_BYTES).saturating_add(act)
            };
            let model_fits_single_gpu = file_size > 0
                && available_gpu_memory > 0
                && file_size.saturating_add(runtime_reserve_bytes) <= available_gpu_memory;
            let model_fits_gpu = file_size > 0
                && total_available_gpu_memory > 0
                && file_size <= total_available_gpu_memory;
            let cuda_available = total_available_gpu_memory > 0;
            let _multi_cuda = cuda_devices.len() > 1;
            let _opencl_available = !opencl_devices.is_empty();

            let device = if model_fits_single_gpu {
                info!("📊 Model ({:.1} GB) fits in single GPU memory ({:.1} GB), loading on GPU {}",
                    file_size as f64 / 1e9, available_gpu_memory as f64 / 1e9, gpu_idx);
                // Use Device::new_cuda (default stream) instead of
                // new_with_stream (non-default). Default stream + tracking
                // disabled is captureable in cudaStreamCaptureModeRelaxed,
                // which the engine uses for CUDA graph capture.
                let dev_result: crate::tensor::Result<Device> = crate::tensor::Device::new_cuda(gpu_idx)
                    .map(|d| {
                        #[cfg(feature = "cuda")]
                        if let Device::Cuda(ref cd) = d {
                            unsafe { cd.context().disable_event_tracking(); }
                        }
                        d
                    });
                dev_result.unwrap_or_else(|_| {
                    warn!("CUDA device {} unavailable, falling back to CPU", gpu_idx);
                    Device::Cpu
                })
            } else {
                // The split plan reads EVERY card's budget, so every card must carry
                // its context overhead before being measured - the warm latches it.
                // Re-probed after warming: the first probe saw the other cards bare,
                // and a budget that moves between loads moves the split layer, which
                // at temperature zero moves the answer.
                #[cfg(feature = "cuda")]
                let cuda_devices: Vec<(usize, u64)> = if disable_cuda {
                    cuda_devices.clone()
                } else {
                    for g in crate::inference::place::device_probe::probe_cuda_gpus_for("llm", 1.0) {
                        warm_card(g.index);
                    }
                    crate::inference::engine::llm_engine::release_cuda_pools();
                    crate::inference::place::device_probe::probe_cuda_gpus_for("llm", max_gpu_memory_fraction)
                        .into_iter()
                        .map(|g| {
                            let free = g.stable_free.saturating_sub(boost);
                            (
                                g.index,
                                ((free as f64 * max_gpu_memory_fraction) as u64)
                                    .saturating_sub(LOADER_OVERHEAD_PER_GPU),
                            )
                        })
                        .filter(|(_, avail)| *avail > 0)
                        .collect()
                };
                // Multi-GPU / spill: base device is CPU; the generic hetero
                // loader places each layer across CUDA GPUs (+ CPU spill) from
                // its own HeteroPlan, exactly like every other multi-GPU arch.
                // The cache is part of what has to fit. Sizing a placement on
                // the weights alone puts a model on the cards that runs out of
                // memory once the context fills, and the generation-time re-plan
                // then produces the SAME plan, because the budget it consults
                // still ignores the cache. Derived here rather than further down
                // where the cache is built: both inputs are known long before,
                // and the decision that needs them is taken here.
                //
                // The bound is the CONFIGURED context, not the model's declared
                // maximum: reserving for a 128k window a model will never open
                // would spill it to the host to prevent an OOM that cannot
                // happen. K and V, per layer, per kv head, over that window.
                let planned_ctx = user_context_length.min(context_length);
                let kv_elem: u64 = if matches!(kv_quant, KvQuant::Off) { 2 } else { 1 };
                let kv_bytes = 2u64
                    .saturating_mul(num_layers as u64)
                    .saturating_mul(num_kv_heads as u64)
                    .saturating_mul(head_dim as u64)
                    .saturating_mul(planned_ctx as u64)
                    .saturating_mul(kv_elem);
                if cuda_available && file_size.saturating_add(kv_bytes) > total_available_gpu_memory {
                    info!("📊 Model ({:.1} GB) > total CUDA GPU memory ({:.1} GB across {} GPUs), CPU spill via generic plan",
                        file_size as f64 / 1e9, total_available_gpu_memory as f64 / 1e9, cuda_devices.len());
                } else if !cuda_available {
                    debug!("No CUDA GPU memory detected, loading on CPU");
                }
                Device::Cpu
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
            let model: BoxedModelBackend = match arch.as_str() {
                // Moondream legacy path: original PyTorch state-dict naming.
                // Ollama's moondream GGUF uses standard llama.cpp naming
                // (token_embd, blk.{i}.attn_qkv) - fall through to the
                // generic phi2 path below instead.
                "moondream" if model_id_clone.to_lowercase().contains("moondream")
                    && content.tensor_infos.contains_key("text_model.transformer.embd.wte.weight") => {
                    debug!("Loading Moondream vision model from GGUF...");
                    let vb = crate::tensor::quantized::QVarBuilder::from_gguf(
                        &model_file, &device,
                    ).map_err(|e| anyhow!("Failed to load Moondream GGUF: {e}"))?;
                    let config = crate::inference::model::moondream::quantized::Config::v2();
                    match moondream::Model::new(&config, vb) {
                        Ok(model) => {
                            info!("Vision model loaded (Moondream v2)");
                            Box::new(MoondreamBackend(model))
                        }
                        Err(e) => {
                            let err_msg = format!("Failed to load Moondream weights: {e}");
                            error!("{}", err_msg);
                            return Err(anyhow!(err_msg));
                        }
                    }
                }
                // Generic transformer: Qwen2/3, Gemma3, Phi3, GLM4, StableLM, etc.
                // Supports CUDA + CPU multi-device via HeteroPlan.
                // All text LLMs go through GenericHetero for multi-device support.
                // This replaces the old QuantizedLlama single-device path for llama/mistral.
                // Qwen3-MoE (qwen3-coder etc): sparse MoE FFN. Handled by a
                // dedicated loader because weight names differ (ffn_gate_exps,
                // ffn_up_exps, ffn_down_exps, ffn_gate_inp) and the forward
                // routes via top-k experts.
                "qwen3moe" => {
                    // Prefer our multi-device loader to handle 30B-A3B-class
                    // models that overflow a single GPU. On a CPU-only build the
                    // MoE GEMM + attention take their fallback CPU paths (plain
                    // F-dtype KV; the Q8/Q4 KV fast paths are CUDA-only).
                    let dtype = crate::tensor::DType::F16;
                    // cuda_devices is Vec<(gpu_idx, free_mem_bytes)> - iterate
                    // sorted by idx so layer-split is deterministic.
                    let mut gpu_ids: Vec<usize> = cuda_devices.iter().map(|(i, _)| *i).collect();
                    gpu_ids.sort_unstable();
                    // Per-device throughput weights for proportional layer
                    // placement. Disabled: uniform split is optimal for
                    // typical homogeneous-tier multi-GPU setups and avoids
                    // the pipeline-bottleneck failure mode when per-layer
                    // cost is launch/scheduling-bound rather than
                    // throughput-bound.
                    let weights_enabled = false;
                    let manual_weights: Option<Vec<f32>> = None;
                    let device_weights: Option<Vec<f32>> = if let Some(w) = manual_weights {
                        info!("📐 qwen3moe placement RATIO (manual): {:?}", w);
                        Some(w)
                    } else if !weights_enabled {
                        None
                    } else {
                        match nvml_wrapper::Nvml::init() {
                            Ok(nvml) => {
                                use nvml_wrapper::enum_wrappers::device::Clock;
                                let mut ws = Vec::with_capacity(gpu_ids.len());
                                let mut all_ok = true;
                                for &idx in &gpu_ids {
                                    match nvml.device_by_index(idx as u32) {
                                        Ok(d) => {
                                            let sm_count = d.num_cores().unwrap_or(0) as f32;
                                            let clock = d.max_clock_info(Clock::SM).unwrap_or(0) as f32;
                                            ws.push((sm_count.max(1.0)) * (clock.max(1.0)));
                                        }
                                        Err(_) => { all_ok = false; break; }
                                    }
                                }
                                if all_ok && ws.len() == gpu_ids.len() {
                                    let total: f32 = ws.iter().sum();
                                    let pct: Vec<String> = ws.iter()
                                        .map(|w| format!("{:.0}%", w / total * 100.0))
                                        .collect();
                                    info!("📐 qwen3moe placement weights (opt-in): {:?} ({})", ws, pct.join("/"));
                                    Some(ws)
                                } else { None }
                            }
                            Err(_) => None,
                        }
                    };
                    let mut gpu_list: Vec<Device> = Vec::new();
                    for idx in &gpu_ids {
                        // Default (NULL) stream rather than `new_with_stream`
                        // so that an external kernel's C bindings - which
                        // hard-code `cudaStream_t stream = 0` - order
                        // correctly against cudarc allocs. With a custom
                        // stream, the FA kernel on NULL races with the
                        // cudaMallocAsync on our stream and produces NaN
                        // output on the first layer.
                        match Device::new_cuda(*idx) {
                            Ok(d) => gpu_list.push(d),
                            Err(e) => warn!("Skipping GPU {idx}: {e}"),
                        }
                    }
                    if gpu_list.is_empty() {
                        // No usable CUDA device (or a CPU-only build) -> run on CPU.
                        gpu_list.push(Device::Cpu);
                    }
                    info!("⬇️  Loading qwen3moe across {} device(s) via MultiDeviceQwen3MoE",
                        gpu_list.len());
                    let content_fresh = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                        .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                    let weights_ref = device_weights.as_deref()
                        .filter(|w| w.len() == gpu_list.len());
                    match crate::inference::model::qwen3::moe_multi::MultiDeviceQwen3MoE::from_gguf_mmap_full(
                        content_fresh,
                        mmap_bytes,
                        &gpu_list,
                        weights_ref,
                        dtype,
                        kv_quant,
                        Some(user_context_length.min(context_length)),
                    ) {
                        Ok(m) => {
                            info!("✅ qwen3moe loaded across {} GPUs (kv_quant={:?})", gpu_list.len(), kv_quant);
                            Box::new(QwenMoEMultiBackend(m))
                        }
                        Err(e) => return Err(anyhow!("MultiDeviceQwen3MoE load failed: {e}")),
                    }
                }
                "gptoss" => {
                    // gpt-oss (OPENAI_MOE): MXFP4 MoE + attention sinks +
                    // sliding-window + NEOX RoPE. The MXFP4 experts are kept
                    // native (the MoE-GEMM has an MXFP4 MMVQ path, case 7), so
                    // expert VRAM stays ~12 GB (vs ~24 GB if requantized to
                    // Q8_0) and decode reads half the expert bytes/token  - 
                    // which lets the whole model fit a single 16 GB GPU
                    // (preferred; see the single-GPU rationale below).
                    // F16 working dtype: gpt-oss's float weights (attn,
                    // lm_head) are requantized to Q8_0 at load, so every matmul
                    // takes the dtype-agnostic MMVQ path (the proven F16
                    // quantized path, as for gemma4/deepcoder). The F32 norm /
                    // bias / sink / cos-sin tensors are cast around their ops
                    // (norms run in F32; attention softmax upcasts to F32).
                    let dtype = crate::tensor::DType::F16;
                    let mut gpu_ids_all: Vec<usize> = cuda_devices.iter().map(|(i, _)| *i).collect();
                    gpu_ids_all.sort_unstable();
                    // CPU run when no GPUs OR this is a non-CUDA build (the GPUs are
                    // still enumerated by NVML in the device manager, but the substrate
                    // can't use them without the cuda feature -> force CPU).
                    let cpu_only = gpu_ids_all.is_empty() || cfg!(not(feature = "cuda"));
                    // Native MXFP4 gpt-oss (~14 GB) fits a single 16 GB GPU, and the
                    // contiguous layer split runs sequentially (no decode
                    // parallelism). Single-GPU MEASURED +29% (117->151 tok/s) by
                    // dropping the per-token cross-GPU hidden-state handoff + sync,
                    // and is also the prerequisite for CUDA-graph capture (capture
                    // is per-stream/device - a 2-GPU forward can't be one graph).
                    // Prefer the fastest single GPU ("use fastest GPU first"); fall
                    // back to a multi-GPU split only if the single-GPU load OOMs on
                    // a smaller card.
                    let force_multi = false;
                    let attempts: Vec<Vec<usize>> = if cpu_only {
                        vec![vec![]] // single CPU attempt
                    } else if force_multi || gpu_ids_all.len() == 1 {
                        vec![gpu_ids_all.clone()]
                    } else {
                        vec![vec![gpu_ids_all[0]], gpu_ids_all.clone()]
                    };
                    let mut loaded: Option<BoxedModelBackend> = None;
                    let mut last_err = String::new();
                    for ids in &attempts {
                        let mut gpu_list: Vec<Device> = Vec::new();
                        for idx in ids {
                            match Device::new_cuda(*idx) {
                                Ok(d) => gpu_list.push(d),
                                Err(e) => warn!("gptoss: skipping GPU {idx}: {e}"),
                            }
                        }
                        if gpu_list.is_empty() {
                            if cpu_only { gpu_list.push(Device::Cpu); } else { continue; }
                        }
                        info!("⬇️  Loading gptoss (native MXFP4 MoE + Q8_0 attn + attn-sinks) on {} CUDA GPU(s)", gpu_list.len());
                        let content_fresh = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                            .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                        let mut reader = Cursor::new(mmap_bytes);
                        match crate::inference::model::gptoss::GptOssModel::from_gguf(
                            &content_fresh, &mut reader, &gpu_list, dtype, Some(&mmap),
                        ) {
                            Ok(mut m) => {
                                info!("✅ gptoss loaded on {} GPU(s)", gpu_list.len());
                                // Diagnostic: probe whether the decode forward
                                // captures + replays as a CUDA graph (settles the
                                // graph-mode question for this arch before the full
                                // device-pos rewrite). Single-GPU only; gated.
                                loaded = Some(Box::new(GptOssBackend(m)));
                                break;
                            }
                            Err(e) => {
                                last_err = format!("{e}");
                                warn!("gptoss load on {} GPU(s) failed: {e}", gpu_list.len());
                            }
                        }
                    }
                    match loaded {
                        Some(m) => m,
                        None => return Err(anyhow!("gptoss load failed: {last_err}")),
                    }
                }
                "nemotron_h_moe" => {
                    // nemotron-H: hybrid Mamba2 + attention + non-gated MoE.
                    // Float weights requantized to Q8_0 at load (same F16-stream
                    // MMVQ rationale as gptoss); SSM recurrence / norms / softmax
                    // / MoE run in F32. Layer-split across all CUDA GPUs.
                    let dtype = crate::tensor::DType::F16;
                    let mut gpu_ids: Vec<usize> = cuda_devices.iter().map(|(i, _)| *i).collect();
                    gpu_ids.sort_unstable();
                    let mut gpu_list: Vec<Device> = Vec::new();
                    for idx in &gpu_ids {
                        match Device::new_cuda(*idx) {
                            Ok(d) => gpu_list.push(d),
                            Err(e) => warn!("nemotron_h: skipping GPU {idx}: {e}"),
                        }
                    }
                    if gpu_list.is_empty() {
                        gpu_list.push(Device::Cpu); // CPU-only build: the host shims
                        info!("⬇️  Loading nemotron_h_moe (Mamba2+attn+MoE) on CPU");
                    } else {
                        info!("⬇️  Loading nemotron_h_moe (Mamba2+attn+MoE) across {} CUDA GPU(s)", gpu_list.len());
                    }
                    // Per-GPU usable VRAM in gpu_list order -> bandwidth/VRAM-weighted
                    // layer split (fill the faster GPU first; the equal-count split
                    // let the slower GPU gate sequential decode).
                    let file_size = mmap_bytes.len() as u64;
                    let gpu_avail: Vec<u64> = gpu_ids.iter().map(|id|
                        cuda_devices.iter().find(|(i, _)| i == id).map(|(_, a)| *a).unwrap_or(0)
                    ).collect();
                    let mut loaded: Option<BoxedModelBackend> = None;
                    let mut last_err = String::new();
                    // Attempt 1: weighted split. Attempt 2 (OOM): even split (empty avail).
                    for avail in [gpu_avail.as_slice(), &[][..]] {
                        let content_fresh = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                            .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                        let mut reader = Cursor::new(mmap_bytes);
                        match crate::inference::model::nemotron_h::NemotronHModel::from_gguf(
                            &content_fresh, &mut reader, &gpu_list, dtype, avail, file_size,
                            runtime_reserve_bytes, Some(&mmap),
                        ) {
                            Ok(m) => {
                                info!("✅ nemotron_h_moe loaded ({} split) across {} GPU(s)",
                                    if avail.is_empty() { "even" } else { "weighted" }, gpu_list.len());
                                loaded = Some(Box::new(NemotronHBackend(m)));
                                break;
                            }
                            Err(e) => {
                                warn!("nemotron_h {} split load failed: {e}", if avail.is_empty() { "even" } else { "weighted" });
                                last_err = e.to_string();
                            }
                        }
                    }
                    match loaded {
                        Some(m) => m,
                        None => return Err(anyhow!("nemotron_h load failed: {last_err}")),
                    }
                }
                "lfm2moe" | "lfm2" => {
                    // LFM2 hybrid short-conv + attention + SwiGLU FFN. `lfm2moe`
                    // uses MoE experts; the dense `lfm2` arch (e.g. lfm2.5-thinking)
                    // uses a plain SwiGLU FFN - both handled by Lfm2MoeModel (its
                    // `Ffn::Dense` path covers all blocks when there are no experts).
                    // F16 stream; shortconv / norms / softmax / FFN in F32.
                    // Experts are Q4_K (block-256, K÷256 ok) so no requant needed.
                    let dtype = crate::tensor::DType::F16;
                    // Single-GPU when the model fits (14.4 GB <= 15.6 GB avail):
                    // splitting lfm2moe across 2 GPUs (the prior unconditional
                    // behaviour) ate cross-device latency every layer (109 tok/s
                    // measured 2-GPU) AND made the decode forward uncapturable
                    // (the CUDA graph path captures one GPU's stream). Mirrors the
                    // gpt-oss/deepcoder single-GPU win - keep all layers on the
                    // fastest GPU (gpu_idx) when weights+KV fit.
                    let mut gpu_ids: Vec<usize> = if model_fits_single_gpu {
                        vec![gpu_idx]
                    } else {
                        cuda_devices.iter().map(|(i, _)| *i).collect()
                    };
                    gpu_ids.sort_unstable();
                    let mut gpu_list: Vec<Device> = Vec::new();
                    for idx in &gpu_ids {
                        match Device::new_cuda(*idx) {
                            Ok(d) => {
                                // Single-GPU: per-slice event tracking OFF (must
                                // precede tensor loads - a slice created with
                                // tracking on keeps recording forever). The lfm2
                                // decode is launch-bound (~840 launches/token);
                                // with tracking on every kernel arg pays a
                                // cuStreamWaitEvent + cuEventRecord (~5+3 per
                                // launch measured) ≈ ms-scale pure CPU overhead
                                // per token. Single stream -> events are self-
                                // stream no-ops. Multi-GPU keeps events (cross-
                                // device sync needs them).
                                #[cfg(feature = "cuda")]
                                if gpu_ids.len() == 1 {
                                    if let Device::Cuda(ref cd) = d {
                                        unsafe { cd.context().disable_event_tracking(); }
                                    }
                                }
                                gpu_list.push(d);
                            }
                            Err(e) => warn!("lfm2moe: skipping GPU {idx}: {e}"),
                        }
                    }
                    if gpu_list.is_empty() {
                        // CPU-only build (or no CUDA GPU): run lfm2moe on the CPU
                        // device - the MoE expert-GEMM + fused ops use the tensor-op
                        // shims (moe_cuda_cpu / fused_kernels_cpu).
                        gpu_list.push(Device::Cpu);
                        info!("⬇️  Loading lfm2moe (shortconv+attn+MoE) on CPU");
                    } else {
                        info!("⬇️  Loading lfm2moe (shortconv+attn+MoE) across {} CUDA GPU(s)", gpu_list.len());
                    }
                    let content_fresh = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                        .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                    let mut reader = Cursor::new(mmap_bytes);
                    match crate::inference::model::lfm2_moe::Lfm2MoeModel::from_gguf(
                        &content_fresh,
                        &mut reader,
                        &gpu_list,
                        dtype,
                        Some(&mmap),
                    ) {
                        Ok(m) => {
                            info!("✅ lfm2moe loaded across {} device(s)", gpu_list.len());
                            Box::new(Lfm2MoeBackend(m))
                        }
                        Err(e) => return Err(anyhow!("lfm2moe load failed: {e}")),
                    }
                }
                "qwen35moe" | "qwen35" | "qwen3next" => {
                    // The dense qwen35 belongs here, not on the generic arm: the family's
                    // block is gated DeltaNet linear attention, which only this path
                    // implements. Dense vs MoE differs in the FFN alone - the same reason
                    // the dense lfm2 was wired onto the lfm2moe path rather than ported
                    // separately.
                    // Qwen3.5-MoE / Qwen3-Next-80B (same family, two GGUF arch tags):
                    // gated DeltaNet linear-attn + attention + softmax MoE
                    // (+ scalar-gated shared expert). F16 stream; deltanet/norms/softmax
                    // /MoE in F32. TEXT path only (vision ViT is CUDA-only). On a CPU
                    // build the DeltaNet recurrence uses the pure tensor-op per-token path.
                    let dtype = crate::tensor::DType::F16;
                    let mut gpu_ids: Vec<usize> = cuda_devices.iter().map(|(i, _)| *i).collect();
                    gpu_ids.sort_unstable();
                    let mut gpu_list: Vec<Device> = Vec::new();
                    for idx in &gpu_ids {
                        match Device::new_cuda(*idx) {
                            Ok(d) => gpu_list.push(d),
                            Err(e) => warn!("qwen35moe: skipping GPU {idx}: {e}"),
                        }
                    }
                    if gpu_list.is_empty() {
                        // No usable CUDA device (or a CPU-only build) -> run on CPU.
                        gpu_list.push(Device::Cpu);
                    }
                    info!("⬇️  Loading qwen35moe (gated-DeltaNet+attn+MoE) across {} device(s)", gpu_list.len());
                    let file_size = mmap_bytes.len() as u64;
                    let mut gpu_avail: Vec<u64> = gpu_ids.iter().map(|id|
                        cuda_devices.iter().find(|(i, _)| i == id).map(|(_, a)| *a).unwrap_or(0)
                    ).collect();
                    // If the model exceeds total GPU VRAM (e.g. Qwen3-Next-80B at 48GB on
                    // 2x16GB), append CPU so plan_layer_devices spills the OVERFLOW to host
                    // RAM instead of piling every remaining layer onto the last GPU (OOM).
                    let total_vram: u64 = gpu_avail.iter().sum();
                    if file_size > total_vram && !gpu_list.iter().any(|d| d.is_cpu()) {
                        gpu_list.push(Device::Cpu);
                        gpu_avail.push(file_size); // CPU budget: hold whatever spills over
                        info!("📦 model {}GB > VRAM {}GB -> offloading overflow to CPU",
                            file_size >> 30, total_vram >> 30);
                    }
                    let mut loaded: Option<BoxedModelBackend> = None;
                    let mut last_err = String::new();
                    for avail in [gpu_avail.as_slice(), &[][..]] {
                        let content_fresh = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                            .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                        let mut reader = Cursor::new(mmap_bytes);
                        match crate::inference::model::qwen35::moe::Qwen35MoeModel::from_gguf(
                            &content_fresh, &mut reader, &gpu_list, dtype, avail, file_size,
                            runtime_reserve_bytes, Some(&mmap),
                        ) {
                            Ok(m) => {
                                info!("✅ qwen35moe loaded ({} split) across {} GPU(s)",
                                    if avail.is_empty() { "even" } else { "weighted" }, gpu_list.len());
                                loaded = Some(Box::new(Qwen35MoeBackend(m)));
                                break;
                            }
                            Err(e) => { warn!("qwen35moe {} split load failed: {e}", if avail.is_empty() { "even" } else { "weighted" }); last_err = e.to_string(); }
                        }
                    }
                    match loaded {
                        Some(m) => m,
                        None => return Err(anyhow!("qwen35moe load failed: {last_err}")),
                    }
                }
                // A model that does not fit one card runs EVERY layer on both, instead of one
                // card per token with the other idle. q/k/v and gate/up are column-parallel,
                // o and down row-parallel joined by an all-reduce. A model that DOES fit one
                // card stays on the single-card path, which is faster still - the guard is the
                // whole point, and it is derived from the weights rather than named per model.
                #[cfg(feature = "cuda")]
                "qwen2" if !model_fits_single_gpu && model_fits_gpu && cuda_devices.len() >= 2 => {
                    info!("⬇️  Loading qwen2 as TpQwen2 - TP=2 across GPU {} + {}",
                        cuda_devices[0].0, cuda_devices[1].0);
                    let g0 = crate::tensor::Device::new_cuda(cuda_devices[0].0)
                        .map_err(|e| anyhow!("TpQwen2 cuda:{}: {e}", cuda_devices[0].0))?;
                    let g1 = crate::tensor::Device::new_cuda(cuda_devices[1].0)
                        .map_err(|e| anyhow!("TpQwen2 cuda:{}: {e}", cuda_devices[1].0))?;
                    match crate::inference::serve::tp_model::TpQwen2::from_gguf(mmap_bytes, &mmap, &arch, g0, g1) {
                        Ok(m) => {
                            info!("✅ qwen2 loaded as TpQwen2 (tensor-parallel across 2 GPUs)");
                            Box::new(TpQwen2Backend(m))
                        }
                        Err(e) => return Err(anyhow!("TpQwen2 load failed: {e}")),
                    }
                }
                // Same for the dense no-bias archs - the 24B llama/mistral family. Dense only:
                // expert_count == 0 excludes mixtral, which also reports arch="llama" and whose
                // routed FFN this decoder does not implement. The QKV bias is read only when the
                // file has one, which is what lets these archs share the qwen2 decoder.
                #[cfg(feature = "cuda")]
                "llama" | "mistral" | "mistral3"
                    if !model_fits_single_gpu && model_fits_gpu && cuda_devices.len() >= 2
                        && get_gguf_u32(&content, &format!("{arch}.expert_count")).unwrap_or(0) == 0 => {
                    info!("⬇️  Loading {} as TpQwen2 (dense) - TP=2 across GPU {} + {}",
                        arch, cuda_devices[0].0, cuda_devices[1].0);
                    let g0 = crate::tensor::Device::new_cuda(cuda_devices[0].0)
                        .map_err(|e| anyhow!("TP dense cuda:{}: {e}", cuda_devices[0].0))?;
                    let g1 = crate::tensor::Device::new_cuda(cuda_devices[1].0)
                        .map_err(|e| anyhow!("TP dense cuda:{}: {e}", cuda_devices[1].0))?;
                    match crate::inference::serve::tp_model::TpQwen2::from_gguf(mmap_bytes, &mmap, &arch, g0, g1) {
                        Ok(m) => {
                            info!("✅ {} loaded as TpQwen2 (tensor-parallel across 2 GPUs)", arch);
                            Box::new(TpQwen2Backend(m))
                        }
                        Err(e) => return Err(anyhow!("TP dense ({arch}) load failed: {e}")),
                    }
                }
                "llama" | "mistral" | "mistral3" |
                "qwen2" | "qwen3" | "gemma" | "gemma2" | "gemma3" |
                "gemma3n" | "gemma4" |
                "phi" | "phi2" | "phi3" | "phi4" | "chatglm" | "glm4" | "stablelm" | "starcoder" |
                "starcoder2" | "falcon" | "bloom" | "mpt" | "refact" | "codeshell" |
                "granite" | "granite3" | "granitemoe" | "dbrx" | "internlm2" | "yi" | "orion" |
                "olmo2" | "olmoe" | "ernie4_5" | "smollm3" => {
                    debug!("⬇️  Loading '{}' as generic hetero transformer (CUDA+CPU)...", arch);
                    // For generic models: CUDA + CPU only (no OpenCL, those need per-model kernels).
                    // Respect force_gpu_layers when set; otherwise auto-calculate from VRAM.
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
                    // gemma4 26B-MoE with kv_quant != Off: SWA layers
                    // keep BOTH F-dtype kv_cache AND Q8 kv_cache (Q8
                    // lazy-pre-alloc at load). Reserve 3 bytes/elem
                    // (F16 + Q8) so the planner doesn't pack more
                    // layers on GPU than fit.
                    let kv_dtype_bytes = if arch == "gemma4"
                        && num_layers == 30
                        && !matches!(kv_quant, KvQuant::Off)
                    {
                        3u64
                    } else {
                        kv_dtype_bytes
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
                    // For Q4/Q8 lazy KV caches (Q4KvCache / Q8KvCache), the initial
                    // allocation is capped at INITIAL_CAPACITY_TOKENS=4096 - the
                    // cache grows on demand up to the user's full context_length.
                    // Pre-planning for the worst-case context overestimates VRAM
                    // need by ~8x at default config (32K config vs 4K bench), which
                    // pushed deepcoder onto an unnecessary 2-GPU split that costs
                    // ~20% decode.
                    //
                    // Use the actual initial allocation as the planning estimate;
                    // when the cache later grows, the NVML-polling adaptive loader
                    // in `inference/generic_transformer/` catches VRAM pressure and
                    // diverts subsequent layers. F-dtype caches still pre-allocate
                    // the full user context, so they keep the worst-case estimate.
                    let lazy_kv_initial_cap =
                        crate::inference::cache::KV_WORKING_WINDOW_TOKENS as u64;
                    // What one layer's cache costs at a given context, sliding windows
                    // included: a windowed layer never holds more than its window.
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
                        2u64 * num_kv_heads as u64 * head_dim as u64 * avg * kv_dtype_bytes
                    };

                    // Choose the context that yields the best PLAN, not the largest number.
                    //
                    // A quantised cache grows on demand, so planning against its initial cap
                    // is already honest. An F-dtype cache pre-allocates in full, and honouring
                    // a context the cards cannot hold does not fail - it pushes layers onto the
                    // host, and a host layer costs far more than a context window nobody asked
                    // for. Measured on gemma4:31b: the configured 32768 sent six of sixty layers
                    // to the processor and the model decoded at a third of ollama's rate, while
                    // its 12b, 26b and latest siblings all won.
                    //
                    // So the hardware decides. Take the largest context whose plan keeps every
                    // layer on the cards, halving down to the cap the lazy path already uses as
                    // its floor. Nothing here is a memory constant: each candidate is priced
                    // from the model's own dimensions and weighed against the cards actually
                    // present, so the same code lands differently on different hardware - which
                    // is the point.
                    let ctx_for_planning: u64 = if matches!(kv_quant, KvQuant::Q4 | KvQuant::Q8) {
                        lazy_kv_initial_cap.min(user_context_length as u64)
                    } else if !cuda_available || cuda_devices.is_empty() {
                        user_context_length as u64
                    } else {
                        let full = user_context_length as u64;
                        let floor = lazy_kv_initial_cap.min(full);
                        let spills = |ctx: u64| {
                            HeteroPlan::calculate_with_kv(
                                num_layers, file_size, &cuda_devices, &[], 1.0, kv_bytes_at(ctx),
                            )
                            .segments
                            .iter()
                            .any(|s| !matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::Cuda(_)))
                        };
                        let host_layers = |ctx: u64| -> usize {
                            HeteroPlan::calculate_with_kv(
                                num_layers, file_size, &cuda_devices, &[], 1.0, kv_bytes_at(ctx),
                            )
                            .segments
                            .iter()
                            .filter(|s| !matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::Cuda(_)))
                            .map(|s| s.num_layers())
                            .sum()
                        };
                        let mut chosen = full;
                        while chosen > floor && spills(chosen) {
                            chosen = (chosen / 2).max(floor);
                        }
                        if chosen < full {
                            info!(
                                "🧠 KV context {} -> {}: at the configured context {} of {} layers would have gone to the host",
                                full, chosen, host_layers(full), num_layers
                            );
                        }
                        chosen
                    };
                    let avg_ctx_per_layer: u64 = if !pattern.is_empty() && sliding_window > 0 {
                        let cap_swa = ctx_for_planning.min(sliding_window);
                        let n_swa = pattern.iter().filter(|&&is_swa| is_swa).count() as u64;
                        let n_global = pattern.len() as u64 - n_swa;
                        let total: u64 = n_swa.saturating_mul(cap_swa)
                                       + n_global.saturating_mul(ctx_for_planning);
                        if !pattern.is_empty() { total / pattern.len() as u64 } else { ctx_for_planning }
                    } else {
                        ctx_for_planning
                    };
                    // Per-layer non-weight cost during first prefill =
                    // Per-layer KV-cache footprint, derived purely from the
                    // model's real dimensions (heads x head_dim x context x
                    // dtype) - no magic activation constant. Earlier revisions
                    // added a static `activation_headroom_per_layer`
                    // (150 MB -> 80 MB for hidden>=4096, 4 MB otherwise) to pad
                    // the plan toward a 2-GPU split so the first prefill's
                    // activation peak wouldn't OOM. That constant was an
                    // arbitrary single-model measurement (qwen3:latest) and is the wrong abstraction: it can't
                    // distinguish a model that genuinely needs 2 GPUs
                    // (devstral, whose WEIGHTS don't fit one card) from one
                    // that doesn't (deepcoder/gpt-oss/qwen3, which fit and win
                    // big single-GPU). The correct mechanism is REACTIVE, not a
                    // guess: place optimistically by the weights-only fit
                    // (`model_fits_single_gpu`), and recover from a real
                    // allocation failure - the load-time OOM retry ladder below
                    // (single-GPU -> forced 2-GPU split -> CPU) plus the NVML
                    // divert-on-pressure loader in generic_transformer handle
                    // genuine overflow without pre-reserving a fictional margin.
                    let kv_bytes_per_layer: u64 = 2u64
                        * num_kv_heads as u64
                        * head_dim as u64
                        * avg_ctx_per_layer
                        * kv_dtype_bytes;
                    // Per-GPU runtime headroom the weights number doesn't include:
                    // the cuBLAS handle/workspace + the chunked-prefill activation
                    // peak. Reserving it on each card stops the pack-first planner
                    // from committing a GPU's last few hundred MB that the first
                    // request then needs (devstral 14.3 GB packed GPU0 -> 436 MB
                    // free -> cuBLAS init OOM, GPU1 idle). The KV cache itself is
                    // already accounted per-layer via `kv_bytes_per_layer`.
                    let gpu_runtime_reserve: u64 = {
                        const CUBLAS_WORKSPACE_BYTES: u64 = 512 * 1024 * 1024; // ~512 MB
                        let widest = hidden_size.max(num_kv_heads * head_dim) as u64;
                        let act = (PREFILL_CHUNK_TOKENS as u64)
                            .saturating_mul(widest)
                            .saturating_mul(4)
                            .saturating_mul(2);
                        CUBLAS_WORKSPACE_BYTES.saturating_add(act)
                    };
                    let hetero_plan = if let Some(forced) = force_gpu_layers {
                        if cuda_available {
                            HeteroPlan::forced_gpu(num_layers, forced, gpu_idx)
                        } else {
                            HeteroPlan::calculate_with_kv(num_layers, file_size, &[], &[], 1.0, kv_bytes_per_layer)
                        }
                    } else if model_fits_single_gpu && cuda_available {
                        // Weights + KV reserve fit ONE GPU -> keep ALL layers on gpu_idx
                        // (single-GPU decode win) instead of letting the planner spread
                        // a fits-1-GPU model across 2 GPUs + CPU, which eats cross-device
                        // sync every layer (e.g. devstral 24B: 14.3 GB fits 15.6 GB but
                        // calculate_with_kv_reserve still planned 2 segments -> -4% vs
                        // ollama's clean single GPU). Mirrors the lfm2moe/gpt-oss/deepcoder
                        // single-GPU path. The OOM-adaptive ladder below reacts (forced
                        // even split, then CPU) if a real prefill OOM surfaces, so this
                        // optimism can't mis-size.
                        HeteroPlan::forced_gpu(num_layers, num_layers, gpu_idx)
                    } else {
                        HeteroPlan::calculate_with_kv_reserve(
                            num_layers, file_size, &cuda_devices, &[], 1.0,
                            kv_bytes_per_layer, gpu_runtime_reserve,
                        )
                    };
                    // Create CUDA devices for ALL detected GPUs, not just
                    // those the planner picked. The loader's per-layer
                    // adaptive placement (see generic_transformer
                    // ::from_gguf load loop) queries NVML free between
                    // layers and diverts to a secondary GPU when the
                    // primary fills up - for that to work it needs a
                    // Device handle on every available GPU even if the
                    // initial plan only used one. Cost: an extra
                    // Device::new_cuda(idx) for unused GPUs (one-time,
                    // ~1 ms each).
                    //
                    // We intentionally create fresh Device handles instead
                    // of reusing the engine's `device`: sharing the
                    // engine's device (which has `enable_graph_capture_mode`
                    // set for potential future capture) breaks the normal
                    // forward path - the stream + async allocator state is
                    // incompatible with regular inference. The engine's
                    // graph capture instead targets the MODEL's stream via
                    // `ModelBackend::model_cuda_stream()`.
                    //
                    // Engine-site CUDA-graph decode additionally requires
                    // per-slice CudaEvent tracking OFF on the model's
                    // handles, disabled at creation BEFORE any tensor
                    // loads: with tracking ON every CudaSlice carries
                    // read/write events recorded on each use, and capturing
                    // the decode forward then records/waits events created
                    // OUTSIDE the capture -> CAPTURE_INVALIDATED
                    // (root-caused; gptoss documents the same
                    // "must precede tensor loads" constraint).
                    //
                    // Scope: every handle, whatever shape the plan has. The
                    // tracking exists to order a slice against a SECOND
                    // stream, and there is none - a device owns exactly one
                    // working stream. Within a device the ops are therefore
                    // already stream-ordered, and a cross-device transfer
                    // records and waits its own event explicitly, so the
                    // per-op events order a stream against itself. On a split
                    // model that is not a small waste: 5600 waits and 3184
                    // records per token, 36% of the CUDA API time, none of it
                    // present on the same model when it fits one card.
                    let mut hetero_cuda_devs: std::collections::HashMap<usize, Device> = std::collections::HashMap::new();
                    for (idx, _) in &cuda_devices {
                        if let std::collections::hash_map::Entry::Vacant(slot) = hetero_cuda_devs.entry(*idx) {
                            if let Ok(d) = Device::new_cuda(*idx) {
                                #[cfg(feature = "cuda")]
                                if let Device::Cuda(ref cd) = d {
                                    unsafe { cd.context().disable_event_tracking(); }
                                }
                                slot.insert(d);
                            }
                        }
                    }
                    // Cap user_context_length at the model's GGUF context.
                    // moondream (GGUF ctx=2048) OOM'd because
                    // server config ctx=32768 forced the F-dtype KV cache
                    // to pre-allocate 24 x [1,32,32768,64] F16 = 12.3 GB
                    // on first append, exceeding the 13 GB GPU budget.
                    // The model can't generate beyond its trained context
                    // anyway, so the larger server cap was pure waste.
                    // The plan above chose this context because the cards can hold it; the
                    // cache must be allocated at that size or the plan promised room the
                    // allocation then takes back.
                    let effective_ctx = user_context_length
                        .min(context_length)
                        .min(ctx_for_planning as usize);
                    // Adaptive OOM-recovery ladder. The plan above places
                    // optimistically - with no fictional activation margin,
                    // a model whose weights fit one card lands single-GPU
                    // (the big decode win). If that optimism is wrong and a
                    // real CUDA_ERROR_OUT_OF_MEMORY surfaces (weights load but
                    // the first-prefill activation peak overflows, or the load
                    // itself can't fit), we *react*: retry with a forced even
                    // CUDA split, then CPU. This replaces the old static
                    // per-layer headroom guess with a measurement-driven
                    // fallback that can't mis-size.
                    // FIXED: qwen2-arch models with hidden_size not a
                    // multiple of 256 (only qwen2.5:0.5b, hidden=896) used to
                    // produce deterministic GPU garbage. ROOT CAUSE = the fused
                    // `rms_norm_then_qmatmul` lm-head kernel mishandles non-256
                    // hidden widths (the transformer layers were proven correct
                    // by a per-layer GPU-vs-CPU trace; only the fused head was
                    // wrong - "duction" vs "Paris"). Fixed by gating that fused
                    // head to hidden%256==0 (inference/generic_transformer/); the
                    // separate rms_norm+mmvq fallback is correct. So the CPU-route
                    // guard is no longer needed - qwen2.5:0.5b now runs coherently
                    // on GPU at ~280 tok/s (was CPU-routed ~27).
                    let mmvq_race_arch = false;
                    if mmvq_race_arch && !hetero_cuda_devs.is_empty() {
                        warn!("⚠️  {} (arch={}, hidden_size={}) triggers the non-512-K CUDA mmvq async race that crashes the server; routing entirely to CPU (coherent, no race).",
                            model_id_clone, arch, hidden_size);
                        load_generic_on_cpu(mmap_bytes, &mmap, &arch, num_layers, file_size,
                            projector_blob.as_deref())?
                    } else {
                    match GenericHeteroTransformer::from_gguf_with_kv_quant(
                        content, mmap_bytes, &hetero_cuda_devs, &hetero_plan, kv_quant,
                        Some(effective_ctx),
                    ) {
                        Ok(m) => {
                            info!("✅ {} loaded: {} segments across {} CUDA GPUs + CPU (kv_quant={:?}, kv_ctx={})",
                                arch, hetero_plan.segments.len(), hetero_cuda_devs.len(), kv_quant, effective_ctx);
                            try_wrap_with_vision(m, &arch, projector_blob.as_deref())?
                        }
                        // Tier 2: OOM on an optimistic placement that did NOT
                        // already span every CUDA device -> retry forced split.
                        Err(e) if is_cuda_oom(&e)
                            && cuda_devices.len() > 1
                            && hetero_plan.segments.iter()
                                .filter(|s| matches!(s.kind, crate::inference::place::layer_executor::DeviceKind::Cuda(_)))
                                .count() < cuda_devices.len() =>
                        {
                            warn!("GenericHetero OOM on optimistic placement ({e}); retrying with forced {}-GPU split",
                                cuda_devices.len());
                            // Return the failed attempt's pool-held memory
                            // before retrying (see load_generic_hybrid_or_cpu).
                            #[cfg(feature = "cuda")]
                            release_cuda_pools();
                            let content_fb = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                                .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                            let split_plan = HeteroPlan::split_across_cuda(num_layers, &cuda_devices);
                            match GenericHeteroTransformer::from_gguf_with_kv_quant(
                                content_fb, mmap_bytes, &hetero_cuda_devs, &split_plan, kv_quant,
                                Some(effective_ctx),
                            ) {
                                Ok(m) => {
                                    info!("✅ {} loaded after OOM-recovery: forced {}-GPU split ({} segments)",
                                        arch, cuda_devices.len(), split_plan.segments.len());
                                    try_wrap_with_vision(m, &arch, projector_blob.as_deref())?
                                }
                                Err(e2) => {
                                    warn!("Forced-split retry also failed ({e2}); trying GPU+CPU hybrid before CPU");
                                    load_generic_hybrid_or_cpu(
                                        mmap_bytes,
                                        &mmap, &arch, num_layers, file_size,
                                        &cuda_devices, &hetero_cuda_devs, kv_bytes_per_layer,
                                        kv_quant, effective_ctx, projector_blob.as_deref())?
                                }
                            }
                        }
                        // Tier 2.5: OOM with a full-GPU plan (the forced-split
                        // Tier 2 above didn't apply because the plan already used
                        // all GPUs). The fill-first planner OVER-PACKS the first
                        // GPU (deepseek-r1:32b: GPU0 got 47/64 layers, GPU1 18 ->
                        // GPU0 OOMs at load). Retry with an EVEN split across GPUs
                        // (`split_across_cuda`, ~balanced layer count) so neither
                        // GPU is over-packed - vastly better than the all-CPU Tier
                        // 3 (deepseek-r1:32b: 1.1 tok/s all-CPU -> GPU-resident).
                        Err(e) if is_cuda_oom(&e) && cuda_devices.len() > 1 => {
                            warn!("GenericHetero OOM with fill-first full-GPU plan ({e}); retrying with an EVEN {}-GPU split", cuda_devices.len());
                            // Return the failed attempt's pool-held memory
                            // before retrying (see load_generic_hybrid_or_cpu).
                            #[cfg(feature = "cuda")]
                            release_cuda_pools();
                            let content_fb = gguf_file::Content::read_mapped(&mut Cursor::new(mmap_bytes), mmap.clone())
                                .map_err(|e| anyhow!("GGUF re-parse failed: {e}"))?;
                            let even_plan = HeteroPlan::split_across_cuda(num_layers, &cuda_devices);
                            match GenericHeteroTransformer::from_gguf_with_kv_quant(
                                content_fb, mmap_bytes, &hetero_cuda_devs, &even_plan, kv_quant,
                                Some(effective_ctx),
                            ) {
                                Ok(m) => {
                                    info!("✅ {} recovered via EVEN {}-GPU split ({} segments)",
                                        arch, cuda_devices.len(), even_plan.segments.len());
                                    try_wrap_with_vision(m, &arch, projector_blob.as_deref())?
                                }
                                Err(e2) => {
                                    warn!("Even-split retry also failed ({e2}); trying GPU+CPU hybrid before CPU");
                                    load_generic_hybrid_or_cpu(
                                        mmap_bytes,
                                        &mmap, &arch, num_layers, file_size,
                                        &cuda_devices, &hetero_cuda_devs, kv_bytes_per_layer,
                                        kv_quant, effective_ctx, projector_blob.as_deref())?
                                }
                            }
                        }
                        // Tier 3: any other failure (non-OOM, single-GPU box,
                        // or split already exhausted) -> CPU.
                        Err(e) => {
                            warn!("GenericHetero load failed: {e}, retrying on CPU only");
                            load_generic_on_cpu(mmap_bytes, &mmap, &arch, num_layers, file_size,
                                projector_blob.as_deref())?
                        }
                    }
                    }
                }
                other => {
                    let err_msg = format!(
                        "Unsupported architecture: {}. Supported: llama, mistral, mistral3, \
                         qwen2, qwen3, qwen3moe, gptoss, nemotron_h_moe, lfm2moe, qwen35moe, qwen35, gemma, gemma2, gemma3, gemma4, phi, phi2, \
                         phi3, phi4, chatglm/glm4, stablelm, starcoder, starcoder2, granite, \
                         granite3, dbrx, internlm2, yi, orion, mpt, moondream",
                        other
                    );
                    error!("❌ {}", err_msg);
                    return Err(anyhow!(err_msg));
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
            let mut new_state = LoadedModelState {
                name: model_id_clone.clone(),
                num_layers,
                hidden_size,
                num_heads,
                vocab_size,
                context_length,
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
            };
            // Route a cb_eligible GPU dense model through the batched
            // continuous-batch worker (no-op for ineligible/CPU models).
            cb_maybe_wrap(&mut new_state);
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
            let bw = crate::inference::place::device_probe::probe_cuda_gpus(1.0)
                .iter()
                .map(|g| g.mem_bw_gbs)
                .fold(0.0, f64::max);
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
