//! Loaded-model registry: LoadedModelEntry, keep-alive/expiration and
//! LRU eviction methods of APIServer.

use super::*;

/// Default keep-alive duration in minutes (same as Ollama)
const DEFAULT_KEEP_ALIVE_MINUTES: i64 = 5;

/// Default cap on simultaneously loaded models. When loading a new model
/// would exceed this cap, the LRU model(s) are unloaded first to free
/// VRAM/RAM. Matches Ollama's default of 1.
const DEFAULT_MAX_LOADED_MODELS: usize = 1;

/// Read max loaded models from `OLLAMA_MAX_LOADED_MODELS` (Ollama-compat
/// env var, kept for parity), else the default.
fn get_max_loaded_models() -> usize {
    std::env::var("OLLAMA_MAX_LOADED_MODELS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(DEFAULT_MAX_LOADED_MODELS)
}

/// Declare the engines that can still be holding weights once a request ends.
///
/// One list, expanded into the enum AND into `ALL`, so an engine cannot exist without
/// being on the walk that `unload_model` performs and without being registered with the
/// pressure protocol. Two hand-maintained lists is how the TTS engine came to be freed
/// under pressure but not on request.
///
/// The documentation and the condition are captured separately because they do not
/// travel to the same places: the prose belongs on the variant, while the condition has
/// to reach every site the variant appears at. Forwarded to only one of them, a family
/// compiled out would vanish from the enum and stay in the walk order.
macro_rules! resident_engines {
    ($( $(#[doc = $doc:literal])* $(#[cfg($cfg:meta)])? $variant:ident => $reclaimer:literal, )*) => {
        /// Every engine whose weights an unload has to be able to reach.
        ///
        /// `holding`, `claims`, `release_resident` and `release_named` all match this
        /// EXHAUSTIVELY: a new engine does not compile until someone has said what it is
        /// holding, what a name of its own looks like, and how it lets go.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) enum ResidentEngine {
            $( $(#[doc = $doc])* $(#[cfg($cfg)])? $variant, )*
        }

        impl ResidentEngine {
            /// The walk order of the unload dispatch, and the set the pressure protocol
            /// is registered from.
            pub(crate) const ALL: &'static [ResidentEngine] =
                &[ $( $(#[cfg($cfg)])? ResidentEngine::$variant, )* ];

            /// Name this engine is registered under with the VRAM manager. These strings
            /// are also what the `touch` / `available_for` call sites name, so they are
            /// part of the contract, not labels.
            pub(crate) fn reclaimer_name(self) -> &'static str {
                match self { $( $(#[cfg($cfg)])? ResidentEngine::$variant => $reclaimer, )* }
            }
        }
    };
}

resident_engines! {
    /// Text models, one entry per loaded checkpoint, with the Ollama keep-alive lifecycle.
    Llm => "llm",
    /// Diffusion image models (one family resident at a time).
    #[cfg(feature = "image")]
    Image => "image",
    /// Whisper.
    #[cfg(feature = "audio")]
    Asr => "asr",
    /// Parler-TTS and the other speech checkpoints.
    #[cfg(feature = "audio")]
    Tts => "tts",
    /// Stable Audio Open, resident in a slot of its own rather than an engine.
    #[cfg(feature = "audio")]
    StableAudio => "stable-audio",
}

/// What an unload actually did.
///
/// `keep_alive:0` answers 200 whatever happens, so this IS the report: a caller freeing
/// VRAM has no other channel, and a resident that stayed resident is invisible from
/// outside. Reporting "unloaded" for weights that were never there is the same lie as
/// reporting it for weights that are still there - the caller believes the card is free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnloadOutcome {
    /// Weights were resident under that name, and are now released.
    Freed,
    /// Nothing of that name was resident. Nothing was freed.
    NotResident,
}

impl UnloadOutcome {
    /// The `done_reason` a client keys on. "unload" means the VRAM came back.
    pub(crate) fn done_reason(self) -> &'static str {
        match self {
            UnloadOutcome::Freed => "unload",
            UnloadOutcome::NotResident => "not_loaded",
        }
    }
}

impl ResidentEngine {
    /// Does `model_id` name what this engine is holding?
    ///
    /// Pure, so the whole table can be tested without standing up an engine (the trick
    /// `image_family_matches_catalog_id` already uses). `loaded` is the engine's own
    /// answer to "what have you got", `None` when it holds nothing - and an engine
    /// holding nothing claims nothing, which is what stops an unload from reporting
    /// freed VRAM that was never allocated.
    pub(crate) fn claims(self, loaded: Option<&str>, model_id: &str) -> bool {
        // Ollama ids arrive normalised to `name:latest`, while the media engines record
        // the name they were asked to load. An exact match failed for every untagged
        // model - and the caller was told the unload had succeeded.
        let bare = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
        let Some(loaded) = loaded else { return false };
        match self {
            // Matched by the caller against the loaded-model registry, so the name is
            // already the entry's own id.
            ResidentEngine::Llm => loaded == model_id,
            #[cfg(feature = "image")]
            ResidentEngine::Image => bare(loaded) == bare(model_id) || is_z_image_model(model_id),
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => {
                bare(loaded) == bare(model_id) || model_id.starts_with("whisper")
            }
            // Through the engine's OWN id resolution, so "tts-1", "parler-tts-mini-v1"
            // and the full repo id all name the checkpoint they load.
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => {
                crate::inference::engine::tts_engine::resolve_tts_model_id(Some(&bare(model_id)))
                    == loaded
            }
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => {
                let id = bare(model_id).to_lowercase().replace('_', "-");
                id == loaded
            }
        }
    }
}

/// Monotonic millis since UNIX epoch for LRU bookkeeping.
pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Parse keep_alive value from string (returns minutes)
/// Supports: "10m", "1h", "30s", plain numbers (minutes), "0" (unload), "-1" (forever)
pub fn parse_keep_alive(value: &str) -> Option<i64> {
    let value = value.trim();

    // Handle special values
    if value == "0" {
        return Some(0);
    }
    if value == "-1" {
        return Some(-1);
    }

    // Parse duration strings with suffix: "10m", "1h", "30s".
    // Ollama semantics: any negative value = "keep forever" sentinel
    // (-1). 0 unloads immediately. Positive sub-minute durations
    // round up to 1 minute - "30s" used to round DOWN to 0 and
    // unload the model the instant it was loaded, which is the
    // opposite of what the caller asked for.
    let normalize = |minutes: i64| -> i64 {
        if minutes < 0 {
            -1
        } else {
            minutes
        }
    };
    if let Some(nums) = value.strip_suffix('s') {
        return nums.parse::<i64>().ok().map(|s| {
            if s < 0 {
                -1
            } else if s == 0 {
                0
            }
            // saturating_add prevents wrap when a client sends an i64::MAX-ish
            // string ("9223372036854775807s") that would overflow `s + 59`.
            else {
                (s.saturating_add(59) / 60).max(1)
            }
        });
    }
    if let Some(nums) = value.strip_suffix('m') {
        return nums.parse::<i64>().ok().map(normalize);
    }
    if let Some(nums) = value.strip_suffix('h') {
        // saturating_mul prevents wrap on h * 60 for huge i64 inputs.
        return nums
            .parse::<i64>()
            .ok()
            .map(|h| if h < 0 { -1 } else { h.saturating_mul(60) });
    }

    // Plain number = minutes (Ollama default)
    value.parse().ok().map(normalize)
}

/// Get default keep_alive from environment or use default
pub(crate) fn get_default_keep_alive() -> i64 {
    std::env::var("OLLAMA_KEEP_ALIVE")
        .ok()
        .and_then(|v| parse_keep_alive(&v))
        .unwrap_or(DEFAULT_KEEP_ALIVE_MINUTES)
}

/// Entry for a loaded model with expiration tracking
pub(crate) struct LoadedModelEntry {
    /// Model ID
    pub(crate) model_id: String,
    /// The inference engine
    pub(crate) engine: Arc<LlmEngine>,
    /// Keep-alive duration in minutes (None = use default, Some(-1) = forever)
    pub(crate) keep_alive_minutes: Option<i64>,
    /// Cancellable expiration task handle
    pub(crate) expire_handle: Option<JoinHandle<()>>,
    /// Chat template loaded from Ollama manifest (Go template syntax)
    pub(crate) chat_template: Option<String>,
    /// Neural-draft speculative decoding. When set, the streaming
    /// generate path runs a draft+verify loop using this engine on a
    /// (typically) different GPU. Tokenizer compat is verified at
    /// attach time; the draft must produce identical token IDs as the
    /// target.
    pub(crate) draft: Option<DraftAttachment>,
    /// UNIX-millis timestamp of last use (request landed on this engine).
    /// Used by auto-eviction to pick the LRU model when we need to free
    /// a slot.
    pub(crate) last_used: Arc<AtomicI64>,
}

/// Sidecar engine that produces speculative draft tokens for the
/// target. Tokenizer fingerprint must match the target's.
pub(crate) struct DraftAttachment {
    /// Draft model ID (for /api/inflight reporting).
    pub model_id: String,
    /// Draft engine on its own GPU. Has its own model_state lock.
    pub engine: Arc<LlmEngine>,
    /// How many tokens the draft proposes per cycle. Tunable per
    /// pair; default 4. The target verifies K + 1 positions.
    pub k: usize,
}

impl LoadedModelEntry {
    /// Create a new entry
    pub(crate) fn new(
        model_id: String,
        engine: Arc<LlmEngine>,
        keep_alive_minutes: Option<i64>,
    ) -> Self {
        Self {
            model_id,
            engine,
            keep_alive_minutes,
            expire_handle: None,
            chat_template: None,
            draft: None,
            last_used: Arc::new(AtomicI64::new(now_millis())),
        }
    }

    /// Record activity on this entry (for LRU eviction ordering).
    pub(crate) fn touch(&self) {
        self.last_used.store(now_millis(), Ordering::Relaxed);
    }
}

impl APIServer {
    /// Get effective keep_alive duration in minutes
    pub(crate) fn get_effective_keep_alive(&self, keep_alive: Option<&str>) -> i64 {
        keep_alive
            .and_then(parse_keep_alive)
            .unwrap_or(self.default_keep_alive)
    }

    /// Schedule model expiration based on keep_alive
    pub(crate) fn schedule_expiration(
        &self,
        model_id: String,
        keep_alive_minutes: i64,
    ) -> JoinHandle<()> {
        let engines = self.engines.clone();

        tokio::spawn(async move {
            // Calculate duration - if keep_alive is -1 (forever), don't schedule
            if keep_alive_minutes == -1 {
                info!("Model {} will stay loaded indefinitely", model_id);
                return;
            }

            // saturating_mul prevents wrap when a client sends a
            // pathological keep_alive like "9223372036854775807" (parses
            // fine as i64::MAX, then i64::MAX as u64 * 60 wraps past
            // u64::MAX into a small value - flipping "stay loaded basically
            // forever" into "unload in a few seconds"). The earlier
            // saturating_mul fix in parse_keep_alive (76312fd) handled
            // the suffixed-string forms (`<huge>h`, `<huge>s`); this
            // catches the plain-number form that arrives here as i64.
            let secs = (keep_alive_minutes.max(0) as u64).saturating_mul(60);
            let duration = Duration::from_secs(secs);
            info!(
                "Model {} will expire in {} minutes",
                model_id, keep_alive_minutes
            );

            tokio::time::sleep(duration).await;

            // Time's up - unload the model
            let mut engines_guard = engines.write().await;
            if let Some(pos) = engines_guard.iter().position(|e| e.model_id == model_id) {
                let entry = engines_guard.remove(pos);
                if let Err(e) = entry.engine.unload().await {
                    info!("Error unloading model {}: {}", entry.model_id, e);
                } else {
                    info!("Model {} unloaded after keep_alive expired", entry.model_id);
                }
            }
        })
    }

    /// Reset model expiration timer (called on each use, like Ollama).
    /// Cancels the existing timer and schedules a fresh one. Also
    /// updates the entry's `last_used` for LRU eviction ordering.
    pub async fn reset_expiration(&self, model_id: &str) {
        let mut engines = self.engines.write().await;
        if let Some(entry) = engines.iter_mut().find(|e| e.model_id == model_id) {
            entry.touch();
            // Cancel existing timer
            if let Some(handle) = entry.expire_handle.take() {
                handle.abort();
            }
            // Schedule new timer with the entry's keep_alive (or server default)
            let keep_alive = entry.keep_alive_minutes.unwrap_or(self.default_keep_alive);
            if keep_alive > 0 {
                let new_handle = self.schedule_expiration(model_id.to_string(), keep_alive);
                entry.expire_handle = Some(new_handle);
            }
            // keep_alive == -1 (forever) or 0 (already handled): no timer
        }
    }

    /// Get or load an engine for a model
    pub(crate) async fn get_engine(&self, model_id: &str) -> Result<Arc<LlmEngine>, ApiError> {
        // Check if engine already loaded (by original ID)
        {
            let engines = self.engines.read().await;
            if let Some(entry) = engines.iter().find(|e| e.model_id == model_id) {
                return Ok(entry.engine.clone());
            }
        }

        // Need to load the engine
        Err(ApiError::NotFound(format!(
            "Model '{model_id}' not loaded. Pull it first with POST /api/pull"
        )))
    }

    /// If `model_id` is loaded but its KV-cache mode differs from `want`,
    /// unload and reload with the new mode. No-op if the engine already
    /// matches, or if it's not currently loaded (caller gets the usual
    /// "not loaded" error from `get_engine` afterward).
    /// A request's `num_ctx` above the context the engine was configured with reloads it
    /// at that context, as Ollama does; the loader still bounds it by what the model
    /// declares and what the cards hold.
    pub(crate) async fn ensure_engine_context(
        &self,
        model_id: &str,
        want: usize,
    ) -> Result<(), ApiError> {
        let (current, had_keep_alive) = {
            let engines = self.engines.read().await;
            match engines.iter().find(|e| e.model_id == model_id) {
                Some(entry) => (entry.engine.config().context_length, entry.keep_alive_minutes),
                None => return Ok(()),
            }
        };
        if want <= current {
            return Ok(());
        }
        info!("num_ctx {want} above the configured context {current} for {model_id} - reloading model");
        self.unload_model(model_id).await.ok();

        let mut config = self.config_for_model(model_id);
        config.context_length = want;
        let engine = Arc::new(LlmEngine::with_config(config));
        engine
            .load_model()
            .await
            .map_err(|e| ApiError::Internal(format!("Reload with num_ctx={want} failed: {e}")))?;

        let keep_alive = had_keep_alive.unwrap_or(self.default_keep_alive);
        let expire_handle = if keep_alive > 0 {
            Some(self.schedule_expiration(model_id.to_string(), keep_alive))
        } else {
            None
        };
        let mut engines = self.engines.write().await;
        let mut entry = LoadedModelEntry::new(model_id.to_string(), engine, Some(keep_alive));
        entry.expire_handle = expire_handle;
        engines.push(entry);
        info!("Model {model_id} reloaded with num_ctx={want}");
        Ok(())
    }

    pub(crate) async fn ensure_engine_kv_quant(
        &self,
        model_id: &str,
        want: crate::inference::engine::llm_engine::KvQuant,
    ) -> Result<(), ApiError> {
        let (current_matches, had_keep_alive) = {
            let engines = self.engines.read().await;
            match engines.iter().find(|e| e.model_id == model_id) {
                Some(entry) => (
                    entry.engine.config().kv_quant == want,
                    entry.keep_alive_minutes,
                ),
                None => return Ok(()),
            }
        };
        if current_matches {
            return Ok(());
        }

        info!(
            "kv_quant override requested ({:?}) for {} - reloading model",
            want, model_id
        );
        self.unload_model(model_id).await.ok();

        let mut config = self.config_for_model(model_id);
        config.kv_quant = want;
        let engine = Arc::new(LlmEngine::with_config(config));
        engine.load_model().await.map_err(|e| {
            ApiError::Internal(format!("Reload with kv_quant={:?} failed: {}", want, e))
        })?;

        let keep_alive = had_keep_alive.unwrap_or(self.default_keep_alive);
        let expire_handle = if keep_alive > 0 {
            Some(self.schedule_expiration(model_id.to_string(), keep_alive))
        } else {
            None
        };
        let mut engines = self.engines.write().await;
        let mut entry = LoadedModelEntry::new(model_id.to_string(), engine, Some(keep_alive));
        entry.expire_handle = expire_handle;
        engines.push(entry);
        info!("Model {} reloaded with kv_quant={:?}", model_id, want);
        Ok(())
    }

    /// What `kind` is holding right now, as the name it would answer with, or `None`
    /// when it holds nothing.
    ///
    /// `model_id` is only read by the LLM arm, which keeps one entry per loaded
    /// checkpoint rather than a single resident.
    async fn holding(&self, kind: ResidentEngine, model_id: &str) -> Option<String> {
        match kind {
            ResidentEngine::Llm => self
                .engines
                .read()
                .await
                .iter()
                .find(|e| e.model_id == model_id)
                .map(|e| e.model_id.clone()),
            #[cfg(feature = "image")]
            ResidentEngine::Image => self.image_engine.model_name().await,
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => self.audio_engine.loaded_name().await,
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => self.tts_engine.loaded_name().await,
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => crate::inference::model::stable_audio::is_resident()
                .then(|| kind.reclaimer_name().to_string()),
        }
    }

    /// Let go of `kind`'s idle resident, whatever it is holding, and report roughly how
    /// many bytes came back.
    ///
    /// THE teardown for every engine but the LLM one: the pressure protocol calls this
    /// through the hook it registers, and an unload naming a model calls the same arm,
    /// so a `keep_alive:0` and a pressure eviction cannot come to mean different things.
    /// (The LLM arm differs by necessity: pressure takes the least-recently-used entry
    /// while an unload names one, so `release_named` handles that engine itself.)
    pub(crate) async fn release_resident(&self, kind: ResidentEngine) -> u64 {
        let freed = self.release_resident_inner(kind).await;
        // GIVE THE MEMORY BACK TO THE DRIVER, not just to the pool.
        //
        // Dropping an engine's state releases its tensors to the allocator, which KEEPS the
        // blocks for its own reuse - so the card still reads as full to anything that probes
        // it. Measured: an unload reported four gigabytes freed and `nvidia-smi` did not move
        // by a byte, and the next model, too large for what was left, was planned onto the
        // slower card. Nobody saw an error; the render was simply 1.7x slower.
        //
        // The bookkeeping was right and the memory was still gone, which is the failure mode
        // a unit test cannot see: it proves the path was called, not that the card came back.
        #[cfg(feature = "cuda")]
        if freed > 0 {
            crate::inference::engine::llm_engine::release_cuda_pools();
        }
        freed
    }

    async fn release_resident_inner(&self, kind: ResidentEngine) -> u64 {
        match kind {
            ResidentEngine::Llm => {
                let victim = {
                    let mut guard = self.engines.write().await;
                    let pos = guard
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_used.load(Ordering::Relaxed))
                        .map(|(i, _)| i);
                    pos.map(|i| guard.remove(i))
                };
                match victim {
                    Some(entry) => {
                        if let Some(h) = entry.expire_handle {
                            h.abort();
                        }
                        let name = entry.model_id.clone();
                        if entry.engine.unload().await.is_ok() {
                            info!("vram_manager: evicted idle LLM '{name}' under pressure");
                            1
                        } else {
                            0
                        }
                    }
                    None => 0,
                }
            }
            #[cfg(feature = "image")]
            ResidentEngine::Image => {
                if self.image_engine.is_loaded().await {
                    self.image_engine.unload().await;
                    1
                } else {
                    0
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => {
                if self.audio_engine.is_loaded().await {
                    self.audio_engine.unload().await;
                    1
                } else {
                    0
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => {
                if self.tts_engine.is_loaded().await {
                    self.tts_engine.unload().await;
                    1
                } else {
                    0
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => {
                crate::inference::model::stable_audio::unload_resident() as u64
            }
        }
    }

    /// Bytes `kind` is holding that the pressure protocol could take back, without
    /// taking them.
    ///
    /// An engine that does not measure its residency reports nothing rather than a
    /// guess: understating capacity only makes the protocol more cautious, while
    /// overstating it would promise room that is not there.
    pub(crate) async fn resident_bytes(&self, kind: ResidentEngine) -> u64 {
        match kind {
            // This engine MEASURES its resident across the load, so it can say what it
            // holds - and that figure is what makes an idle image model count as
            // capacity rather than as memory that is gone.
            #[cfg(feature = "image")]
            ResidentEngine::Image => self.image_engine.resident_bytes().await,
            ResidentEngine::Llm => 0,
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => 0,
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => 0,
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => 0,
        }
    }

    /// Release `kind`'s resident IF `model_id` names it. `Ok(false)` = not this engine's.
    async fn release_named(&self, kind: ResidentEngine, model_id: &str) -> Result<bool, ApiError> {
        let loaded = self.holding(kind, model_id).await;
        if !kind.claims(loaded.as_deref(), model_id) {
            return Ok(false);
        }
        match kind {
            // Named, not least-recently-used: an unload frees the model the caller asked
            // for, and only that one. This is also the only arm that can fail.
            ResidentEngine::Llm => {
                let entry = {
                    let mut engines = self.engines.write().await;
                    match engines.iter().position(|e| e.model_id == model_id) {
                        Some(pos) => engines.remove(pos),
                        // Lost a race with another unload; nothing left to free.
                        None => return Ok(false),
                    }
                };
                if let Some(handle) = entry.expire_handle {
                    handle.abort();
                }
                entry
                    .engine
                    .unload()
                    .await
                    .map_err(|e| ApiError::Internal(format!("Failed to unload model: {}", e)))?;
            }
            #[cfg(feature = "image")]
            ResidentEngine::Image => {
                if self.release_resident(kind).await == 0 {
                    // It said it was holding this model a moment ago and now holds
                    // nothing: a concurrent reclaim got there first. Nothing was freed
                    // HERE, and this call has no business claiming it was.
                    return Ok(false);
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => {
                if self.release_resident(kind).await == 0 {
                    // It said it was holding this model a moment ago and now holds
                    // nothing: a concurrent reclaim got there first. Nothing was freed
                    // HERE, and this call has no business claiming it was.
                    return Ok(false);
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => {
                if self.release_resident(kind).await == 0 {
                    // It said it was holding this model a moment ago and now holds
                    // nothing: a concurrent reclaim got there first. Nothing was freed
                    // HERE, and this call has no business claiming it was.
                    return Ok(false);
                }
            }
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => {
                if self.release_resident(kind).await == 0 {
                    // It said it was holding this model a moment ago and now holds
                    // nothing: a concurrent reclaim got there first. Nothing was freed
                    // HERE, and this call has no business claiming it was.
                    return Ok(false);
                }
            }
        }
        info!(
            "Unloaded {} from the {} engine",
            model_id,
            kind.reclaimer_name()
        );
        Ok(true)
    }

    /// Unload, and report honestly what happened.
    ///
    /// THE ONE PLACE that decides what an unload result means, because getting it wrong
    /// is invisible from outside. A caller freeing VRAM that is told it worked, while
    /// the weights stay resident, has no way to learn otherwise, and the card stays
    /// full. It cost a 1.7x image render: 4.5 GB held by a TTS model this function did
    /// not know about pushed a DiT onto the second card, and nothing said so.
    ///
    /// Every engine on the walk is asked, in order; the first one that recognises the
    /// name frees it. "Nothing recognised the name" is NOT an error - there is nothing
    /// to free - but it is not an unload either, and `UnloadOutcome` keeps the two apart.
    ///
    /// Do NOT write `unload_model(..).await.ok()`. That is how five endpoints came to
    /// answer "unloaded" unconditionally - one of them while a name mismatch meant
    /// nothing was ever freed.
    pub(crate) async fn unload_model(&self, model_id: &str) -> Result<UnloadOutcome, ApiError> {
        for kind in ResidentEngine::ALL {
            if self.release_named(*kind, model_id).await? {
                return Ok(UnloadOutcome::Freed);
            }
        }
        info!("unload: {model_id} was not loaded; nothing to free");
        Ok(UnloadOutcome::NotResident)
    }

    /// Free slots for a new model by unloading the LRU entries that
    /// aren't `incoming_id` until the loaded count is below the cap.
    /// Generic: applies to every architecture and every accelerator;
    /// prior engines hold weights, KV blocks, scratch buffers and pool
    /// allocations that prevent the next model's HeteroPlan from
    /// claiming GPU memory.
    pub(crate) async fn evict_to_make_room(&self, incoming_id: &str) {
        let max_loaded = get_max_loaded_models();
        // Snapshot candidates (skip the incoming model itself) sorted
        // by oldest-use first.
        let victims: Vec<String> = {
            let engines = self.engines.read().await;
            // Already counts incoming as +1 once we push it; we want
            // existing != incoming_id count <= max_loaded - 1.
            let mut others: Vec<(i64, String)> = engines
                .iter()
                .filter(|e| e.model_id != incoming_id)
                .map(|e| (e.last_used.load(Ordering::Relaxed), e.model_id.clone()))
                .collect();
            let to_remove = others.len().saturating_sub(max_loaded.saturating_sub(1));
            if to_remove == 0 {
                Vec::new()
            } else {
                others.sort_by_key(|(ts, _)| *ts);
                others
                    .into_iter()
                    .take(to_remove)
                    .map(|(_, id)| id)
                    .collect()
            }
        };
        for victim in victims {
            info!(
                "⚖️  Auto-evicting LRU model {} to make room for {} (max_loaded={})",
                victim, incoming_id, max_loaded
            );
            // unload_model logs success/failure; ignore the result so
            // a stale entry can't block the new load.
            let _ = self.unload_model(&victim).await;
        }
    }

    /// Refresh the LRU timestamp on every resident model in `ids`
    /// (Stage-B conversation-aware residency - keep a conversation's whole
    /// working set warm so a one-off request from another caller can't
    /// evict a model the active conversation still needs).
    pub(crate) async fn touch_models(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        let engines = self.engines.read().await;
        for e in engines.iter() {
            if ids.iter().any(|id| id == &e.model_id) {
                e.touch();
            }
        }
    }

    /// Atomically swap a loaded model with a new model
    pub(crate) async fn swap_model(
        &self,
        current_model_id: &str,
        new_model_id: &str,
        keep_alive: Option<&str>,
    ) -> Result<(), ApiError> {
        info!(
            "Starting atomic model swap: {} -> {}",
            current_model_id, new_model_id
        );

        // 1. Check if current model is loaded
        {
            let engines = self.engines.read().await;
            if !engines.iter().any(|e| e.model_id == current_model_id) {
                return Err(ApiError::NotFound(format!(
                    "Current model {} not loaded",
                    current_model_id
                )));
            }
        }

        // 2. Load the new model
        let model_path = self
            .model_manager
            .resolve_path(new_model_id, "ollama")
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to resolve new model path: {}", e)))?;

        if !model_path.exists() {
            info!(
                "New model not found locally, pulling first: {}",
                new_model_id
            );
            self.model_manager
                .pull_model_with_source(new_model_id, "ollama", None)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to pull new model: {}", e)))?;
        }

        // Use server's default inference config with resolved models dir
        let config = self.config_for_model(new_model_id);

        let new_engine = Arc::new(LlmEngine::with_config(config));
        new_engine
            .load_model()
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to load new model: {}", e)))?;

        info!("New model {} loaded successfully", new_model_id);

        // 3. Atomically swap in the engines list
        let keep_alive_minutes = self.get_effective_keep_alive(keep_alive);
        let old_entry = {
            let mut engines = self.engines.write().await;

            // Find and remove old entry
            let pos = engines
                .iter()
                .position(|e| e.model_id == current_model_id)
                .ok_or_else(|| ApiError::Internal("Model disappeared during swap".to_string()))?;

            let old_entry = engines.remove(pos);

            // Add new entry
            let expire_handle =
                self.schedule_expiration(new_model_id.to_string(), keep_alive_minutes);
            let mut new_entry = LoadedModelEntry::new(
                new_model_id.to_string(),
                new_engine,
                Some(keep_alive_minutes),
            );
            new_entry.expire_handle = Some(expire_handle);
            engines.push(new_entry);

            old_entry
        };

        info!("Model swap completed in engines list");

        // 4. Clean up old model (cancel expiration timer and unload)
        if let Some(handle) = old_entry.expire_handle {
            handle.abort();
        }

        if let Err(e) = old_entry.engine.unload().await {
            warn!("Failed to unload old model {}: {}", old_entry.model_id, e);
        } else {
            info!("Old model {} unloaded successfully", old_entry.model_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row per engine: the name it would answer with while holding something, an id
    /// a client can reach it by, and whether that id names it.
    ///
    /// An exhaustive match, so an engine added to `ResidentEngine` cannot compile until
    /// someone has said how a request would ever name it - which is the question that
    /// went unasked for the TTS engine.
    fn name_rule(kind: ResidentEngine) -> (&'static str, &'static str, bool) {
        match kind {
            ResidentEngine::Llm => ("llama3.2:1b", "llama3.2:1b", true),
            #[cfg(feature = "image")]
            ResidentEngine::Image => ("qwen-image", "qwen-image:latest", true),
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => ("openai/whisper-small", "openai/whisper-small", true),
            // The id /api/ps publishes for a warm voice, tagged the way an Ollama
            // client normalises it.
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => (
                "parler-tts/parler-tts-mini-v1",
                "parler-tts/parler-tts-mini-v1:latest",
                true,
            ),
            #[cfg(feature = "audio")]
            ResidentEngine::StableAudio => ("stable-audio", "stable_audio", true),
        }
    }

    /// THE defect, at the level where it lived: `unload_model` knew the LLM, image and
    /// ASR engines and not the TTS one, so `keep_alive:0` against a resident parler-tts
    /// answered `done_reason:"unload"` and freed nothing - 4.5 GB that pushed the next
    /// image render's DiT onto the slower card, for a 1.7x loss nothing reported.
    #[test]
    fn a_resident_tts_checkpoint_is_named_by_the_ids_a_client_has_for_it() {
        assert!(ResidentEngine::ALL.contains(&ResidentEngine::Tts));
        let loaded = Some("parler-tts/parler-tts-mini-v1");
        for id in [
            "parler-tts/parler-tts-mini-v1",
            "parler-tts/parler-tts-mini-v1:latest",
            "parler-tts-mini-v1",
            // The OpenAI alias, resolved by the engine's own id rules rather than a
            // second table that could disagree with them.
            "tts-1",
        ] {
            assert!(
                ResidentEngine::Tts.claims(loaded, id),
                "the TTS engine does not answer to {id}"
            );
        }
        // And only that engine: an id nothing else recognises must not be swallowed by
        // an earlier arm of the walk, or the TTS weights stay resident again.
        for kind in ResidentEngine::ALL {
            if *kind == ResidentEngine::Tts {
                continue;
            }
            let (its_own, _, _) = name_rule(*kind);
            assert!(
                !kind.claims(Some(its_own), "parler-tts/parler-tts-mini-v1"),
                "{kind:?} claimed a TTS checkpoint"
            );
        }
    }

    /// Every engine that can hold weights answers to a name of its own, and none of them
    /// answers while holding nothing. The second half is the honesty rule: a report of
    /// "unloaded" for a model that was never resident is the same lie as one for a model
    /// that is still resident - the caller believes the card is free.
    #[test]
    fn an_engine_holding_nothing_claims_nothing() {
        for kind in ResidentEngine::ALL {
            let (loaded, id, claimed) = name_rule(*kind);
            assert_eq!(
                kind.claims(Some(loaded), id),
                claimed,
                "{kind:?} disagrees with its own name rule"
            );
            assert!(
                !kind.claims(None, id),
                "{kind:?} claimed {id} while holding nothing"
            );
        }
    }

    /// The inventory is the ONE list: the unload walk reads it, and so does the pressure
    /// protocol's registration. The names are the contract - `touch("image")`,
    /// `available_for("stable-audio")` and the rest of the VRAM manager's call sites
    /// address engines by exactly these strings.
    #[test]
    fn every_engine_registers_under_a_name_of_its_own() {
        let mut seen = std::collections::HashSet::new();
        for kind in ResidentEngine::ALL {
            assert!(
                seen.insert(kind.reclaimer_name()),
                "{kind:?} shares a reclaimer name with another engine"
            );
        }
        // The engines on the walk. A reclaimer the VRAM manager knows by some other route
        // is deliberately absent: this walk is what a named unload iterates, and it cannot
        // name an engine it does not know exists.
        for name in ["llm", "image", "tts", "stable-audio"] {
            assert!(
                seen.contains(name),
                "the pressure protocol addresses '{name}', which is not on the walk"
            );
        }
    }

    /// A server holding nothing frees nothing, and says so - for every engine on the
    /// walk, named the way a client would name it. Answering "unload" here is what made
    /// a stuck resident indistinguishable from a freed one.
    #[tokio::test]
    async fn an_unload_of_a_model_nothing_is_holding_reports_that_it_freed_nothing() {
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            "/nonexistent-hf-models".to_string(),
        );
        for kind in ResidentEngine::ALL {
            let (_, id, _) = name_rule(*kind);
            assert_eq!(
                state.unload_model(id).await.expect("unload must not fail"),
                UnloadOutcome::NotResident,
                "{kind:?}: {id} was reported as freed by a server holding nothing"
            );
        }
        assert_eq!(UnloadOutcome::NotResident.done_reason(), "not_loaded");
        assert_eq!(UnloadOutcome::Freed.done_reason(), "unload");
    }

    #[test]
    fn parse_keep_alive_sub_minute_rounds_up() {
        // Anything under a minute floors to one: rounding it down means unload immediately.
        assert_eq!(parse_keep_alive("30s"), Some(1));
        assert_eq!(parse_keep_alive("59s"), Some(1));
        assert_eq!(parse_keep_alive("60s"), Some(1));
        assert_eq!(parse_keep_alive("120s"), Some(2));
        // Explicit zero still unloads.
        assert_eq!(parse_keep_alive("0s"), Some(0));
        assert_eq!(parse_keep_alive("0"), Some(0));
        // Negative durations all collapse to the keep-forever sentinel.
        assert_eq!(parse_keep_alive("-1"), Some(-1));
        assert_eq!(parse_keep_alive("-5m"), Some(-1));
        assert_eq!(parse_keep_alive("-2h"), Some(-1));
        assert_eq!(parse_keep_alive("-10s"), Some(-1));
        // Normal cases stay as-is.
        assert_eq!(parse_keep_alive("10m"), Some(10));
        assert_eq!(parse_keep_alive("1h"), Some(60));
        assert_eq!(parse_keep_alive("5"), Some(5));
        // Pathological i64-near-MAX inputs no longer wrap into the
        // small-positive / negative range. Hours: i64::MAX * 60
        // saturates instead of wrapping. Seconds: i64::MAX + 59
        // saturates instead of overflowing into a small/negative value.
        let huge = i64::MAX.to_string();
        assert_eq!(parse_keep_alive(&format!("{huge}h")), Some(i64::MAX));
        assert_eq!(parse_keep_alive(&format!("{huge}s")), Some(i64::MAX / 60));
    }
}
