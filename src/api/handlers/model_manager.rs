//! Loaded-model registry: LoadedModelEntry, keep-alive/expiration and
//! LRU eviction methods of APIServer.

use super::*;

/// Default keep-alive duration in minutes (same as Ollama)
pub(crate) const DEFAULT_KEEP_ALIVE_MINUTES: i64 = 5;

/// A file's length and modification time: a header read under one stamp answers for the
/// file while the stamp holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
}

impl FileStamp {
    pub(crate) fn of(path: &std::path::Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// The context length each checkpoint declares, keyed by its path and remembered while the
/// file's stamp holds.
#[derive(Default)]
pub(crate) struct DeclaredContexts {
    read: std::sync::Mutex<std::collections::HashMap<PathBuf, (FileStamp, Option<usize>)>>,
}

impl DeclaredContexts {
    /// The value remembered for `path` under `stamp`, else the one `read` returns, which is
    /// then remembered.
    pub(crate) fn get_or_read(
        &self,
        path: &std::path::Path,
        stamp: FileStamp,
        read: impl FnOnce() -> Option<usize>,
    ) -> Option<usize> {
        if let Some((at, value)) = self
            .read
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(path)
        {
            if *at == stamp {
                return *value;
            }
        }
        let value = read();
        self.read
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.to_path_buf(), (stamp, value));
        value
    }
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
    /// Pure, so the whole table can be tested without standing up an engine. `loaded` is
    /// the engine's own
    /// answer to "what have you got", `None` when it holds nothing - and an engine
    /// holding nothing claims nothing, which is what stops an unload from reporting
    /// freed VRAM that was never allocated.
    pub(crate) fn claims(self, hf_models_dir: &str, loaded: Option<&str>, model_id: &str) -> bool {
        // Ollama ids arrive normalised to `name:latest`, while the media engines record
        // the name they were asked to load. An exact match failed for every untagged
        // model - and the caller was told the unload had succeeded.
        let bare = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
        #[cfg(not(feature = "audio"))]
        let _ = hf_models_dir;
        let Some(loaded) = loaded else { return false };
        match self {
            // Matched by the caller against the loaded-model registry, so the name is
            // already the entry's own id.
            ResidentEngine::Llm => loaded == model_id,
            #[cfg(feature = "image")]
            ResidentEngine::Image => bare(loaded).eq_ignore_ascii_case(&bare(model_id)),
            // Through the catalogue id the loads record, so an alias, a bare name and the
            // full repository id all name the checkpoint they load.
            #[cfg(feature = "audio")]
            ResidentEngine::Asr => {
                crate::api::handlers::canonical_id(hf_models_dir, model_id) == loaded
            }
            #[cfg(feature = "audio")]
            ResidentEngine::Tts => {
                crate::api::handlers::speech_canonical(hf_models_dir, model_id) == loaded
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

/// What a model load has read so far, as its reporter gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadProgress {
    pub(crate) phase: String,
    pub(crate) done: usize,
    pub(crate) total: usize,
}

/// The loads in flight, by model id.
///
/// A model has a record from `begin` until its guard drops, whichever way the load ends.
/// The record holds the last count the load reported.
///
/// Each record carries the generation `begin` gave it.
/// A reporter and a guard act only on the record of their own generation.
#[derive(Default)]
pub(crate) struct LoadsInFlight {
    records: std::sync::Mutex<std::collections::HashMap<String, (u64, LoadProgress)>>,
    next_generation: std::sync::atomic::AtomicU64,
}

/// Holds the record of one load; dropping it clears the record.
pub(crate) struct LoadGuard {
    loads: Arc<LoadsInFlight>,
    model_id: String,
    generation: u64,
}

impl LoadsInFlight {
    /// Opens the record of `model_id` at the start of its load, with nothing counted.
    pub(crate) fn begin(self: &Arc<Self>, model_id: &str) -> LoadGuard {
        let generation = self
            .next_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.lock().insert(
            model_id.to_string(),
            (
                generation,
                LoadProgress {
                    phase: crate::inference::serve::progress::phase::LOAD_MODEL.to_string(),
                    done: 0,
                    total: 0,
                },
            ),
        );
        LoadGuard {
            loads: self.clone(),
            model_id: model_id.to_string(),
            generation,
        }
    }

    /// The ids of the models loading now.
    pub(crate) fn models(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// The last count `model_id` reported, when it is loading.
    pub(crate) fn progress(&self, model_id: &str) -> Option<LoadProgress> {
        self.lock().get(model_id).map(|(_, p)| p.clone())
    }

    fn lock(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, (u64, LoadProgress)>> {
        self.records.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LoadGuard {
    /// A reporter that writes into the record of this guard's generation. Safe to call from any thread.
    pub(crate) fn reporter(&self) -> crate::inference::serve::progress::SharedProgressFn {
        let loads = self.loads.clone();
        let model_id = self.model_id.clone();
        let generation = self.generation;
        Arc::new(move |phase: &str, done: usize, total: usize| {
            if let Some((_, record)) = loads
                .lock()
                .get_mut(&model_id)
                .filter(|(g, _)| *g == generation)
            {
                record.phase.clear();
                record.phase.push_str(phase);
                record.done = done;
                record.total = total;
            }
        })
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        let mut records = self.loads.lock();
        if records
            .get(&self.model_id)
            .is_some_and(|(g, _)| *g == self.generation)
        {
            records.remove(&self.model_id);
        }
    }
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
    /// The chat format read from the model's files, once read.
    pub(crate) chat_format: Option<super::chat_prompt::ChatFormat>,
    /// UNIX-millis timestamp of last use (request landed on this engine).
    /// Used by auto-eviction to pick the LRU model when we need to free
    /// a slot.
    pub(crate) last_used: Arc<AtomicI64>,
    /// Further copies of the model, each whole on a card of its own, served beside `engine`.
    pub(crate) replicas: Vec<Arc<LlmEngine>>,
    /// Admits the model's requests, as many at once as the model has copies.
    /// Each model has its own, so 2 models run their requests at once.
    pub(crate) gate: Arc<crate::api::gate::RequestGate>,
    /// When a request was last sent to each copy, `engine` first.
    sent: std::sync::Mutex<Vec<std::time::Instant>>,
    /// The copy the last request went to.
    last: std::sync::atomic::AtomicUsize,
    /// The other models and whether each was busy, when a load in full of this one was last
    /// priced or when it loaded; `None` before either.
    pub(crate) priced: std::sync::Mutex<Option<Vec<(String, bool)>>>,
}

/// How long a copy counts as taken after a request was sent to it, before the request has
/// been admitted and shows in the copy's own count.
const PICK_WINDOW: std::time::Duration = std::time::Duration::from_millis(25);

/// The copy to send a request to, given each copy's requests in flight and whether one was
/// sent to it a moment ago: the least taken, the one that served the last request on a tie,
/// so a conversation goes on where its cache is while that copy is free.
pub(crate) fn pick_copy(in_flight: &[usize], just_sent: &[bool], last: usize) -> usize {
    let taken = |i: usize| in_flight[i] + usize::from(just_sent[i] && in_flight[i] == 0);
    (0..in_flight.len())
        .min_by_key(|&i| (taken(i), usize::from(i != last), i))
        .unwrap_or(0)
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
            chat_format: None,
            last_used: Arc::new(AtomicI64::new(now_millis())),
            replicas: Vec::new(),
            gate: crate::api::gate::RequestGate::new(1),
            sent: std::sync::Mutex::new(Vec::new()),
            last: std::sync::atomic::AtomicUsize::new(0),
            priced: std::sync::Mutex::new(None),
        }
    }

    /// Serve the model on `replicas` as well as on its own engine.
    pub(crate) fn with_replicas(mut self, replicas: Vec<Arc<LlmEngine>>) -> Self {
        if !replicas.is_empty() {
            self.gate = crate::api::gate::RequestGate::new(1 + replicas.len());
            let now = std::time::Instant::now() - PICK_WINDOW;
            *self.sent.lock().unwrap_or_else(|e| e.into_inner()) = vec![now; 1 + replicas.len()];
        }
        self.replicas = replicas;
        self
    }

    /// The engine the next request goes to.
    pub(crate) fn serving(&self) -> Arc<LlmEngine> {
        if self.replicas.is_empty() {
            return self.engine.clone();
        }
        let copies = || std::iter::once(&self.engine).chain(self.replicas.iter());
        let in_flight: Vec<usize> = copies().map(|e| e.in_flight()).collect();
        let mut sent = self.sent.lock().unwrap_or_else(|e| e.into_inner());
        let just_sent: Vec<bool> = sent.iter().map(|t| t.elapsed() < PICK_WINDOW).collect();
        let last = self.last.load(Ordering::Relaxed);
        let at = pick_copy(&in_flight, &just_sent, last);
        sent[at] = std::time::Instant::now();
        self.last.store(at, Ordering::Relaxed);
        copies()
            .nth(at)
            .cloned()
            .unwrap_or_else(|| self.engine.clone())
    }

    /// Unload every copy of the model.
    pub(crate) async fn unload_all(&self) -> Result<(), String> {
        let mut first_error = None;
        for engine in std::iter::once(&self.engine).chain(self.replicas.iter()) {
            if let Err(e) = engine.unload().await {
                first_error.get_or_insert(e.to_string());
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Whether a request is running on any copy of the model.
    pub(crate) fn busy(&self) -> bool {
        std::iter::once(&self.engine)
            .chain(self.replicas.iter())
            .any(|e| e.in_flight() > 0)
    }

    /// Every copy of the model, `engine` first.
    pub(crate) fn copies(&self) -> Vec<Arc<LlmEngine>> {
        std::iter::once(&self.engine)
            .chain(self.replicas.iter())
            .cloned()
            .collect()
    }

    /// Seconds until the requests running on every copy are expected to end, `None` when a copy
    /// cannot say.
    pub(crate) fn expected_end_secs(&self) -> Option<f64> {
        expected_end_of(&self.copies())
    }

    /// Decode tokens per second for the main engine, or `None` when no rate has been measured.
    pub(crate) fn decode_tok_s(&self) -> Option<f64> {
        self.engine.pace().decode_tok_s()
    }

    /// Record activity on this entry (for LRU eviction ordering).
    pub(crate) fn touch(&self) {
        self.last_used.store(now_millis(), Ordering::Relaxed);
    }
}

/// The models of `engines` other than `model_id`, by name, each with whether it serves a request.
pub(crate) fn neighbours(engines: &[LoadedModelEntry], model_id: &str) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = engines
        .iter()
        .filter(|e| e.model_id != model_id)
        .map(|e| (e.model_id.clone(), e.busy()))
        .collect();
    out.sort();
    out
}

/// Seconds until the requests running on `copies` are expected to end, `None` when a copy cannot
/// say.
pub(crate) fn expected_end_of(copies: &[Arc<LlmEngine>]) -> Option<f64> {
    copies.iter().try_fold(0.0_f64, |longest, e| {
        Some(longest.max(e.pace().expected_end_secs()?))
    })
}

/// The seconds a load of `model_id` took, `None` before one was measured.
fn load_secs(model_id: &str) -> Option<f64> {
    let secs = crate::distributed::rate_meter::snapshot(model_id).model_load_s;
    (secs.is_finite() && secs > 0.0).then_some(secs)
}

/// The tokens a request writes: its `max_tokens`, else `mean`, the mean of the model's answers,
/// else `expected`, the tokens the rate meter expects; `None` when none is known.
pub(crate) fn request_tokens(
    max_tokens: Option<u64>,
    mean: Option<f64>,
    expected: u32,
) -> Option<f64> {
    // The token limit bounds an answer: the mean length, when known, is what it writes.
    max_tokens
        .map(|n| mean.map_or(n as f64, |m| m.min(n as f64)))
        .or(mean)
        .or_else(|| (expected > 0).then_some(expected as f64))
}

/// `busy`, the figures of a busy model, with those of the incoming model `incoming_id` loaded
/// with `config`: what its earlier loads and answers measured, and the plan it would build now
/// beside the resident models.
/// The request's `max_tokens` gives the tokens it writes.
async fn incoming_figures(
    incoming_id: &str,
    config: &crate::inference::engine::llm_engine::InferenceConfig,
    busy: ShareInputs,
    max_tokens: Option<u64>,
) -> ShareInputs {
    let kept = crate::inference::engine::llm_engine::Pace::load(incoming_id).unwrap_or_default();
    let tokens_b = request_tokens(
        max_tokens,
        kept.mean_answer_tokens(),
        crate::distributed::rate_meter::expected_output_tokens(incoming_id),
    );
    let dry = LlmEngine::with_config(config.clone()).dry_plan().await;
    if let Err(e) = &dry {
        info!("⚖️  {incoming_id}: no plan beside the busy models ({e})");
    }
    let dry = dry.ok();
    let shares_card = dry.as_ref().is_none_or(|d| {
        d.cards
            .iter()
            .any(|&card| crate::inference::engine::llm_engine::requests_on(Some(card)).1 > 0)
    });
    ShareInputs {
        tokens_b,
        load_b: load_secs(incoming_id),
        rate_b: kept.decode_tok_s(),
        rate_b_beside: dry
            .as_ref()
            .and_then(|d| kept.rate_on(d.predicted_tok_s, Some(&d.signature))),
        beside_on_cards: dry.as_ref().is_some_and(|d| d.on_cards),
        shares_card,
        ..busy
    }
}

/// `secs` as the log says it, "unknown" when it is not known.
fn seconds(secs: Option<f64>) -> String {
    secs.map_or_else(|| "unknown".to_string(), |s| format!("{s:.1} s"))
}

/// `tok_s` as the log says it, "unknown" when it is not known.
fn rate(tok_s: Option<f64>) -> String {
    tok_s.map_or_else(|| "unknown".to_string(), |r| format!("{r:.1} tok/s"))
}

/// What the choice between a busy model A and an incoming model B reads.
/// Seconds and tokens per second, each `None` when it is not measured.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ShareInputs {
    /// Seconds until A's running requests are expected to end.
    pub(crate) end_a: Option<f64>,
    /// Seconds a load of A takes.
    pub(crate) load_a: Option<f64>,
    /// A's decode rate alone.
    pub(crate) rate_a: Option<f64>,
    /// A's decode rate while another model runs.
    pub(crate) rate_a_beside: Option<f64>,
    /// Mean seconds between 2 requests of A.
    pub(crate) gap_a: Option<f64>,
    /// Tokens B's request writes.
    pub(crate) tokens_b: Option<f64>,
    /// Seconds a load of B takes.
    pub(crate) load_b: Option<f64>,
    /// B's decode rate on its full plan.
    pub(crate) rate_b: Option<f64>,
    /// B's decode rate on its plan beside A.
    pub(crate) rate_b_beside: Option<f64>,
    /// B's plan beside A keeps no block and no routed expert on the host.
    pub(crate) beside_on_cards: bool,
    /// B's plan beside A uses a card a running request uses.
    pub(crate) shares_card: bool,
}

/// What the incoming model does about a busy model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Share {
    /// Wait for A's end, unload A, load B on the room A held.
    WaitEvict,
    /// Load B beside A now, and run both at once.
    Beside,
    /// Wait for A's end, then load B beside A.
    Wait,
}

impl Share {
    /// The choice as the log says it.
    fn label(self) -> &'static str {
        match self {
            Share::WaitEvict => "wait and evict",
            Share::Beside => "run beside",
            Share::Wait => "wait",
        }
    }
}

/// The estimated seconds until B's request ends under each choice, in the order of [`Share`]:
/// wait and evict, run beside, wait; and the choice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ShareChoice {
    pub(crate) costs: [Option<f64>; 3],
    pub(crate) choice: Share,
}

/// The choice between waiting for busy model A and loading incoming model B beside it.
///
/// Wait and evict costs `T_A + L_B + n_B/r_B + q_A*L_A`.
/// `q_A = min(1, (L_B + n_B/r_B) / g_A)` is the chance that A is asked for while B runs.
/// An unknown `g_A` counts A as asked for, and an unknown `L_A` as a free reload.
/// Run beside costs `L_B + n_B/r_B' + T_A*(r_A/r_A' - 1)`: B on its plan beside A, A slowed.
/// Wait costs `T_A + L_B + n_B/r_B`, offered only when B's plan beside A holds no host block.
///
/// A plan beside A on cards no running request uses runs both at their own rates: B runs beside.
/// Otherwise the lowest known cost wins.
/// Wait and evict wins when its own cost is not known, and on a tie.
pub(crate) fn choose_share(i: &ShareInputs) -> ShareChoice {
    let alone = i.beside_on_cards && !i.shares_card;
    let fine = |v: Option<f64>| v.filter(|v| v.is_finite() && *v >= 0.0);
    let rate = |v: Option<f64>| v.filter(|v| v.is_finite() && *v > 0.0);
    let run_b = fine(i.tokens_b).zip(rate(i.rate_b)).map(|(n, r)| n / r);
    let end_a = fine(i.end_a);
    let load_b = fine(i.load_b);
    let wait_evict = end_a.zip(load_b).zip(run_b).map(|((t, l), run)| {
        let asked = rate(i.gap_a).map_or(1.0, |g| ((l + run) / g).min(1.0));
        t + l + run + asked * fine(i.load_a).unwrap_or(0.0)
    });
    let rate_b_beside = match alone {
        true => rate(i.rate_b_beside).or(rate(i.rate_b)),
        false => rate(i.rate_b_beside),
    };
    let slowed_a = match alone {
        true => Some(0.0),
        false => end_a
            .zip(rate(i.rate_a))
            .zip(rate(i.rate_a_beside))
            .map(|((t, r), r_beside)| t * (r / r_beside - 1.0)),
    };
    let beside = load_b
        .zip(fine(i.tokens_b))
        .zip(rate_b_beside)
        .zip(slowed_a)
        .map(|(((l, n), r), slowed)| l + n / r + slowed);
    let wait = end_a
        .zip(load_b)
        .zip(run_b)
        .filter(|_| i.beside_on_cards)
        .map(|((t, l), run)| t + l + run);
    let costs = [wait_evict, beside, wait];
    let choice = if alone {
        Share::Beside
    } else if let Some(first) = wait_evict {
        [(beside, Share::Beside), (wait, Share::Wait)]
            .into_iter()
            .filter_map(|(cost, share)| cost.map(|c| (c, share)))
            .fold((first, Share::WaitEvict), |best, next| {
                if next.0 < best.0 {
                    next
                } else {
                    best
                }
            })
            .1
    } else {
        Share::WaitEvict
    };
    ShareChoice { costs, choice }
}

/// What the choice to load an idle model B again in full reads.
/// Seconds, tokens and tokens per second, each `None` when it is not measured.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReplaceInputs {
    /// Seconds a load of B takes.
    pub(crate) load_b: Option<f64>,
    /// B's decode rate on the plan it holds.
    pub(crate) rate_now: Option<f64>,
    /// B's decode rate on the plan a load in full builds.
    pub(crate) rate_full: Option<f64>,
    /// Tokens the request on its way to B writes, `None` when no request is on its way.
    pub(crate) tokens_arriving: Option<f64>,
    /// Mean tokens of B's answers.
    pub(crate) tokens_b: Option<f64>,
    /// Mean seconds between 2 requests of B.
    pub(crate) gap_b: Option<f64>,
    /// Seconds until B's keep-alive ends, infinite when B stays loaded.
    pub(crate) keep_left: f64,
    /// The idle models a load in full may evict: their load seconds and mean gap between
    /// requests.
    pub(crate) evicted: Vec<(Option<f64>, Option<f64>)>,
}

impl Default for ReplaceInputs {
    fn default() -> Self {
        Self {
            load_b: None,
            rate_now: None,
            rate_full: None,
            tokens_arriving: None,
            tokens_b: None,
            gap_b: None,
            keep_left: 0.0,
            evicted: Vec::new(),
        }
    }
}

/// The load in full priced: its cost and its gain in seconds, the requests the gain counts, and
/// the choice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReplaceChoice {
    pub(crate) cost: Option<f64>,
    pub(crate) gain: Option<f64>,
    pub(crate) requests: Option<f64>,
    pub(crate) replace: bool,
}

/// The choice to load idle model B again in full, or to keep the plan it holds.
///
/// The load costs `L_B`.
/// Each request gains `n_B * (1/r_B' - 1/r_B)`: `r_B'` is the rate on the plan B holds and `r_B`
/// the rate on the plan in full.
/// The requests are the one on its way, with its own tokens, and `K / g_B` more of `n_B` tokens:
/// `K` is the keep-alive B has left and `g_B` the mean gap between its requests.
/// An unknown `g_B` counts the request on its way alone.
/// Each idle model X the load may evict adds `q_X * L_X`, as the wait and evict cost does:
/// `q_X = min(1, K / g_X)` is the chance X is asked again while B holds the cards.
/// An unknown `g_X` counts `q_X` as 1, an unknown `L_X` as 0.
/// B loads again when the gain exceeds the cost.
/// An unknown cost, rate or count of requests keeps B where it is.
pub(crate) fn choose_replace(i: &ReplaceInputs) -> ReplaceChoice {
    let fine = |v: Option<f64>| v.filter(|v| v.is_finite() && *v >= 0.0);
    let rate = |v: Option<f64>| v.filter(|v| v.is_finite() && *v > 0.0);
    let reloads: f64 = i
        .evicted
        .iter()
        .map(|(load, gap)| {
            let q = rate(*gap).map_or(1.0, |gap| (i.keep_left.max(0.0) / gap).min(1.0));
            q * fine(*load).unwrap_or(0.0)
        })
        .sum();
    let cost = fine(i.load_b).map(|load| load + reloads);
    let saved = rate(i.rate_now)
        .zip(rate(i.rate_full))
        .map(|(now, full)| (1.0 / now - 1.0 / full).max(0.0));
    let arriving = fine(i.tokens_arriving);
    let later = rate(i.gap_b)
        .zip(rate(i.tokens_b))
        .map(|(gap, tokens)| (i.keep_left.max(0.0) / gap, tokens));
    let requests = match (arriving, later) {
        (None, None) => None,
        (a, l) => Some(f64::from(u8::from(a.is_some())) + l.map_or(0.0, |(r, _)| r)),
    };
    let tokens = match (arriving, later) {
        (None, None) => None,
        (a, l) => Some(a.unwrap_or(0.0) + l.map_or(0.0, |(r, n)| r * n)),
    };
    let gain = saved
        .zip(tokens)
        .map(|(s, t)| if s > 0.0 && t > 0.0 { s * t } else { 0.0 });
    let replace = matches!((gain, cost), (Some(g), Some(c)) if g > c);
    ReplaceChoice {
        cost,
        gain,
        requests,
        replace,
    }
}

/// Waits `period`, then until no request runs on the model, and returns the engine list locked.
/// A model is idle when no request runs on it: a prompt read for minutes keeps the model, and
/// the wait starts over once the request is done.
async fn wait_until_idle<'a>(
    engines: &'a tokio::sync::RwLock<Vec<LoadedModelEntry>>,
    model_id: &str,
    period: Duration,
) -> tokio::sync::RwLockWriteGuard<'a, Vec<LoadedModelEntry>> {
    loop {
        tokio::time::sleep(period).await;
        let guard = engines.write().await;
        if !guard.iter().any(|e| e.model_id == model_id && e.busy()) {
            return guard;
        }
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
        let server = self.clone();

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

            let mut engines_guard = wait_until_idle(&engines, &model_id, duration).await;
            if let Some(pos) = engines_guard.iter().position(|e| e.model_id == model_id) {
                let entry = engines_guard.remove(pos);
                // The image is taken while the weights are resident, so it reads no disk.
                let slept = match entry.engine.sleep_files() {
                    Some(files) => {
                        let sleepers = server.sleepers.clone();
                        let id = entry.model_id.clone();
                        tokio::task::spawn_blocking(move || sleepers.sleep(&id, &files))
                            .await
                            .ok()
                            .flatten()
                    }
                    None => None,
                };
                if let Err(e) = entry.unload_all().await {
                    info!("Error unloading model {}: {}", entry.model_id, e);
                    server.sleepers.release(&entry.model_id);
                } else if let Some((bytes, pinned)) = slept {
                    info!(
                        "Model {} asleep after keep_alive expired: {} MB held in host memory ({}), card memory released",
                        entry.model_id,
                        bytes / 1_000_000,
                        if pinned { "pinned" } else { "resident" }
                    );
                } else {
                    info!("Model {} unloaded after keep_alive expired", entry.model_id);
                }
            }
            drop(engines_guard);
            server.migrate_idle_models().await;
        })
    }

    /// Reloads idle language models onto a card that now serves them faster: the memory an
    /// expired or unloaded model gave back can hold a resident that went to a slower card, or
    /// to several, or partly to the host, because that card was full when it loaded. One pass
    /// at a time; a model serving a request is never moved, and one that gets a request while
    /// it reloads waits for the load like any cold request.
    /// A model no card serves better loads again in full when [`Self::replace_in_full`] finds
    /// that it pays.
    pub(crate) async fn migrate_idle_models(&self) {
        static MIGRATING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let Ok(_one_pass) = MIGRATING.try_lock() else {
            return;
        };
        loop {
            let cards = crate::inference::place::room::card_rooms();
            let idle: Vec<(String, Arc<LlmEngine>, i64)> = {
                let engines = self.engines.read().await;
                engines
                    .iter()
                    .filter(|e| e.engine.in_flight() == 0)
                    .map(|e| {
                        (
                            e.model_id.clone(),
                            e.engine.clone(),
                            e.keep_alive_minutes.unwrap_or(self.default_keep_alive),
                        )
                    })
                    .collect()
            };
            let mut moved = false;
            for (model_id, engine, keep_alive) in idle {
                let Some(placement) = engine.placement().await else {
                    continue;
                };
                let config = self.config_for_model(&model_id);
                let need = self.model_blob_size(&model_id, &config)
                    + self.model_load_reserve(&model_id, &config);
                let Some(target) = placement.better_card(&cards, need) else {
                    continue;
                };
                if engine.in_flight() > 0 {
                    continue;
                }
                info!(
                    "⚖️  Moving idle model {model_id} to card {target} (now on cards {:?}, {} \
                     layers on the host)",
                    placement.cards, placement.host_layers
                );
                drop(engine);
                let _ = self.unload_model(&model_id).await;
                if let Err(e) = self.ensure_loaded(&model_id, keep_alive).await {
                    warn!("⚖️  {model_id} could not be reloaded after its move: {e}");
                }
                moved = true;
                break;
            }
            if !moved {
                let idle: Vec<(String, i64)> = {
                    let engines = self.engines.read().await;
                    engines
                        .iter()
                        .filter(|e| !e.busy())
                        .map(|e| {
                            let keep_alive =
                                e.keep_alive_minutes.unwrap_or(self.default_keep_alive);
                            // The load in full keeps what was left of the keep-alive.
                            let idle_minutes = (now_millis() - e.last_used.load(Ordering::Relaxed))
                                .max(0)
                                / 60_000;
                            let left = match keep_alive {
                                k if k < 0 => k,
                                k => (k - idle_minutes).max(1),
                            };
                            (e.model_id.clone(), left)
                        })
                        .collect()
                };
                for (model_id, keep_alive) in idle {
                    let Some(window) = self.replace_in_full(&model_id, None).await else {
                        continue;
                    };
                    let window = (window > 0).then_some(window);
                    if let Err(e) = self
                        .ensure_loaded_with_window(&model_id, keep_alive, window, None)
                        .await
                    {
                        warn!("⚖️  {model_id} could not be loaded again in full: {e}");
                    }
                    moved = true;
                    break;
                }
            }
            if !moved {
                return;
            }
        }
    }

    /// Unloads `model_id` when a load of it in full pays, priced by [`choose_replace`]; returns
    /// the window it held then, for the load in full to open again, or `None` when it stays.
    ///
    /// `arriving` holds the `max_tokens` of a request on its way to the model, `None` when no
    /// request is on its way.
    /// The plan in full is the plan a load builds once the model and the idle models give back
    /// what they hold on the cards.
    /// A load unloads the idle models whose room it needs.
    /// It is priced only when the other models or their requests changed since the model loaded
    /// or was last priced, and only while no request runs on the model.
    /// A model with copies on other cards is whole on its card and stays.
    pub(crate) async fn replace_in_full(
        &self,
        model_id: &str,
        arriving: Option<Option<u64>>,
    ) -> Option<usize> {
        let (engine, state, given_back, keep_left, idle, holders) = {
            let engines = self.engines.read().await;
            let entry = engines.iter().find(|e| e.model_id == model_id)?;
            if entry.busy() || !entry.replicas.is_empty() {
                return None;
            }
            let state = neighbours(&engines, model_id);
            if entry
                .priced
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                == Some(&state)
            {
                return None;
            }
            // A plan whole on one card with nothing on the host is already in full.
            match entry.engine.pace().plan() {
                Some(plan) if plan.host_bytes > 0 || plan.cards.len() > 1 => {}
                _ => return None,
            }
            let mut given_back = entry.engine.pace().held();
            let mut idle: Vec<(Vec<usize>, Option<f64>, Option<f64>)> = Vec::new();
            for other in engines
                .iter()
                .filter(|e| e.model_id != model_id && !e.busy())
            {
                let cards = other
                    .copies()
                    .into_iter()
                    .flat_map(|copy| copy.pace().held().into_iter().map(|(card, _)| card))
                    .collect();
                idle.push((
                    cards,
                    load_secs(&other.model_id),
                    other.engine.pace().mean_gap_secs(),
                ));
                for copy in other.copies() {
                    for (card, bytes) in copy.pace().held() {
                        match given_back.iter_mut().find(|(c, _)| *c == card) {
                            Some(held) => held.1 = held.1.saturating_add(bytes),
                            None => given_back.push((card, bytes)),
                        }
                    }
                }
            }
            let keep_alive = entry.keep_alive_minutes.unwrap_or(self.default_keep_alive);
            let idle_secs =
                (now_millis() - entry.last_used.load(Ordering::Relaxed)).max(0) as f64 / 1000.0;
            let keep_left = match (keep_alive, arriving) {
                (k, _) if k < 0 => f64::INFINITY,
                // The request on its way starts the keep-alive again.
                (k, Some(_)) => (k * 60) as f64,
                (k, None) => ((k * 60) as f64 - idle_secs).max(0.0),
            };
            (
                entry.engine.clone(),
                state,
                given_back,
                keep_left,
                idle,
                // The holders of the engine now, this function's own copy included.
                Arc::strong_count(&entry.engine),
            )
        };
        let full = match engine.dry_plan_given(given_back).await {
            Ok(full) => full,
            Err(e) => {
                tracing::debug!("⚖️  {model_id}: no plan in full ({e})");
                return None;
            }
        };
        if let Some(entry) = self
            .engines
            .read()
            .await
            .iter()
            .find(|e| e.model_id == model_id)
        {
            *entry.priced.lock().unwrap_or_else(|e| e.into_inner()) = Some(state);
        }
        let pace = engine.pace();
        let kept = pace.kept();
        let plan = pace.plan();
        let mean = kept.mean_answer_tokens();
        let expected = crate::distributed::rate_meter::expected_output_tokens(model_id);
        let inputs = ReplaceInputs {
            load_b: load_secs(model_id),
            rate_now: kept.rate_on(kept.planned_tok_s, plan.as_ref()),
            rate_full: kept.rate_on(full.predicted_tok_s, Some(&full.signature)),
            tokens_arriving: arriving.and_then(|max| request_tokens(max, mean, expected)),
            tokens_b: request_tokens(None, mean, expected),
            gap_b: pace.mean_gap_secs(),
            keep_left,
            // The idle models on the cards the plan in full uses may give their room to it.
            evicted: idle
                .into_iter()
                .filter(|(cards, _, _)| cards.iter().any(|c| full.cards.contains(c)))
                .map(|(_, load, gap)| (load, gap))
                .collect(),
        };
        let choice = choose_replace(&inputs);
        info!(
            "⚖️  {model_id} in full: load {}, gain {} over {} requests at {} now and {} in full \
             ({} MB on the host now, {} MB in full): {}",
            seconds(choice.cost),
            seconds(choice.gain),
            choice
                .requests
                .map_or_else(|| "unknown".to_string(), |r| format!("{r:.1}")),
            rate(inputs.rate_now),
            rate(inputs.rate_full),
            plan.as_ref().map_or(0, |p| p.host_bytes >> 20),
            full.signature.host_bytes >> 20,
            if choice.replace { "load again" } else { "keep" }
        );
        if !choice.replace {
            return None;
        }
        let lock = self.loading_lock(model_id).await;
        let _g = lock.lock().await;
        let window = engine.config().context_length;
        drop(engine);
        // A request that took the engine since the pricing holds one more copy of it, before
        // the engine counts it as running: the model stays for it.
        let busy = {
            let engines = self.engines.read().await;
            engines
                .iter()
                .find(|e| e.model_id == model_id)
                .is_none_or(|e| e.busy() || Arc::strong_count(&e.engine) >= holders)
        };
        if busy {
            return None;
        }
        matches!(self.unload_model(model_id).await, Ok(UnloadOutcome::Freed)).then_some(window)
    }

    /// Reset model expiration timer (called on each use, like Ollama).
    /// Cancels the existing timer and schedules a fresh one. Also
    /// updates the entry's `last_used` for LRU eviction ordering.
    pub async fn reset_expiration(&self, model_id: &str) {
        let mut engines = self.engines.write().await;
        if let Some(entry) = engines.iter_mut().find(|e| e.model_id == model_id) {
            entry.touch();
            // Cancel existing timer
            if let Some(handle) = &entry.expire_handle.take() {
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
    /// Whether a conversation's prefix is resident for `model_id` right now.
    ///
    /// False when the model is not loaded at all: there is nothing to lose, so nothing to
    /// protect. Asked without loading anything, because the answer decides whether to do
    /// the work rather than how.
    pub(crate) async fn holds_a_conversation(&self, model_id: &str) -> bool {
        let engines = self.engines.read().await;
        let Some(entry) = engines.iter().find(|e| e.model_id == model_id) else {
            return false;
        };
        let engine = entry.engine.clone();
        drop(engines);
        engine.holds_a_conversation().await
    }

    pub(crate) async fn get_engine(&self, model_id: &str) -> Result<Arc<LlmEngine>, ApiError> {
        // Check if engine already loaded (by original ID)
        {
            let engines = self.engines.read().await;
            if let Some(entry) = engines.iter().find(|e| e.model_id == model_id) {
                return Ok(entry.serving());
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
    /// The context length a model's GGUF header declares, read without loading it.
    /// Read once per file: a request asking past the loaded window would otherwise parse
    /// the whole header, vocabulary included, on every call.
    pub(crate) fn declared_context(&self, model_id: &str) -> Option<usize> {
        let path = self.manifest_layer_path(model_id, "model")?;
        let stamp = FileStamp::of(&path)?;
        self.declared_contexts.get_or_read(&path, stamp, || {
            let (meta, _, _, _) = super::ollama::gguf_facts(&path);
            let meta = meta?;
            let arch = meta.get("general.architecture")?.as_str()?.to_string();
            meta.get(&format!("{arch}.context_length"))?
                .as_u64()
                .map(|v| v as usize)
        })
    }

    /// A request's `num_ctx` above the context the engine was configured with reloads it
    /// at that context, as Ollama does; the loader still bounds it by what the model
    /// declares and what the cards hold.
    pub(crate) async fn ensure_engine_context(
        &self,
        model_id: &str,
        want: usize,
    ) -> Result<(), ApiError> {
        let (engine, current, current_quant, had_keep_alive) = {
            let engines = self.engines.read().await;
            match engines.iter().find(|e| e.model_id == model_id) {
                Some(entry) => (
                    entry.engine.clone(),
                    entry.engine.config().context_length,
                    entry.engine.config().kv_quant,
                    entry.keep_alive_minutes,
                ),
                None => return Ok(()),
            }
        };
        // A window the engine already has needs nothing; answering that first keeps the
        // file's header, which `declared_context` parses, off the path of every request.
        if want <= current {
            return Ok(());
        }
        // A configured context of zero opens the window the model declares, which the
        // loaded engine reports.
        let open = engine.context_window().await;
        let Some(want) = widened_context(want, current, open, || self.declared_context(model_id))
        else {
            return Ok(());
        };
        info!("num_ctx {want} above the configured context {current} for {model_id} - reloading model");
        self.unload_model(model_id).await.ok();

        // Rebuilt from the file's defaults, so anything the engine in place was asked
        // for has to be carried over: a widening that reset the cache format would
        // silently double what the wider window costs.
        let mut config = self.config_for_model(model_id);
        config.context_length = want;
        config.kv_quant = current_quant;
        let engine = Arc::new(LlmEngine::with_config(config));
        // A window is clamped by what the checkpoint declares, not by what the cards can
        // hold, so a request can ask for a KV cache that does not fit. The model has
        // already been unloaded by this point: without putting it back, one such request
        // leaves the node serving nothing until somebody asks for the model again.
        // Kept as text from the start: the loader's error is not `Send`, and holding it
        // across the reload below would make this whole future unspawnable.
        let failure = engine
            .load_model_for_request()
            .await
            .err()
            .map(|e| e.to_string());
        if let Some(e) = failure {
            let mut back = self.config_for_model(model_id);
            back.context_length = current;
            back.kv_quant = current_quant;
            let restored = Arc::new(LlmEngine::with_config(back));
            if restored.load_model().await.is_ok() {
                let keep_alive = had_keep_alive.unwrap_or(self.default_keep_alive);
                let expire_handle = if keep_alive > 0 {
                    Some(self.schedule_expiration(model_id.to_string(), keep_alive))
                } else {
                    None
                };
                let mut engines = self.engines.write().await;
                let mut entry =
                    LoadedModelEntry::new(model_id.to_string(), restored, Some(keep_alive));
                entry.expire_handle = expire_handle;
                engines.push(entry);
            }
            return Err(ApiError::Internal(format!(
                "Reload with num_ctx={want} failed: {e}"
            )));
        }

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
        #[cfg(feature = "gpu")]
        if freed > 0 {
            crate::inference::engine::llm_engine::release_cuda_pools();
        }
        crate::inference::engine::llm_engine::release_idle_cards();
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
                        if let Some(h) = &entry.expire_handle {
                            h.abort();
                        }
                        let name = entry.model_id.clone();
                        if entry.unload_all().await.is_ok() {
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
        if !kind.claims(&self.huggingface_models_dir, loaded.as_deref(), model_id) {
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
                if let Some(handle) = &entry.expire_handle {
                    handle.abort();
                }
                entry
                    .unload_all()
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
        // An unload asked for gives the host memory back too.
        if self.sleepers.release(model_id) {
            info!("unload: {model_id} was asleep; host memory released");
        }
        for kind in ResidentEngine::ALL {
            if self.release_named(*kind, model_id).await? {
                return Ok(UnloadOutcome::Freed);
            }
        }
        info!("unload: {model_id} was not loaded; nothing to free");
        Ok(UnloadOutcome::NotResident)
    }

    /// Makes room for the `need` bytes the incoming model's weights take (or, for a model larger
    /// than any card, for them across the cards), while no card has it: first what the other
    /// models keep for their later requests, then the least recently used language models that
    /// serve no request, other than `incoming_id`.
    ///
    /// A model serving a request is never unloaded under it.
    /// When only busy models are left, [`choose_share`] weighs the least recently used one
    /// against the incoming model, whose `config` gives the plan it would build beside it:
    /// wait for its end and unload it, place the incoming model around it now, or wait for
    /// its end and keep it.
    /// A model still busy after its wait stays, and the incoming model is placed around it.
    /// Resident models hold weights, KV blocks and pools that would otherwise push the new
    /// model's placement onto the host; as many stay resident as the memory allows.
    ///
    /// `per_card` is what the load keeps free on each card it uses beyond the weights: the
    /// KV cache, workspaces and staging the loader reserves.
    /// `max_tokens` is what the request that asks for the incoming model may write.
    pub(crate) async fn evict_to_make_room(
        &self,
        incoming_id: &str,
        need: u64,
        per_card: u64,
        config: &crate::inference::engine::llm_engine::InferenceConfig,
        max_tokens: Option<u64>,
    ) {
        let mut tried: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut waited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut asked = false;
        loop {
            let demand = crate::inference::place::ladder::Demand {
                block_bytes: need,
                runtime_reserve: per_card,
                ..Default::default()
            };
            // The cards as the incoming model is offered them, each settled: less what other
            // owners declared.
            let cards = crate::inference::place::room::card_rooms_for(incoming_id);
            for card in &cards {
                crate::inference::place::room::settle(card.index);
            }
            let cards = crate::inference::place::room::card_rooms_for(incoming_id);
            let free = crate::inference::place::room::room_for(&demand, &cards);
            if need <= free {
                return;
            }
            if !asked {
                asked = true;
                let mut given = false;
                for card in &cards {
                    given |= crate::inference::place::reclaim::make_room_for(
                        card.index,
                        need.saturating_add(per_card),
                        incoming_id,
                    );
                }
                if given {
                    continue;
                }
            }
            let victim = {
                let engines = self.engines.read().await;
                engines
                    .iter()
                    .filter(|e| e.model_id != incoming_id && !tried.contains(&e.model_id))
                    .filter(|e| !e.busy())
                    .min_by_key(|e| e.last_used.load(Ordering::Relaxed))
                    .map(|e| e.model_id.clone())
            };
            let Some(victim) = victim else {
                let busy = {
                    let engines = self.engines.read().await;
                    engines
                        .iter()
                        .filter(|e| e.model_id != incoming_id && e.busy())
                        .filter(|e| !tried.contains(&e.model_id) && !waited.contains(&e.model_id))
                        .min_by_key(|e| e.last_used.load(Ordering::Relaxed))
                        .map(|e| {
                            let pace = format!(
                                "{} at {}",
                                seconds(e.expected_end_secs()),
                                rate(e.decode_tok_s())
                            );
                            let figures = ShareInputs {
                                end_a: e.expected_end_secs(),
                                load_a: load_secs(&e.model_id),
                                rate_a: e.decode_tok_s(),
                                rate_a_beside: e.engine.pace().beside_tok_s(),
                                gap_a: e.engine.pace().mean_gap_secs(),
                                ..ShareInputs::default()
                            };
                            (e.model_id.clone(), e.copies(), pace, figures)
                        })
                };
                let Some((id, copies, pace, figures)) = busy else {
                    let engines = self.engines.read().await;
                    if engines
                        .iter()
                        .any(|e| e.model_id != incoming_id && e.busy())
                    {
                        info!("⚖️  {incoming_id} is placed around the busy models");
                    }
                    return;
                };
                let inputs = incoming_figures(incoming_id, config, figures, max_tokens).await;
                let choice = choose_share(&inputs);
                let [wait_evict, beside, wait] = choice.costs.map(seconds);
                info!(
                    "⚖️  {incoming_id} beside busy {id}: wait and evict {wait_evict}, \
                     run beside {beside}, wait {wait}: {}",
                    choice.choice.label()
                );
                if choice.choice == Share::Beside {
                    info!("⚖️  {incoming_id} is placed around {id}, both run at once");
                    return;
                }
                info!("⚖️  {id} is busy, expected to end in {pace}: {incoming_id} waits for it");
                let started = std::time::Instant::now();
                // The wait is bounded as a request's wait at its gate is.
                let ended = tokio::time::timeout(super::GATE_WAIT, async {
                    for copy in &copies {
                        copy.until_no_request().await;
                    }
                })
                .await
                .is_ok();
                info!(
                    "⚖️  {id} {} after {:.1} s",
                    if ended { "ended" } else { "still runs" },
                    started.elapsed().as_secs_f64()
                );
                if choice.choice == Share::Wait {
                    return;
                }
                waited.insert(id);
                continue;
            };
            info!(
                "⚖️  Evicting LRU model {victim} to make room for {incoming_id} \
                 ({} MB needed, {} MB free beyond {} MB kept per card)",
                need >> 20,
                free >> 20,
                per_card >> 20
            );
            tried.insert(victim.clone());
            let _ = self.unload_model(&victim).await;
        }
    }

    /// Unloads idle language models, least recently used first, while the host has no room
    /// left beyond its safety headroom: a load then needs the memory they hold.
    /// A machine with host memory to spare unloads nothing.
    pub(crate) async fn evict_for_host_room(&self, incoming_id: &str) {
        while crate::inference::place::layer_executor::host_spill_budget_bytes() == 0 {
            if self.evict_idle_llm(incoming_id).await == 0 {
                return;
            }
            // What the unloaded model freed goes back to the kernel past the burst headroom.
            crate::inference::place::heap::release_retained();
        }
    }

    /// Unloads the least recently used language model that is not `except` and serves no
    /// request; returns the bytes its weights took, 0 when every other model is busy or none
    /// is loaded.
    pub(crate) async fn evict_idle_llm(&self, except: &str) -> u64 {
        let victim = {
            let engines = self.engines.read().await;
            engines
                .iter()
                .filter(|e| e.model_id != except && e.engine.in_flight() == 0)
                .min_by_key(|e| e.last_used.load(Ordering::Relaxed))
                .map(|e| e.model_id.clone())
        };
        let Some(victim) = victim else {
            return 0;
        };
        let bytes = self.model_blob_size(&victim, &self.config_for_model(&victim));
        info!("⚖️  Unloading idle model {victim} to give {except} room");
        match self.unload_model(&victim).await {
            Ok(_) => bytes.max(1),
            Err(_) => 0,
        }
    }

    /// Installs the evictor the engines call when a generation runs out of memory.
    pub fn register_memory_hooks(&self) {
        let server = self.clone();
        crate::inference::place::vram_manager::register_idle_llm_evictor(Box::new(move |except| {
            let server = server.clone();
            Box::pin(async move { server.evict_idle_llm(&except).await })
        }));
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

        if let Some(refusal) = self.companion_refusal(new_model_id).await {
            return Err(ApiError::NotFound(refusal));
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
        if let Some(handle) = &old_entry.expire_handle {
            handle.abort();
        }

        if let Err(e) = old_entry.unload_all().await {
            warn!("Failed to unload old model {}: {}", old_entry.model_id, e);
        } else {
            info!("Old model {} unloaded successfully", old_entry.model_id);
        }

        Ok(())
    }
}

/// The window a request for `want` tokens needs a reload to reach, or None when the engine
/// serves it as it is. `current` is the configured context (zero opens the window the model
/// declares), `open` the window the loaded engine reports and `declared` what the checkpoint
/// declares: the model cannot open a window past that, so a larger request is clamped there
/// before it can cost a reload that would change nothing.
fn widened_context(
    want: usize,
    current: usize,
    open: Option<usize>,
    declared: impl FnOnce() -> Option<usize>,
) -> Option<usize> {
    if want <= current || open.is_some_and(|open| want <= open) {
        return None;
    }
    let want = declared().map_or(want, |declared| want.min(declared));
    (want > current.max(open.unwrap_or(0))).then_some(want)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_request_past_the_declared_window_does_not_reload_a_model_that_has_it() {
        // Configured zero, window open at the declared 4096: any request is served as is.
        assert_eq!(widened_context(131072, 0, Some(4096), || Some(4096)), None);
        assert_eq!(widened_context(4096, 0, Some(4096), || Some(4096)), None);
        // A wider window the checkpoint declares reloads at the clamped size.
        assert_eq!(
            widened_context(131072, 4096, Some(4096), || Some(32768)),
            Some(32768)
        );
        // Nothing declared: the request is taken as asked.
        assert_eq!(widened_context(8192, 4096, Some(4096), || None), Some(8192));
        // A window already configured needs nothing.
        assert_eq!(widened_context(2048, 4096, Some(4096), || Some(4096)), None);
    }

    #[test]
    fn a_declared_context_is_read_once_per_file_stamp() {
        use super::{DeclaredContexts, FileStamp};
        let memo = DeclaredContexts::default();
        let path = std::path::Path::new("model.gguf");
        let first = FileStamp {
            len: 10,
            modified: None,
        };
        let reads = std::cell::Cell::new(0);
        let read = |v| {
            reads.set(reads.get() + 1);
            v
        };
        assert_eq!(
            memo.get_or_read(path, first, || read(Some(32768))),
            Some(32768)
        );
        assert_eq!(memo.get_or_read(path, first, || read(Some(1))), Some(32768));
        assert_eq!(reads.get(), 1);
        // A file rewritten under the same path is read again.
        let rewritten = FileStamp {
            len: 11,
            modified: None,
        };
        assert_eq!(
            memo.get_or_read(path, rewritten, || read(Some(8192))),
            Some(8192)
        );
        assert_eq!(reads.get(), 2);
        // A header with no context is remembered as such.
        let other = std::path::Path::new("other.gguf");
        assert_eq!(memo.get_or_read(other, first, || read(None)), None);
        assert_eq!(memo.get_or_read(other, first, || read(Some(1))), None);
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn a_request_goes_to_the_least_taken_copy_and_a_conversation_stays_put() {
        use super::pick_copy;
        // Everything idle: the copy that served the last request.
        assert_eq!(pick_copy(&[0, 0], &[false, false], 1), 1);
        assert_eq!(pick_copy(&[0, 0, 0], &[false; 3], 0), 0);
        // A busy copy gives way, whichever served last.
        assert_eq!(pick_copy(&[1, 0], &[false, false], 0), 1);
        assert_eq!(pick_copy(&[2, 1, 3], &[false; 3], 2), 1);
        // A request sent a moment ago, not yet admitted, counts as taking the copy.
        assert_eq!(pick_copy(&[0, 0], &[true, false], 0), 1);
        // Once admitted it shows in the count and the flag adds nothing: a tie.
        assert_eq!(pick_copy(&[1, 1], &[true, false], 0), 0);
        // Every copy taken: the one with the fewest, the first on a tie.
        assert_eq!(pick_copy(&[2, 2], &[false, false], 1), 1);
        assert_eq!(pick_copy(&[3], &[false], 0), 0);
    }

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
    /// A store holding the speech checkpoint the TTS rows name.
    fn store() -> crate::inference::media_facts::fixture::Scratch {
        use crate::inference::media_facts::fixture::{json, Scratch};
        let s = Scratch::new("resident-names");
        let dir = s.0.join("hub/models--parler-tts--parler-tts-mini-v1");
        std::fs::create_dir_all(dir.join("refs")).unwrap();
        std::fs::write(dir.join("refs/main"), "r").unwrap();
        json(
            &dir.join("snapshots/r/config.json"),
            serde_json::json!({"model_type": "parler_tts"}),
        );
        std::fs::write(dir.join("snapshots/r/model.safetensors"), b"").unwrap();
        s
    }

    #[cfg(feature = "audio")]
    #[test]
    fn a_resident_tts_checkpoint_is_named_by_the_ids_a_client_has_for_it() {
        let s = store();
        let hf = s.0.to_str().unwrap();
        assert!(ResidentEngine::ALL.contains(&ResidentEngine::Tts));
        let loaded = Some("parler-tts/parler-tts-mini-v1");
        for id in [
            "parler-tts/parler-tts-mini-v1",
            "parler-tts/parler-tts-mini-v1:latest",
            "parler-tts-mini-v1",
            // The OpenAI alias, resolved by the catalogue the load resolved it by.
            "tts-1",
        ] {
            assert!(
                ResidentEngine::Tts.claims(hf, loaded, id),
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
                !kind.claims(hf, Some(its_own), "parler-tts/parler-tts-mini-v1"),
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
        let s = store();
        let hf = s.0.to_str().unwrap();
        for kind in ResidentEngine::ALL {
            let (loaded, id, claimed) = name_rule(*kind);
            assert_eq!(
                kind.claims(hf, Some(loaded), id),
                claimed,
                "{kind:?} disagrees with its own name rule"
            );
            assert!(
                !kind.claims(hf, None, id),
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
        let mut expected = vec!["llm"];
        #[cfg(feature = "image")]
        expected.push("image");
        #[cfg(feature = "audio")]
        expected.extend(["tts", "stable-audio"]);
        for name in expected {
            assert!(
                seen.contains(name),
                "the pressure protocol addresses '{name}', which is not on the walk"
            );
        }
    }

    /// The wait for a busy model returns only once its last request has ended.
    #[tokio::test]
    async fn the_wait_for_a_busy_engine_ends_with_its_last_request() {
        let engine = Arc::new(LlmEngine::with_config(
            crate::inference::engine::llm_engine::InferenceConfig::default(),
        ));
        let first = engine.admit_request().await;
        let second = engine.admit_request().await;
        let waiting = engine.until_no_request();
        tokio::pin!(waiting);
        let early = tokio::time::timeout(Duration::from_millis(200), &mut waiting).await;
        assert!(early.is_err(), "returned under 2 running requests");
        drop(first);
        let early = tokio::time::timeout(Duration::from_millis(200), &mut waiting).await;
        assert!(early.is_err(), "returned under 1 running request");
        drop(second);
        tokio::time::timeout(Duration::from_secs(5), &mut waiting)
            .await
            .expect("the wait ends with the last request");
    }

    /// A model with a request running is not unloaded when its keep-alive runs out: a cold
    /// prompt of a big model is read for longer than the keep-alive.
    #[tokio::test]
    async fn the_keep_alive_waits_for_a_request_running_on_the_model() {
        let engine = Arc::new(LlmEngine::with_config(
            crate::inference::engine::llm_engine::InferenceConfig::default(),
        ));
        let engines = tokio::sync::RwLock::new(vec![LoadedModelEntry::new(
            "m".to_string(),
            engine.clone(),
            Some(1),
        )]);
        assert!(!engines.read().await[0].busy());
        let request = engine.admit_request().await;
        assert!(engines.read().await[0].busy());
        let period = Duration::from_millis(20);
        let waiting = wait_until_idle(&engines, "m", period);
        tokio::pin!(waiting);
        let early = tokio::time::timeout(Duration::from_millis(200), &mut waiting).await;
        assert!(early.is_err(), "returned under a running request");
        drop(request);
        let guard = tokio::time::timeout(Duration::from_secs(5), &mut waiting)
            .await
            .expect("idle once the request is done");
        assert_eq!(guard.len(), 1);
    }

    /// The server's figures read its own gate and the gate of each model: a model with one copy
    /// adds 1 lane, a model with 2 copies 2 lanes.
    #[tokio::test]
    async fn the_gate_figures_add_the_gates_of_the_models() {
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            "/nonexistent-hf-models".to_string(),
        );
        let engine = || {
            Arc::new(LlmEngine::with_config(
                crate::inference::engine::llm_engine::InferenceConfig::default(),
            ))
        };
        assert_eq!(state.gates().await.len(), 1);
        assert_eq!(state.lanes().await, 1);
        state
            .engines
            .write()
            .await
            .push(LoadedModelEntry::new("a".to_string(), engine(), Some(1)));
        let gates = state.gates().await;
        assert_eq!(gates.len(), 2);
        assert!(Arc::ptr_eq(&gates[0], &state.request_gate));
        assert_eq!(state.lanes().await, 1);
        state.engines.write().await.push(
            LoadedModelEntry::new("b".to_string(), engine(), Some(1)).with_replicas(vec![engine()]),
        );
        let gates = state.gates().await;
        assert_eq!(gates.len(), 3);
        assert_eq!(state.lanes().await, 3);
        let own = gates[2]
            .acquire_with_info_timeout(
                crate::api::gate::Priority::Interactive,
                "b".to_string(),
                "/test",
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert_eq!(state.request_gate.snapshot().await.in_flight, 0);
        assert_eq!(
            crate::api::gate::snapshot_of(&gates).await.in_flight,
            1,
            "a request on the model's own gate shows in the server's figures"
        );
        drop(own);
    }

    /// 2 models admit a request each at once on their own gates, and a second request of one
    /// model waits for its first, while the server's figures count all 3.
    #[tokio::test]
    async fn two_models_are_admitted_at_once_on_their_own_gates() {
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            "/nonexistent-hf-models".to_string(),
        );
        let engine = || {
            Arc::new(LlmEngine::with_config(
                crate::inference::engine::llm_engine::InferenceConfig::default(),
            ))
        };
        let (a, b) = (engine(), engine());
        {
            let mut engines = state.engines.write().await;
            engines.push(LoadedModelEntry::new("a".to_string(), a.clone(), Some(1)));
            engines.push(LoadedModelEntry::new("b".to_string(), b.clone(), Some(1)));
        }
        let admit = |engine: Arc<LlmEngine>, model: &'static str| {
            let state = state.clone();
            async move {
                state
                    .admit(
                        engine,
                        model,
                        crate::api::gate::Priority::Interactive,
                        "/test",
                        &[],
                    )
                    .await
            }
        };
        let quick = Duration::from_secs(5);
        let on_a = tokio::time::timeout(quick, admit(a.clone(), "a"))
            .await
            .expect("a is admitted at once")
            .expect("a is admitted");
        let on_b = tokio::time::timeout(quick, admit(b.clone(), "b"))
            .await
            .expect("b is admitted beside a")
            .expect("b is admitted");
        let second = tokio::spawn(admit(a.clone(), "a"));
        let mut queued = false;
        for _ in 0..100 {
            let g = crate::api::gate::snapshot_of(&state.gates().await).await;
            if g.queue_depth == 1 {
                assert_eq!(g.in_flight, 2);
                queued = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(queued, "the second request of a waits at a's gate");
        assert!(!second.is_finished(), "a admits one request at a time");
        drop(on_b);
        assert!(!second.is_finished(), "b's end does not admit a's request");
        drop(on_a);
        let again = tokio::time::timeout(quick, second)
            .await
            .expect("a's second request is admitted once its first ends")
            .unwrap()
            .expect("a is admitted");
        let g = crate::api::gate::snapshot_of(&state.gates().await).await;
        assert_eq!((g.in_flight, g.queue_depth), (1, 0));
        drop(again);
    }

    /// The figures of the F36 measure: A ends in 65 s at 6.1 tok/s, B loads in 7 s and writes
    /// 16 tokens at 96 tok/s alone and 38 tok/s on its plan beside A, which holds host blocks.
    fn f36() -> ShareInputs {
        ShareInputs {
            end_a: Some(65.0),
            rate_a: Some(6.1),
            tokens_b: Some(16.0),
            load_b: Some(7.0),
            rate_b: Some(96.0),
            rate_b_beside: Some(38.0),
            ..ShareInputs::default()
        }
    }

    /// With A's rate beside another model unknown, the F36 measure waits for A and evicts it.
    #[test]
    fn the_f36_measure_waits_and_evicts() {
        let choice = choose_share(&f36());
        assert_eq!(choice.choice, Share::WaitEvict);
        let wait_evict = choice.costs[0].unwrap();
        assert!((wait_evict - (65.0 + 7.0 + 16.0 / 96.0)).abs() < 1e-9);
        assert_eq!(choice.costs[1], None);
        assert_eq!(choice.costs[2], None, "the plan beside A holds host blocks");
    }

    /// Nothing measured waits and evicts, as a server with one gate did.
    #[test]
    fn nothing_measured_waits_and_evicts() {
        let choice = choose_share(&ShareInputs::default());
        assert_eq!(choice.choice, Share::WaitEvict);
        assert_eq!(choice.costs, [None; 3]);
    }

    /// A slowed a little beside B runs beside; A slowed much waits and evicts.
    #[test]
    fn run_beside_wins_while_a_slows_less_than_the_wait() {
        let little = choose_share(&ShareInputs {
            rate_a_beside: Some(5.5),
            ..f36()
        });
        assert_eq!(little.choice, Share::Beside);
        let beside = 7.0 + 16.0 / 38.0 + 65.0 * (6.1 / 5.5 - 1.0);
        assert!((little.costs[1].unwrap() - beside).abs() < 1e-9);
        let much = choose_share(&ShareInputs {
            rate_a_beside: Some(3.0),
            ..f36()
        });
        assert_eq!(much.choice, Share::WaitEvict);
    }

    /// The reload of A is priced by the chance A is asked for while B runs.
    #[test]
    fn the_reload_of_a_counts_as_often_as_a_is_asked_for() {
        let run = 7.0 + 16.0 / 96.0;
        let often = choose_share(&ShareInputs {
            load_a: Some(20.0),
            gap_a: Some(1.0),
            ..f36()
        });
        assert!((often.costs[0].unwrap() - (65.0 + run + 20.0)).abs() < 1e-9);
        let rare = choose_share(&ShareInputs {
            load_a: Some(20.0),
            gap_a: Some(100.0),
            ..f36()
        });
        assert!((rare.costs[0].unwrap() - (65.0 + run + run / 100.0 * 20.0)).abs() < 1e-9);
        let unknown_gap = choose_share(&ShareInputs {
            load_a: Some(20.0),
            ..f36()
        });
        assert!((unknown_gap.costs[0].unwrap() - (65.0 + run + 20.0)).abs() < 1e-9);
    }

    /// A plan beside A without host blocks on a card A shares waits without evicting when
    /// running both slows them more than the wait costs.
    #[test]
    fn a_plan_on_shared_cards_waits_when_both_slow_too_much() {
        let choice = choose_share(&ShareInputs {
            beside_on_cards: true,
            shares_card: true,
            rate_a_beside: Some(2.0),
            rate_b_beside: Some(30.0),
            load_a: Some(20.0),
            gap_a: Some(10.0),
            ..f36()
        });
        assert_eq!(choice.choice, Share::Wait);
        let wait = choice.costs[2].unwrap();
        assert!((wait - (65.0 + 7.0 + 16.0 / 96.0)).abs() < 1e-9);
        assert!(choice.costs[0].unwrap() > wait);
        assert!(choice.costs[1].unwrap() > wait);
    }

    /// A plan beside A on cards no running request uses runs beside with nothing measured.
    #[test]
    fn a_plan_on_free_cards_runs_beside_without_a_measure() {
        let alone = ShareInputs {
            beside_on_cards: true,
            ..ShareInputs::default()
        };
        assert_eq!(choose_share(&alone).choice, Share::Beside);
        let measured = choose_share(&ShareInputs {
            beside_on_cards: true,
            rate_b_beside: None,
            ..f36()
        });
        assert_eq!(measured.choice, Share::Beside);
        assert!((measured.costs[1].unwrap() - (7.0 + 16.0 / 96.0)).abs() < 1e-9);
    }

    /// An unknown input on the side of running beside never chooses it.
    #[test]
    fn an_unknown_input_beside_does_not_run_beside() {
        let known = ShareInputs {
            rate_a_beside: Some(6.0),
            ..f36()
        };
        assert_eq!(choose_share(&known).choice, Share::Beside);
        for unknown in [
            ShareInputs {
                rate_b_beside: None,
                ..known
            },
            ShareInputs {
                rate_a: None,
                ..known
            },
            ShareInputs {
                load_b: None,
                ..known
            },
            ShareInputs {
                tokens_b: None,
                ..known
            },
        ] {
            assert_ne!(choose_share(&unknown).choice, Share::Beside, "{unknown:?}");
        }
    }

    /// B of the F36 measure, left beside A: 38 tok/s on its plan, 96 tok/s in full, a load of
    /// 7 s, answers of 16 tokens, a request every 60 s and 5 minutes of keep-alive left.
    fn beside_b() -> ReplaceInputs {
        ReplaceInputs {
            load_b: Some(7.0),
            rate_now: Some(38.0),
            rate_full: Some(96.0),
            tokens_b: Some(16.0),
            gap_b: Some(60.0),
            keep_left: 300.0,
            ..ReplaceInputs::default()
        }
    }

    /// B loads again in full when the requests it expects gain more than the load costs.
    #[test]
    fn b_loads_again_when_the_gain_exceeds_the_load() {
        // 5 requests of 16 tokens gain 5 * 16 * (1/38 - 1/96) = 1.3 s: less than 7 s.
        let short = choose_replace(&beside_b());
        assert!(!short.replace, "{short:?}");
        assert!((short.requests.unwrap() - 5.0).abs() < 1e-9);
        let gain = 5.0 * 16.0 * (1.0 / 38.0 - 1.0 / 96.0);
        assert!((short.gain.unwrap() - gain).abs() < 1e-9);
        // Answers of 512 tokens gain 40 s.
        let long = choose_replace(&ReplaceInputs {
            tokens_b: Some(512.0),
            ..beside_b()
        });
        assert!(long.replace, "{long:?}");
        // A request on its way of 2048 tokens gains 34 s alone.
        let arriving = choose_replace(&ReplaceInputs {
            tokens_arriving: Some(2048.0),
            gap_b: None,
            ..beside_b()
        });
        assert!(arriving.replace, "{arriving:?}");
        assert_eq!(arriving.requests, Some(1.0));
        // A model kept loaded for ever loads again as soon as the plan in full is faster.
        let forever = choose_replace(&ReplaceInputs {
            keep_left: f64::INFINITY,
            ..beside_b()
        });
        assert!(forever.replace, "{forever:?}");
        // A plan in full no faster gains nothing.
        let same = choose_replace(&ReplaceInputs {
            rate_full: Some(38.0),
            keep_left: f64::INFINITY,
            ..beside_b()
        });
        assert_eq!(same.gain, Some(0.0));
        assert!(!same.replace);
    }

    /// An unknown load, rate or count of requests keeps B where it is.
    /// An idle model the load in full evicts adds its reload, weighed by the chance it is
    /// asked again while B holds the cards.
    #[test]
    fn an_evicted_idle_model_adds_its_reload() {
        let base = ReplaceInputs {
            load_b: Some(10.0),
            rate_now: Some(40.0),
            rate_full: Some(100.0),
            tokens_arriving: Some(1000.0),
            keep_left: 300.0,
            ..Default::default()
        };
        // 1000 tokens save 1000 * (1/40 - 1/100) = 15 s against a load of 10 s.
        assert!(choose_replace(&base).replace);
        // An idle model asked every 100 s is asked again within 300 s: its 20 s reload count.
        let busy_neighbour = ReplaceInputs {
            evicted: vec![(Some(20.0), Some(100.0))],
            ..base.clone()
        };
        assert_eq!(choose_replace(&busy_neighbour).cost, Some(30.0));
        assert!(!choose_replace(&busy_neighbour).replace);
        // One asked every 3000 s counts a tenth of its reload.
        let rare = ReplaceInputs {
            evicted: vec![(Some(20.0), Some(3000.0))],
            ..base
        };
        assert_eq!(choose_replace(&rare).cost, Some(12.0));
        assert!(choose_replace(&rare).replace);
    }

    #[test]
    fn an_unknown_input_keeps_b_where_it_is() {
        let known = ReplaceInputs {
            tokens_b: Some(512.0),
            ..beside_b()
        };
        assert!(choose_replace(&known).replace);
        for unknown in [
            ReplaceInputs {
                load_b: None,
                ..known.clone()
            },
            ReplaceInputs {
                rate_now: None,
                ..known.clone()
            },
            ReplaceInputs {
                rate_full: Some(f64::NAN),
                ..known.clone()
            },
            ReplaceInputs {
                gap_b: None,
                ..known.clone()
            },
            ReplaceInputs {
                tokens_b: None,
                ..known.clone()
            },
        ] {
            assert!(!choose_replace(&unknown).replace, "{unknown:?}");
        }
        assert_eq!(choose_replace(&ReplaceInputs::default()).requests, None);
    }

    /// The tokens of a request are its `max_tokens`, else the mean of the model's answers, else
    /// what the rate meter expects.
    #[test]
    fn the_request_max_tokens_comes_before_the_mean() {
        // A token limit bounds the answer: the mean length, when known, is what it writes.
        assert_eq!(request_tokens(Some(2048), Some(16.0), 40), Some(16.0));
        assert_eq!(request_tokens(Some(8), Some(16.0), 40), Some(8.0));
        assert_eq!(request_tokens(Some(2048), None, 40), Some(2048.0));
        assert_eq!(request_tokens(None, Some(16.0), 40), Some(16.0));
        assert_eq!(request_tokens(None, None, 40), Some(40.0));
        assert_eq!(request_tokens(None, None, 0), None);
        // A request's bound reaches the choice as the tokens B writes.
        let inputs = ShareInputs {
            tokens_b: request_tokens(Some(8), Some(16.0), 0),
            ..f36()
        };
        assert_eq!(inputs.tokens_b, Some(8.0));
    }

    /// A load records the other models and their requests, so that it is priced again once they
    /// change.
    #[test]
    fn the_neighbours_name_the_other_models_and_their_requests() {
        let engine = || {
            Arc::new(LlmEngine::with_config(
                crate::inference::engine::llm_engine::InferenceConfig::default(),
            ))
        };
        let engines = vec![
            LoadedModelEntry::new("b".to_string(), engine(), Some(1)),
            LoadedModelEntry::new("a".to_string(), engine(), Some(1)),
        ];
        assert_eq!(neighbours(&engines, "b"), vec![("a".to_string(), false)]);
        assert_eq!(neighbours(&engines, "c").len(), 2);
    }

    /// A refused request is told to come back when the model's running requests end, never
    /// sooner than the gate measured.
    #[test]
    fn the_retry_after_follows_the_expected_end() {
        assert_eq!(super::super::retry_after(Some(12.3), None), Some(13));
        assert_eq!(super::super::retry_after(Some(12.3), Some(40)), Some(40));
        assert_eq!(super::super::retry_after(Some(0.0), Some(4)), Some(4));
        assert_eq!(super::super::retry_after(None, None), None);
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

    /// A record lives from `begin` to the drop of its guard, and holds the last count.
    #[test]
    fn a_load_record_follows_the_reports_and_clears_when_the_guard_drops() {
        let loads: Arc<LoadsInFlight> = Arc::default();
        assert_eq!(loads.progress("m"), None);
        let guard = loads.begin("m");
        assert_eq!(loads.models(), vec!["m".to_string()]);
        assert_eq!(loads.progress("m").map(|p| (p.done, p.total)), Some((0, 0)));
        let report = guard.reporter();
        report("load-model", 3, 10);
        report("load-model", 4, 10);
        assert_eq!(
            loads.progress("m"),
            Some(LoadProgress {
                phase: "load-model".to_string(),
                done: 4,
                total: 10
            })
        );
        assert_eq!(loads.progress("other"), None);
        drop(guard);
        assert_eq!(loads.progress("m"), None);
        assert!(loads.models().is_empty());
        // A reporter that outlives its load writes nothing back.
        report("load-model", 5, 10);
        assert_eq!(loads.progress("m"), None);
    }

    /// A reporter or a guard from a dropped load touches nothing of a newer load of the same model.
    #[test]
    fn a_stale_reporter_and_guard_leave_the_new_record_alone() {
        let loads: Arc<LoadsInFlight> = Arc::default();
        let old = loads.begin("m");
        let stale = old.reporter();
        // The old load is cancelled from the outside while its thread keeps running.
        drop(old);
        let newer = loads.begin("m");
        stale("load-model", 7, 9);
        assert_eq!(loads.progress("m").map(|p| (p.done, p.total)), Some((0, 0)));
        newer.reporter()("load-model", 2, 9);
        stale("load-model", 8, 9);
        assert_eq!(loads.progress("m").map(|p| (p.done, p.total)), Some((2, 9)));
        drop(newer);
        assert_eq!(loads.progress("m"), None);
    }

    /// A guard dropped after a newer `begin` of the same model keeps the newer record.
    #[test]
    fn a_late_guard_drop_keeps_the_new_record() {
        let loads: Arc<LoadsInFlight> = Arc::default();
        let old = loads.begin("m");
        let newer = loads.begin("m");
        drop(old);
        assert!(loads.progress("m").is_some());
        drop(newer);
        assert_eq!(loads.progress("m"), None);
    }

    /// A load that fails or unwinds drops its guard, so the record never outlives it.
    #[test]
    fn a_load_record_clears_when_the_load_fails() {
        let loads: Arc<LoadsInFlight> = Arc::default();
        let run = |fail: bool| -> Result<(), String> {
            let guard = loads.begin("m");
            guard.reporter()("load-model", 1, 2);
            if fail {
                return Err("out of memory".to_string());
            }
            Ok(())
        };
        assert!(run(true).is_err());
        assert_eq!(loads.progress("m"), None);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = loads.begin("m");
            guard.reporter()("load-model", 1, 2);
            panic!("loader panicked");
        }));
        assert!(panicked.is_err());
        assert_eq!(loads.progress("m"), None);
        assert!(run(false).is_ok());
        assert_eq!(loads.progress("m"), None);
    }

    /// The reporter a load publishes on its blocking thread is the one the weight loaders
    /// count into, and a pool thread given a clone explicitly reaches it too.
    #[test]
    fn a_reporter_published_on_the_loading_thread_reaches_the_record() {
        use crate::inference::serve::progress::{phase, scoped};
        let loads: Arc<LoadsInFlight> = Arc::default();
        let guard = loads.begin("m");
        let report = guard.reporter();
        let seen = loads.clone();
        std::thread::spawn(move || {
            scoped::with(report, || {
                scoped::note(phase::LOAD_MODEL, 2, 5);
                let carried = scoped::current().expect("published");
                std::thread::scope(|s| {
                    s.spawn(|| scoped::note(phase::LOAD_MODEL, 9, 9));
                    s.spawn(move || carried(phase::LOAD_MODEL, 3, 5));
                });
            });
            assert!(scoped::current().is_none());
        })
        .join()
        .unwrap();
        // The thread without a clone published nothing, so only the carried count landed.
        let got = seen.progress("m").expect("still loading");
        assert_eq!(got.phase, phase::LOAD_MODEL);
        assert_eq!((got.done, got.total), (3, 5));
        drop(guard);
    }
}
