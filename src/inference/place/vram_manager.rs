//! Central VRAM authority - the ONE place placement decisions read GPU state from.
//!
//! Months of per-engine placement code accumulated the same latent bug in many shapes: each
//! engine probed NVML on its own, at its own moment, with its own reserve - and none of them
//! could see the whole system. Freed-but-pooled memory read as "used", idle residents of OTHER
//! engines read as immovable, and the planner squeezed hot models into slivers that OOMed at
//! generation time. This module replaces that patchwork with an OS-like memory authority:
//!
//! 1. [`probe`] - the single probing choke point. Trims the CUDA mempools FIRST (returning
//!    freed-but-retained memory to the driver) and then returns the throughput-ranked device
//!    list. A placement that reads through here can no longer be lied to by pool retention.
//! 2. A RECLAIM REGISTRY - engines holding evictable residents (a resident image DiT, a TTS
//!    voice) register a reclaim hook at startup. The registry knows who holds reclaimable VRAM;
//!    nothing else needs to.
//! 3. [`ensure_gpu_headroom`] - the pressure protocol, hetero-first: callers ask for headroom
//!    for a HOT component before loading. If some GPU already has room, nothing happens (idle
//!    residents are placed AROUND, never touched - the hetero contract). Only when NO GPU can
//!    host the hot component - which would otherwise crawl on CPU - are idle residents
//!    reclaimed, least-recently-used first, until one card has room. Maximum performance,
//!    never an OOM, and eviction strictly as a last resort.
//!
//! The LLM engine is deliberately NOT registered as reclaimable: its keep-alive lifecycle is
//! user-facing (Ollama semantics) and its reload cost is high. Media residents reload on their
//! next request in tens of seconds and register here.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

type ReclaimFuture = Pin<Box<dyn Future<Output = u64> + Send>>;
/// Returns an estimate of the bytes it released (0 if it held nothing).
type ReclaimHook = Box<dyn Fn() -> ReclaimFuture + Send + Sync>;
/// What this engine WOULD release, without releasing it.
type HeldFuture = Pin<Box<dyn Future<Output = u64> + Send>>;
type HeldHook = Box<dyn Fn() -> HeldFuture + Send + Sync>;

struct Reclaimer {
    hook: ReclaimHook,
    /// Bytes this engine is holding that the pressure protocol could take back.
    ///
    /// The registry could only ever ASK an engine to let go - there was no way to
    /// learn what it held without taking it. So every decision was made against free
    /// VRAM at that instant, and an idle resident read as memory that was simply gone.
    /// That is why a machine with 34 GB of cards refused a render at 7.8 GB free, and
    /// why a plan spilled layers to the host while another engine sat idle on a full
    /// card.
    held: HeldHook,
    /// Monotonic use counter value at last touch - the LRU key.
    last_used: AtomicU64,
}

static RECLAIMERS: OnceLock<Mutex<HashMap<&'static str, Reclaimer>>> = OnceLock::new();
static USE_CLOCK: AtomicU64 = AtomicU64::new(1);

fn reclaimers() -> &'static Mutex<HashMap<&'static str, Reclaimer>> {
    RECLAIMERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The single probing choke point: return pooled-but-free VRAM to the driver, then probe.
/// Fastest-first (device_probe throughput ranking); `reserve` is subtracted per card.
/// EVERY load-time placement must read GPU state through here.
/// The cards a placement may consider right now, fastest first, with the pressure the
/// fleet is under already taken off what each reports free.
///
/// [`probe`] answers what the hardware has; this answers what a placement is allowed to
/// believe. The difference is the degrade ladder: after an exhaustion the same card must
/// look smaller, or the next attempt plans exactly the placement that just failed.
pub fn probe_under_pressure(reserve: u64) -> Vec<(usize, u64, crate::tensor::Device)> {
    if vram_force_cpu() {
        // The ladder has escalated past every split worth trying. The host is slower and
        // it fits, which at this point is the only property left that matters.
        return Vec::new();
    }
    probe(reserve.saturating_add(vram_reserve_boost()))
}

pub fn probe(reserve: u64) -> Vec<(usize, u64, crate::tensor::Device)> {
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::trim_cuda_pools();
    crate::inference::place::device_probe::probe_cuda_devices(reserve)
}

/// Total FREE VRAM across every CUDA device.
///
/// The footprint of a load is the DROP in this between entering a loader and
/// installing its state. Measured rather than derived, because a model is several
/// networks on possibly different cards and a formula over one of them misses the
/// rest - which is why `/api/ps` used to publish 0 for them.
///
/// `probe` trims the pools first, so both readings see true free memory rather than
/// blocks the allocator is still holding; that is what keeps the delta honest.
///
/// Lives HERE, not in each engine: three engines had grown their own copy of this
/// sum, two of them named the same thing. Three copies of one definition is three
/// chances for them to disagree about what "free" means.
/// Bumped whenever what the cards hold changes: a model loaded, a model unloaded, the pools
/// returned. Readings derived from free VRAM are cached against it, because `free_total`
/// trims the pools before it measures - so asking it per request does not merely read a
/// counter, it hands memory back to the driver that the next forward has to take again.
static RESIDENCY_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Call after anything that changes residency. Cheap, and wrong only in the safe direction:
/// an extra bump costs one re-probe, a missing one serves a stale figure.
pub fn residency_changed() {
    RESIDENCY_EPOCH.fetch_add(1, Ordering::Relaxed);
}

pub fn residency_epoch() -> u64 {
    RESIDENCY_EPOCH.load(Ordering::Relaxed)
}

pub fn free_total() -> u64 {
    probe(0).into_iter().map(|(_, free, _)| free).sum()
}

/// What the cards hold in total, whatever is currently on them.
///
/// The CAPACITY of the machine, as against `free_total`'s "right now". A statement
/// about what the hardware can ever do needs this one: free VRAM at any instant
/// includes whatever model is resident, which a request that switches models is
/// about to give back.
pub fn total_vram() -> u64 {
    #[cfg(feature = "cuda")]
    {
        crate::inference::place::device_probe::probe_cuda_gpus(1.0)
            .iter()
            .map(|g| g.total)
            .sum()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// The biggest single card's physical capacity, or 0 with no GPU.
///
/// The ceiling on what any amount of waiting could deliver to ONE card, which is what
/// separates "busy right now" from "will always need to span devices".
pub fn largest_gpu_capacity() -> u64 {
    #[cfg(feature = "cuda")]
    {
        crate::inference::place::device_probe::probe_cuda_gpus(1.0)
            .iter()
            .map(|g| g.total)
            .max()
            .unwrap_or(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Pick the home device for a model that is about to be LOADED: the fastest card
/// whose free VRAM covers `want_bytes`, else the fastest card, else nothing (CPU).
///
/// THE FLEET RULE, and the one place it is implemented. `want_bytes` must be the
/// resident weights PLUS the runtime reserve that generation will need on the same
/// card - "fits" has to mean "will place AND run there". Ranking is by compute
/// throughput, so this returns the fastest card that fits, never the emptiest and
/// never a fixed index.
///
/// Taking `probe(...).first()` directly is the bug this exists to prevent: it picks
/// the fastest card even when it is full, and the loader then either splits a model
/// that would fit whole elsewhere or spills the whole thing to the CPU - silently,
/// with no OOM to point at it. Call this instead; do not re-implement the ranking.
pub fn pick_device_for(tag: &str, want_bytes: u64) -> Option<(usize, u64, crate::tensor::Device)> {
    pick_device_tiered(tag, want_bytes, want_bytes)
}

/// Place a small AUXILIARY model that will stay resident for the process's life.
///
/// The face detector, the recognisers and the swap generator are each a fraction of a
/// gigabyte, are never the dominant component, and - unlike an engine's resident model
/// - are held in statics that nothing can reclaim. Placing them by the fleet rule put
/// every one of them on the FASTEST card, which is also the card the image models want,
/// so they quietly took ~2 GB off the hot component and left the server retrying OOMs
/// in matmul.
///
/// So these go where there is most room, which is the opposite of the rule for a hot
/// component and correct for the same reason: they must not compete with it. This is
/// not the "most-free GPU" heuristic applied to a model's home - it is applied to
/// permanent ballast.
pub fn pick_device_for_aux(
    tag: &str,
    want_bytes: u64,
) -> Option<(usize, u64, crate::tensor::Device)> {
    let mut probe = probe(0);
    // PLACEMENT-EXEMPT: permanent ballast, not a hot component's home. The fleet rule is
    // "fastest card that fits" precisely so the component that runs every step gets the
    // fastest card; a detector that is loaded once and never freed must do the opposite,
    // or it takes that card away. Ranking by room is correct HERE and nowhere a model
    // lives - see the doc comment above.
    probe.sort_by_key(|(_, free, _)| std::cmp::Reverse(*free));
    let hit = probe.into_iter().find(|(_, free, _)| *free >= want_bytes)?;
    tracing::info!(
        "vram_manager: auxiliary '{tag}' placed on GPU{} ({:.1} GB free - the roomiest, so it \
         does not sit on the card a model wants)",
        hit.0,
        hit.1 as f64 / 1e9
    );
    Some(hit)
}

/// Place a ONE-SHOT component while leaving the hot component's card alone.
///
/// The fleet rule is "fastest card that fits", and for the dominant component that is
/// exactly right. For a component that runs once per request - a text tower, a VAE
/// decode - it is not: taking the last gigabytes of the hot component's card starves
/// the thing that runs every step. Measured on SDXL at 1024^2, the VAE took the
/// denoiser's remaining headroom and the next generation died on its first activation.
///
/// So: the fastest card that fits and is NOT `avoid`; failing that the fastest that
/// fits at all (sharing beats spilling to the host); failing that, nothing.
pub fn pick_device_elsewhere(
    tag: &str,
    want_bytes: u64,
    avoid: &crate::tensor::Device,
) -> Option<(usize, u64, crate::tensor::Device)> {
    // PLACEMENT-EXEMPT: the ordinal read here is the card to STAY OFF, handed in by the
    // caller - it is not a card being chosen by its number. The choice itself is the
    // throughput-ranked scan below, and the fallback is `pick_device_for`.
    let avoid_ord = crate::inference::place::plan::ordinal(avoid);
    let probe = probe(0);
    if let Some(hit) = probe
        .iter()
        .find(|(idx, free, _)| Some(*idx) != avoid_ord && *free >= want_bytes)
    {
        tracing::info!(
            "vram_manager: one-shot '{tag}' placed on GPU{} ({:.1} GB free), leaving {} to the \
             hot component",
            hit.0,
            hit.1 as f64 / 1e9,
            avoid_ord.map_or_else(|| "the CPU".to_string(), |o| format!("GPU{o}"))
        );
        return Some(hit.clone());
    }
    pick_device_for(tag, want_bytes)
}

/// [`pick_device_for`] with a COMFORTABLE demand and a MINIMUM viable one.
///
/// A single rigid demand makes the rule brittle at the margin: a 12.1 GB checkpoint
/// wanting a 4.3 GB reserve skipped the fastest card because it had 16.2 GB free
/// instead of 16.4, and landed on a card half as fast - trading a 2x slowdown for
/// 0.2 GB of slack it never used. Ranking is still by throughput and a card that
/// cannot run the model is still never chosen; `min_bytes` only says how much slack
/// is genuinely required, as opposed to preferred.
pub fn pick_device_tiered(
    tag: &str,
    want_bytes: u64,
    min_bytes: u64,
) -> Option<(usize, u64, crate::tensor::Device)> {
    let ranked = probe(0);
    let floor = min_bytes.min(want_bytes);
    // The fastest card that can RUN it. `pick_ranked` falls back to the fastest card
    // when nothing fits, so this is always a usable answer.
    let viable = pick_ranked(&ranked, floor)?;
    // Prefer the comfortable demand only when it does not cost throughput.
    let chosen = match pick_ranked(&ranked, want_bytes) {
        Some(c) if c <= viable => c,
        _ => viable,
    };
    let (idx, free, dev) = &ranked[chosen];
    let note = if *free >= want_bytes {
        String::new()
    } else if *free >= floor {
        format!(
            " - below the {:.1} GB preferred reserve, above the {:.1} GB it needs",
            want_bytes as f64 / 1e9,
            floor as f64 / 1e9
        )
    } else {
        " - NO card fits, fastest card used".to_string()
    };
    tracing::info!(
        "vram_manager: '{tag}' placed on GPU{idx} ({:.1} GB free, wanted {:.1} GB){note}",
        *free as f64 / 1e9,
        want_bytes as f64 / 1e9,
    );
    Some((*idx, *free, dev.clone()))
}

/// The ranking decision alone, so it can be tested without a GPU: index into
/// `ranked` (already fastest-first) of the first card that fits, else 0, else None.
fn pick_ranked<T>(ranked: &[(usize, u64, T)], want_bytes: u64) -> Option<usize> {
    if ranked.is_empty() {
        return None;
    }
    Some(
        ranked
            .iter()
            .position(|(_, free, _)| *free >= want_bytes)
            .unwrap_or(0),
    )
}

/// Run one model stage (an encoder, a vision tower, a VAE pass, ...) on the best
/// device for its `demand` in bytes, cascading on memory pressure so a stage can
/// never surface a CUDA OOM to the user.
///
/// Order of attempts:
///  1. every GPU whose CURRENT free VRAM covers `demand`, fastest-first;
///  2. every remaining GPU, fastest-first (the demand estimate may be pessimistic,
///     and pooled memory returned between attempts can change the answer);
///  3. the CPU, which always has room.
///
/// After each pressure failure the CUDA pools are released so the next candidate
/// probes TRUE free VRAM. Non-pressure errors surface immediately - a genuine bug
/// must not be masked by a silent (and much slower) CPU run.
///
/// `f` receives the device to build/run on; it must place ALL of the stage's
/// tensors there, since a partially-placed stage cannot be retried elsewhere.
pub fn run_staged<T>(
    tag: &str,
    demand: u64,
    f: impl Fn(&crate::tensor::Device) -> crate::tensor::Result<T>,
) -> crate::tensor::Result<T> {
    let ranked = probe(0);
    let mut candidates: Vec<crate::tensor::Device> = ranked
        .iter()
        .filter(|(_, free, _)| *free >= demand)
        .map(|(_, _, d)| d.clone())
        .collect();
    for (_, free, d) in &ranked {
        if *free < demand {
            candidates.push(d.clone());
        }
    }
    candidates.push(crate::tensor::Device::Cpu);

    let mut last: Option<crate::tensor::Error> = None;
    let total = candidates.len();
    for (i, dev) in candidates.into_iter().enumerate() {
        let is_cpu = dev.is_cpu();
        match f(&dev) {
            Ok(v) => {
                if i > 0 {
                    tracing::info!(
                        "vram_manager: '{tag}' ran on {:?} (candidate {}/{total}, {:.1} GB demand)",
                        dev.location(),
                        i + 1,
                        demand as f64 / 1e9
                    );
                }
                return Ok(v);
            }
            Err(e) => {
                if !e.is_oom() || is_cpu {
                    return Err(e);
                }
                tracing::warn!(
                    "vram_manager: '{tag}' hit memory pressure on {:?} ({e}); trying the next device",
                    dev.location()
                );
                #[cfg(feature = "cuda")]
                crate::inference::engine::llm_engine::release_cuda_pools();
                last = Some(e);
            }
        }
    }
    Err(last
        .unwrap_or_else(|| crate::tensor::Error::msg("run_staged: no device produced a result")))
}

/// Ask the other engines for room, from a thread that cannot await.
///
/// The pressure protocol is async because the reclaim hooks are, but the decode and prefill
/// paths run inside `spawn_blocking` - and those are exactly the places that discover, at the
/// worst moment, that a card filled up behind them. Without this they have nothing to call: the
/// prefill ladder halves its chunk down to a floor and then surfaces the OOM, while an idle
/// image model sits on gigabytes next to it waiting to be asked.
///
/// Returns whether anything was freed. A `false` here is not "no memory" - it is "nobody else
/// is holding any", which is a different situation and a caller may still want to try.
pub fn ensure_gpu_headroom_blocking(caller: &'static str, hot_bytes: u64) -> bool {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        // No runtime - a test, or a tool built on the engine directly. Nothing to ask.
        return false;
    };
    let before = free_total();
    handle.block_on(ensure_gpu_headroom(caller, hot_bytes, 0));
    free_total() > before
}

/// Register (or replace) an engine's reclaim hook. The hook is called ONLY under the pressure
/// protocol and must release the engine's idle resident (no-op when nothing is loaded),
/// returning roughly how many bytes it freed.
pub fn register_reclaimer(name: &'static str, hook: ReclaimHook, held: HeldHook) {
    let mut map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
    map.insert(
        name,
        Reclaimer {
            hook,
            held,
            last_used: AtomicU64::new(USE_CLOCK.fetch_add(1, Ordering::Relaxed)),
        },
    );
}

/// What the machine could give this caller: free VRAM PLUS what the other engines
/// would hand back if asked.
///
/// The distinction the manager did not draw. "Where does this go right now" is a
/// question about free VRAM. "Can this machine serve this at all" is a question about
/// CAPACITY, and answering it with free VRAM declines work the machine can do - the
/// resident that a request is about to replace counts against the request replacing
/// it, which is exactly backwards.
///
/// `exclude` is the caller's own engine: it is about to unload its own resident, so
/// counting that as reclaimable AND as free would count it twice.
pub async fn available_for(exclude: &str) -> u64 {
    let free: u64 = probe(0).iter().map(|(_, f, _)| *f).sum();
    let names: Vec<&'static str> = {
        let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
        map.keys().copied().filter(|n| *n != exclude).collect()
    };
    let mut reclaimable = 0u64;
    for n in names {
        // The future is built under the lock and awaited outside it: a hook that
        // touches an engine's async state would deadlock against a held mutex.
        let fut = {
            let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
            map.get(n).map(|r| (r.held)())
        };
        if let Some(f) = fut {
            reclaimable += f.await;
        }
    }
    free + reclaimable
}

/// Serialises the LOAD phase of heavy components across subsystems.
///
/// Every placement decision reads free VRAM to decide where a component goes. That
/// reading is only true while nobody else is allocating: two loaders that start
/// together both see the same free card, both plan onto it, and the second allocation
/// fails - which is how an image encoder and an LLM, requested seconds apart, OOMed on
/// the same GPU while the planner had told each of them the card was empty.
///
/// It does not reject and it does not reserve: the second loader WAITS, then plans
/// against a machine that now holds the first one. Waiting a few seconds for an honest
/// number is the whole point; the alternative is planning against a stale one.
///
/// Hold it around the ALLOCATION, never around a whole request - an image request that
/// held it while calling an LLM (the prompt enhancer does) would deadlock against the
/// LLM's own load.
static LOAD_ADMISSION: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

pub async fn load_admission() -> tokio::sync::MutexGuard<'static, ()> {
    let m = LOAD_ADMISSION.get_or_init(|| tokio::sync::Mutex::new(()));
    match m.try_lock() {
        Ok(g) => g,
        Err(_) => {
            tracing::info!(
                "vram: another component is loading - waiting for it rather than planning \
                 against VRAM it is in the middle of taking"
            );
            m.lock().await
        }
    }
}

/// VRAM a subsystem is about to need but has not allocated yet.
///
/// The admission lock covers loader against loader. It cannot cover a loader against a
/// RUNNING generation: the resident model's weights are visible to NVML, its activation
/// peak is not, so a model loading mid-render reads the card as far emptier than it is
/// about to be. A declared demand makes that peak visible for as long as the work runs.
struct Demand {
    owner: &'static str,
    bytes: u64,
}

static DEMANDS: OnceLock<Mutex<HashMap<u64, Demand>>> = OnceLock::new();
static DEMAND_ID: AtomicU64 = AtomicU64::new(1);

fn demands() -> &'static Mutex<HashMap<u64, Demand>> {
    DEMANDS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Live for as long as the work it describes; the demand disappears when it drops, so
/// an aborted or panicking request cannot leave the machine looking permanently full.
pub struct DemandLease {
    id: u64,
}

impl Drop for DemandLease {
    fn drop(&mut self) {
        demands()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

/// Announce that `owner` is about to need `bytes` of VRAM, until the lease drops.
pub fn declare_demand(owner: &'static str, bytes: u64) -> DemandLease {
    let id = DEMAND_ID.fetch_add(1, Ordering::Relaxed);
    demands()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, Demand { owner, bytes });
    DemandLease { id }
}

/// Bytes other subsystems have announced. A caller never subtracts its OWN demand -
/// it is the one about to use it, and counting it twice would spill its own work to
/// the host.
pub fn pending_demand_excluding(owner: &str) -> u64 {
    demands()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter(|d| d.owner != owner)
        .map(|d| d.bytes)
        .sum()
}

/// Watch free VRAM on one card for as long as this lives, and report the lowest reading.
///
/// What a render actually takes is only visible WHILE it runs, and the samplers that could
/// see it lived inside per-step callbacks - which not every engine has. So the measurement
/// was wired to one entry point of one variant, silently did nothing everywhere else, and
/// the demand system that was supposed to learn from it kept planning against an estimate:
/// 4.29 GB reserved at 1024^2 for a denoise that measured 2.61 GB. A reserve 60% too high
/// splits a model that fits one card, and a split runs the cards in SEQUENCE.
///
/// A watcher needs nothing from the code it measures, so it cannot be wired to some paths
/// and not others.
pub struct VramWatch {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    low: std::sync::Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VramWatch {
    /// Begin watching `gpu_index`. Returns `None` when there is nothing to watch.
    pub fn start(gpu_index: usize) -> Option<Self> {
        let start_free = free_on(gpu_index)?;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let low = std::sync::Arc::new(AtomicU64::new(start_free));
        let (s, l) = (stop.clone(), low.clone());
        let handle = std::thread::Builder::new()
            .name("vram-watch".into())
            .spawn(move || {
                // Often enough to catch the peak of a denoise step, rare enough that the
                // NVML reads are nothing against the work being measured.
                let interval = std::time::Duration::from_millis(100);
                while !s.load(Ordering::Relaxed) {
                    if let Some(free) = free_on(gpu_index) {
                        l.fetch_min(free, Ordering::Relaxed);
                    }
                    std::thread::sleep(interval);
                }
            })
            .ok()?;
        Some(Self {
            stop,
            low,
            handle: Some(handle),
        })
    }

    /// Stop watching and return `(free at the start, lowest seen)`.
    pub fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.low.load(Ordering::Relaxed)
    }
}

impl Drop for VramWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Free VRAM on one card, read WITHOUT trimming the pools.
///
/// Deliberately not [`probe`]: trimming returns freed blocks to the driver, so sampling
/// through it during a render would hand memory back a hundred times a second for a
/// number that is only being observed.
pub fn free_on(gpu_index: usize) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        crate::inference::place::device_probe::probe_cuda_gpus(1.0)
            .into_iter()
            .find(|g| g.index == gpu_index)
            .map(|g| g.stable_free)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = gpu_index;
        None
    }
}

/// Mark an engine's resident as freshly used (bumps it to most-recently-used so the pressure
/// protocol reclaims it LAST). Engines call this on every request they serve.
pub fn touch(name: &'static str) {
    let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = map.get(name) {
        r.last_used
            .store(USE_CLOCK.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// Outcome of [`ensure_gpu_headroom_within`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Headroom {
    /// Some GPU can host the hot component.
    Ready,
    /// There is no CUDA device at all - the caller's CPU path is the right one.
    NoGpu,
    /// CUDA exists but stayed full for the whole wait: every card is held by work
    /// that is IN FLIGHT, so nothing could be reclaimed.
    Busy,
    /// The cards of this machine, all of them empty, would not hold it: a matter of
    /// capacity, not occupancy, so there is no gap to wait for. The caller's plan
    /// spills to the host.
    Spills,
}

/// What every card of this machine holds together, bytes.
pub fn total_gpu_capacity() -> u64 {
    #[cfg(feature = "cuda")]
    {
        crate::inference::place::device_probe::probe_cuda_gpus(1.0)
            .iter()
            .map(|g| g.total)
            .sum()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// [`ensure_gpu_headroom`] with a bounded wait.
///
/// A resident can only be reclaimed when it is IDLE, and a chat model under steady
/// load is busy almost continuously. One pass then reports "no room", the caller
/// spills a 12 GB DiT onto the CPU, and a 1024^2 render never finishes - observed
/// live: with a chat loop running, an image request sat on the CPU path until the
/// server's own 600 s timeout killed it, while 90 chat requests sailed through.
/// The memory is not gone, it is merely busy: waiting for the gap between two
/// requests turns that into a normal GPU render.
///
/// Callers should treat [`Headroom::Busy`] as "tell the user to retry", never as
/// "load it on the CPU anyway".
pub async fn ensure_gpu_headroom_within(
    caller: &'static str,
    hot_bytes: u64,
    reserve: u64,
    max_wait: std::time::Duration,
) -> Headroom {
    if probe(0).is_empty() {
        return Headroom::NoGpu;
    }
    // "No single card fits it" is NOT "it cannot run": a model larger than one GPU
    // runs SPLIT across them, which is the hetero contract and the normal path for
    // the biggest checkpoints. Only the COMBINED free VRAM falling short means there
    // is nowhere to run.
    let combined_free = || -> u64 { probe(0).iter().map(|(_, free, _)| *free).sum() };
    let spans_devices = |why: &str, total: u64| -> Headroom {
        if total >= hot_bytes {
            tracing::info!(
                "vram_manager: no single GPU fits '{caller}' ({:.1} GB, {why}), but {:.1} GB is \
                 free across the cards - the load will span devices",
                hot_bytes as f64 / 1e9,
                total as f64 / 1e9
            );
            return Headroom::Ready;
        }
        Headroom::Busy
    };
    // WAIT FOR THE CONDITION THAT CAN ACTUALLY BECOME TRUE. When the hot component
    // exceeds the biggest card's CAPACITY, no gap between two requests will ever make
    // ONE card fit it - that answer is the same on an idle machine as on a busy one, so
    // waiting for it burned the whole deadline on every render of such a model
    // (measured: 90 s of dead wait against 16 s of actual sampling). What CAN become
    // true is the COMBINED capacity, which is what the split placement needs, so that
    // is what these models wait on. Skipping the wait entirely instead is equally
    // wrong: a request arriving while another generation holds VRAM then fails
    // outright rather than taking its turn.
    let deadline = std::time::Instant::now() + max_wait;
    if hot_bytes > total_gpu_capacity() {
        tracing::info!(
            "vram_manager: '{caller}' needs {:.1} GB and the cards hold {:.1} GB together; \
             the plan spills to the host",
            hot_bytes as f64 / 1e9,
            total_gpu_capacity() as f64 / 1e9
        );
        return Headroom::Spills;
    }
    if hot_bytes > largest_gpu_capacity() {
        let mut announced = false;
        loop {
            let total = combined_free();
            if total >= hot_bytes {
                return spans_devices("larger than any single card", total);
            }
            // Idle residents of other engines are what stands between the cards and the
            // render; they are reclaimed before any waiting, as on a single card.
            if reclaim_idle_for(caller).await > 0 {
                continue;
            }
            if std::time::Instant::now() >= deadline {
                return Headroom::Busy;
            }
            if !announced {
                tracing::info!(
                    "vram_manager: '{caller}' needs {:.1} GB across the cards and only {:.1} GB \
                     is free; waiting up to {:.0}s for a gap",
                    hot_bytes as f64 / 1e9,
                    total as f64 / 1e9,
                    max_wait.as_secs_f64()
                );
                announced = true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        }
    }
    let mut waited = false;
    loop {
        if ensure_gpu_headroom(caller, hot_bytes, reserve).await {
            if waited {
                tracing::info!("vram_manager: '{caller}' got its headroom after waiting for a gap");
            }
            return Headroom::Ready;
        }
        if std::time::Instant::now() >= deadline {
            // The cards never freed up in time; a split is still the right answer if
            // the combined capacity covers it (this rejected Z-Image on an EMPTY
            // machine when it was a single-card test).
            return spans_devices("waited for a gap and none came", combined_free());
        }
        if !waited {
            tracing::info!(
                "vram_manager: every GPU is busy; '{caller}' waits up to {:.0}s for {:.1} GB",
                max_wait.as_secs_f64(),
                hot_bytes as f64 / 1e9
            );
            waited = true;
        }
        // Long enough that a decode step or a short generation can finish, short
        // enough to catch the gap between two requests.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    }
}

/// Pressure protocol, hetero-first. Ensure at least one GPU has `hot_bytes` free (after
/// `reserve` headroom): if the post-trim probe already shows room, do nothing - idle residents
/// stay resident and the caller's hetero plan places around them. Otherwise reclaim registered
/// idle residents LEAST-recently-used first, re-probing after each, until a card fits or no
/// reclaimers remain. Returns true when a card has room; false means the caller's plan should
/// spill (hetero split / CPU) - which its OOM cascade already handles, so this can never
/// hard-fail a request.
/// Reclaim every idle resident of another engine, least recently used first, until one
/// gives something back. Returns the bytes freed, zero when nothing could be.
async fn reclaim_idle_for(caller: &'static str) -> u64 {
    let order: Vec<&'static str> = {
        let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(&'static str, u64)> = map
            .iter()
            .filter(|(n, _)| **n != caller)
            .map(|(n, r)| (*n, r.last_used.load(Ordering::Relaxed)))
            .collect();
        v.sort_by_key(|(_, t)| *t);
        v.into_iter().map(|(n, _)| n).collect()
    };
    for name in order {
        let fut = {
            let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
            match map.get(name) {
                Some(r) => (r.hook)(),
                None => continue,
            }
        };
        let freed = fut.await;
        #[cfg(feature = "cuda")]
        crate::inference::engine::llm_engine::release_cuda_pools();
        if freed > 0 {
            tracing::info!(
                "vram_manager: reclaimed idle '{name}' ({freed} component(s)) for {caller}'s hot component"
            );
            return freed;
        }
    }
    0
}

pub async fn ensure_gpu_headroom(caller: &'static str, hot_bytes: u64, reserve: u64) -> bool {
    let fits = |probe: &[(usize, u64, crate::tensor::Device)]| {
        probe.iter().any(|(_, free, _)| *free >= hot_bytes)
    };
    if fits(&probe(reserve)) {
        return true;
    }
    // LRU order over the OTHER engines' reclaimers (never reclaim the caller's own resident:
    // the caller is about to replace it itself).
    let order: Vec<&'static str> = {
        let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(&'static str, u64)> = map
            .iter()
            .filter(|(n, _)| **n != caller)
            .map(|(n, r)| (*n, r.last_used.load(Ordering::Relaxed)))
            .collect();
        v.sort_by_key(|(_, t)| *t);
        v.into_iter().map(|(n, _)| n).collect()
    };
    for name in order {
        let fut = {
            let map = reclaimers().lock().unwrap_or_else(|e| e.into_inner());
            match map.get(name) {
                Some(r) => (r.hook)(),
                None => continue,
            }
        };
        let freed = fut.await;
        // The engine just dropped its resident: the FULL release (workspaces + caches + pool
        // trim) is safe and maximally effective here - nothing of that engine is in flight.
        #[cfg(feature = "cuda")]
        crate::inference::engine::llm_engine::release_cuda_pools();
        if freed > 0 {
            // `freed` is a component COUNT from the hook, not bytes.
            tracing::info!(
                "vram_manager: reclaimed idle '{name}' ({freed} component(s)) for {caller}'s hot component"
            );
        }
        if fits(&probe(reserve)) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------------------------
// Host-RAM cache registry - the system-RAM twin of the reclaim registry above.
//
// Long-lived HOST caches (the umT5 f16 staging ~11 GB, CPU-staged VAEs, ...) speed up repeat
// renders but accumulate for the life of the process and can strangle a later CPU-spill plan
// (the host-RAM guard then refuses the segment and a load degrades or fails). Cache owners
// register a DROP hook (sync; it must try_lock and no-op if the cache is in use) with an
// estimated size; [`reclaim_host_ram`] frees registered caches, biggest first, until the
// requested amount is recovered. Dropped caches simply re-fill on their next use.
// ---------------------------------------------------------------------------------------------

type HostCacheHook = Box<dyn Fn() -> u64 + Send + Sync>;

struct HostCache {
    hook: HostCacheHook,
    est_bytes: u64,
}

static HOST_CACHES: OnceLock<Mutex<HashMap<&'static str, HostCache>>> = OnceLock::new();

fn host_caches() -> &'static Mutex<HashMap<&'static str, HostCache>> {
    HOST_CACHES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register (or refresh) a host cache. `hook` drops the cache and returns the bytes it actually
/// freed (0 when empty or currently in use - use try_lock inside). Idempotent.
pub fn register_host_cache(name: &'static str, est_bytes: u64, hook: HostCacheHook) {
    host_caches()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name, HostCache { hook, est_bytes });
}

/// Free registered host caches, biggest estimate first, until ~`needed_bytes` are recovered
/// (or the registry is exhausted). Returns the total bytes the hooks reported freeing. Safe to
/// call from sync contexts (the layer-planner's host-RAM guard).
pub fn reclaim_host_ram(needed_bytes: u64) -> u64 {
    let order: Vec<&'static str> = {
        let map = host_caches().lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(&'static str, u64)> = map.iter().map(|(n, c)| (*n, c.est_bytes)).collect();
        v.sort_by_key(|(_, b)| std::cmp::Reverse(*b));
        v.into_iter().map(|(n, _)| n).collect()
    };
    let mut freed = 0u64;
    for name in order {
        if freed >= needed_bytes {
            break;
        }
        let hook_freed = {
            let map = host_caches().lock().unwrap_or_else(|e| e.into_inner());
            match map.get(name) {
                Some(c) => (c.hook)(),
                None => 0,
            }
        };
        if hook_freed > 0 {
            tracing::info!(
                "vram_manager: dropped host cache '{name}' (~{:.1} GB) to relieve RAM pressure",
                hook_freed as f64 / 1e9
            );
            freed += hook_freed;
        }
    }
    freed
}

#[cfg(test)]
mod concurrent_admission_tests {
    use super::*;

    /// The whole point of the ledger: a subsystem must not plan AROUND its own demand.
    /// Subtracting it from itself is not conservative, it spills the very work that
    /// declared it onto the host.
    #[test]
    fn a_declared_demand_is_invisible_to_its_own_owner() {
        let _lease = declare_demand("media-test-owner", 4 << 30);
        assert_eq!(pending_demand_excluding("media-test-owner"), 0);
        assert_eq!(pending_demand_excluding("llm-test-owner"), 4 << 30);
    }

    /// A request that dies mid-render must not leave the machine looking permanently
    /// full - which is what any ledger keyed on anything but a guard would do.
    #[test]
    fn a_demand_disappears_with_its_lease() {
        let before = pending_demand_excluding("someone-else");
        {
            let _lease = declare_demand("transient-owner", 1 << 30);
            assert_eq!(pending_demand_excluding("someone-else"), before + (1 << 30));
        }
        assert_eq!(pending_demand_excluding("someone-else"), before);
    }

    /// Two loaders must not both be told the card is free. The second one WAITS - it is
    /// never refused, because a refusal is a failed request where a wait is a slow one.
    #[tokio::test]
    async fn the_admission_lock_serialises_two_loaders() {
        let first = load_admission().await;
        let second = tokio::spawn(async {
            let _g = load_admission().await;
            true
        });
        // While the first holds it, the second cannot have finished.
        tokio::task::yield_now().await;
        assert!(
            !second.is_finished(),
            "the second loader planned while the first allocated"
        );
        drop(first);
        assert!(
            second.await.unwrap(),
            "the second loader never got its turn"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::pick_ranked;

    /// `ranked` is fastest-first, so index 0 is the fastest card - which is exactly
    /// the one a naive `.first()` would take even when it is full.
    fn ranked(free: &[u64]) -> Vec<(usize, u64, ())> {
        free.iter().enumerate().map(|(i, f)| (i, *f, ())).collect()
    }

    #[test]
    fn picks_the_fastest_card_that_fits_not_the_fastest_card() {
        // The Flux regression, exactly: fastest card 10.3 GB free, second 16.5 GB,
        // a 12.1 GB checkpoint + 4.3 GB reserve. `.first()` -> GPU0 -> split -> CPU.
        let r = ranked(&[10_300_000_000, 16_500_000_000]);
        assert_eq!(pick_ranked(&r, 16_400_000_000), Some(1));
    }

    #[test]
    fn prefers_the_fastest_card_when_it_fits() {
        let r = ranked(&[16_000_000_000, 24_000_000_000]);
        assert_eq!(pick_ranked(&r, 8_000_000_000), Some(0));
    }

    #[test]
    fn falls_back_to_the_fastest_card_when_none_fits() {
        // No card holds it whole: the loader still needs a home to split or spill from,
        // and the fastest card is the right base for that decision.
        let r = ranked(&[10_000_000_000, 12_000_000_000]);
        assert_eq!(pick_ranked(&r, 40_000_000_000), Some(0));
    }

    #[test]
    fn no_gpu_means_no_pick() {
        let r = ranked(&[]);
        assert_eq!(pick_ranked(&r, 1), None);
    }
}

/// THE INVARIANT GATE for the placement rule above.
///
/// Three separate loaders (AWQ, Flux, Boogu) shipped the same bug - "take the fastest
/// GPU" without asking whether the model fits on it - and each was found only after a
/// user hit a symptom, because a bad placement does not raise an error: it splits, or
/// silently spills to the CPU, and just runs slowly or wrongly. Convention plus review
/// did not catch it three times, so it is a test now.
///
/// A load-time placement must go through [`pick_device_for`]. Reading `probe()` and
/// taking the first entry is only legitimate when it is NOT a placement, and then it
/// must say so with a `PLACEMENT-EXEMPT:` comment stating why.
/// A CPU fallback must be justified by a MEASUREMENT or by a real failure - never by
/// a constant.
///
/// Three separate bugs of this exact shape, each found only because a user noticed
/// something was slow:
///  * the Z-Image VAE, placed on the CPU by a load-time demand that charged the whole
///    decode peak, and never re-checked for the process's life (116 s of a 176 s
///    request);
///  * the Flux EDIT decode, which read `if width.max(height) > 512 -> CPU`
///    unconditionally, with both cards empty;
///  * the Wan video decoder, disabled for a whole render by one clip's OOM.
///
/// None raised an error. A wrong fallback never does - it just runs on the host and
/// looks like the model being slow, which is why review kept missing it and why it is
/// a test now.
///
/// So: any call into a resident `*_cpu` decoder/encoder must have a free-VRAM probe or
/// an error branch within sight of it. A size threshold is not a reason.
#[cfg(test)]
mod cpu_fallback_invariant {
    use std::path::Path;

    /// How far around the call a justification may sit. Wide enough for a cascade
    /// spread over a match, tight enough that an unrelated `Err(` elsewhere in a long
    /// function does not launder an unjustified fallback.
    const WINDOW: usize = 16;

    fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                rust_files(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    fn is_cpu_fallback_call(line: &str) -> bool {
        let t = line.trim_start();
        if t.starts_with("//") || t.starts_with("///") {
            return false;
        }
        [
            "_cpu.decode(",
            "_cpu.decode_",
            "_cpu.encode(",
            "_cpu().decode(",
            "_cpu.forward(",
        ]
        .iter()
        .any(|pat| line.contains(pat))
    }

    /// A probe, an OOM test, or an error branch - any of these means the fallback was
    /// reached by finding out rather than by assuming.
    fn justified(window: &[&str]) -> bool {
        window.iter().any(|l| {
            l.contains("probe(")
                || l.contains("is_oom")
                || l.contains("vram_manager")
                || l.contains("Err(")
                || l.contains("CPU-FALLBACK-OK")
        })
    }

    #[test]
    fn no_cpu_fallback_is_chosen_without_measuring_or_failing_first() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        let mut offenders = Vec::new();
        for f in files {
            let Ok(src) = std::fs::read_to_string(&f) else {
                continue;
            };
            let lines: Vec<&str> = src.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if !is_cpu_fallback_call(line) {
                    continue;
                }
                let lo = i.saturating_sub(WINDOW);
                let hi = (i + WINDOW).min(lines.len());
                if !justified(&lines[lo..hi]) {
                    offenders.push(format!("{}:{}: {}", f.display(), i + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a CPU fallback is being chosen without a free-VRAM probe or an error \
             branch near it - a size threshold is not a reason, because the host decode \
             it selects is minutes where the device is seconds. Probe, or mark the line \
             `CPU-FALLBACK-OK: <why>`.\n{}",
            offenders.join("\n")
        );
    }
}

#[cfg(test)]
mod placement_invariant {
    use std::path::Path;

    fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                rust_files(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    #[test]
    fn no_placement_takes_the_fastest_gpu_without_checking_that_it_fits() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        // Binaries under src/bin are probes and benchmarks: they place by intent, on a
        // quiet machine, and are not part of the served fleet.
        files.retain(|p| !p.components().any(|c| c.as_os_str() == "bin"));

        let mut offenders = Vec::new();
        for path in &files {
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if !line.contains("vram_manager::probe(") && !line.contains(" probe(") {
                    continue;
                }
                // The choke point and this gate are allowed to name it.
                if path.ends_with("vram_manager.rs") {
                    continue;
                }
                let exempt = lines[i.saturating_sub(5)..=i]
                    .iter()
                    .any(|l| l.contains("PLACEMENT-EXEMPT"));
                if exempt {
                    continue;
                }
                // A probe result is consumed either by a chained expression right here,
                // or through a binding used further down the function - the second shape
                // is how a fit check and its `.first()` drift apart until nobody sees
                // that they contradict each other.
                let binding = line
                    .split_once("let ")
                    .map(|(_, rest)| {
                        rest.trim_start_matches("mut ")
                            .split(|c: char| !c.is_alphanumeric() && c != '_')
                            .next()
                            .unwrap_or("")
                            .to_string()
                    })
                    .filter(|b| !b.is_empty());
                let end = (i + binding.as_ref().map_or(4, |_| 40)).min(lines.len());
                let window = lines[i..end].join(" ");
                let takes_first = match &binding {
                    // Only flag `.first()`/`.next()` reached through THIS binding.
                    Some(b) => {
                        window.contains(&format!("{b}.first()"))
                            || window.contains(&format!("{b}.iter().next()"))
                            || window.contains(&format!("{b}.into_iter().next()"))
                            || lines[i..(i + 4).min(lines.len())]
                                .join(" ")
                                .contains(".first()")
                    }
                    None => window.contains(".first()") || window.contains(".next()"),
                };
                // A free-VRAM comparison in the same region means the site does filter;
                // iterating every card (no .first()/.next()) is fine too.
                let checks_fit = window.contains(">=") || window.contains(" < ");
                if takes_first && !checks_fit {
                    offenders.push(format!(
                        "{}:{}",
                        path.strip_prefix(&src).unwrap_or(path).display(),
                        i + 1
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these sites take the fastest GPU without checking that the model fits:\n  {}\n\
             Use vram_manager::pick_device_for(tag, weights + runtime_reserve), or add a \
             `PLACEMENT-EXEMPT: <why this is not a placement>` comment above the probe.",
            offenders.join("\n  ")
        );
    }
}

#[cfg(test)]
mod tiered_tests {
    use super::pick_ranked;

    fn ranked(free: &[u64]) -> Vec<(usize, u64, ())> {
        free.iter().enumerate().map(|(i, f)| (i, *f, ())).collect()
    }

    /// The decision `pick_device_tiered` makes, without the probe: prefer the
    /// comfortable demand only when it does not cost throughput.
    fn tiered(free: &[u64], want: u64, min: u64) -> usize {
        let r = ranked(free);
        let viable = pick_ranked(&r, min.min(want)).unwrap();
        match pick_ranked(&r, want) {
            Some(c) if c <= viable => c,
            _ => viable,
        }
    }

    #[test]
    fn a_marginal_reserve_never_costs_the_faster_card() {
        // The rayflux regression: 12.1 GB checkpoint, 4.3 GB preferred reserve, fast
        // card 16.2 GB free and slow card 16.5. Preferring the reserve moved it to a
        // card half as fast and doubled the denoise.
        assert_eq!(
            tiered(
                &[16_200_000_000, 16_500_000_000],
                16_400_000_000,
                15_100_000_000
            ),
            0
        );
    }

    #[test]
    fn the_faster_card_still_wins_when_it_fits_comfortably() {
        assert_eq!(
            tiered(
                &[20_000_000_000, 16_500_000_000],
                16_400_000_000,
                15_100_000_000
            ),
            0
        );
    }

    #[test]
    fn a_card_that_cannot_run_the_model_is_never_chosen_over_one_that_can() {
        // Fast card holds the weights but not the minimum reserve: the slower card wins.
        assert_eq!(
            tiered(
                &[12_500_000_000, 16_500_000_000],
                16_400_000_000,
                15_100_000_000
            ),
            1
        );
    }

    #[test]
    fn when_nothing_fits_the_fastest_card_is_the_base_for_splitting() {
        assert_eq!(
            tiered(
                &[8_000_000_000, 9_000_000_000],
                16_400_000_000,
                15_100_000_000
            ),
            0
        );
    }
}

// ============================================================================
// PRESSURE ESCALATION
// ============================================================================
//
// How much a placement gives up after an exhaustion, so the next attempt spills across
// cards instead of re-packing the one that just failed. Read by the language engine's
// retry loop and by every media loader, which is why it lives with the VRAM authority
// rather than in the family whose loader first needed it.

/// Process-global VRAM-degradation level for the ACE-Step render. Zero on the
/// no-pressure path (placement unchanged -> numerics bit-identical). A render stage
/// that hits CUDA OOM under memory pressure (e.g. a concurrent VRAM consumer) bumps
/// this so the NEXT (re)placement reserves more headroom or falls back to CPU  -
/// graceful degradation instead of a crash. Read by `probe_under_pressure` (shared by
/// the LM / DiT / VAE placement probes). This is automatic internal state, not an
/// env-var knob.
static VRAM_DEGRADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Escalate degradation one notch and return the new level. Each notch widens the
/// per-card reserve so the next placement spills layers across GPU0->GPU1->CPU (a real
/// multi-GPU split); only once the GPUs are exhausted (see `vram_force_cpu`) does
/// placement fall back to all-CPU (the guaranteed-to-fit floor).
pub fn vram_degrade() -> u64 {
    VRAM_DEGRADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// Extra reserve bytes the degradation level adds to a placement probe. Sized so each
/// degrade notch shrinks every card's usable budget enough to push a chunk of layers
/// onto the next device - first GPU0->GPU1 (a real 2-GPU split), then GPU->CPU - rather
/// than re-packing the starved GPU0 and OOM-looping: the placement estimate omits the
/// runtime peak (activations + cuBLAS workspace + graph arena, several GB), so a coarse
/// reserve is what actually moves layers off a card.
/// Current degradation level (0 = none).
pub fn vram_degrade_level() -> u64 {
    VRAM_DEGRADE.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn vram_reserve_boost() -> u64 {
    // The ladder of reserves an escalation walks. Its table lives with the audio
    // demand model; without that family the escalation still has to move, so it falls
    // back to doubling - coarse, and coarse is what makes layers leave a card.
    #[cfg(feature = "audio")]
    {
        crate::inference::place::audio_demand::degrade_reserve_bytes(vram_degrade_level())
    }
    #[cfg(not(feature = "audio"))]
    {
        (1u64 << 30) * vram_degrade_level()
    }
}

/// Whether degradation has escalated past the multi-GPU-split attempts to forcing
/// all-CPU placement. Deferred well past the first OOM so the DiT/LM/VAE first spread
/// across every GPU (HeteroPlan greedy-fill under the escalating reserve); only when
/// even a 2-GPU split can't hold the weights do layers land on CPU.
pub fn vram_force_cpu() -> bool {
    vram_degrade_level() >= 4
}

/// Reset the degradation level to 0 (optimistic placement). Called at the START of a
/// render invocation so each run begins at the pack-first no-pressure path; a fresh
/// process already starts at 0, but this makes multi-clip single-process renders (which
/// load models once and never re-enter) explicitly begin un-degraded.
pub fn vram_degrade_reset() {
    VRAM_DEGRADE.store(0, std::sync::atomic::Ordering::Relaxed);
}
