//! Energy + carbon sampler for one timed decode window.
//!
//! Measures the energy drawn over a window (start -> stop) across three domains
//! so the bench can rank engines by joules-per-generated-token:
//!
//!   * GPU   - NVML `nvmlDeviceGetTotalEnergyConsumption` (a monotonic mJ
//!             counter, summed over every CUDA device). When the driver/GPU
//!             doesn't expose that counter we fall back to integrating
//!             `power.draw` sampled at ~10 Hz (P.dt). The path actually used
//!             is recorded in `gpu_path`.
//!   * CPU   - Intel RAPL package energy: the `intel-rapl:0` (+ `:1`, ...)
//!             `energy_uj` counters under /sys/class/powercap. Wraparound is
//!             handled via `max_energy_range_uj`.
//!   * DRAM  - the RAPL `dram` subdomain (`intel-rapl:N:*` where name=="dram"),
//!             when present.
//!
//! RAPL `energy_uj` is root-only on recent kernels (a perf side-channel
//! mitigation). When it can't be read we degrade gracefully: GPU energy is
//! still reported and `cpu_rapl_available` is set false with a note. To enable
//! CPU/DRAM energy without running the bench as root, grant read on the
//! powercap counters, e.g.:
//!
//!   sudo setcap cap_dac_read_search+ep ./assay           # not reliable for sysfs
//!   # or, more robustly, a udev rule / one-shot chmod:
//!   sudo chmod -R a+r /sys/class/powercap/intel-rapl:*/energy_uj
//!   # (resets on reboot; for persistence add a tmpfiles.d/udev rule)
//!
//! All fields are optional/zero-cost: when no counter is available the window
//! simply reports `domains_counted` empty and the per-token energy is None.

use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

/// Default grid carbon intensity (gCO2eq / kWh). A documented global-average
/// placeholder - only affects the *absolute* gCO2/token number, not the J/token
/// ranking (CI is a common multiplicative factor across engines). Override with
/// `--carbon-intensity` for your region (e.g. France ~50, world avg ~480,
/// coal-heavy grid ~800).
pub const DEFAULT_CARBON_INTENSITY: f64 = 480.0;

/// Which GPU-energy measurement path was used for a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuEnergyPath {
    /// NVML `nvmlDeviceGetTotalEnergyConsumption` monotonic counter (preferred).
    NvmlCounter,
    /// Integrated `power.draw` (P.dt) sampled at ~10 Hz (fallback).
    PowerIntegration,
    /// No GPU energy available (no NVML, no GPU).
    None,
}

/// Energy drawn over one timed window, plus the derived per-token metrics.
#[derive(Debug, Clone, Serialize)]
pub struct EnergyWindow {
    /// GPU energy over the window (J), summed across all devices.
    pub gpu_energy_j: f64,
    /// CPU package energy over the window (J), summed across all packages.
    pub cpu_pkg_energy_j: f64,
    /// DRAM energy over the window (J), summed across all dram subdomains.
    pub dram_energy_j: f64,
    /// Total counted energy = sum of the available domains (J).
    pub energy_j: f64,
    /// Window duration (s) - for cross-checking against power.
    pub duration_s: f64,
    /// Which GPU path produced `gpu_energy_j`.
    pub gpu_path: GpuEnergyPath,
    /// True if RAPL package energy was readable this window.
    pub cpu_rapl_available: bool,
    /// True if a RAPL dram subdomain was readable this window.
    pub dram_rapl_available: bool,
    /// Human-readable list of domains that contributed to `energy_j`
    /// (e.g. ["gpu", "cpu_pkg", "dram"]).
    pub domains_counted: Vec<String>,
    /// Note explaining any unavailable domain (e.g. RAPL needs root).
    pub note: Option<String>,
    /// GPU energy over the DECODE phase only (J) - the window from the first
    /// generated token to the end, excluding prefill. `None` when no decode
    /// boundary was supplied or the time-series was empty. This is the honest
    /// per-token decode energy: it is immune to the prompt-cache confound (an
    /// engine that serves a repeated long prompt from cache pays ~0 prefill on
    /// warm iters, which would otherwise deflate its full-window J/tok).
    pub gpu_decode_j: Option<f64>,
    /// CPU package energy over the decode phase only (J).
    pub cpu_pkg_decode_j: Option<f64>,
    /// DRAM energy over the decode phase only (J).
    pub dram_decode_j: Option<f64>,
    /// Total counted decode-phase energy (J) = sum of available decode domains.
    pub decode_energy_j: Option<f64>,
    /// Decode-phase duration (s) = window end - first-token time.
    pub decode_duration_s: Option<f64>,
}

impl EnergyWindow {
    /// Energy per generated token (J/tok) over this window.
    pub fn j_per_tok(&self, tokens: u64) -> Option<f64> {
        if tokens == 0 || self.energy_j <= 0.0 {
            return None;
        }
        Some(self.energy_j / tokens as f64)
    }

    /// Energy per generated token in watt-hours (Wh/tok).
    pub fn wh_per_tok(&self, tokens: u64) -> Option<f64> {
        self.j_per_tok(tokens).map(|j| j / 3600.0)
    }

    /// Carbon per generated token (gCO2eq/tok) for the given grid intensity
    /// (gCO2/kWh). gCO2/tok = Wh/tok x CI / 1000.
    pub fn gco2_per_tok(&self, tokens: u64, carbon_intensity: f64) -> Option<f64> {
        self.wh_per_tok(tokens)
            .map(|wh| wh * carbon_intensity / 1000.0)
    }
}

/// Lightweight **synchronous** energy meter: snapshot CPU RAPL + GPU NVML energy
/// counters at `start()`, diff at `stop()`. No background threads, no tokio runtime
/// - usable directly inside async request handlers *and* sync CLI render bins.
///
/// Unlike [`EnergySampler`] it does NOT do the prefill/decode split, and it relies
/// on the NVML *energy counter* (no power-integration fallback). That counter is
/// present on modern GPUs; when absent, GPU energy is reported as unavailable.
// Cache the RAPL domain discovery + a single NVML handle so a per-request meter
// pays only two sysfs reads + one cached-handle counter read - no `Nvml::init()`
// or sysfs directory walk per request (keeps the instrumentation ~free).
static RAPL_CACHE: OnceLock<(Vec<RaplDomain>, bool)> = OnceLock::new();
static NVML_CACHE: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
fn rapl_domains_cached() -> &'static (Vec<RaplDomain>, bool) {
    RAPL_CACHE.get_or_init(discover_rapl)
}
fn nvml_cached() -> Option<&'static nvml_wrapper::Nvml> {
    NVML_CACHE
        .get_or_init(|| nvml_wrapper::Nvml::init().ok())
        .as_ref()
}

pub struct EnergyMeter {
    t0: Instant,
    rapl_start: Vec<Option<u128>>,
    gpu_start_j: Option<f64>,
}

impl EnergyMeter {
    /// Begin a measurement window (cheap: cached RAPL domains + cached NVML handle).
    pub fn start() -> Self {
        let (domains, _) = rapl_domains_cached();
        let rapl_start = rapl_snapshot(domains);
        let gpu_start_j = nvml_cached().and_then(nvml_counter_sum_j);
        Self {
            t0: Instant::now(),
            rapl_start,
            gpu_start_j,
        }
    }

    /// Close the window and return the energy drawn across the available domains.
    pub fn stop(self) -> EnergyWindow {
        let duration_s = self.t0.elapsed().as_secs_f64();
        let (domains, rapl_readable) = rapl_domains_cached();
        let rapl_end = rapl_snapshot(domains);
        let (pkg_j, dram_j) = rapl_delta_j(domains, &self.rapl_start, &rapl_end);
        let gpu_end_j = nvml_cached().and_then(nvml_counter_sum_j);
        let (gpu_energy_j, gpu_path) = match (self.gpu_start_j, gpu_end_j) {
            (Some(s), Some(e)) => ((e - s).max(0.0), GpuEnergyPath::NvmlCounter),
            _ => (0.0, GpuEnergyPath::None),
        };
        let cpu_rapl_available =
            *rapl_readable && domains.iter().any(|d| d.kind == RaplKind::Package);
        let dram_rapl_available =
            *rapl_readable && domains.iter().any(|d| d.kind == RaplKind::Dram);
        let mut domains_counted = Vec::new();
        if gpu_path == GpuEnergyPath::NvmlCounter {
            domains_counted.push("gpu".to_string());
        }
        if cpu_rapl_available {
            domains_counted.push("cpu_pkg".to_string());
        }
        if dram_rapl_available {
            domains_counted.push("dram".to_string());
        }
        let energy_j = gpu_energy_j + pkg_j + dram_j;
        EnergyWindow {
            gpu_energy_j,
            cpu_pkg_energy_j: pkg_j,
            dram_energy_j: dram_j,
            energy_j,
            duration_s,
            gpu_path,
            cpu_rapl_available,
            dram_rapl_available,
            domains_counted,
            note: None,
            gpu_decode_j: None,
            cpu_pkg_decode_j: None,
            dram_decode_j: None,
            decode_energy_j: None,
            decode_duration_s: None,
        }
    }
}

// --- RAPL ---------------------------------------------------------------------

/// One RAPL energy domain: its `energy_uj` file, wrap range, and a label.
#[derive(Clone)]
struct RaplDomain {
    energy_uj_path: PathBuf,
    max_range_uj: u128,
    /// "cpu_pkg" or "dram"
    kind: RaplKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RaplKind {
    Package,
    Dram,
}

/// Read a u128 from a sysfs file, trimming whitespace.
fn read_u128(path: &std::path::Path) -> Option<u128> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u128>()
        .ok()
}

/// Discover all RAPL package + dram domains under /sys/class/powercap.
/// Returns (domains, readable) - `readable` is false if energy_uj exists but is
/// permission-denied (the root-only mitigation).
fn discover_rapl() -> (Vec<RaplDomain>, bool) {
    let base = std::path::Path::new("/sys/class/powercap");
    let mut domains = Vec::new();
    let mut any_eacces = false;
    let mut any_readable = false;
    let Ok(entries) = std::fs::read_dir(base) else {
        return (domains, false);
    };
    // Top-level packages look like intel-rapl:0, intel-rapl:1, ...
    for e in entries.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("intel-rapl:") || name.matches(':').count() != 1 {
            continue;
        }
        let pkg_dir = e.path();
        // The package domain itself.
        push_domain(
            &pkg_dir,
            RaplKind::Package,
            &mut domains,
            &mut any_eacces,
            &mut any_readable,
        );
        // Sub-domains intel-rapl:N:M - look for one named "dram".
        if let Ok(subs) = std::fs::read_dir(&pkg_dir) {
            for s in subs.flatten() {
                let sname = s.file_name();
                let sname = sname.to_string_lossy();
                if !sname.starts_with("intel-rapl:") {
                    continue;
                }
                let sub_dir = s.path();
                let label = std::fs::read_to_string(sub_dir.join("name")).unwrap_or_default();
                if label.trim() == "dram" {
                    push_domain(
                        &sub_dir,
                        RaplKind::Dram,
                        &mut domains,
                        &mut any_eacces,
                        &mut any_readable,
                    );
                }
            }
        }
    }
    // "readable" means at least one counter could actually be read. If we found
    // domains but every read hit EACCES, report unreadable so callers degrade.
    let readable = any_readable && !domains.is_empty();
    let _ = any_eacces;
    (domains, readable)
}

fn push_domain(
    dir: &std::path::Path,
    kind: RaplKind,
    out: &mut Vec<RaplDomain>,
    any_eacces: &mut bool,
    any_readable: &mut bool,
) {
    let energy_uj_path = dir.join("energy_uj");
    if !energy_uj_path.exists() {
        return;
    }
    // Probe readability now so we can report a clean "needs root" flag.
    match std::fs::read_to_string(&energy_uj_path) {
        Ok(_) => *any_readable = true,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            *any_eacces = true;
        }
        Err(_) => {}
    }
    let max_range_uj = read_u128(&dir.join("max_energy_range_uj")).unwrap_or(u128::MAX);
    out.push(RaplDomain {
        energy_uj_path,
        max_range_uj,
        kind,
    });
}

/// Read all domains' current energy (µJ). None entries are domains that failed.
fn rapl_snapshot(domains: &[RaplDomain]) -> Vec<Option<u128>> {
    domains
        .iter()
        .map(|d| read_u128(&d.energy_uj_path))
        .collect()
}

/// Delta between two RAPL snapshots in joules, split by kind, handling wrap.
fn rapl_delta_j(
    domains: &[RaplDomain],
    start: &[Option<u128>],
    end: &[Option<u128>],
) -> (f64, f64) {
    let mut pkg_j = 0.0;
    let mut dram_j = 0.0;
    for (i, d) in domains.iter().enumerate() {
        let (Some(s), Some(e)) = (
            start.get(i).copied().flatten(),
            end.get(i).copied().flatten(),
        ) else {
            continue;
        };
        let delta_uj = if e >= s {
            e - s
        } else {
            // Counter wrapped: add the range back.
            d.max_range_uj.saturating_sub(s) + e
        };
        let j = delta_uj as f64 / 1_000_000.0;
        match d.kind {
            RaplKind::Package => pkg_j += j,
            RaplKind::Dram => dram_j += j,
        }
    }
    (pkg_j, dram_j)
}

// --- NVML GPU energy ----------------------------------------------------------

/// Sum of per-device total-energy counters (mJ -> J). Returns None if NVML or
/// the energy counter is unavailable (so the caller can fall back to power).
fn nvml_energy_snapshot_j() -> Option<f64> {
    use nvml_wrapper::Nvml;
    let nvml = Nvml::init().ok()?;
    let count = nvml.device_count().ok()?;
    if count == 0 {
        return None;
    }
    let mut total_mj: u128 = 0;
    let mut any = false;
    for i in 0..count {
        if let Ok(dev) = nvml.device_by_index(i) {
            if let Ok(mj) = dev.total_energy_consumption() {
                total_mj += mj as u128;
                any = true;
            }
        }
    }
    if any {
        Some(total_mj as f64 / 1000.0)
    } else {
        None
    }
}

// --- Sampler ------------------------------------------------------------------

/// Active energy measurement over one window. Construct with `start`, then call
/// `stop` to obtain the `EnergyWindow`.
/// One cumulative-energy sample (deltas from window start), used to split the
/// window into prefill and decode phases post-hoc at the first-token boundary.
#[derive(Clone, Copy)]
struct EnergyTrace {
    t_s: f64,
    gpu_j: f64,
    pkg_j: f64,
    dram_j: f64,
}

pub struct EnergySampler {
    start_instant: Instant,
    // RAPL
    rapl_domains: Vec<RaplDomain>,
    rapl_readable: bool,
    rapl_start: Vec<Option<u128>>,
    // GPU NVML counter path
    gpu_start_j: Option<f64>,
    // GPU power-integration fallback (only spawned when the counter is absent)
    power_stop: Option<Arc<AtomicBool>>,
    power_handle: Option<JoinHandle<f64>>,
    // Lightweight cumulative time-series for the prefill/decode split (one NVML
    // init reused across ticks + RAPL snapshots). Runs only on the NVML-counter
    // path; on the power-integration fallback the series stays empty and the
    // decode split is unavailable (full-window energy is still reported).
    trace: Arc<Mutex<Vec<EnergyTrace>>>,
    trace_stop: Arc<AtomicBool>,
    trace_handle: Option<JoinHandle<()>>,
}

impl EnergySampler {
    /// Begin a measurement window. `power_interval_ms` is the fallback
    /// power-integration sample period (~100ms = 10 Hz) - only used if the NVML
    /// energy counter is unavailable.
    pub fn start(power_interval_ms: u64) -> Self {
        let start_instant = Instant::now();
        let (rapl_domains, rapl_readable) = discover_rapl();
        let rapl_start = rapl_snapshot(&rapl_domains);

        let gpu_start_j = nvml_energy_snapshot_j();

        // Only run the power-integration thread when we have no energy counter.
        let (power_stop, power_handle) = if gpu_start_j.is_none() {
            let stop = Arc::new(AtomicBool::new(false));
            let stop_c = stop.clone();
            let h = tokio::task::spawn_blocking(move || {
                power_integrate_loop(stop_c, power_interval_ms)
            });
            (Some(stop), Some(h))
        } else {
            (None, None)
        };

        // Cumulative time-series for the prefill/decode split - only meaningful on
        // the NVML-counter path (deterministic monotonic energy). One NVML init
        // is reused across ticks; RAPL is re-snapshotted each tick.
        let trace = Arc::new(Mutex::new(Vec::new()));
        let trace_stop = Arc::new(AtomicBool::new(false));
        let trace_handle = if gpu_start_j.is_some() {
            let trace_c = trace.clone();
            let stop_c = trace_stop.clone();
            let domains_c = rapl_domains.clone();
            let rapl_start_c = rapl_start.clone();
            let gpu0 = gpu_start_j.unwrap_or(0.0);
            Some(tokio::task::spawn_blocking(move || {
                energy_trace_loop(
                    stop_c,
                    trace_c,
                    power_interval_ms,
                    start_instant,
                    gpu0,
                    domains_c,
                    rapl_start_c,
                )
            }))
        } else {
            None
        };

        Self {
            start_instant,
            rapl_domains,
            rapl_readable,
            rapl_start,
            gpu_start_j,
            power_stop,
            power_handle,
            trace,
            trace_stop,
            trace_handle,
        }
    }

    /// Close the window and compute the energy summary. `decode_start_s` is the
    /// wall-clock time (from window start) of the first generated token; when
    /// supplied, the decode-phase energy (first token -> end) is computed from the
    /// time-series, excluding prefill. Pass `None` (or on the power-integration
    /// fallback) to skip the split - full-window energy is always reported.
    pub async fn stop(mut self, decode_start_s: Option<f64>) -> EnergyWindow {
        let duration_s = self.start_instant.elapsed().as_secs_f64();

        // Stop and collect the time-series before re-reading the end counters.
        self.trace_stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.trace_handle.take() {
            let _ = h.await;
        }
        let trace = self.trace.lock().map(|t| t.clone()).unwrap_or_default();

        // GPU energy
        let (gpu_energy_j, gpu_path) = if let Some(start_j) = self.gpu_start_j {
            // NVML counter path: re-read and diff (monotonic, no wrap in practice
            // over a bench window).
            let end_j = nvml_energy_snapshot_j().unwrap_or(start_j);
            ((end_j - start_j).max(0.0), GpuEnergyPath::NvmlCounter)
        } else if let Some(stop) = self.power_stop.take() {
            stop.store(true, Ordering::Relaxed);
            let j = match self.power_handle.take() {
                Some(h) => h.await.unwrap_or(0.0),
                None => 0.0,
            };
            if j > 0.0 {
                (j, GpuEnergyPath::PowerIntegration)
            } else {
                (0.0, GpuEnergyPath::None)
            }
        } else {
            (0.0, GpuEnergyPath::None)
        };

        // CPU/DRAM via RAPL
        let rapl_end = rapl_snapshot(&self.rapl_domains);
        let (cpu_pkg_energy_j, dram_energy_j) =
            rapl_delta_j(&self.rapl_domains, &self.rapl_start, &rapl_end);
        let cpu_rapl_available = self.rapl_readable
            && self
                .rapl_domains
                .iter()
                .any(|d| d.kind == RaplKind::Package)
            && cpu_pkg_energy_j > 0.0;
        let dram_rapl_available = self.rapl_readable
            && self.rapl_domains.iter().any(|d| d.kind == RaplKind::Dram)
            && dram_energy_j > 0.0;

        // Aggregate only the domains that actually counted.
        let mut domains_counted = Vec::new();
        let mut energy_j = 0.0;
        if gpu_path != GpuEnergyPath::None && gpu_energy_j > 0.0 {
            energy_j += gpu_energy_j;
            domains_counted.push("gpu".to_string());
        }
        if cpu_rapl_available {
            energy_j += cpu_pkg_energy_j;
            domains_counted.push("cpu_pkg".to_string());
        }
        if dram_rapl_available {
            energy_j += dram_energy_j;
            domains_counted.push("dram".to_string());
        }

        let note = build_note(&self.rapl_domains, self.rapl_readable, gpu_path);

        // Decode-phase split: energy consumed AFTER the first token. cum_at(t)
        // linearly interpolates the cumulative trace; decode = full - cum_at(t0).
        let (gpu_decode_j, cpu_pkg_decode_j, dram_decode_j, decode_energy_j, decode_duration_s) =
            match decode_start_s {
                Some(t0) if !trace.is_empty() && t0 < duration_s => {
                    let at = cum_at(&trace, t0);
                    let g = if gpu_path == GpuEnergyPath::NvmlCounter {
                        Some((gpu_energy_j - at.gpu_j).max(0.0))
                    } else {
                        None
                    };
                    let c = if cpu_rapl_available {
                        Some((cpu_pkg_energy_j - at.pkg_j).max(0.0))
                    } else {
                        None
                    };
                    let d = if dram_rapl_available {
                        Some((dram_energy_j - at.dram_j).max(0.0))
                    } else {
                        None
                    };
                    let tot = g.unwrap_or(0.0) + c.unwrap_or(0.0) + d.unwrap_or(0.0);
                    let tot = if tot > 0.0 { Some(tot) } else { None };
                    (g, c, d, tot, Some(duration_s - t0))
                }
                _ => (None, None, None, None, None),
            };

        EnergyWindow {
            gpu_energy_j,
            cpu_pkg_energy_j,
            dram_energy_j,
            energy_j,
            duration_s,
            gpu_path,
            cpu_rapl_available,
            dram_rapl_available,
            domains_counted,
            note,
            gpu_decode_j,
            cpu_pkg_decode_j,
            dram_decode_j,
            decode_energy_j,
            decode_duration_s,
        }
    }
}

/// Linearly interpolate the cumulative trace at time `t` (s from window start).
/// Clamps to the first/last sample outside the recorded range.
fn cum_at(trace: &[EnergyTrace], t: f64) -> EnergyTrace {
    if let Some(first) = trace.first() {
        if t <= first.t_s {
            return *first;
        }
    } else {
        return EnergyTrace {
            t_s: t,
            gpu_j: 0.0,
            pkg_j: 0.0,
            dram_j: 0.0,
        };
    }
    let last = *trace.last().unwrap();
    if t >= last.t_s {
        return last;
    }
    // Find the bracketing pair and interpolate.
    for w in trace.windows(2) {
        let (a, b) = (w[0], w[1]);
        if t >= a.t_s && t <= b.t_s {
            let span = (b.t_s - a.t_s).max(1e-9);
            let f = (t - a.t_s) / span;
            return EnergyTrace {
                t_s: t,
                gpu_j: a.gpu_j + (b.gpu_j - a.gpu_j) * f,
                pkg_j: a.pkg_j + (b.pkg_j - a.pkg_j) * f,
                dram_j: a.dram_j + (b.dram_j - a.dram_j) * f,
            };
        }
    }
    last
}

/// Background loop: record cumulative energy deltas (from window start) every
/// `interval_ms`, reusing one NVML handle. Stops when `stop` is set.
fn energy_trace_loop(
    stop: Arc<AtomicBool>,
    trace: Arc<Mutex<Vec<EnergyTrace>>>,
    interval_ms: u64,
    start: Instant,
    gpu_start_j: f64,
    domains: Vec<RaplDomain>,
    rapl_start: Vec<Option<u128>>,
) {
    use nvml_wrapper::Nvml;
    let nvml = Nvml::init().ok();
    let interval = Duration::from_millis(interval_ms.max(20));
    loop {
        let t_s = start.elapsed().as_secs_f64();
        let gpu_j = nvml
            .as_ref()
            .and_then(nvml_counter_sum_j)
            .map(|now| (now - gpu_start_j).max(0.0))
            .unwrap_or(0.0);
        let snap = rapl_snapshot(&domains);
        let (pkg_j, dram_j) = rapl_delta_j(&domains, &rapl_start, &snap);
        if let Ok(mut g) = trace.lock() {
            g.push(EnergyTrace {
                t_s,
                gpu_j,
                pkg_j,
                dram_j,
            });
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(interval);
    }
}

/// Sum the per-device NVML total-energy counters (mJ -> J) using an existing
/// handle (no re-init). Returns None if no device exposes the counter.
fn nvml_counter_sum_j(nvml: &nvml_wrapper::Nvml) -> Option<f64> {
    let count = nvml.device_count().ok()?;
    let mut total_mj: u128 = 0;
    let mut any = false;
    for i in 0..count {
        if let Ok(dev) = nvml.device_by_index(i) {
            if let Ok(mj) = dev.total_energy_consumption() {
                total_mj += mj as u128;
                any = true;
            }
        }
    }
    if any {
        Some(total_mj as f64 / 1000.0)
    } else {
        None
    }
}

impl Drop for EnergySampler {
    fn drop(&mut self) {
        if let Some(stop) = self.power_stop.as_ref() {
            stop.store(true, Ordering::Relaxed);
        }
        self.trace_stop.store(true, Ordering::Relaxed);
    }
}

/// Compose a human-readable note about any unavailable domain.
fn build_note(
    domains: &[RaplDomain],
    rapl_readable: bool,
    gpu_path: GpuEnergyPath,
) -> Option<String> {
    let mut parts = Vec::new();
    let have_rapl_files = !domains.is_empty();
    if have_rapl_files && !rapl_readable {
        parts.push(
            "CPU RAPL unavailable (needs root): energy_uj is permission-denied - \
             run as root or `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`"
                .to_string(),
        );
    } else if !have_rapl_files {
        parts.push("CPU RAPL unavailable (no intel-rapl powercap domains)".to_string());
    }
    match gpu_path {
        GpuEnergyPath::None => parts.push("GPU energy unavailable (no NVML/GPU)".to_string()),
        GpuEnergyPath::PowerIntegration => parts
            .push("GPU energy via power.draw integration (NVML energy counter absent)".to_string()),
        GpuEnergyPath::NvmlCounter => {}
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("; "))
    }
}

/// Integrate GPU `power.draw` (P.dt) over a window. Used only when the NVML
/// energy counter is unavailable. Sums every CUDA device.
fn power_integrate_loop(stop: Arc<AtomicBool>, interval_ms: u64) -> f64 {
    use nvml_wrapper::Nvml;
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return 0.0,
    };
    let count = nvml.device_count().unwrap_or(0);
    if count == 0 {
        return 0.0;
    }
    let interval = Duration::from_millis(interval_ms.max(10));
    let mut energy_j = 0.0;
    let mut last = Instant::now();
    // Trapezoid-ish: use the dt since the previous sample x instantaneous power.
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(interval);
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        let mut watts = 0.0;
        for i in 0..count {
            if let Ok(dev) = nvml.device_by_index(i) {
                if let Ok(mw) = dev.power_usage() {
                    watts += mw as f64 / 1000.0;
                }
            }
        }
        energy_j += watts * dt;
    }
    energy_j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_token_metrics_chain() {
        // 36 J over 60 tokens = 0.6 J/tok = 0.6/3600 Wh = 1.6667e-4 Wh/tok.
        let w = EnergyWindow {
            gpu_energy_j: 30.0,
            cpu_pkg_energy_j: 6.0,
            dram_energy_j: 0.0,
            energy_j: 36.0,
            duration_s: 1.0,
            gpu_path: GpuEnergyPath::NvmlCounter,
            cpu_rapl_available: true,
            dram_rapl_available: false,
            domains_counted: vec!["gpu".into(), "cpu_pkg".into()],
            note: None,
            gpu_decode_j: None,
            cpu_pkg_decode_j: None,
            dram_decode_j: None,
            decode_energy_j: None,
            decode_duration_s: None,
        };
        assert!((w.j_per_tok(60).unwrap() - 0.6).abs() < 1e-9);
        assert!((w.wh_per_tok(60).unwrap() - 0.6 / 3600.0).abs() < 1e-12);
        // gCO2/tok at CI=480: 1.6667e-4 Wh x 480 / 1000 = 8.0e-5 g.
        let g = w.gco2_per_tok(60, 480.0).unwrap();
        assert!((g - (0.6 / 3600.0) * 480.0 / 1000.0).abs() < 1e-12);
    }

    #[test]
    fn zero_tokens_or_energy_yields_none() {
        let w = EnergyWindow {
            gpu_energy_j: 0.0,
            cpu_pkg_energy_j: 0.0,
            dram_energy_j: 0.0,
            energy_j: 0.0,
            duration_s: 1.0,
            gpu_path: GpuEnergyPath::None,
            cpu_rapl_available: false,
            dram_rapl_available: false,
            domains_counted: vec![],
            note: None,
            gpu_decode_j: None,
            cpu_pkg_decode_j: None,
            dram_decode_j: None,
            decode_energy_j: None,
            decode_duration_s: None,
        };
        assert!(w.j_per_tok(60).is_none());
        let w2 = EnergyWindow {
            energy_j: 10.0,
            ..w
        };
        assert!(w2.j_per_tok(0).is_none());
    }

    #[test]
    fn rapl_wraparound_handled() {
        let dom = vec![RaplDomain {
            energy_uj_path: PathBuf::from("/nonexistent"),
            max_range_uj: 1000,
            kind: RaplKind::Package,
        }];
        // start near the top, end wrapped past 0.
        let start = vec![Some(900u128)];
        let end = vec![Some(100u128)];
        let (pkg, _dram) = rapl_delta_j(&dom, &start, &end);
        // (1000 - 900) + 100 = 200 µJ = 2.0e-4 J
        assert!((pkg - 200.0 / 1_000_000.0).abs() < 1e-12);
    }

    #[test]
    fn rapl_normal_delta() {
        let dom = vec![RaplDomain {
            energy_uj_path: PathBuf::from("/nonexistent"),
            max_range_uj: 1_000_000_000,
            kind: RaplKind::Dram,
        }];
        let (_pkg, dram) =
            rapl_delta_j(&dom, &vec![Some(1_000_000u128)], &vec![Some(3_000_000u128)]);
        // 2_000_000 µJ = 2.0 J
        assert!((dram - 2.0).abs() < 1e-9);
    }
}
