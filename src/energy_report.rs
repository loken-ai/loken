//! Human-facing energy report - turns an [`EnergyWindow`] into everyday-equipment
//! equivalences + a CO₂ estimate, so every request can show "what did this cost?".
//!
//! The goal is awareness: a paragraph of generated text draws a few joules (≈ a
//! second of an LED bulb), while a video render draws kilojoules (≈ minutes of a
//! kettle). Relating the number to familiar appliances makes the impact legible.
//!
//! Measurement comes from [`crate::energy`] (CPU RAPL + GPU NVML). When a domain
//! can't be read (RAPL is root-only on recent kernels), we optionally estimate the
//! CPU share from a configured package TDP so the report still says something, and
//! mark it `estimated`.

use crate::energy::{EnergyWindow, GpuEnergyPath};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Default grid carbon intensity (gCO₂eq / kWh). France's low-carbon grid; override
/// per region via config (`carbon_intensity`). World average ≈ 480, coal-heavy ≈ 800.
pub const DEFAULT_CARBON_INTENSITY: f64 = 50.0;

/// Default water footprint of electricity generation (liters / kWh) - the water
/// consumed (cooling, evaporation) to produce the energy. Highly grid-dependent;
/// ~1.8 L/kWh is a representative figure. Override via config (`water_l_per_kwh`).
pub const DEFAULT_WATER_L_PER_KWH: f64 = 1.8;

// -- reference appliance constants (generic, documented; not machine-specific) --
const LED_BULB_W: f64 = 7.0; // a typical LED bulb
const SMARTPHONE_CHARGE_WH: f64 = 15.0; // one full smartphone charge (~4000 mAh)
const EV_WH_PER_KM: f64 = 180.0; // electric-car consumption
const KETTLE_W: f64 = 2000.0; // electric kettle
const CUP_WH: f64 = 23.3; // heat 0.25 L water by ~80 °C
const HAIRDRYER_W: f64 = 1800.0; // hair dryer
const MICROWAVE_W: f64 = 1200.0; // microwave oven
const PETROL_CAR_GCO2_PER_KM: f64 = 120.0; // thermal-car tailpipe CO₂

/// Everyday-equipment equivalences for a given amount of energy.
#[derive(Debug, Clone, Serialize)]
pub struct EnergyEquivalents {
    pub led_bulb_minutes: f64,
    pub smartphone_charges: f64,
    pub ev_meters: f64,
    pub kettle_seconds: f64,
    pub hot_water_cups: f64,
    pub hairdryer_seconds: f64,
    pub microwave_seconds: f64,
}

impl EnergyEquivalents {
    pub fn from_joules(j: f64) -> Self {
        let wh = j / 3600.0;
        Self {
            led_bulb_minutes: wh / LED_BULB_W * 60.0,
            smartphone_charges: wh / SMARTPHONE_CHARGE_WH,
            ev_meters: wh / EV_WH_PER_KM * 1000.0,
            kettle_seconds: j / KETTLE_W,
            hot_water_cups: wh / CUP_WH,
            hairdryer_seconds: j / HAIRDRYER_W,
            microwave_seconds: j / MICROWAVE_W,
        }
    }
}

/// A complete, human-presentable energy report for one request/render.
#[derive(Debug, Clone, Serialize)]
pub struct EnergyReport {
    pub energy_j: f64,
    pub wh: f64,
    pub gco2: f64,
    /// Water consumed to generate this energy (liters), via `water_l_per_kwh`.
    pub water_liters: f64,
    pub duration_s: f64,
    /// True if any part of `energy_j` was estimated (a domain counter was unreadable).
    pub estimated: bool,
    /// True if nothing at all could be measured or estimated.
    pub unmeasured: bool,
    pub domains: Vec<String>,
    pub equiv: EnergyEquivalents,
}

impl EnergyReport {
    /// Build from a measured window. When CPU RAPL is unavailable and `cpu_tdp_w > 0`,
    /// estimate the CPU share as `cpu_tdp_w x duration` (a full-TDP upper bound) so the
    /// report degrades gracefully instead of under-counting to zero.
    pub fn from_window(
        w: &EnergyWindow,
        carbon_intensity: f64,
        cpu_tdp_w: f64,
        water_l_per_kwh: f64,
    ) -> Self {
        let mut energy_j = w.energy_j;
        let mut estimated = false;
        let mut domains = w.domains_counted.clone();
        if !w.cpu_rapl_available && cpu_tdp_w > 0.0 && w.duration_s > 0.0 {
            energy_j += cpu_tdp_w * w.duration_s;
            estimated = true;
            domains.push("cpu_est".into());
        }
        if w.gpu_path == GpuEnergyPath::PowerIntegration {
            estimated = true; // integrated power.draw, not a hardware counter
        }
        let unmeasured = energy_j <= 0.0;
        let wh = energy_j / 3600.0;
        Self {
            energy_j,
            wh,
            gco2: wh * carbon_intensity / 1000.0,
            water_liters: wh / 1000.0 * water_l_per_kwh,
            duration_s: w.duration_s,
            estimated,
            unmeasured,
            domains,
            equiv: EnergyEquivalents::from_joules(energy_j),
        }
    }

    /// Build from one request's share of the samples it was active in.
    pub fn from_share(share: &EnergyShare, carbon_intensity: f64, water_l_per_kwh: f64) -> Self {
        let energy_j = share.joules().unwrap_or(0.0);
        let mut domains = Vec::new();
        if share.gpu_j.is_some() {
            domains.push("gpu".to_string());
        }
        match (share.cpu_j, share.cpu_estimated) {
            (Some(_), false) => domains.push("cpu_pkg".to_string()),
            (Some(_), true) => domains.push("cpu_est".to_string()),
            (None, _) => {}
        }
        let wh = energy_j / 3600.0;
        Self {
            energy_j,
            wh,
            gco2: wh * carbon_intensity / 1000.0,
            water_liters: wh / 1000.0 * water_l_per_kwh,
            duration_s: share.duration_s,
            estimated: share.cpu_estimated && share.cpu_j.is_some(),
            unmeasured: energy_j <= 0.0,
            domains,
            equiv: EnergyEquivalents::from_joules(energy_j),
        }
    }

    /// Compact one-liner for the per-request log line.
    pub fn human_line(&self) -> String {
        if self.unmeasured {
            return format!(
                "⚡ energy not measured ({}; RAPL root-only / no NVML)",
                fmt_dur(self.duration_s)
            );
        }
        let approx = if self.estimated { "~" } else { "" };
        let e = &self.equiv;
        format!(
            "⚡ {approx}{} ({:.0} J) . {approx}{:.2} gCO₂ . {approx}{} water ≈ {} of an LED bulb . {} . {} in an EV . {}",
            fmt_wh(self.wh), self.energy_j, self.gco2,
            fmt_water(self.water_liters),
            fmt_dur(e.led_bulb_minutes * 60.0),
            fmt_charges(e.smartphone_charges, "phone charge"),
            fmt_dist(e.ev_meters),
            fmt_kettle(e.hot_water_cups, e.kettle_seconds),
        )
    }

    /// Multi-line readable block (CLI render tools, session summaries).
    pub fn human_block(&self) -> String {
        if self.unmeasured {
            return format!(
                "⚡ Energy: not measured ({}). Enable RAPL (read /sys/.../energy_uj) or use an NVML GPU.",
                fmt_dur(self.duration_s)
            );
        }
        let approx = if self.estimated { " (estimated)" } else { "" };
        let e = &self.equiv;
        format!(
            "⚡ Energy for this operation{approx}: {} ({:.0} J) in {}\n   CO₂ ≈ {:.3} gCO₂eq (≈ {} in a petrol car)\n   Water ≈ {} (to generate the electricity)\n   Everyday equivalents:\n     • {} of an LED bulb (7 W)\n     • {}\n     • {} in an electric car\n     • {} of a kettle ({})\n     • {} of a hair dryer / {} of a microwave",
            fmt_wh(self.wh), self.energy_j, fmt_dur(self.duration_s),
            self.gco2, fmt_dist(self.gco2 / PETROL_CAR_GCO2_PER_KM * 1000.0),
            fmt_water(self.water_liters),
            fmt_dur(e.led_bulb_minutes * 60.0),
            fmt_charges(e.smartphone_charges, "smartphone charge"),
            fmt_dist(e.ev_meters),
            fmt_dur(e.kettle_seconds), fmt_cups(e.hot_water_cups),
            fmt_dur(e.hairdryer_seconds), fmt_dur(e.microwave_seconds),
        )
    }
}

// -- readable formatting (auto unit scaling) --
fn fmt_wh(wh: f64) -> String {
    if wh >= 1000.0 {
        format!("{:.2} kWh", wh / 1000.0)
    } else if wh >= 1.0 {
        format!("{:.2} Wh", wh)
    } else {
        format!("{:.1} mWh", wh * 1000.0)
    }
}
/// Human-friendly duration: compound units (`1 h 36 min`, `2 min 54 s`, `20 s`)
/// via integer decomposition (no `1.6 h` / no `2 min 60 s` rounding glitches).
fn fmt_dur(seconds: f64) -> String {
    if seconds < 1.0 {
        return format!("{:.0} ms", seconds * 1000.0);
    }
    if seconds < 10.0 {
        return format!("{:.1} s", seconds);
    }
    let total = seconds.round() as u64;
    if total < 60 {
        return format!("{total} s");
    }
    if total < 3600 {
        let (m, s) = (total / 60, total % 60);
        return if s > 0 {
            format!("{m} min {s} s")
        } else {
            format!("{m} min")
        };
    }
    if total < 86_400 {
        let (h, m) = (total / 3600, (total % 3600) / 60);
        return if m > 0 {
            format!("{h} h {m} min")
        } else {
            format!("{h} h")
        };
    }
    let (d, h) = (total / 86_400, (total % 86_400) / 3600);
    if h > 0 {
        format!("{d} d {h} h")
    } else {
        format!("{d} days")
    }
}
/// Human-friendly distance. "meters" is spelled out so it can't be read as "min".
fn fmt_dist(m: f64) -> String {
    if m >= 1000.0 {
        format!("{:.1} km", m / 1000.0)
    } else if m >= 10.0 {
        format!("{:.0} meters", m)
    } else if m >= 1.0 {
        format!("{:.1} meters", m)
    } else {
        format!("{:.0} cm", m * 100.0)
    }
}
/// Human-friendly count (e.g. smartphone charges): sensible, non-noisy precision.
fn fmt_num(n: f64) -> String {
    if n >= 10.0 {
        format!("{:.0}", n)
    } else if n >= 1.0 {
        format!("{:.1}", n)
    } else if n >= 0.1 {
        format!("{:.2}", n)
    } else if n >= 0.001 {
        format!("{:.3}", n)
    } else {
        format!("{:.1e}", n)
    }
}
fn fmt_cups(c: f64) -> String {
    if c >= 1.0 {
        format!("{} cups of hot water", fmt_num(c))
    } else {
        format!("{:.0}% of a cup", c * 100.0)
    }
}
/// A count of discrete `noun`s, but fractions (< 1) read as a percentage of one
/// - "60% of a phone charge" instead of the abstract "0.60 phone charges".
fn fmt_charges(n: f64, noun: &str) -> String {
    if n >= 1.0 {
        return format!("{} {noun}s", fmt_num(n));
    }
    let pct = n * 100.0;
    if pct >= 1.0 {
        format!("{:.0}% of a {noun}", pct)
    } else {
        format!("<1% of a {noun}")
    }
}
fn fmt_kettle(cups: f64, kettle_s: f64) -> String {
    if cups >= 0.2 {
        fmt_cups(cups)
    } else {
        format!("{} of a kettle", fmt_dur(kettle_s))
    }
}
/// Human-friendly water volume, bare quantity (a "glass" = 0.25 L). The "water"
/// wording is added by the caller to avoid "Water ≈ ... of water" redundancy.
fn fmt_water(liters: f64) -> String {
    if liters >= 1.0 {
        format!("{:.1} L", liters)
    } else if liters >= 0.25 {
        format!("{} glasses", fmt_num(liters / 0.25))
    } else {
        format!("{:.0} mL", liters * 1000.0)
    }
}

// -- session accumulator (shared via Arc in the server state) --

/// JSON-able snapshot of cumulative session energy (for the GUI / stats endpoint).
#[derive(Debug, Clone, Default, Serialize)]
pub struct EnergySnapshot {
    pub total_j: f64,
    pub total_wh: f64,
    pub total_gco2: f64,
    /// Water consumed to generate the session's energy (liters), via `water_l_per_kwh`.
    pub total_water_l: f64,
    pub requests: u64,
    pub by_modality: HashMap<String, f64>, // modality -> joules
    pub last_line: Option<String>,
    pub session_equiv: EnergyEquivalentsLite,
}

/// Lite equivalences for the cumulative session card (a few headline numbers).
#[derive(Debug, Clone, Default, Serialize)]
pub struct EnergyEquivalentsLite {
    pub led_bulb_minutes: f64,
    pub smartphone_charges: f64,
    pub ev_meters: f64,
    pub hot_water_cups: f64,
}

/// Thread-safe cumulative energy tracker for the server session.
#[derive(Default)]
pub struct EnergyTracker {
    inner: Mutex<TrackerInner>,
}

#[derive(Default)]
struct TrackerInner {
    total_j: f64,
    total_gco2: f64,
    requests: u64,
    by_modality: HashMap<String, f64>,
    last_line: Option<String>,
}

impl EnergyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one request's report under a modality label ("text", "image", "audio", "video", "tts").
    pub fn record(&self, modality: &str, report: &EnergyReport) {
        if report.unmeasured {
            return;
        }
        let mut g = self.inner.lock().unwrap();
        g.total_j += report.energy_j;
        g.total_gco2 += report.gco2;
        g.requests += 1;
        *g.by_modality.entry(modality.to_string()).or_insert(0.0) += report.energy_j;
        g.last_line = Some(report.human_line());
    }

    pub fn snapshot(&self) -> EnergySnapshot {
        let g = self.inner.lock().unwrap();
        let eq = EnergyEquivalents::from_joules(g.total_j);
        let total_wh = g.total_j / 3600.0;
        EnergySnapshot {
            total_j: g.total_j,
            total_wh,
            total_gco2: g.total_gco2,
            total_water_l: total_wh / 1000.0 * settings().water_l_per_kwh,
            requests: g.requests,
            by_modality: g.by_modality.clone(),
            last_line: g.last_line.clone(),
            session_equiv: EnergyEquivalentsLite {
                led_bulb_minutes: eq.led_bulb_minutes,
                smartphone_charges: eq.smartphone_charges,
                ev_meters: eq.ev_meters,
                hot_water_cups: eq.hot_water_cups,
            },
        }
    }
}

// -- global settings + session tracker (initialised once from config at startup) --

/// Server-wide energy-reporting settings, set once from config.toml at startup.
#[derive(Debug, Clone)]
pub struct EnergySettings {
    pub enabled: bool,
    pub carbon_intensity: f64,
    /// CPU package TDP (W) used to *estimate* CPU energy when RAPL is unreadable
    /// (root-only kernels). 0 = no estimate (report what's measured only).
    pub cpu_tdp_w: f64,
    /// Water footprint of electricity generation (liters / kWh).
    pub water_l_per_kwh: f64,
}
impl Default for EnergySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            carbon_intensity: DEFAULT_CARBON_INTENSITY,
            cpu_tdp_w: 0.0,
            water_l_per_kwh: DEFAULT_WATER_L_PER_KWH,
        }
    }
}

static SETTINGS: OnceLock<EnergySettings> = OnceLock::new();
static TRACKER: OnceLock<EnergyTracker> = OnceLock::new();

/// Initialise the global energy settings (call once at server startup, after config load).
pub fn init_settings(s: EnergySettings) {
    let _ = SETTINGS.set(s);
}
/// Current settings (defaults if not yet initialised - reporting on, France CI).
pub fn settings() -> EnergySettings {
    SETTINGS.get().cloned().unwrap_or_default()
}
/// Whether energy reporting is enabled (cheap gate before starting a sampler).
pub fn is_enabled() -> bool {
    settings().enabled
}
/// The process-wide cumulative session tracker.
pub fn tracker() -> &'static EnergyTracker {
    TRACKER.get_or_init(EnergyTracker::new)
}

// -- per-request shares of the energy drawn while requests overlap --

/// One request's part of the energy drawn while it ran.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnergyShare {
    /// GPU joules from the hardware counter; `None` when no counter was read.
    pub gpu_j: Option<f64>,
    /// CPU package and DRAM joules, read from RAPL or estimated from the configured TDP;
    /// `None` when neither reads them.
    pub cpu_j: Option<f64>,
    /// The CPU part was not read from RAPL.
    pub cpu_estimated: bool,
    /// Another request was active during at least one of its samples.
    pub shared: bool,
    /// Seconds from its start to its end.
    pub duration_s: f64,
}

impl EnergyShare {
    /// The domains read, summed; `None` when no domain was read.
    pub fn joules(&self) -> Option<f64> {
        sum(self.gpu_j, self.cpu_j)
    }
}

fn sum(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
    }
}

/// One metered request: what it has drawn so far, and its work as last sampled.
#[derive(Default)]
struct Entry {
    share: EnergyShare,
    /// The request's work counter (tokens computed); `None` when nothing reports it.
    probe: Option<Arc<crate::request_metrics::RequestProbe>>,
    work_sampled: u64,
}

/// The requests being metered and what each has drawn so far. A sample is the energy read
/// between two consecutive starts or ends of requests, so the set of active requests is the
/// same over the whole sample; it is split among them by the work each did over it.
#[derive(Default)]
struct Ledger {
    active: HashMap<u64, Entry>,
}

impl Ledger {
    /// Splits the sample `w` among the requests active over it, each by its part of the tokens
    /// computed over the sample; evenly when none reports any. Without RAPL the CPU part is
    /// `cpu_tdp_w` over the sample's duration when a TDP is configured.
    fn sample(&mut self, w: &EnergyWindow, cpu_tdp_w: f64) {
        let n = self.active.len();
        if n == 0 {
            return;
        }
        let gpu = (w.gpu_path == GpuEnergyPath::NvmlCounter).then_some(w.gpu_energy_j);
        let cpu = if w.cpu_rapl_available {
            Some(w.cpu_pkg_energy_j + w.dram_energy_j)
        } else if cpu_tdp_w > 0.0 {
            Some(cpu_tdp_w * w.duration_s)
        } else {
            None
        };
        let done: Vec<(u64, u64)> = self
            .active
            .iter()
            .map(|(&id, e)| {
                let now = e.probe.as_ref().map_or(e.work_sampled, |p| p.work());
                (id, now.saturating_sub(e.work_sampled))
            })
            .collect();
        let total: u64 = done.iter().map(|&(_, d)| d).sum();
        for (id, d) in done {
            let part = if total > 0 {
                d as f64 / total as f64
            } else {
                1.0 / n as f64
            };
            let e = self.active.get_mut(&id).expect("an active request");
            e.work_sampled += d;
            let s = &mut e.share;
            s.gpu_j = sum(s.gpu_j, gpu.map(|j| j * part));
            s.cpu_j = sum(s.cpu_j, cpu.map(|j| j * part));
            s.cpu_estimated |= !w.cpu_rapl_available;
            s.shared |= n > 1;
        }
    }
}

/// The meter read at every start and end of a metered request, open while one is active.
#[derive(Default)]
struct Meters {
    meter: Option<crate::energy::EnergyMeter>,
    ledger: Ledger,
    next_id: u64,
}

static METERS: OnceLock<Mutex<Meters>> = OnceLock::new();

fn meters() -> std::sync::MutexGuard<'static, Meters> {
    METERS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// One request's place in the shared meter, from [`begin`] to [`end_share`]. Dropped without
/// being ended, it leaves the meter and its share is discarded.
pub struct RequestEnergy {
    id: u64,
    started: std::time::Instant,
    open: bool,
}

impl RequestEnergy {
    fn open(probe: Option<Arc<crate::request_metrics::RequestProbe>>) -> Self {
        let tdp = settings().cpu_tdp_w;
        let mut guard = meters();
        let m = &mut *guard;
        match m.meter.as_mut() {
            Some(meter) => m.ledger.sample(&meter.lap(), tdp),
            None => m.meter = Some(crate::energy::EnergyMeter::start()),
        }
        let id = m.next_id;
        m.next_id += 1;
        let work_sampled = probe.as_ref().map_or(0, |p| p.work());
        m.ledger.active.insert(
            id,
            Entry {
                share: EnergyShare::default(),
                probe,
                work_sampled,
            },
        );
        Self {
            id,
            started: std::time::Instant::now(),
            open: true,
        }
    }

    /// Leaves the meter, returning the share drawn since [`Self::open`].
    fn close(mut self) -> EnergyShare {
        self.open = false;
        let mut share = close_entry(self.id);
        share.duration_s = self.started.elapsed().as_secs_f64();
        share
    }
}

impl Drop for RequestEnergy {
    fn drop(&mut self) {
        if self.open {
            let _ = close_entry(self.id);
        }
    }
}

/// Reads the meter, splits the sample, and removes request `id`; the meter closes with the
/// last active request.
fn close_entry(id: u64) -> EnergyShare {
    let tdp = settings().cpu_tdp_w;
    let mut guard = meters();
    let m = &mut *guard;
    if let Some(meter) = m.meter.as_mut() {
        m.ledger.sample(&meter.lap(), tdp);
    }
    let share = m
        .ledger
        .active
        .remove(&id)
        .map(|e| e.share)
        .unwrap_or_default();
    if m.ledger.active.is_empty() {
        m.meter = None;
    }
    share
}

/// Start metering one request iff energy reporting is enabled (else `None`, zero cost).
/// Pair with [`end`] around the inference call:
/// ```ignore
/// let m = energy_report::begin();
/// // ... run inference ...
/// energy_report::end(m, "text", "[/api/generate]");
/// ```
pub fn begin() -> Option<RequestEnergy> {
    begin_with(None)
}

/// [`begin`] for a request whose probe counts its work, so that energy drawn while it overlaps
/// others is shared by the work each did.
pub fn begin_with(
    probe: Option<Arc<crate::request_metrics::RequestProbe>>,
) -> Option<RequestEnergy> {
    is_enabled().then(|| RequestEnergy::open(probe))
}

/// Close a meter from [`begin`], emit the per-request log line + record to the session.
pub fn end(meter: Option<RequestEnergy>, modality: &str, label: &str) {
    let _ = end_share(meter, modality, label);
}

/// [`end`] returning the joules of the request's share (total across available domains),
/// so handlers can include the figure in their response payloads.
pub fn end_measured(meter: Option<RequestEnergy>, modality: &str, label: &str) -> Option<f64> {
    end_share(meter, modality, label).map(|s| s.joules().unwrap_or(0.0))
}

/// [`end`] returning the request's share by domain and how it was shared.
pub fn end_share(meter: Option<RequestEnergy>, modality: &str, label: &str) -> Option<EnergyShare> {
    let share = meter?.close();
    let s = settings();
    let r = EnergyReport::from_share(&share, s.carbon_intensity, s.water_l_per_kwh);
    tracing::info!("{label} {}", r.human_line());
    tracker().record(modality, &r);
    Some(share)
}

/// Build a report from a measured window using global settings, emit the per-request
/// log line (prefixed by `label`), and record it under `modality` in the session tracker.
/// Returns the report (callers may also use `human_block()` for CLI tools).
pub fn report_window(modality: &str, label: &str, w: &EnergyWindow) -> EnergyReport {
    let s = settings();
    let r = EnergyReport::from_window(w, s.carbon_intensity, s.cpu_tdp_w, s.water_l_per_kwh);
    tracing::info!("{label} {}", r.human_line());
    tracker().record(modality, &r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equivalents_scale_sensibly() {
        // 1 Wh = 3600 J -> ~8.6 min of a 7 W LED, ~5.6 m in an EV.
        let e = EnergyEquivalents::from_joules(3600.0);
        assert!((e.led_bulb_minutes - 8.571).abs() < 0.01);
        assert!((e.ev_meters - 5.556).abs() < 0.01);
        assert!((e.smartphone_charges - 1.0 / 15.0).abs() < 1e-6);
    }

    #[test]
    fn durations_and_distances_are_human_friendly() {
        assert_eq!(fmt_dur(174.0), "2 min 54 s");
        assert_eq!(fmt_dur(5740.0), "1 h 35 min");
        assert_eq!(fmt_dur(20.08), "20 s");
        assert_eq!(fmt_dur(3.4), "3.4 s");
        assert_eq!(fmt_dur(3600.0), "1 h");
        assert_eq!(fmt_dist(62.0), "62 meters");
        assert_eq!(fmt_dist(4.6), "4.6 meters");
        assert_eq!(fmt_dist(2500.0), "2.5 km");
        assert_eq!(fmt_num(0.744), "0.74");
        // Fractional counts read as a percentage of one, not "0.60 charges".
        assert_eq!(fmt_charges(0.60, "phone charge"), "60% of a phone charge");
        assert_eq!(fmt_charges(1.5, "phone charge"), "1.5 phone charges");
        assert_eq!(fmt_charges(0.0003, "phone charge"), "<1% of a phone charge");
    }

    #[test]
    fn report_human_line_nonempty_and_co2_scales_with_ci() {
        let w = EnergyWindow {
            gpu_energy_j: 30.0,
            cpu_pkg_energy_j: 6.0,
            dram_energy_j: 0.0,
            energy_j: 36.0,
            duration_s: 5.0,
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
        let r50 = EnergyReport::from_window(&w, 50.0, 0.0, 1.8);
        let r800 = EnergyReport::from_window(&w, 800.0, 0.0, 1.8);
        assert!(!r50.human_line().is_empty());
        assert!(!r50.estimated && !r50.unmeasured);
        assert!((r800.gco2 / r50.gco2 - 16.0).abs() < 1e-6); // CO₂ linear in carbon intensity
    }

    /// A sample of `gpu` joules from the counter (none when `None`) and `cpu` joules from
    /// RAPL (unreadable when `None`), over `duration_s`.
    fn sample(gpu: Option<f64>, cpu: Option<f64>, duration_s: f64) -> EnergyWindow {
        EnergyWindow {
            gpu_energy_j: gpu.unwrap_or(0.0),
            cpu_pkg_energy_j: cpu.unwrap_or(0.0),
            dram_energy_j: 0.0,
            energy_j: gpu.unwrap_or(0.0) + cpu.unwrap_or(0.0),
            duration_s,
            gpu_path: if gpu.is_some() {
                GpuEnergyPath::NvmlCounter
            } else {
                GpuEnergyPath::None
            },
            cpu_rapl_available: cpu.is_some(),
            dram_rapl_available: false,
            domains_counted: Vec::new(),
            note: None,
            gpu_decode_j: None,
            cpu_pkg_decode_j: None,
            dram_decode_j: None,
            decode_energy_j: None,
            decode_duration_s: None,
        }
    }

    /// A request alone keeps every sample whole and is not marked shared.
    #[test]
    fn a_request_alone_keeps_every_sample() {
        let mut l = Ledger::default();
        l.active.insert(1, Entry::default());
        l.sample(&sample(Some(10.0), Some(4.0), 1.0), 0.0);
        l.sample(&sample(Some(2.0), Some(1.0), 1.0), 0.0);
        let s = l.active.remove(&1).unwrap().share;
        assert_eq!((s.gpu_j, s.cpu_j), (Some(12.0), Some(5.0)));
        assert!(!s.shared && !s.cpu_estimated);
        assert_eq!(s.joules(), Some(17.0));
    }

    /// Each sample is split evenly among the requests active over it when none reports its work, and every request that
    /// met another is marked shared; the shares add up to what was drawn.
    #[test]
    fn a_sample_is_split_among_the_requests_active_over_it() {
        let mut l = Ledger::default();
        l.active.insert(1, Entry::default());
        l.sample(&sample(Some(6.0), Some(2.0), 1.0), 0.0);
        l.active.insert(2, Entry::default());
        l.sample(&sample(Some(8.0), Some(4.0), 1.0), 0.0);
        let a = l.active.remove(&1).unwrap().share;
        l.sample(&sample(Some(3.0), Some(1.0), 1.0), 0.0);
        let b = l.active.remove(&2).unwrap().share;
        assert_eq!((a.gpu_j, a.cpu_j), (Some(10.0), Some(4.0)));
        assert_eq!((b.gpu_j, b.cpu_j), (Some(7.0), Some(3.0)));
        assert!(a.shared && b.shared);
        assert_eq!(a.joules().unwrap() + b.joules().unwrap(), 24.0);
    }

    /// Without RAPL the CPU part is the configured TDP over the sample, split like the rest
    /// and marked estimated; with no TDP it is not reported. Without a GPU counter the GPU
    /// part is not reported.
    /// Requests that report their work share each sample by the tokens each computed over it.
    #[test]
    fn a_sample_is_shared_by_the_work_done_over_it() {
        use crate::request_metrics::RequestProbe;
        let (long, short) = (
            Arc::new(RequestProbe::default()),
            Arc::new(RequestProbe::default()),
        );
        let mut l = Ledger::default();
        for (id, p) in [(1, &long), (2, &short)] {
            l.active.insert(
                id,
                Entry {
                    probe: Some(p.clone()),
                    ..Entry::default()
                },
            );
        }
        long.note_prefill(300, 300);
        short.note_decoded(100);
        l.sample(&sample(Some(40.0), Some(8.0), 1.0), 0.0);
        let a = l.active.remove(&1).unwrap().share;
        let b = l.active.remove(&2).unwrap().share;
        assert_eq!((a.gpu_j, a.cpu_j), (Some(30.0), Some(6.0)));
        assert_eq!((b.gpu_j, b.cpu_j), (Some(10.0), Some(2.0)));
        assert!(a.shared && b.shared);
    }

    #[test]
    fn an_unread_domain_is_estimated_or_left_out() {
        let mut l = Ledger::default();
        l.active.insert(1, Entry::default());
        l.active.insert(2, Entry::default());
        l.sample(&sample(None, None, 2.0), 50.0);
        let s = l.active.remove(&1).unwrap().share;
        assert_eq!((s.gpu_j, s.cpu_j), (None, Some(50.0)));
        assert!(s.cpu_estimated && s.shared);
        let mut l = Ledger::default();
        l.active.insert(1, Entry::default());
        l.sample(&sample(None, None, 2.0), 0.0);
        let s = l.active.remove(&1).unwrap().share;
        assert_eq!((s.gpu_j, s.cpu_j, s.joules()), (None, None, None));
        assert!(s.cpu_estimated);
    }

    /// A metered request leaves the shared meter when it ends or is dropped, so a later
    /// request alone is not marked shared.
    #[test]
    fn a_dropped_request_leaves_the_meter() {
        let dropped = RequestEnergy::open(None);
        drop(dropped);
        let alone = RequestEnergy::open(None);
        let id = alone.id;
        assert!(!meters().ledger.active.contains_key(&(id - 1)));
        let share = alone.close();
        assert!(!meters().ledger.active.contains_key(&id));
        assert!(share.duration_s >= 0.0);
    }
}
