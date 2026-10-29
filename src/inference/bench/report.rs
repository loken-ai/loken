//! The record of one run: the build, the machine, the controls, what was loaded and placed, and
//! what was measured. Written as JSON so two commits compare, and shown as a table.

use super::counters::{ProcessCounters, StepCount};
use super::{Controls, PageCache, Placement};
use serde::{Deserialize, Serialize};

/// The binary that ran.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Build {
    pub version: String,
    pub commit: String,
    pub features: Vec<String>,
}

impl Build {
    pub fn this() -> Self {
        let mut features = Vec::new();
        if cfg!(feature = "cuda") {
            features.push("cuda".to_string());
        }
        if cfg!(feature = "rocm") {
            features.push("rocm".to_string());
        }
        Self {
            version: env!("LOKEN_VERSION").to_string(),
            commit: env!("LOKEN_COMMIT").to_string(),
            features,
        }
    }
}

/// One card of the machine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Card {
    pub ordinal: usize,
    pub total_bytes: u64,
    pub memory_gbs: f64,
    pub perf_gflops: u64,
}

/// The disk a file lives on: what a read of it can reach at best is set by the drive and by the
/// link it sits on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Disk {
    /// The block device, `nvme1n1` for instance.
    pub device: String,
    pub model: String,
    /// The bus link the drive negotiated, `16.0 GT/s PCIe x4` for instance; empty when the sysfs
    /// tree does not say.
    pub link: String,
    /// A spinning drive.
    pub rotational: bool,
}

impl Disk {
    /// The disk holding `path`, read from the kernel's block device tree.
    pub fn of(path: &std::path::Path) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let dev = std::fs::metadata(path).ok()?.dev();
        let (major, minor) = ((dev >> 8) & 0xfff, (dev & 0xff) | ((dev >> 12) & 0xfff00));
        let node = std::fs::canonicalize(format!("/sys/dev/block/{major}:{minor}")).ok()?;
        // A partition's node sits inside its disk's.
        let disk = if node.join("partition").exists() {
            node.parent()?.to_path_buf()
        } else {
            node
        };
        let read = |p: std::path::PathBuf| {
            std::fs::read_to_string(p)
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let device = disk.file_name()?.to_string_lossy().into_owned();
        // An NVMe drive's controller is `device`, and the PCI function it sits on is its parent.
        let pci = std::fs::canonicalize(disk.join("device/device")).ok();
        let link = pci
            .map(|p| {
                let (speed, width) = (
                    read(p.join("current_link_speed")),
                    read(p.join("current_link_width")),
                );
                if speed.is_empty() {
                    String::new()
                } else {
                    format!("{speed} x{width}")
                }
            })
            .unwrap_or_default();
        let model = [disk.join("device/model"), disk.join("device/device/model")]
            .into_iter()
            .map(read)
            .find(|m| !m.is_empty())
            .unwrap_or_default();
        Some(Self {
            device,
            model,
            link,
            rotational: read(disk.join("queue/rotational")) == "1",
        })
    }

    /// One line for a table.
    pub fn describe(&self) -> String {
        let kind = if self.rotational {
            "rotational"
        } else {
            "solid state"
        };
        format!("{} {} ({kind}) {}", self.device, self.model, self.link)
    }
}

/// The machine that ran.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Machine {
    pub host: String,
    pub cpu: String,
    pub cores: usize,
    pub memory_bytes: u64,
    pub cards: Vec<Card>,
}

impl Machine {
    pub fn this() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();
        let cards = crate::inference::place::room::card_rooms()
            .iter()
            .map(|c| Card {
                ordinal: c.index,
                total_bytes: c.total,
                memory_gbs: 0.0,
                perf_gflops: c.perf,
            })
            .collect();
        Self {
            host: sysinfo::System::host_name().unwrap_or_default(),
            cpu: sys
                .cpus()
                .first()
                .map(|c| c.brand().to_string())
                .unwrap_or_default(),
            cores: sys.cpus().len(),
            memory_bytes: sys.total_memory(),
            cards,
        }
    }
}

/// The controls as recorded.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ControlsRecord {
    pub file: String,
    pub layers: (usize, usize),
    pub tokens: usize,
    pub warm: usize,
    pub placement: String,
    pub page_cache: String,
    pub readahead: Option<usize>,
    pub card_experts: Option<usize>,
    pub guess_hit: f64,
    pub seed: u64,
    pub expert_profile: Option<String>,
}

impl From<&Controls> for ControlsRecord {
    fn from(c: &Controls) -> Self {
        Self {
            file: c.file.display().to_string(),
            layers: c.layers,
            tokens: c.tokens,
            warm: c.warm,
            placement: c.placement.name().to_string(),
            page_cache: c.page_cache.name().to_string(),
            readahead: c.readahead,
            card_experts: c.card_experts,
            guess_hit: c.guess_hit,
            seed: c.seed,
            expert_profile: c.expert_profile.as_ref().map(|p| p.display().to_string()),
        }
    }
}

/// What was loaded and where its experts went.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelRecord {
    pub architecture: String,
    pub layers: usize,
    pub held: (usize, usize),
    pub routed_experts: usize,
    pub activated: usize,
    pub expert_bytes: u64,
    /// Whether a card placement was opened.
    pub cards_opened: bool,
    /// Per held layer: experts the cards hold, and the host tier's slots.
    pub card_experts_per_layer: Vec<usize>,
    pub host_slots_per_layer: Vec<usize>,
    /// Where the routing heat came from: the profile, the file's prior, or uniform.
    pub heat_source: String,
}

/// The time of the decode steps.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Timing {
    pub ms_per_token: Vec<f64>,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl Timing {
    pub fn of(ms: Vec<f64>) -> Self {
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = if n == 0 {
            0.0
        } else if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        Self {
            median_ms: median,
            mean_ms: if n == 0 {
                0.0
            } else {
                sorted.iter().sum::<f64>() / n as f64
            },
            min_ms: sorted.first().copied().unwrap_or(0.0),
            max_ms: sorted.last().copied().unwrap_or(0.0),
            ms_per_token: ms,
        }
    }
}

/// One stage of the profile.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StageRow {
    pub stage: String,
    pub total_us: u64,
    pub calls: u64,
}

/// One kind of step at the card seam.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepRow {
    pub step: String,
    #[serde(flatten)]
    pub count: StepCount,
}

/// The counters over the timed steps.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Counters {
    pub steps: Vec<StepRow>,
    pub expert_uploads: u64,
    pub tier_hits: u64,
    pub tier_misses: u64,
    pub tier_evictions: u64,
    pub process: ProcessCounters,
}

/// The whole record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub build: Build,
    pub machine: Machine,
    /// The disk of the model file, whose speed bounds the reads of the experts a host keeps out of
    /// memory.
    #[serde(default)]
    pub disk: Option<Disk>,
    pub controls: ControlsRecord,
    pub model: ModelRecord,
    pub timing: Timing,
    pub stages: Vec<StageRow>,
    pub counters: Counters,
}

impl Record {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// The record as a table for a terminal: the run, the time per token, the stages by time,
    /// then the counters.
    pub fn table(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let tokens = self.controls.tokens.max(1) as f64;
        if let Some(d) = &self.disk {
            let _ = writeln!(s, "disk of the file: {}", d.describe());
        }
        let _ = writeln!(
            s,
            "layer bench: {} layers [{}, {}) of {}, {} tokens after {} warm, placement {}, page cache {}",
            self.model.architecture,
            self.model.held.0,
            self.model.held.1,
            self.model.layers,
            self.controls.tokens,
            self.controls.warm,
            self.controls.placement,
            self.controls.page_cache,
        );
        let _ = writeln!(
            s,
            "build {} ({}), {} on {}, {} cards",
            self.build.version,
            self.build.commit,
            self.machine.host,
            self.machine.cpu,
            self.machine.cards.len()
        );
        let _ = writeln!(
            s,
            "experts: {} routed, {} per token, {:.1} MB each; cards hold {:?} per layer, host tier {:?} slots; heat from {}",
            self.model.routed_experts,
            self.model.activated,
            self.model.expert_bytes as f64 / 1e6,
            self.model.card_experts_per_layer,
            self.model.host_slots_per_layer,
            self.model.heat_source,
        );
        let _ = writeln!(
            s,
            "per token: {:.2} ms median, {:.2} mean, {:.2} min, {:.2} max",
            self.timing.median_ms, self.timing.mean_ms, self.timing.min_ms, self.timing.max_ms
        );
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "{:<32} {:>12} {:>12} {:>8}",
            "stage", "ms/token", "calls/token", "share"
        );
        let total: u64 = self.stages.iter().map(|r| r.total_us).sum();
        let mut stages = self.stages.clone();
        stages.sort_by_key(|r| std::cmp::Reverse(r.total_us));
        for r in &stages {
            let _ = writeln!(
                s,
                "{:<32} {:>12.3} {:>12.2} {:>7.1}%",
                r.stage,
                r.total_us as f64 / 1e3 / tokens,
                r.calls as f64 / tokens,
                if total > 0 {
                    r.total_us as f64 * 100.0 / total as f64
                } else {
                    0.0
                }
            );
        }
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "{:<32} {:>12} {:>12} {:>14} {:>14}",
            "step at the card seam", "taken/token", "declined/tok", "MB out/token", "MB back/token"
        );
        let c = &self.counters;
        for r in &c.steps {
            let _ = writeln!(
                s,
                "{:<32} {:>12.2} {:>12.2} {:>14.3} {:>14.3}",
                r.step,
                r.count.taken as f64 / tokens,
                r.count.declined as f64 / tokens,
                r.count.bytes_out as f64 / 1e6 / tokens,
                r.count.bytes_back as f64 / 1e6 / tokens,
            );
        }
        let taken: u64 = c.steps.iter().map(|r| r.count.taken).sum();
        let _ = writeln!(s);
        let _ = writeln!(s, "{:<32} {:>12} {:>12}", "counter", "per token", "total");
        let row = |s: &mut String, name: &str, v: u64| {
            let _ = writeln!(s, "{:<32} {:>12.2} {:>12}", name, v as f64 / tokens, v);
        };
        row(&mut s, "steps taken by a card (waits)", taken);
        row(&mut s, "expert uploads to the cards", c.expert_uploads);
        row(&mut s, "host tier hits", c.tier_hits);
        row(&mut s, "host tier misses", c.tier_misses);
        row(&mut s, "host tier evictions", c.tier_evictions);
        row(&mut s, "major page faults", c.process.major_faults);
        let _ = writeln!(
            s,
            "{:<32} {:>12.2} {:>12.1}",
            "MB read from disk",
            c.process.read_bytes as f64 / 1e6 / tokens,
            c.process.read_bytes as f64 / 1e6
        );
        s
    }
}

impl Placement {
    pub fn name(&self) -> &'static str {
        match self {
            Placement::Auto => "auto",
            Placement::Host => "host",
            Placement::Cards => "cards",
        }
    }
}

impl PageCache {
    pub fn name(&self) -> &'static str {
        match self {
            PageCache::Warm => "warm",
            PageCache::Cold => "cold",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path on this machine is placed on a disk, and the disk's line starts with its device.
    #[test]
    fn a_file_is_placed_on_a_disk() {
        let Some(d) = Disk::of(&std::env::temp_dir()) else {
            eprintln!("no block device tree for the temp directory; not covered by this run");
            return;
        };
        assert!(!d.device.is_empty());
        assert!(d.describe().starts_with(&d.device));
    }

    fn record() -> Record {
        Record {
            disk: None,
            build: Build {
                version: "0.5.0-dev+gabc".into(),
                commit: "abc".into(),
                features: vec!["cuda".into()],
            },
            machine: Machine {
                host: "ia".into(),
                cpu: "a cpu".into(),
                cores: 16,
                memory_bytes: 64 << 30,
                cards: vec![Card {
                    ordinal: 0,
                    total_bytes: 16 << 30,
                    memory_gbs: 0.0,
                    perf_gflops: 1,
                }],
            },
            controls: ControlsRecord {
                file: "m.gguf".into(),
                layers: (10, 13),
                tokens: 4,
                warm: 1,
                placement: "auto".into(),
                page_cache: "warm".into(),
                readahead: Some(2),
                card_experts: None,
                guess_hit: 1.0,
                seed: 1,
                expert_profile: None,
            },
            model: ModelRecord {
                architecture: "arch".into(),
                layers: 40,
                held: (10, 13),
                routed_experts: 384,
                activated: 6,
                expert_bytes: 9_500_000,
                cards_opened: true,
                card_experts_per_layer: vec![50, 50, 50],
                host_slots_per_layer: vec![90, 90, 90],
                heat_source: "prior".into(),
            },
            timing: Timing::of(vec![4.0, 2.0, 3.0, 9.0]),
            stages: vec![
                StageRow {
                    stage: "attention".into(),
                    total_us: 8000,
                    calls: 12,
                },
                StageRow {
                    stage: "moe expert prep".into(),
                    total_us: 12000,
                    calls: 12,
                },
            ],
            counters: Counters {
                steps: vec![StepRow {
                    step: "projection".into(),
                    count: StepCount {
                        taken: 40,
                        declined: 4,
                        bytes_out: 4_000_000,
                        bytes_back: 2_000_000,
                    },
                }],
                expert_uploads: 8,
                tier_hits: 20,
                tier_misses: 4,
                tier_evictions: 1,
                process: ProcessCounters {
                    major_faults: 3,
                    read_bytes: 40_000_000,
                },
            },
        }
    }

    /// The timing summarises an even and an odd count of steps.
    #[test]
    fn timing_summarises_the_steps() {
        let t = Timing::of(vec![4.0, 2.0, 3.0, 9.0]);
        assert_eq!(
            (t.median_ms, t.mean_ms, t.min_ms, t.max_ms),
            (3.5, 4.5, 2.0, 9.0)
        );
        assert_eq!(Timing::of(vec![5.0, 1.0, 3.0]).median_ms, 3.0);
        assert_eq!(Timing::of(Vec::new()).median_ms, 0.0);
    }

    /// The record round-trips through JSON.
    #[test]
    fn the_record_round_trips_through_json() {
        let r = record();
        let back: Record = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back, r);
        assert!(r.to_json().contains("\"bytes_out\": 4000000"));
    }

    /// The table names the run, the stages by time and the counters per token.
    #[test]
    fn the_table_shows_stages_by_time_and_counters_per_token() {
        let t = record().table();
        assert!(t.contains("layers [10, 13) of 40, 4 tokens after 1 warm, placement auto"));
        assert!(t.contains("3.50 ms median"));
        let prep = t.find("moe expert prep").unwrap();
        let attn = t.find("attention").unwrap();
        assert!(prep < attn, "the longer stage comes first");
        // The columns are read with their runs of spaces collapsed: the figures matter, the
        // widths do not.
        let rows: Vec<String> = t
            .lines()
            .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect();
        assert!(
            rows.contains(&"expert uploads to the cards 2.00 8".to_string()),
            "{t}"
        );
        assert!(
            rows.contains(&"projection 10.00 1.00 1.000 0.500".to_string()),
            "{t}"
        );
        assert!(
            rows.contains(&"MB read from disk 10.00 40.0".to_string()),
            "{t}"
        );
    }
}
