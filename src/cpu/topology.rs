//! CPU topology detection and core affinity management

use std::collections::HashSet;
use tracing::info;
#[cfg(target_os = "windows")]
use tracing::warn;

/// CPU core type (for hybrid CPUs like Intel Alder Lake/Raptor Lake)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    /// Performance core (P-core)
    Performance,
    /// Efficiency core (E-core)
    Efficiency,
    /// Unknown core type (fallback)
    Unknown,
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Total number of logical cores
    pub total_cores: usize,
    /// Performance core IDs
    pub p_cores: Vec<usize>,
    /// Efficiency core IDs
    pub e_cores: Vec<usize>,
    /// Whether hybrid CPU detection succeeded
    pub is_hybrid: bool,
}

impl CpuTopology {
    /// Get the optimal cores for inference workloads (P-cores if available, otherwise all cores)
    pub fn get_inference_cores(&self) -> Vec<usize> {
        if !self.p_cores.is_empty() {
            self.p_cores.clone()
        } else {
            (0..self.total_cores).collect()
        }
    }
}

/// Detect CPU topology (P-cores vs E-cores)
pub fn detect_cpu_topology() -> CpuTopology {
    let total_cores = num_cpus::get();

    #[cfg(target_os = "linux")]
    {
        if let Some(topology) = detect_linux_topology(total_cores) {
            info!(
                "CPU topology: {} total cores, {} P-cores, {} E-cores (hybrid: {})",
                topology.total_cores,
                topology.p_cores.len(),
                topology.e_cores.len(),
                topology.is_hybrid
            );
            let mut detected_simd = simd_capabilities();
            if !detected_simd.is_empty() {
                detected_simd.sort();
                info!("CPU SIMD features: {}", detected_simd.join(", "));
            }
            return topology;
        }
    }

    #[cfg(target_os = "windows")]
    {
        match detect_windows_topology(total_cores) {
            Ok(topology) => {
                info!(
                    "CPU topology: {} total cores, {} P-cores, {} E-cores (hybrid: {})",
                    topology.total_cores,
                    topology.p_cores.len(),
                    topology.e_cores.len(),
                    topology.is_hybrid
                );
                return topology;
            }
            Err(e) => {
                #[allow(unused_imports)]
                use tracing::warn;
                warn!(
                    "Failed to detect Windows CPU topology: {}. Using fallback.",
                    e
                );
            }
        }
    }

    // Fallback: assume all cores are performance cores
    let fallback = CpuTopology {
        total_cores,
        p_cores: (0..total_cores).collect(),
        e_cores: Vec::new(),
        is_hybrid: false,
    };

    info!(
        "CPU topology (fallback): {} cores (assuming all performance cores)",
        fallback.total_cores
    );

    fallback
}

/// Linux P/E core detection via `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`.
/// Hybrid CPUs (Alder Lake, Raptor Lake, Meteor Lake) ship P and E cores at
/// different max frequencies; non-hybrid CPUs have a single freq tier.
///
/// Algorithm:
///   1. Read each CPU's max freq.
///   2. Group cores by max-freq value.
///   3. If exactly one group: not hybrid - all P-cores.
///   4. Otherwise: the highest-freq group is P-cores, the rest E-cores.
///
/// Returns None if cpufreq isn't readable (containers, certain virt setups);
/// caller falls back to the all-P-cores default.
#[cfg(target_os = "linux")]
/// One logical CPU per PHYSICAL core: the lowest SMT-sibling of each core.
/// `thread_siblings_list` lists the logical CPUs sharing a physical core
/// (e.g. "0,10"); we keep the leader of each distinct sibling group. This is
/// what "P-core" should mean for a memory-bound workload - running a second
/// thread on a core's hyperthread sibling adds contention, not bandwidth.
fn physical_core_leaders(total_cores: usize) -> Vec<usize> {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut leaders = Vec::new();
    for cpu in 0..total_cores {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let leader = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| {
                s.trim()
                    .split([',', '-'])
                    .next()
                    .and_then(|x| x.trim().parse::<usize>().ok())
            })
            .unwrap_or(cpu);
        if seen.insert(leader) {
            leaders.push(leader);
        }
    }
    leaders.sort_unstable();
    leaders
}

fn detect_linux_topology(total_cores: usize) -> Option<CpuTopology> {
    use std::collections::BTreeMap;
    let mut by_freq: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for cpu in 0..total_cores {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
        let raw = std::fs::read_to_string(&path).ok()?;
        let freq: u64 = raw.trim().parse().ok()?;
        by_freq.entry(freq).or_default().push(cpu);
    }
    if by_freq.is_empty() {
        return None;
    }
    // Physical-core leaders (collapse SMT siblings). Empty if topology is
    // unreadable, in which case fall back to the raw logical lists below.
    let leaders = physical_core_leaders(total_cores);
    let is_leader = |c: usize| leaders.is_empty() || leaders.contains(&c);
    let is_hybrid = by_freq.len() > 1;
    if !is_hybrid {
        // Non-hybrid SMT CPU: the "P-cores" are the physical cores (one
        // logical per core), NOT every hyperthread. Reporting all logical
        // threads as P-cores over-subscribes pools onto SMT siblings.
        let p_cores: Vec<usize> = if leaders.is_empty() {
            (0..total_cores).collect()
        } else {
            leaders
        };
        return Some(CpuTopology {
            total_cores,
            p_cores,
            e_cores: Vec::new(),
            is_hybrid: false,
        });
    }
    // Multiple freq tiers -> highest tier is P-cores, others E-cores. Collapse
    // SMT siblings within each tier so a P-core is counted once (Alder-Lake
    // P-cores have SMT; E-cores don't).
    let max_freq = *by_freq.keys().last().unwrap();
    let mut p_cores: Vec<usize> = by_freq
        .remove(&max_freq)
        .unwrap_or_default()
        .into_iter()
        .filter(|&c| is_leader(c))
        .collect();
    let mut e_cores: Vec<usize> = by_freq
        .into_values()
        .flatten()
        .filter(|&c| is_leader(c))
        .collect();
    p_cores.sort();
    e_cores.sort();
    Some(CpuTopology {
        total_cores,
        p_cores,
        e_cores,
        is_hybrid: true,
    })
}

/// Best-effort enumeration of the SIMD features the current x86/ARM
/// CPU advertises at runtime. Used at server start so operators can
/// see whether the build will exercise AVX-512 / VNNI / NEON paths.
/// Returns an empty Vec on non-x86/ARM targets.
pub fn simd_capabilities() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, present) in [
            ("sse4.2", is_x86_feature_detected!("sse4.2")),
            ("avx", is_x86_feature_detected!("avx")),
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
            ("avx512vnni", is_x86_feature_detected!("avx512vnni")),
            ("avx512bf16", is_x86_feature_detected!("avx512bf16")),
            ("avx512_bf16", is_x86_feature_detected!("avx512bf16")),
        ] {
            if present {
                out.push(name);
            }
        }
        out.sort();
        out.dedup();
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64; advertise the optional ones.
        out.push("neon");
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            out.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            out.push("i8mm");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            out.push("fp16");
        }
    }
    out
}

#[cfg(target_os = "windows")]
fn detect_windows_topology(total_cores: usize) -> Result<CpuTopology, String> {
    // Windows API for CPU topology detection would go here
    // For now, use a heuristic based on core count patterns
    // Real implementation would use GetSystemCpuSetInformation

    // Heuristic: Intel hybrid CPUs (12th gen+) typically have patterns like:
    // - 12th gen: 8P+8E (16 cores), 6P+8E (14 cores), 6P+4E (10 cores)
    // - 13th/14th gen: similar patterns

    // Simple heuristic: if total_cores >= 12, assume hybrid with roughly 1:1 or 2:3 P:E ratio
    let is_hybrid = total_cores >= 12;

    let (p_cores, e_cores) = if is_hybrid {
        // Assume first 60% are P-cores, rest are E-cores (rough heuristic)
        let p_count = (total_cores as f32 * 0.6).ceil() as usize;
        let p_cores: Vec<usize> = (0..p_count).collect();
        let e_cores: Vec<usize> = (p_count..total_cores).collect();
        (p_cores, e_cores)
    } else {
        // All cores are P-cores
        let p_cores: Vec<usize> = (0..total_cores).collect();
        (p_cores, Vec::new())
    };

    Ok(CpuTopology {
        total_cores,
        p_cores,
        e_cores,
        is_hybrid,
    })
}

/// Set thread affinity to specific CPU cores
pub fn set_thread_affinity(core_ids: &[usize]) -> Result<(), String> {
    if core_ids.is_empty() {
        return Err("No core IDs provided".to_string());
    }

    // Build core affinity set
    let core_set: HashSet<usize> = core_ids.iter().copied().collect();

    // Get core IDs that exist on this system
    let available_cores = core_affinity::get_core_ids()
        .ok_or_else(|| "Failed to get available core IDs".to_string())?;

    // Filter to only valid cores
    let valid_cores: Vec<_> = available_cores
        .into_iter()
        .filter(|id| core_set.contains(&id.id))
        .collect();

    if valid_cores.is_empty() {
        return Err("No valid core IDs in the requested set".to_string());
    }

    // Set affinity to the first valid core (core_affinity crate limitation)
    // Note: This sets affinity for the current thread only
    if let Some(first_core) = valid_cores.first() {
        if core_affinity::set_for_current(*first_core) {
            Ok(())
        } else {
            Err("Failed to set thread affinity".to_string())
        }
    } else {
        Err("No valid cores available".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(total: usize, p: Vec<usize>, e: Vec<usize>, hybrid: bool) -> CpuTopology {
        CpuTopology {
            total_cores: total,
            p_cores: p,
            e_cores: e,
            is_hybrid: hybrid,
        }
    }

    #[test]
    fn get_inference_cores_prefers_p_cores_when_available() {
        // On a hybrid CPU (Alder Lake / Raptor Lake) we want inference
        // pinned to P-cores. Without this, the OS scheduler might park
        // workers on E-cores under load, halving throughput.
        let t = topology(
            16,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15],
            true,
        );
        assert_eq!(t.get_inference_cores(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn get_inference_cores_returns_all_cores_on_homogeneous_cpu() {
        // Non-hybrid CPU (most server/workstation chips) - no P/E
        // distinction, so use everything.
        let t = topology(8, vec![], vec![], false);
        assert_eq!(t.get_inference_cores(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn get_inference_cores_returns_all_when_p_cores_empty_even_if_e_cores_present() {
        // Defensive: if topology detection found E-cores but no P-cores
        // (would be a detection bug), fall back to all cores rather
        // than leaving the worker with no scheduling targets.
        let t = topology(4, vec![], vec![0, 1, 2, 3], true);
        assert_eq!(
            t.get_inference_cores(),
            vec![0, 1, 2, 3],
            "no P-cores -> fall back to all cores, not empty"
        );
    }

    #[test]
    fn get_inference_cores_handles_zero_cores_edge() {
        // Pathological: total_cores=0 (NUMA detection failure mode).
        // Must return empty, not panic or wrap.
        let t = topology(0, vec![], vec![], false);
        assert!(t.get_inference_cores().is_empty());
    }

    #[test]
    fn core_type_equality_matches_documented_semantics() {
        // CoreType drives the hybrid-cpu logic; pin equality so a
        // future refactor adding a new variant doesn't accidentally
        // overload Performance/Efficiency comparisons.
        assert_eq!(CoreType::Performance, CoreType::Performance);
        assert_eq!(CoreType::Efficiency, CoreType::Efficiency);
        assert_eq!(CoreType::Unknown, CoreType::Unknown);
        assert_ne!(CoreType::Performance, CoreType::Efficiency);
        assert_ne!(CoreType::Performance, CoreType::Unknown);
        assert_ne!(CoreType::Efficiency, CoreType::Unknown);
    }
}
