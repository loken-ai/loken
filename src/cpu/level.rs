//! Runtime selection of the SIMD level the CPU kernels run at.
//!
//! The level is detected once at first use and every kernel dispatches on it. `[cpu] level` in
//! the configuration caps it below what the hardware offers, so each path can be exercised on a
//! machine that has more. A host below the build's baseline is refused at start by
//! [`check_baseline`].

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// SIMD tiers, ordered: a higher tier implies every feature of the lower ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuLevel {
    /// Scalar reference code, selected only by `[cpu] level = "generic"`.
    Generic,
    /// `avx2`, `fma`, `f16c`, `bmi2`.
    Avx2,
    /// Avx2 plus `avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512vnni`.
    Avx512,
}

impl CpuLevel {
    pub const ALL: [CpuLevel; 3] = [CpuLevel::Generic, CpuLevel::Avx2, CpuLevel::Avx512];

    pub fn name(self) -> &'static str {
        match self {
            CpuLevel::Generic => "generic",
            CpuLevel::Avx2 => "avx2",
            CpuLevel::Avx512 => "avx512",
        }
    }

    pub fn parse(s: &str) -> Option<CpuLevel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "generic" | "scalar" => Some(CpuLevel::Generic),
            "avx2" => Some(CpuLevel::Avx2),
            "avx512" => Some(CpuLevel::Avx512),
            _ => None,
        }
    }
}

/// Fails with a message naming the missing features when the CPU lacks what the build was
/// compiled for (x86-64-v3). Called first thing in each binary.
pub fn check_baseline() -> Result<(), String> {
    #[cfg(target_arch = "x86_64")]
    {
        let missing: Vec<&str> = [
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("f16c", is_x86_feature_detected!("f16c")),
            ("bmi1", is_x86_feature_detected!("bmi1")),
            ("bmi2", is_x86_feature_detected!("bmi2")),
            ("lzcnt", is_x86_feature_detected!("lzcnt")),
            ("movbe", is_x86_feature_detected!("movbe")),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect();
        if !missing.is_empty() {
            return Err(format!(
                "this build needs an x86-64-v3 CPU; this one lacks {}",
                missing.join(", ")
            ));
        }
    }
    Ok(())
}

/// Highest level the running CPU supports, ignoring any override.
pub fn hardware_level() -> CpuLevel {
    static HW: OnceLock<CpuLevel> = OnceLock::new();
    *HW.get_or_init(detect_hardware_level)
}

fn detect_hardware_level() -> CpuLevel {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("bmi2");
        if !avx2 {
            return CpuLevel::Generic;
        }
        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vnni");
        if avx512 {
            CpuLevel::Avx512
        } else {
            CpuLevel::Avx2
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        CpuLevel::Generic
    }
}

const UNSET: u8 = u8::MAX;

/// The level kernels dispatch on, as `CpuLevel as u8`: the hardware level, lowered by
/// [`set_level_cap`]. One relaxed load per dispatch.
static EFFECTIVE: AtomicU8 = AtomicU8::new(UNSET);

fn from_u8(v: u8) -> CpuLevel {
    CpuLevel::ALL[v as usize]
}

/// Level the kernels dispatch on: the hardware level, capped by the configuration.
#[inline]
pub fn cpu_level() -> CpuLevel {
    match EFFECTIVE.load(Ordering::Relaxed) {
        UNSET => {
            let hw = hardware_level() as u8;
            // A concurrent `set_level_cap` wins over this default.
            let _ = EFFECTIVE.compare_exchange(UNSET, hw, Ordering::Relaxed, Ordering::Relaxed);
            from_u8(EFFECTIVE.load(Ordering::Relaxed))
        }
        v => from_u8(v),
    }
}

/// Cap the dispatch level at `cap` (never above what the hardware has), or lift the cap with
/// `None`. Called once at start from `[cpu] level`, before any weight is repacked: the repacked
/// layouts depend on the level.
pub fn set_level_cap(cap: Option<CpuLevel>) {
    let hw = hardware_level();
    let level = cap.map_or(hw, |c| c.min(hw));
    EFFECTIVE.store(level as u8, Ordering::Relaxed);
}

/// Whether kernels may dispatch to the AVX2 tier (hardware allows it and the override does not forbid it).
#[inline]
pub fn dispatch_avx2() -> bool {
    cpu_level() >= CpuLevel::Avx2
}

/// Whether kernels may dispatch to the AVX-512 tier.
#[inline]
pub fn dispatch_avx512() -> bool {
    cpu_level() >= CpuLevel::Avx512
}

/// Whether the AVX-512 kernels can be executed at all, whatever the cap says. The guard in front
/// of every AVX-512 `#[target_feature]` entry point.
#[inline]
pub fn hardware_has_avx512() -> bool {
    hardware_level() >= CpuLevel::Avx512
}

/// Whether the AVX2 kernels can be executed at all, whatever the override says. The guard in
/// front of every `#[target_feature]` entry point: running one without the feature is undefined.
#[inline]
pub fn hardware_has_avx2() -> bool {
    hardware_level() >= CpuLevel::Avx2
}

/// Every level this host can execute, lowest first. Parity tests iterate over it.
pub fn supported_levels() -> Vec<CpuLevel> {
    let hw = hardware_level();
    CpuLevel::ALL.into_iter().filter(|l| *l <= hw).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host that built and runs this test has the baseline it was compiled for, and the check
    /// agrees with the level detection: AVX2 present means the level is not `generic`.
    #[test]
    fn this_host_meets_the_baseline_it_was_built_for() {
        assert_eq!(check_baseline(), Ok(()));
        assert!(hardware_level() >= CpuLevel::Avx2);
    }

    #[test]
    fn parse_accepts_every_name_and_rejects_the_rest() {
        for l in CpuLevel::ALL {
            assert_eq!(CpuLevel::parse(l.name()), Some(l));
        }
        assert_eq!(CpuLevel::parse(" AVX512 "), Some(CpuLevel::Avx512));
        assert_eq!(CpuLevel::parse("avx1024"), None);
    }

    #[test]
    fn supported_levels_end_at_the_hardware_level_and_the_cap_never_exceeds_it() {
        let s = supported_levels();
        assert_eq!(s.first(), Some(&CpuLevel::Generic));
        assert_eq!(s.last(), Some(&hardware_level()));
        assert!(cpu_level() <= hardware_level());
    }
}
