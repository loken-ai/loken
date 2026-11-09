//! What a decode step moves and waits on at the card seam, and what the kernel counts of the
//! process.
//!
//! The crossings are the calls a layer offers a card, by kind, with the bytes handed over and read
//! back. A route with no such seam leaves them empty.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// One kind of step, as counted at the seam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StepCount {
    /// Calls the card took.
    pub taken: u64,
    /// Calls the card declined, run on the host.
    pub declined: u64,
    /// Bytes handed to the card over the taken calls.
    pub bytes_out: u64,
    /// Bytes read back from the card over the taken calls.
    pub bytes_back: u64,
}

/// The counts of every kind of step since the counting began.
#[derive(Default)]
pub struct Crossings {
    steps: Mutex<BTreeMap<&'static str, StepCount>>,
}

impl Crossings {
    /// Note one call of `step` that handed `out` bytes and got `back` bytes, or was declined.
    pub fn note(&self, step: &'static str, taken: bool, out: u64, back: u64) {
        let mut m = self.steps.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(step).or_default();
        if taken {
            e.taken += 1;
            e.bytes_out += out;
            e.bytes_back += back;
        } else {
            e.declined += 1;
        }
    }

    /// Every kind of step and its counts, by name.
    pub fn snapshot(&self) -> Vec<(&'static str, StepCount)> {
        self.steps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// Zero every count.
    pub fn reset(&self) {
        self.steps.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// What the kernel counts for this process: the page faults that read from the disk, and the
/// bytes the process had read from storage. Zero on a system that does not say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessCounters {
    pub major_faults: u64,
    pub read_bytes: u64,
}

impl ProcessCounters {
    /// The counters now.
    pub fn now() -> Self {
        Self {
            major_faults: major_faults().unwrap_or(0),
            read_bytes: read_bytes().unwrap_or(0),
        }
    }

    /// What was counted between `earlier` and this.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            major_faults: self.major_faults.saturating_sub(earlier.major_faults),
            read_bytes: self.read_bytes.saturating_sub(earlier.read_bytes),
        }
    }
}

/// The process's major page faults, from the kernel's status line of the process.
#[cfg(target_os = "linux")]
fn major_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name is in parentheses and may hold spaces: the fields start after it.
    let after = &stat[stat.rfind(')')? + 1..];
    // The fields after the name: state, ppid, pgrp, session, tty, tpgid, flags, minflt, cminflt,
    // majflt. The major faults are the tenth.
    after.split_whitespace().nth(9)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
fn major_faults() -> Option<u64> {
    None
}

/// Bytes the process, every thread of it, had read from storage.
#[cfg(target_os = "linux")]
fn read_bytes() -> Option<u64> {
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    io.lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
}

#[cfg(not(target_os = "linux"))]
fn read_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A taken step counts its bytes each way; a declined one counts only that it was declined.
    #[test]
    fn taken_and_declined_steps_are_counted_with_their_bytes() {
        let c = Crossings::default();
        c.note("projection", true, 32, 16);
        c.note("projection", true, 16, 8);
        c.note("sparse attention", false, 0, 0);
        let snap: BTreeMap<&str, StepCount> = c.snapshot().into_iter().collect();
        assert_eq!(
            snap["projection"],
            StepCount {
                taken: 2,
                declined: 0,
                bytes_out: 48,
                bytes_back: 24,
            }
        );
        assert_eq!(
            snap["sparse attention"],
            StepCount {
                taken: 0,
                declined: 1,
                bytes_out: 0,
                bytes_back: 0,
            }
        );
        c.reset();
        assert!(c.snapshot().is_empty());
    }

    /// The process counters read and never go backwards.
    #[test]
    fn process_counters_are_read_and_monotonic() {
        let a = ProcessCounters::now();
        let b = ProcessCounters::now();
        let d = b.since(&a);
        assert!(d.major_faults <= b.major_faults);
        assert!(d.read_bytes <= b.read_bytes);
        assert_eq!(a.since(&b).major_faults, 0);
    }
}
