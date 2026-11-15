//! The size of the host's kept expert tier, moved while the server runs by what the machine
//! reports of its memory.
//!
//! The tier grows by one expert per layer while the host has the room for that step beside what
//! the model needs and the memory stalls have not risen with the last growth.
//! It falls back one step when they have.
//! The kernel's own count of the time tasks wait for memory is the signal.
//!
//! Nothing here locks a page. The tier that owns the locks applies the size.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Whether tasks waited for memory during the last interval the tier was looked at.
///
/// Pages of experts the tier does not keep are dropped from the page cache only while it is true.
/// With the host's memory not short, they stay for the next use, and the kernel takes them back
/// itself when it needs the room.
///
/// True until a look says otherwise, and where the system reports no memory stalls.
static MEMORY_STALLED: AtomicBool = AtomicBool::new(true);

/// A tuned host tier exists: set when the first [`TierTune`] is made.
static HOST_TIER: AtomicBool = AtomicBool::new(false);

/// See [`MEMORY_STALLED`].
pub(crate) fn memory_stalled() -> bool {
    MEMORY_STALLED.load(Ordering::Relaxed)
}

/// Whether the pages of an expert the host tier does not keep go once it has been used: a tuned
/// tier exists and the host's memory is short.
pub(crate) fn drops_unkept_pages() -> bool {
    HOST_TIER.load(Ordering::Relaxed) && memory_stalled()
}

/// Whether a stall share means the host's memory is short: tasks waited for it at all.
fn memory_short(stall: f64) -> bool {
    stall > 0.0
}

/// What the next size of the tier is.
///
/// `slots` experts per layer are kept now, at most `ceiling`.
/// `stall` is the share of the time since the last look that tasks waited for memory.
/// `stall_before` is the same share at the size below the current one, `None` before the first
/// growth.
/// `available` is the memory the host can still give.
/// `need` is what the model keeps outside the tier.
/// `step_bytes` is what one more expert in every layer takes.
pub(crate) fn next_slots(
    slots: usize,
    ceiling: usize,
    stall: f64,
    stall_before: Option<f64>,
    available: u64,
    need: u64,
    step_bytes: u64,
) -> usize {
    // The memory waits rose with the last growth: the tier is past what the host holds.
    if stall_before.is_some_and(|before| stall > before) {
        return slots.saturating_sub(1).max(1);
    }
    if slots < ceiling && available >= need.saturating_add(step_bytes) {
        return slots + 1;
    }
    slots
}

/// The tier's size over time, with what it needs to move it.
pub(crate) struct TierTune {
    pub(crate) slots: usize,
    pub(crate) ceiling: usize,
    pub(crate) need: u64,
    pub(crate) step_bytes: u64,
    /// The memory-stall share measured before the last growth.
    stall_before: Option<f64>,
    /// The kernel's stall total and the clock at the last look.
    last: Option<(u64, Instant)>,
}

impl TierTune {
    pub(crate) fn new(slots: usize, ceiling: usize, need: u64, step_bytes: u64) -> Self {
        HOST_TIER.store(true, Ordering::Relaxed);
        // A system that reports memory stalls starts with the pages kept: the first look says
        // whether that was right.
        if stall_total_us().is_some() {
            MEMORY_STALLED.store(false, Ordering::Relaxed);
        }
        Self {
            slots,
            ceiling,
            need,
            step_bytes,
            stall_before: None,
            last: None,
        }
    }

    /// Look at the memory now and give the size the tier should have, `None` where the system
    /// does not report memory stalls.
    pub(crate) fn look(&mut self, available: u64) -> Option<usize> {
        let total = stall_total_us()?;
        let now = Instant::now();
        let Some((was, at)) = self.last.replace((total, now)) else {
            return Some(self.slots);
        };
        let wall = now.duration_since(at).as_micros().max(1) as f64;
        let stall = total.saturating_sub(was) as f64 / wall;
        MEMORY_STALLED.store(memory_short(stall), Ordering::Relaxed);
        let next = next_slots(
            self.slots,
            self.ceiling,
            stall,
            self.stall_before,
            available,
            self.need,
            self.step_bytes,
        );
        self.stall_before = (next > self.slots).then_some(stall);
        self.slots = next;
        Some(next)
    }
}

/// Microseconds that some task waited for memory since boot, from the kernel's pressure report.
fn stall_total_us() -> Option<u64> {
    let report = std::fs::read_to_string("/proc/pressure/memory").ok()?;
    report
        .lines()
        .find(|l| l.starts_with("some"))?
        .split_whitespace()
        .find_map(|w| w.strip_prefix("total="))?
        .parse()
        .ok()
}

/// Bytes of memory the host can still give this process, its control group included.
pub(crate) fn available_bytes() -> u64 {
    crate::inference::place::layer_executor::available_host_ram_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// The tier grows while the host has room and no stall rose.
    #[test]
    fn it_grows_into_free_room() {
        assert_eq!(next_slots(100, 384, 0.0, None, 20 * GB, 8 * GB, GB), 101);
        assert_eq!(
            next_slots(100, 384, 0.0, Some(0.0), 20 * GB, 8 * GB, GB),
            101
        );
    }

    /// It holds when the step would eat into what the model needs, and at the ceiling.
    #[test]
    fn it_holds_without_room_and_at_the_ceiling() {
        assert_eq!(next_slots(100, 384, 0.0, None, 8 * GB, 8 * GB, GB), 100);
        assert_eq!(next_slots(384, 384, 0.0, None, 99 * GB, 8 * GB, GB), 384);
    }

    /// It steps back when the stalls rose after a growth.
    #[test]
    fn it_steps_back_when_the_stalls_rose() {
        assert_eq!(
            next_slots(101, 384, 0.02, Some(0.0), 20 * GB, 8 * GB, GB),
            100
        );
        assert_eq!(next_slots(1, 384, 0.5, Some(0.0), 20 * GB, 8 * GB, GB), 1);
    }

    /// Where the system reports memory stalls, a tuner starts with the pages of unkept experts
    /// left in the cache, and the first look sets the flag from the stalls it measures.
    #[test]
    fn pages_stay_until_memory_is_short() {
        if !std::path::Path::new("/proc/pressure/memory").exists() {
            return;
        }
        let _tune = TierTune::new(10, 384, 0, GB);
        assert!(!memory_stalled());
        assert!(!drops_unkept_pages());
    }

    /// Any wait for memory in an interval makes it short, and none leaves it not short.
    #[test]
    fn memory_is_short_when_tasks_waited_for_it() {
        assert!(!memory_short(0.0));
        assert!(memory_short(0.0001));
    }

    /// The pressure report of this machine, where there is one, parses.
    #[test]
    fn the_stall_total_reads() {
        if std::path::Path::new("/proc/pressure/memory").exists() {
            assert!(stall_total_us().is_some());
        }
    }

    /// The first look keeps the size and starts the interval the next look measures.
    #[test]
    fn the_first_look_keeps_the_size() {
        if !std::path::Path::new("/proc/pressure/memory").exists() {
            return;
        }
        let mut tune = TierTune::new(10, 384, 0, GB);
        assert_eq!(tune.look(20 * GB), Some(10));
        assert_eq!(tune.slots, 10);
    }
}
