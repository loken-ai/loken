//! The size of the host's kept expert tier, moved while the server runs by what the machine
//! reports of its memory. The tier grows by one expert per layer while the host has the room for
//! that step beside what the model needs and the memory stalls have not risen with the last
//! growth, and falls back one step when they have. The kernel's own count of the time tasks wait
//! for memory is the signal, so no share of the memory is set aside by a rule of thumb.

use std::time::Instant;

/// What the next size of the tier is.
///
/// `slots` experts per layer are kept now, at most `ceiling`. `stall` is the share of the time
/// since the last look that tasks waited for memory and `stall_before` the same share at the size
/// below the current one (`None` before the first growth). `available` is the memory the host
/// can still give, `need` what the model keeps outside the tier and `step_bytes` what one more
/// expert in every layer takes.
pub fn next_slots(
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
pub struct TierTune {
    pub slots: usize,
    pub ceiling: usize,
    pub need: u64,
    pub step_bytes: u64,
    /// The memory-stall share measured before the last growth.
    stall_before: Option<f64>,
    /// The kernel's stall total and the clock at the last look.
    last: Option<(u64, Instant)>,
}

impl TierTune {
    pub fn new(slots: usize, ceiling: usize, need: u64, step_bytes: u64) -> Self {
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
    pub fn look(&mut self, available: u64) -> Option<usize> {
        let total = stall_total_us()?;
        let now = Instant::now();
        let Some((was, at)) = self.last.replace((total, now)) else {
            return Some(self.slots);
        };
        let wall = now.duration_since(at).as_micros().max(1) as f64;
        let stall = total.saturating_sub(was) as f64 / wall;
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

    /// The pressure report of this machine, where there is one, parses.
    #[test]
    fn the_stall_total_reads() {
        if std::path::Path::new("/proc/pressure/memory").exists() {
            assert!(stall_total_us().is_some());
        }
    }
}
