//! What the decode-state caches derive instead of being told: the block size of a model's hash
//! chain, the spacing of the states a prefill keeps, the budget, whether a stored prefix is worth
//! restoring, and whether a write may start.
//!
//! Every value comes from a quantity the machine or the model reports. The model gives the bytes a
//! token adds to a state and the granule a block must be a multiple of; the disk gives its free
//! space, its per-file cost and its throughput, measured on the cache directory; the engine gives
//! the prefill chunk it really uses and, as it runs, the seconds a token of prefill and a restored
//! byte cost. Rates are volume-weighted (all bytes over all seconds seen), so no smoothing constant
//! stands in for a tuned one. Each function is pure so a test can move one input and see the
//! output move.

use std::path::Path;
use std::sync::Mutex;

/// Seconds and units observed, accumulated: the rate is their ratio.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rate {
    units: f64,
    secs: f64,
}

impl Rate {
    pub fn add(&mut self, units: f64, secs: f64) {
        if units > 0.0 && secs > 0.0 {
            self.units += units;
            self.secs += secs;
        }
    }

    /// Units per second, `None` before anything was observed.
    pub fn per_sec(&self) -> Option<f64> {
        (self.secs > 0.0).then(|| self.units / self.secs)
    }

    /// Seconds per unit, `None` before anything was observed.
    pub fn secs_per(&self) -> Option<f64> {
        (self.units > 0.0).then(|| self.secs / self.units)
    }
}

/// What the running engine has measured of itself and of the disk.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measured {
    /// Tokens prefilled per second spent prefilling them.
    pub prefill: Rate,
    /// Bytes written per second, a put's whole duration over its bytes.
    pub write: Rate,
    /// Bytes restored per second, a restore's whole duration (read and decode) over its bytes.
    pub restore: Rate,
    /// Largest single state written, manifest and blocks, in bytes.
    pub largest_state: u64,
    /// Bytes of one manifest, as last written.
    pub manifest_bytes: u64,
}

/// The measurements one store carries, shared by every cache that writes to it.
#[derive(Default)]
pub struct Tuning(Mutex<Measured>);

impl Tuning {
    pub fn get(&self) -> Measured {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with(&self, f: impl FnOnce(&mut Measured)) {
        f(&mut self.0.lock().unwrap_or_else(|e| e.into_inner()));
    }

    pub fn prefilled(&self, tokens: usize, secs: f64) {
        self.with(|m| m.prefill.add(tokens as f64, secs));
    }

    /// A state stored: its whole size, manifest and blocks, and the size of its manifest.
    pub fn stored(&self, state_bytes: u64, manifest: u64) {
        self.with(|m| {
            m.largest_state = m.largest_state.max(state_bytes);
            m.manifest_bytes = manifest;
        });
    }

    /// Bytes a put wrote and the seconds it took.
    pub fn write_rate(&self, bytes: u64, secs: f64) {
        self.with(|m| m.write.add(bytes as f64, secs));
    }

    pub fn restored(&self, bytes: u64, secs: f64) {
        self.with(|m| m.restore.add(bytes as f64, secs));
    }
}

/// Smallest multiple of `unit` that is at least `n`.
fn round_up(n: usize, unit: usize) -> usize {
    n.div_ceil(unit.max(1)) * unit.max(1)
}

/// The tokens per block of a model's hash chain: a multiple of `granule` (what the model's state
/// divides into whole groups by), large enough that a block file's transfer time covers the
/// disk's per-file cost (`file_cost_s` at `write_bps`), and no larger than half of
/// `cap_bytes`, so a block always fits the budget with room for another. `token_bytes` is what a
/// token adds to the blocks. Powers of two times the granule, so a block size is stable under small
/// changes of the measurements.
pub fn block_tokens(
    granule: usize,
    token_bytes: usize,
    file_cost_s: f64,
    write_bps: f64,
    cap_bytes: u64,
) -> usize {
    let granule = granule.max(1);
    let unit_bytes = (granule * token_bytes.max(1)) as f64;
    let wanted = (file_cost_s * write_bps / unit_bytes).ceil().max(1.0) as usize;
    let mut mult = wanted.next_power_of_two();
    let fits = |m: usize| (m as f64) * unit_bytes <= (cap_bytes / 2) as f64;
    while mult > 1 && !fits(mult) {
        mult /= 2;
    }
    granule * mult
}

/// The spacing, in tokens, of the states a prefill keeps: the least a write of one manifest
/// (`manifest_bytes` at `write_bps`) needs for the disk to finish it within the prefill time of the
/// tokens in between (`prefill_s_per_token`), so a write never trails the compute that made it. A
/// multiple of `block`, and of `chunk` once it spans more than one: states are cut where the
/// prefill's own forwards end. Before the disk or the prefill has been measured the spacing is the
/// chunk itself, the finest the prefill passes.
pub fn grid_stride(
    block: usize,
    chunk: usize,
    manifest_bytes: u64,
    write_bps: Option<f64>,
    prefill_s_per_token: Option<f64>,
) -> usize {
    let block = block.max(1);
    let base = round_up(chunk.max(block), block);
    let (Some(w), Some(t)) = (write_bps, prefill_s_per_token) else {
        return base;
    };
    if manifest_bytes == 0 || w <= 0.0 || t <= 0.0 {
        return base;
    }
    let needed = (manifest_bytes as f64 / (w * t)).ceil() as usize;
    if needed <= base {
        return base;
    }
    round_up(needed, base)
}

/// The bytes a store may hold when the operator set none: what it holds already plus what the
/// filesystem still offers, less room for one more state of the largest size written.
pub fn derived_budget(held: u64, free: u64, largest_state: u64) -> u64 {
    (held + free).saturating_sub(largest_state)
}

/// Whether restoring `bytes` of stored state to skip `covered` tokens of prefill is expected to
/// beat prefilling them. True while either side is unmeasured: the first restore is how it gets
/// measured.
pub fn restore_pays(
    covered: usize,
    bytes: u64,
    restore_s_per_byte: Option<f64>,
    prefill_s_per_token: Option<f64>,
) -> bool {
    match (restore_s_per_byte, prefill_s_per_token) {
        (Some(r), Some(p)) => bytes as f64 * r < covered as f64 * p,
        _ => true,
    }
}

/// Whether a write of `bytes` may queue behind `pending` bytes already waiting: the queue must
/// drain, at `write_bps`, within `window_s`, the compute time of the prompt that produced the
/// state. Unmeasured, a write is allowed only on an empty queue.
pub fn write_may_queue(
    pending: u64,
    bytes: u64,
    write_bps: Option<f64>,
    window_s: Option<f64>,
) -> bool {
    if pending == 0 {
        return true;
    }
    match (write_bps, window_s) {
        (Some(w), Some(win)) if w > 0.0 => (pending + bytes) as f64 / w <= win,
        _ => false,
    }
}

/// The disk's per-file cost and throughput, from writing and syncing files in `dir`: time is
/// `file_cost + bytes / throughput`. It starts at `small` bytes and doubles the size until the time
/// is twice the first (the transfer then dominates and the rate is resolved) or `cap` is reached.
/// `None` when the directory cannot be written. The probe file is removed.
pub fn probe_disk(dir: &Path, small: usize, cap: usize) -> Option<(f64, f64)> {
    use std::io::Write;
    let time_one = |bytes: usize| -> Option<f64> {
        let path = dir.join(".tune-probe");
        let buf = vec![0x5Au8; bytes];
        let t0 = std::time::Instant::now();
        let mut f = std::fs::File::create(&path).ok()?;
        f.write_all(&buf).ok()?;
        f.sync_all().ok()?;
        let secs = t0.elapsed().as_secs_f64();
        let _ = std::fs::remove_file(&path);
        Some(secs)
    };
    let small = small.max(1);
    let ts = time_one(small)?;
    let (mut size, mut tl) = (small, ts);
    while size < cap && tl < 2.0 * ts {
        size = (size * 2).min(cap);
        tl = time_one(size)?;
    }
    let dt = tl - ts;
    if dt <= 0.0 || size == small {
        // Throughput is not resolved by this probe: the transfer of the sizes tried is within
        // the per-file cost.
        return Some((ts.max(tl), f64::INFINITY));
    }
    let bps = (size - small) as f64 / dt;
    let cost = (ts - small as f64 / bps).max(0.0);
    Some((cost, bps))
}

/// The block size of the chain under `scope`: settled once per store and scope from the model's
/// `granule` and `token_bytes` and the store's directory, measured. `None` when the directory
/// cannot be written to measure, and the cache stays off.
pub fn settle_block_tokens(
    store: &super::state_disk::StateStore,
    scope: &[u8; 32],
    granule: usize,
    token_bytes: usize,
) -> Option<usize> {
    let mut usable = true;
    let block = store.settle(scope, |s| {
        let room = s.room();
        let unit = granule.max(1) * token_bytes.max(1);
        let cap = usize::try_from(room / 2).unwrap_or(usize::MAX);
        match probe_disk(s.dir(), unit, cap) {
            Some((cost, bps)) => {
                let b = block_tokens(granule, token_bytes, cost, bps, room);
                tracing::info!(
                    "state cache: blocks of {b} tokens ({token_bytes} B a token; the directory costs {:.1} ms a file and writes {:.0} MB/s)",
                    cost * 1e3,
                    bps / 1e6
                );
                b
            }
            None => {
                usable = false;
                granule.max(1)
            }
        }
    });
    usable.then_some(block)
}

/// Free bytes on the filesystem holding `dir`.
pub fn free_space(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid NUL-terminated path and `st` a valid out parameter.
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_grows_with_the_disks_fixed_cost_and_shrinks_with_the_state_per_token() {
        let base = block_tokens(4, 1000, 0.002, 1.0e9, u64::MAX);
        assert!(
            block_tokens(4, 1000, 0.020, 1.0e9, u64::MAX) > base,
            "slower per file"
        );
        assert!(
            block_tokens(4, 4000, 0.002, 1.0e9, u64::MAX) < base,
            "more bytes per token"
        );
        assert!(
            block_tokens(4, 1000, 0.002, 4.0e9, u64::MAX) > base,
            "faster disk"
        );
        assert_eq!(base % 4, 0, "a multiple of the granule");
        assert_eq!(
            block_tokens(1, 10, 0.0, 1.0e9, u64::MAX),
            1,
            "no fixed cost: the granule"
        );
    }

    #[test]
    fn a_block_never_exceeds_half_the_room_it_must_fit() {
        let big = block_tokens(1, 1000, 1.0, 1.0e9, u64::MAX);
        let capped = block_tokens(1, 1000, 1.0, 1.0e9, 64 << 20);
        assert!(big > capped);
        assert!(capped as u64 * 1000 <= 32 << 20);
    }

    #[test]
    fn the_stride_follows_the_disk_and_the_prefill_it_must_not_trail() {
        let (block, chunk, manifest) = (256, 512, 90 << 20);
        assert_eq!(
            grid_stride(block, chunk, manifest, None, None),
            chunk,
            "unmeasured"
        );
        let fast = grid_stride(block, chunk, manifest, Some(2.0e9), Some(0.02));
        assert_eq!(fast, chunk, "a disk that keeps up: every chunk");
        let slow = grid_stride(block, chunk, manifest, Some(5.0e6), Some(0.02));
        assert!(slow > fast, "a slower disk spaces the states out");
        assert_eq!(slow % chunk, 0);
        let slow_prefill = grid_stride(block, chunk, manifest, Some(5.0e6), Some(0.002));
        assert!(
            slow_prefill > slow,
            "a faster prefill leaves the disk less time"
        );
        let bigger = grid_stride(block, chunk, 4 * manifest, Some(5.0e6), Some(0.02));
        assert!(bigger > slow, "a larger state");
    }

    #[test]
    fn the_budget_is_what_is_held_and_what_is_free_less_one_more_state() {
        assert_eq!(derived_budget(10, 100, 30), 80);
        assert!(derived_budget(10, 200, 30) > derived_budget(10, 100, 30));
        assert!(derived_budget(10, 100, 60) < derived_budget(10, 100, 30));
        assert_eq!(derived_budget(0, 5, 30), 0);
    }

    #[test]
    fn a_restore_is_used_while_it_beats_the_prefill_it_replaces() {
        assert!(
            restore_pays(1000, 1 << 30, None, Some(0.01)),
            "unmeasured: try"
        );
        // 1000 tokens at 10 ms is 10 s; 1 GiB at 1 GB/s is about 1 s.
        assert!(restore_pays(1000, 1 << 30, Some(1.0e-9), Some(0.01)));
        assert!(
            !restore_pays(1000, 1 << 30, Some(1.0e-7), Some(0.01)),
            "a slow disk loses"
        );
        assert!(
            !restore_pays(10, 1 << 30, Some(1.0e-9), Some(0.01)),
            "too few tokens covered"
        );
    }

    #[test]
    fn a_write_queues_only_while_the_disk_drains_within_the_compute_that_made_it() {
        assert!(write_may_queue(0, 1 << 30, None, None));
        assert!(!write_may_queue(1, 1 << 30, None, None));
        assert!(write_may_queue(1 << 20, 1 << 20, Some(1.0e9), Some(1.0)));
        assert!(!write_may_queue(1 << 30, 1 << 30, Some(1.0e8), Some(1.0)));
    }

    #[test]
    fn rates_are_volume_weighted() {
        let mut r = Rate::default();
        r.add(100.0, 1.0);
        r.add(900.0, 9.0);
        assert!((r.per_sec().unwrap() - 100.0).abs() < 1e-9);
        r.add(1000.0, 5.0);
        assert!(r.per_sec().unwrap() > 100.0);
        assert!(Rate::default().per_sec().is_none());
    }

    #[test]
    fn a_disk_probe_reports_a_cost_and_a_rate_and_leaves_nothing() {
        let dir = std::env::temp_dir().join(format!("loken-tune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (cost, bps) = probe_disk(&dir, 4096, 8 << 20).expect("writable");
        assert!(cost >= 0.0 && bps > 0.0);
        assert!(std::fs::read_dir(&dir).unwrap().next().is_none());
        assert!(free_space(&dir).unwrap() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
