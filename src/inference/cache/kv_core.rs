//! The lifecycle every device KV cache shares, whatever precision it stores.
//!
//! A cache is a length inside a capacity inside a hard cap, a device-side position the
//! replay-safe kernels read, and the state of any CUDA graph recorded against its buffers.
//! Each of those has rules that hold for every storage format:
//!
//! - a trim or a reset leaves the device position unprimed, so the appends that follow take
//!   the host-offset path until the first decode token primes it again;
//! - a live graph freezes the capacity (growth would move the buffers under the recorded
//!   kernels) and bounds the sequence the replays may reach;
//! - replayed tokens advance the cache on the device only, so the host length is synced
//!   from the engine after each one;
//! - the device position lives at one address for the life of the cache, so a graph that
//!   recorded it never reads a freed slot.
//!
//! `KvCore` holds the host side of those rules and `DevPos` the device slot; a storage
//! format keeps only its buffers and kernels.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Length, capacity, device-position and graph bookkeeping of one KV cache.
#[derive(Debug)]
pub struct KvCore {
    len: usize,
    cap: usize,
    max: usize,
    /// Slot index the device position currently names: the length before the append of the
    /// token being decoded. Kernels attend over `slot + 1` positions. `None` = unprimed.
    primed: Option<usize>,
    /// A CUDA graph recorded against the buffers is live.
    captured: bool,
    /// Longest sequence the live graph may replay to, frozen when it was marked.
    seq_limit: Option<usize>,
    /// Padded stride most recently used by a fixed-stride attention chain since the last
    /// capture cycle; 0 = none. Written from `&self` attention calls, hence atomic.
    chain_stride_seen: AtomicUsize,
}

impl KvCore {
    pub fn new(initial_cap: usize, max: usize) -> Self {
        Self {
            len: 0,
            cap: initial_cap.min(max),
            max,
            primed: None,
            captured: false,
            seq_limit: None,
            chain_stride_seen: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn max(&self) -> usize {
        self.max
    }

    /// Checks that `n_new` tokens fit under the hard cap; returns the capacity they need when
    /// that exceeds the current one.
    pub fn room_for(&self, n_new: usize) -> Result<Option<usize>, String> {
        let need = self.len + n_new;
        if need > self.max {
            return Err(format!(
                "would exceed max_seq_len {} (have {}, adding {n_new})",
                self.max, self.len
            ));
        }
        Ok((need > self.cap).then_some(need))
    }

    /// The capacity to grow to for `need` tokens: doubling, clamped to the cap. `None` when the
    /// current capacity already holds them. Refused while a graph is live.
    pub fn grow_target(&self, need: usize) -> Result<Option<usize>, String> {
        if self.captured {
            return Err(format!(
                "cannot grow while a captured graph references the buffers (capacity {}, requested {need})",
                self.cap
            ));
        }
        let target = need.max(self.cap * 2).min(self.max);
        Ok((target > self.cap).then_some(target))
    }

    /// The smallest capacity that still holds `need` tokens, for a retry after the doubled
    /// target did not fit in memory.
    pub fn minimal_target(&self, need: usize) -> usize {
        need.max(self.cap + 1).min(self.max)
    }

    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap.min(self.max);
    }

    /// Records `n_new` tokens written by an eager append.
    pub fn commit(&mut self, n_new: usize) {
        self.len += n_new;
    }

    /// Shortens to `new_len` (never lengthens). Unprimes the device position and releases the
    /// capture bookkeeping: a graph recorded against the longer cache no longer describes it.
    /// Returns the length before the trim.
    pub fn trim_to(&mut self, new_len: usize) -> usize {
        let old = self.len;
        self.len = new_len.min(old);
        self.primed = None;
        self.clear_captured();
        old
    }

    pub fn reset(&mut self) {
        self.trim_to(0);
    }

    pub fn is_primed(&self) -> bool {
        self.primed.is_some()
    }

    /// Names `slot` as the position the next replay writes. Returns whether the device copy
    /// has to be refreshed: every layer of one token primes the same slot.
    pub fn prime(&mut self, slot: usize) -> bool {
        if self.primed == Some(slot) {
            return false;
        }
        self.primed = Some(slot);
        true
    }

    pub fn is_captured(&self) -> bool {
        self.captured
    }

    /// A graph recorded against the buffers is now live: freezes growth, and bounds replays by
    /// the capacity and by the stride of a fixed-stride chain if one was recorded.
    pub fn mark_captured(&mut self) {
        self.captured = true;
        let stride = self.chain_stride_seen.load(Ordering::Relaxed);
        self.seq_limit = Some(if stride > 0 {
            self.cap.min(stride)
        } else {
            self.cap
        });
    }

    pub fn clear_captured(&mut self) {
        self.captured = false;
        self.seq_limit = None;
        self.chain_stride_seen.store(0, Ordering::Relaxed);
    }

    pub fn seq_limit(&self) -> Option<usize> {
        self.seq_limit
    }

    /// A fixed-stride attention chain ran at `stride`; a graph recorded now is bound by it.
    pub fn note_chain_stride(&self, stride: usize) {
        self.chain_stride_seen.store(stride, Ordering::Relaxed);
    }

    /// Host length after a replayed token, which advanced the cache on the device only.
    pub fn sync_len(&mut self, len: usize) {
        self.len = len.min(self.max);
    }
}

/// The device-side position the replay-safe kernels read, at one address for the life of the
/// cache.
#[cfg(feature = "cuda")]
pub struct DevPos {
    slot: crate::tensor::cuda_ext::CudaSlice<i32>,
}

#[cfg(feature = "cuda")]
impl DevPos {
    pub fn new(dev: &crate::tensor::cuda_ext::RawCudaDevice) -> anyhow::Result<Self> {
        let slot = dev
            .cuda_stream()
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("DevPos alloc: {e}"))?;
        Ok(Self { slot })
    }

    /// Writes `slot` into the device position. Must run outside a capture: the value is read
    /// by the replays, not recorded into them.
    pub fn write(
        &mut self,
        dev: &crate::tensor::cuda_ext::RawCudaDevice,
        slot: usize,
    ) -> anyhow::Result<()> {
        dev.cuda_stream()
            .memcpy_htod(&[slot as i32], &mut self.slot)
            .map_err(|e| anyhow::anyhow!("DevPos write: {e}"))
    }

    pub fn slice(&self) -> &crate::tensor::cuda_ext::CudaSlice<i32> {
        &self.slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trim_unprimes_and_releases_the_graph() {
        let mut c = KvCore::new(64, 1024);
        c.commit(40);
        assert!(c.prime(40));
        c.mark_captured();
        assert_eq!(c.trim_to(35), 40);
        assert_eq!(c.len(), 35);
        assert!(!c.is_primed());
        assert!(!c.is_captured());
        assert_eq!(c.seq_limit(), None);
    }

    #[test]
    fn a_trim_never_lengthens() {
        let mut c = KvCore::new(64, 1024);
        c.commit(10);
        c.trim_to(20);
        assert_eq!(c.len(), 10);
    }

    #[test]
    fn growth_doubles_to_the_cap_and_is_refused_under_a_graph() {
        let mut c = KvCore::new(64, 100);
        c.commit(60);
        assert_eq!(c.room_for(10), Ok(Some(70)));
        assert_eq!(c.grow_target(70), Ok(Some(100)));
        assert_eq!(c.minimal_target(70), 70);
        assert!(c.room_for(41).is_err());
        c.mark_captured();
        assert!(c.grow_target(70).is_err());
    }

    #[test]
    fn a_graph_is_bounded_by_capacity_and_recorded_stride() {
        let mut c = KvCore::new(4096, 32768);
        c.mark_captured();
        assert_eq!(c.seq_limit(), Some(4096));
        c.clear_captured();
        c.note_chain_stride(512);
        c.mark_captured();
        assert_eq!(c.seq_limit(), Some(512));
    }

    #[test]
    fn priming_the_same_slot_twice_needs_one_copy() {
        let mut c = KvCore::new(64, 1024);
        assert!(c.prime(7));
        assert!(!c.prime(7));
        assert!(c.prime(8));
    }

    #[test]
    fn replays_sync_the_host_length_under_the_cap() {
        let mut c = KvCore::new(64, 100);
        c.sync_len(50);
        assert_eq!(c.len(), 50);
        c.sync_len(500);
        assert_eq!(c.len(), 100);
    }
}
