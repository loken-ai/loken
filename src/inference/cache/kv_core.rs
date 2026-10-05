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

/// Channels one quantised KV block spans; a head is a whole number of blocks.
const KV_BLOCK: usize = crate::tensor::quantized::GgmlDType::Q8_0.block_size();
/// Positions a fixed-stride score buffer grows by.
const STRIDE_STEP: usize = 256;
/// Largest head size the quantised attention kernels are instantiated for: one warp lane per
/// block of channels.
pub const SHAPE_MAX_HEAD_DIM: usize = 512;
/// Largest number of query heads per KV head: each thread keeps one accumulator per query head
/// it serves, so this bounds the registers a shape kernel holds.
pub const SHAPE_MAX_Q_PER_KV: usize = 16;

/// Whether the quantised KV attention kernels can serve this head shape.
pub fn kv_shape_supported(head_dim: usize, n_q_per_kv: usize) -> bool {
    head_dim > 0
        && head_dim.is_multiple_of(KV_BLOCK)
        && head_dim <= SHAPE_MAX_HEAD_DIM
        && (1..=SHAPE_MAX_Q_PER_KV).contains(&n_q_per_kv)
}

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
    /// Longest sequence the kernels recorded since the last capture cycle serve as an eager
    /// step would; 0 = none. Written from `&self` attention calls.
    kernel_bound_seen: AtomicUsize,
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
            kernel_bound_seen: AtomicUsize::new(0),
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
    /// the capacity, by the stride of a fixed-stride chain if one was recorded, and by the
    /// longest sequence its recorded kernels serve as an eager step would.
    pub fn mark_captured(&mut self) {
        self.captured = true;
        let limit = [
            self.chain_stride_seen.load(Ordering::Relaxed),
            self.kernel_bound_seen.load(Ordering::Relaxed),
        ]
        .into_iter()
        .filter(|&b| b > 0)
        .fold(self.cap, usize::min);
        self.seq_limit = Some(limit);
    }

    pub fn clear_captured(&mut self) {
        self.captured = false;
        self.seq_limit = None;
        self.chain_stride_seen.store(0, Ordering::Relaxed);
        self.kernel_bound_seen.store(0, Ordering::Relaxed);
    }

    /// A kernel ran that serves sequences of at most `bound` keys as an eager step would; a
    /// graph recorded now replays no further.
    pub fn note_kernel_bound(&self, bound: usize) {
        let _ = self
            .kernel_bound_seen
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |seen| {
                (seen == 0 || bound < seen).then_some(bound)
            });
    }

    pub fn seq_limit(&self) -> Option<usize> {
        self.seq_limit
    }

    /// A fixed-stride attention chain ran at `stride`; a graph recorded now is bound by it.
    pub fn note_chain_stride(&self, stride: usize) {
        self.chain_stride_seen.store(stride, Ordering::Relaxed);
    }

    /// Stride of the score buffer the fixed-stride attention kernels use: the positions the
    /// next token attends (length plus the slot it writes), rounded up to `STRIDE_STEP` so the
    /// stride, and with it a captured graph, changes once per step rather than per token, and
    /// never past the capacity rounded to whole `block`s. Positions between the length and the
    /// stride score -INF.
    pub fn padded_stride(&self, block: usize) -> usize {
        let used = (self.len + 1).min(self.cap);
        let cap_aligned = self.cap.div_ceil(block) * block;
        (used.div_ceil(STRIDE_STEP) * STRIDE_STEP)
            .min(cap_aligned)
            .max(STRIDE_STEP.min(cap_aligned))
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
    fn a_graph_stops_where_a_recorded_kernel_would_be_left() {
        let mut c = KvCore::new(4096, 32768);
        c.note_kernel_bound(1023);
        c.note_kernel_bound(511);
        c.note_kernel_bound(2047);
        c.mark_captured();
        assert_eq!(c.seq_limit(), Some(511));
        c.clear_captured();
        c.mark_captured();
        assert_eq!(c.seq_limit(), Some(4096));
    }

    #[test]
    fn priming_the_same_slot_twice_needs_one_copy() {
        let mut c = KvCore::new(64, 1024);
        assert!(c.prime(7));
        assert!(!c.prime(7));
        assert!(c.prime(8));
    }

    #[test]
    fn the_stride_follows_the_length_past_any_fixed_width() {
        let mut c = KvCore::new(8192, 131072);
        assert_eq!(c.padded_stride(32), 256);
        c.commit(2160);
        assert_eq!(c.padded_stride(32), 2304);
        c.commit(4000);
        assert_eq!(c.padded_stride(32), 6400);
        let mut small = KvCore::new(100, 100);
        small.commit(99);
        assert_eq!(small.padded_stride(32), 128);
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
