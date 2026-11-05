//! When a card's expert cache takes an expert on: once it has been routed to as many times as an
//! upload needs uses to pay for itself.
//!
//! An upload costs a measured time on the copy stream.
//! Each later use of the expert on the card saves the host's measured time for one expert.
//! The uses that repay an upload are the one over the other, rounded up.
//!
//! The routes an expert must show before it is taken on are that number plus a learned bias.
//! Experts that leave the cache used fewer times than the payback raise the bias.
//! Experts that repaid it lower the bias again.
//! The bias moves in steps of one, within `[0, payback]`.
//!
//! Nothing here touches a card. The tier that owns the cache times the copies and the host.

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::Relaxed};

pub(crate) struct Admission {
    bias: AtomicU32,
    /// Evictions that repaid minus those that did not, since the bias last moved.
    credit: AtomicI64,
    /// Evictions of one kind in surplus that move the bias: the cached layers a token crosses
    /// on the card, the uses the card sees of its working set in a token.
    window: i64,
    /// Smoothed nanoseconds one upload took.
    upload_ns: AtomicU64,
}

/// Each sample moves a smoothed time by this share (an eighth).
pub(crate) const SMOOTHING: u64 = 8;

impl Admission {
    pub(crate) fn new(window: usize) -> Self {
        Self {
            bias: AtomicU32::new(0),
            credit: AtomicI64::new(0),
            window: window.max(1) as i64,
            upload_ns: AtomicU64::new(0),
        }
    }

    /// Uses that repay an upload when the host takes `host_ns` for one expert.
    pub(crate) fn payback(&self, host_ns: u64) -> u32 {
        let upload = self.upload_ns.load(Relaxed);
        if upload == 0 {
            return 1;
        }
        (upload.div_ceil(host_ns.max(1)) as u32).max(1)
    }

    /// The routes an expert must show before it is taken on now.
    pub(crate) fn sightings(&self, host_ns: u64) -> u32 {
        let p = self.payback(host_ns);
        p + self.bias.load(Relaxed).min(p)
    }

    /// One upload took `ns`.
    pub(crate) fn note_upload(&self, ns: u64) {
        let old = self.upload_ns.load(Relaxed);
        self.upload_ns
            .store(old - old / SMOOTHING + ns / SMOOTHING, Relaxed);
    }

    /// An expert left the cache after `uses` uses there.
    pub(crate) fn settle(&self, uses: u64, host_ns: u64) {
        let paid = uses >= self.payback(host_ns) as u64;
        let step = if paid { 1 } else { -1 };
        let credit = self.credit.fetch_add(step, Relaxed) + step;
        if credit >= self.window {
            self.credit.store(0, Relaxed);
            let b = self.bias.load(Relaxed);
            self.bias.store(b.saturating_sub(1), Relaxed);
        } else if credit <= -self.window {
            self.credit.store(0, Relaxed);
            self.bias.store(self.bias.load(Relaxed) + 1, Relaxed);
        }
    }
}

/// The heat a missing expert needs to take a slot: the routes `admission` asks for at a host time
/// of `host_ns` an expert, never below `floor`.
/// Without an admission the floor alone decides.
pub(crate) fn heat_to_take_a_slot(admission: Option<&Admission>, host_ns: u64, floor: f32) -> f32 {
    admission.map_or(floor, |a| (a.sightings(host_ns) as f32).max(floor))
}

/// Nanoseconds in `us` microseconds, a measured figure that is `0.0` before a measure.
pub(crate) fn ns_of_us(us: f64) -> u64 {
    (us * 1e3).max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::{heat_to_take_a_slot, ns_of_us, Admission};

    /// The tier's bar is the floor until the payback and the bias rise above it.
    #[test]
    fn the_heat_bar_is_the_payback_above_the_floor() {
        let a = Admission::new(4);
        assert_eq!(heat_to_take_a_slot(None, 1_000_000, 2.0), 2.0);
        assert_eq!(
            heat_to_take_a_slot(Some(&a), 1_000_000, 2.0),
            2.0,
            "nothing measured"
        );
        for _ in 0..64 {
            a.note_upload(3_000_000);
        }
        assert_eq!(heat_to_take_a_slot(Some(&a), 1_000_000, 2.0), 3.0);
        for _ in 0..8 {
            a.settle(1, 1_000_000);
        }
        assert_eq!(heat_to_take_a_slot(Some(&a), 1_000_000, 2.0), 5.0);
        assert_eq!(ns_of_us(0.0), 0);
        assert_eq!(ns_of_us(1.5), 1_500);
    }

    /// The payback follows the measured times; evictions that did not repay raise the bar above it
    /// and ones that did bring it back.
    #[test]
    fn sightings_follow_the_payback_and_what_evictions_earned() {
        let a = Admission::new(4);
        assert_eq!(
            a.sightings(1_000_000),
            1,
            "nothing measured: admit on sight"
        );
        for _ in 0..64 {
            a.note_upload(3_000_000);
        }
        assert_eq!(a.payback(1_000_000), 3);
        assert_eq!(a.sightings(1_000_000), 3);
        for _ in 0..8 {
            a.settle(1, 1_000_000);
        }
        assert_eq!(a.sightings(1_000_000), 5, "two windows of waste");
        for _ in 0..40 {
            a.settle(9, 1_000_000);
        }
        assert_eq!(a.sightings(1_000_000), 3, "repaid evictions bring it back");
        assert_eq!(
            a.sightings(3_000_000),
            1,
            "a slower host makes uploads pay sooner"
        );
    }
}
