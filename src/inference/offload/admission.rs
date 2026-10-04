//! When a card takes an expert on: once it has been seen as many times as an upload needs uses to
//! pay for itself. An upload costs a measured time on the lane; each later use of the expert on a
//! card saves the host's measured time for one expert, so the number of uses that repay an upload
//! is the one over the other, rounded up. The sightings an expert must show before it is taken on
//! are that number, plus a bias learned from what evictions earned: experts that leave a card
//! having been used fewer times than the payback raise the bias, experts that repaid it lower it
//! again, in steps of one and within `[0, payback]`.

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::Relaxed};

pub struct Admission {
    bias: AtomicU32,
    /// Evictions that repaid minus those that did not, since the bias last moved.
    credit: AtomicI64,
    /// Evictions of one kind in surplus that move the bias: one per layer a token crosses, the
    /// number of uses a card sees of its working set in a token.
    window: i64,
    /// Smoothed nanoseconds one upload took.
    upload_ns: AtomicU64,
}

/// Each sample moves a smoothed time by this share (an eighth).
pub(crate) const SMOOTHING: u64 = 8;

impl Admission {
    pub fn new(window: usize) -> Self {
        Self {
            bias: AtomicU32::new(0),
            credit: AtomicI64::new(0),
            window: window.max(1) as i64,
            upload_ns: AtomicU64::new(0),
        }
    }

    /// Uses that repay an upload when the host takes `host_ns` for one expert.
    pub fn payback(&self, host_ns: u64) -> u32 {
        let upload = self.upload_ns.load(Relaxed);
        if upload == 0 {
            return 1;
        }
        (upload.div_ceil(host_ns.max(1)) as u32).max(1)
    }

    /// The sightings an expert must show before it is taken on now.
    pub fn sightings(&self, host_ns: u64) -> u32 {
        let p = self.payback(host_ns);
        p + self.bias.load(Relaxed).min(p)
    }

    /// One upload took `ns`.
    pub fn note_upload(&self, ns: u64) {
        let old = self.upload_ns.load(Relaxed);
        self.upload_ns
            .store(old - old / SMOOTHING + ns / SMOOTHING, Relaxed);
    }

    /// An expert left a card after `uses` uses there.
    pub fn settle(&self, uses: u64, host_ns: u64) {
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

#[cfg(test)]
mod tests {
    use super::Admission;

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
