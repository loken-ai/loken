//! When a model serves requests one at a time and when it batches them.
//!
//! Serial decode is the faster path for one stream; the batch worker is the faster path once
//! requests overlap. The engine counts the requests it has admitted and, at each admission,
//! records whether the new request arrived while others were running. Several overlaps in a
//! row move the model into its batch worker; as many lone arrivals in a row take it back.
//! Requiring a run of the same signal keeps one accidental overlap, or one quiet moment in
//! a busy period, from switching the mode.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Consecutive arrivals of the same kind that switch the mode.
const SWITCH_AFTER: usize = 3;

/// What an admission should do to the serving mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Switch {
    Stay,
    ToBatched,
    ToSerial,
}

/// Counts of consecutive overlapping and lone arrivals.
#[derive(Debug, Default)]
pub struct BatchingSignal {
    overlapping: usize,
    lone: usize,
    /// The model was offered the batch worker and could not take it.
    cannot_batch: bool,
}

impl BatchingSignal {
    /// Records an arrival that found `others` requests running and says whether the mode
    /// should change, given whether the model is batching now.
    pub fn arrive(&mut self, others: usize, batched: bool) -> Switch {
        if others > 0 {
            self.overlapping += 1;
            self.lone = 0;
        } else {
            self.lone += 1;
            self.overlapping = 0;
        }
        if !batched && !self.cannot_batch && self.overlapping >= SWITCH_AFTER {
            self.overlapping = 0;
            return Switch::ToBatched;
        }
        if batched && self.lone >= SWITCH_AFTER {
            self.lone = 0;
            return Switch::ToSerial;
        }
        Switch::Stay
    }

    pub fn mark_cannot_batch(&mut self) {
        self.cannot_batch = true;
    }
}

/// One admitted request, counted until it is dropped.
pub struct ActiveRequest {
    count: Arc<AtomicUsize>,
    others: usize,
}

impl ActiveRequest {
    pub fn enter(count: &Arc<AtomicUsize>) -> Self {
        let others = count.fetch_add(1, Ordering::AcqRel);
        Self {
            count: Arc::clone(count),
            others,
        }
    }

    /// Requests that were running when this one was admitted.
    pub fn others(&self) -> usize {
        self.others
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_of_overlaps_batches_and_a_run_of_lone_requests_goes_back() {
        let mut s = BatchingSignal::default();
        assert_eq!(s.arrive(1, false), Switch::Stay);
        assert_eq!(s.arrive(1, false), Switch::Stay);
        assert_eq!(s.arrive(2, false), Switch::ToBatched);
        assert_eq!(s.arrive(0, true), Switch::Stay);
        assert_eq!(s.arrive(0, true), Switch::Stay);
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
    }

    #[test]
    fn one_quiet_arrival_breaks_the_run() {
        let mut s = BatchingSignal::default();
        s.arrive(1, false);
        s.arrive(1, false);
        assert_eq!(s.arrive(0, false), Switch::Stay);
        assert_eq!(s.arrive(1, false), Switch::Stay);
    }

    #[test]
    fn a_model_that_cannot_batch_is_not_offered_again() {
        let mut s = BatchingSignal::default();
        s.mark_cannot_batch();
        for _ in 0..10 {
            assert_eq!(s.arrive(3, false), Switch::Stay);
        }
    }

    #[test]
    fn active_requests_count_until_dropped() {
        let n = Arc::new(AtomicUsize::new(0));
        let a = ActiveRequest::enter(&n);
        let b = ActiveRequest::enter(&n);
        assert_eq!((a.others(), b.others()), (0, 1));
        drop(a);
        assert_eq!(ActiveRequest::enter(&n).others(), 1);
    }
}
