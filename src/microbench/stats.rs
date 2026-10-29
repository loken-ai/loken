//! The statistics of one variant: the time per call over a run sized to a target wall time, its
//! median and spread, and the outliers above Tukey's upper fence, counted rather than dropped.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// The time per call of one variant, in microseconds, over `calls` timed calls.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub calls: usize,
    pub median_us: f64,
    pub mean_us: f64,
    pub p10_us: f64,
    pub p90_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    /// Calls above the upper fence `q3 + 1.5 (q3 - q1)`; they stay in every figure above.
    pub outliers: usize,
}

/// The value at rank `q`, 0 to 1, of `sorted`, interpolated between the two nearest samples.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n => {
            let at = q * (n - 1) as f64;
            let lo = at.floor() as usize;
            let hi = at.ceil() as usize;
            sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo as f64)
        }
    }
}

impl Summary {
    /// The summary of `samples_us`, each one call's time in microseconds.
    pub fn of(samples_us: &[f64]) -> Self {
        let mut sorted = samples_us.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let (q1, q3) = (quantile(&sorted, 0.25), quantile(&sorted, 0.75));
        let fence = q3 + 1.5 * (q3 - q1);
        Self {
            calls: n,
            median_us: quantile(&sorted, 0.5),
            mean_us: if n == 0 {
                0.0
            } else {
                sorted.iter().sum::<f64>() / n as f64
            },
            p10_us: quantile(&sorted, 0.1),
            p90_us: quantile(&sorted, 0.9),
            min_us: sorted.first().copied().unwrap_or(0.0),
            max_us: sorted.last().copied().unwrap_or(0.0),
            outliers: sorted.iter().filter(|&&s| s > fence).count(),
        }
    }
}

/// Run `call` for about `target`: a warm-up of a tenth of it, then timed calls until it has
/// elapsed. Each call answers its own time in microseconds. The warm-up and the timed run make
/// at least one call each, so a call longer than the target is still measured once.
pub fn run_for(target: Duration, mut call: impl FnMut() -> f64) -> Vec<f64> {
    let warm = Instant::now();
    loop {
        call();
        if warm.elapsed() * 10 >= target {
            break;
        }
    }
    let mut samples = Vec::new();
    let timed = Instant::now();
    loop {
        samples.push(call());
        if timed.elapsed() >= target {
            break;
        }
    }
    samples
}

/// `f` run under the host clock: its time in microseconds, for a call whose cost is what the
/// calling thread waits.
pub fn timed(mut f: impl FnMut()) -> impl FnMut() -> f64 {
    move || {
        let t = Instant::now();
        f();
        t.elapsed().as_secs_f64() * 1e6
    }
}

/// Gigabytes per second of `bytes` moved in `us` microseconds.
pub fn gbs(bytes: u64, us: f64) -> f64 {
    if us <= 0.0 {
        0.0
    } else {
        bytes as f64 / us / 1e3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_reads_the_median_and_the_spread() {
        let s = Summary::of(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(s.calls, 5);
        assert_eq!(s.median_us, 3.0);
        assert_eq!(s.mean_us, 3.0);
        assert_eq!(s.min_us, 1.0);
        assert_eq!(s.max_us, 5.0);
        assert!((s.p10_us - 1.4).abs() < 1e-9);
        assert!((s.p90_us - 4.6).abs() < 1e-9);
        assert_eq!(s.outliers, 0);
    }

    #[test]
    fn an_outlier_is_counted_and_kept_in_the_maximum() {
        let mut v = vec![10.0; 40];
        v.push(10.5);
        v.push(900.0);
        let s = Summary::of(&v);
        assert_eq!(s.outliers, 2);
        assert_eq!(s.max_us, 900.0);
        assert_eq!(s.median_us, 10.0);
    }

    #[test]
    fn an_empty_run_summarises_to_zero() {
        let s = Summary::of(&[]);
        assert_eq!(s.calls, 0);
        assert_eq!(s.median_us, 0.0);
    }

    #[test]
    fn the_run_fills_the_target_and_makes_at_least_one_call() {
        let samples = run_for(
            Duration::from_millis(20),
            timed(|| std::hint::black_box(())),
        );
        assert!(!samples.is_empty());
        let one = run_for(
            Duration::from_micros(1),
            timed(|| std::thread::sleep(Duration::from_millis(2))),
        );
        assert_eq!(one.len(), 1);
        assert!(one[0] >= 2000.0);
    }

    #[test]
    fn bandwidth_is_bytes_over_time() {
        assert!((gbs(1_000_000_000, 1_000_000.0) - 1.0).abs() < 1e-12);
        assert_eq!(gbs(10, 0.0), 0.0);
    }
}
