//! What each request spent, by model: its wait at the admission gate, its wait before the model
//! took it, the reading of its prompt, the writing of its answer, and the tokens on each side,
//! the prompt's share served from the cache among them.
//!
//! Each serving path records a request once, where it already holds these figures, and the
//! metrics endpoint reads them as histograms. Nothing here measures.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// The upper bound of the first time bucket, in seconds; each next bound doubles it.
const SECONDS_FIRST_BOUND: f64 = 0.001;
/// Time buckets: from a millisecond to past four minutes.
const SECONDS_BUCKETS: usize = 19;
/// Each next token bound is this many times the previous one, from one token.
const TOKENS_RATIO: f64 = 4.0;
/// Token buckets: from one token to a quarter of a million.
const TOKENS_BUCKETS: usize = 10;

/// What one finished request spent and read.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RequestFigures {
    /// Seconds from its arrival at the model to the start of its prompt.
    pub queue_s: f64,
    /// Seconds reading its prompt, to its first token.
    pub prefill_s: f64,
    /// Seconds writing its answer after the first token.
    pub decode_s: f64,
    /// Tokens of its prompt.
    pub prompt_tokens: u64,
    /// Tokens of its prompt served from the cache instead of read.
    pub cached_prompt_tokens: u64,
    /// Tokens of its answer.
    pub generated_tokens: u64,
}

/// Observations in buckets with upper bounds `bounds`, the last bucket open.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub bounds: Vec<f64>,
    /// Observations per bucket, the open one last: `bounds.len() + 1` entries.
    counts: Vec<u64>,
    pub sum: f64,
    pub count: u64,
}

impl Histogram {
    fn new(first: f64, ratio: f64, buckets: usize) -> Self {
        let bounds: Vec<f64> = (0..buckets).map(|i| first * ratio.powi(i as i32)).collect();
        Self {
            counts: vec![0; bounds.len() + 1],
            bounds,
            sum: 0.0,
            count: 0,
        }
    }

    fn seconds() -> Self {
        Self::new(SECONDS_FIRST_BOUND, 2.0, SECONDS_BUCKETS)
    }

    fn tokens() -> Self {
        Self::new(1.0, TOKENS_RATIO, TOKENS_BUCKETS)
    }

    /// Takes `v` in; a value that is not a finite non-negative number is taken as zero.
    fn observe(&mut self, v: f64) {
        let v = if v.is_finite() && v > 0.0 { v } else { 0.0 };
        let at = self.bounds.partition_point(|&b| b < v);
        self.counts[at] += 1;
        self.sum += v;
        self.count += 1;
    }

    /// The observations at or below each bound, then all of them: what a histogram's
    /// buckets publish.
    pub fn cumulative(&self) -> Vec<u64> {
        self.counts
            .iter()
            .scan(0u64, |total, &c| {
                *total += c;
                Some(*total)
            })
            .collect()
    }
}

/// The finished requests of one model, and the waits of its admitted ones at the gate.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRequests {
    pub gate_s: Histogram,
    pub queue_s: Histogram,
    pub prefill_s: Histogram,
    pub decode_s: Histogram,
    pub prompt_tokens: Histogram,
    pub cached_prompt_tokens: Histogram,
    pub generated_tokens: Histogram,
}

impl Default for ModelRequests {
    fn default() -> Self {
        Self {
            gate_s: Histogram::seconds(),
            queue_s: Histogram::seconds(),
            prefill_s: Histogram::seconds(),
            decode_s: Histogram::seconds(),
            prompt_tokens: Histogram::tokens(),
            cached_prompt_tokens: Histogram::tokens(),
            generated_tokens: Histogram::tokens(),
        }
    }
}

impl ModelRequests {
    fn record(&mut self, f: &RequestFigures) {
        self.queue_s.observe(f.queue_s);
        self.prefill_s.observe(f.prefill_s);
        self.decode_s.observe(f.decode_s);
        self.prompt_tokens.observe(f.prompt_tokens as f64);
        self.cached_prompt_tokens
            .observe(f.cached_prompt_tokens as f64);
        self.generated_tokens.observe(f.generated_tokens as f64);
    }

    /// Requests finished.
    pub fn requests(&self) -> u64 {
        self.queue_s.count
    }
}

static BY_MODEL: Mutex<BTreeMap<String, ModelRequests>> = Mutex::new(BTreeMap::new());

/// Takes in one finished request of `model`.
pub fn record(model: &str, figures: &RequestFigures) {
    let mut all = BY_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    all.entry(model.to_string()).or_default().record(figures);
}

/// Takes in the wait of one request of `model` at the admission gate, once it is admitted.
pub fn record_gate_wait(model: &str, seconds: f64) {
    let mut all = BY_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    all.entry(model.to_string())
        .or_default()
        .gate_s
        .observe(seconds);
}

/// The finished requests so far, by model.
pub fn snapshot() -> BTreeMap<String, ModelRequests> {
    BY_MODEL.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value lands in the first bucket whose bound holds it, and the cumulative counts end
    /// on every observation.
    #[test]
    fn a_value_lands_in_the_first_bucket_that_holds_it() {
        let mut h = Histogram::seconds();
        h.observe(SECONDS_FIRST_BOUND);
        h.observe(SECONDS_FIRST_BOUND * 1.5);
        h.observe(1e9);
        let c = h.cumulative();
        assert_eq!(c.len(), h.bounds.len() + 1);
        assert_eq!((c[0], c[1]), (1, 2));
        assert_eq!(c[h.bounds.len() - 1], 2);
        assert_eq!(*c.last().unwrap(), 3);
        assert_eq!(h.count, 3);
    }

    /// A figure that is not a number is counted as zero, so the sum stays a number.
    #[test]
    fn a_figure_that_is_not_a_number_counts_as_zero() {
        let mut h = Histogram::tokens();
        h.observe(f64::NAN);
        h.observe(-3.0);
        h.observe(f64::INFINITY);
        assert_eq!(h.sum, 0.0);
        assert_eq!(h.cumulative()[0], 3);
    }

    /// Each model keeps its own requests.
    #[test]
    fn each_model_keeps_its_own_requests() {
        let a = "request-metrics-test-a";
        let b = "request-metrics-test-b";
        let figures = RequestFigures {
            queue_s: 0.002,
            prefill_s: 0.1,
            decode_s: 2.0,
            prompt_tokens: 100,
            cached_prompt_tokens: 64,
            generated_tokens: 128,
        };
        record(a, &figures);
        record(a, &figures);
        record(b, &figures);
        record_gate_wait(b, 0.5);
        let all = snapshot();
        assert_eq!(all[a].requests(), 2);
        assert_eq!(all[b].requests(), 1);
        assert_eq!((all[a].gate_s.count, all[b].gate_s.sum), (0, 0.5));
        assert_eq!(all[a].cached_prompt_tokens.sum, 128.0);
        assert_eq!(all[a].decode_s.sum, 4.0);
    }
}
