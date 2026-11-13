//! What each request spent, by model: its wait at the admission gate, its wait before the model
//! took it, the reading of its prompt, the writing of its answer, and the tokens on each side,
//! the prompt's share served from the cache among them.
//!
//! Each serving path records a request once, where it already holds these figures, and the
//! metrics endpoint reads them as histograms. A request that carries a [`RequestProbe`] also
//! gets its own figures back through it. Nothing here measures.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

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

/// The instants a request served by the batch worker passed, seen from its client.
#[derive(Clone, Copy, Debug)]
pub struct BatchedTimeline {
    /// The request was handed to the worker.
    pub submitted: Instant,
    /// The worker started its prompt.
    pub admitted: Instant,
    /// The first token reached the client; `None` for an answer with no token.
    pub first_token: Option<Instant>,
    /// The answer ended.
    pub done: Instant,
}

impl RequestFigures {
    /// Tokens of its prompt read, the share served from the cache left out.
    pub fn prompt_read(&self) -> u64 {
        self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)
    }

    /// The figures of a request served by the batch worker: the wait from submission to
    /// admission, the prompt from admission to the first token, the answer from the first
    /// token to its end. An answer with no token reads its prompt until it ends.
    /// An instant that precedes the one before it counts as equal to it.
    pub fn batched(
        t: &BatchedTimeline,
        prompt_tokens: u64,
        cached_prompt_tokens: u64,
        generated_tokens: u64,
    ) -> Self {
        let first = t.first_token.unwrap_or(t.done);
        Self {
            queue_s: t
                .admitted
                .saturating_duration_since(t.submitted)
                .as_secs_f64(),
            prefill_s: first.saturating_duration_since(t.admitted).as_secs_f64(),
            decode_s: t.done.saturating_duration_since(first).as_secs_f64(),
            prompt_tokens,
            cached_prompt_tokens,
            generated_tokens,
        }
    }
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

/// A request's sequence in the KV cache when it ended, and the tokens the cache holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvPlace {
    pub used_tokens: u64,
    pub capacity_tokens: u64,
}

impl KvPlace {
    /// The place, when the backend reports both its sequence length and its capacity.
    pub fn of(used: Option<usize>, capacity: Option<usize>) -> Option<Self> {
        Some(Self {
            used_tokens: used? as u64,
            capacity_tokens: capacity? as u64,
        })
    }
}

/// What one request's decode path saw once it ended.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RequestOutcome {
    pub figures: RequestFigures,
    /// Tokens proposed by speculation and verified by the model.
    pub drafted: u64,
    /// Proposed tokens the model kept.
    pub accepted: u64,
    /// `None` when the backend does not report its cache.
    pub kv: Option<KvPlace>,
}

/// One request's line from its decode path back to its handler: how far the prompt is read
/// while it is read, and the outcome once the request ends. Shared by the clones of the
/// request's parameters.
#[derive(Debug, Default)]
pub struct RequestProbe {
    prefill_done: AtomicU64,
    prefill_total: AtomicU64,
    prefill_started: OnceLock<Instant>,
    /// Tokens generated so far.
    decoded: AtomicU64,
    outcome: Mutex<Option<RequestOutcome>>,
}

impl RequestProbe {
    /// Tokens of the prompt read so far and tokens to read; `(0, 0)` before the reading starts.
    pub fn prefill(&self) -> (u64, u64) {
        (
            self.prefill_done.load(Ordering::Relaxed),
            self.prefill_total.load(Ordering::Relaxed),
        )
    }

    /// Time since the reading of the prompt started; `None` before it starts.
    pub fn prefill_elapsed(&self) -> Option<Duration> {
        self.prefill_started.get().map(Instant::elapsed)
    }

    pub fn note_prefill(&self, done: u64, total: u64) {
        self.prefill_started.get_or_init(Instant::now);
        self.prefill_total.store(total, Ordering::Relaxed);
        self.prefill_done.store(done, Ordering::Relaxed);
    }

    pub fn note_decoded(&self, generated: u64) {
        self.decoded.store(generated, Ordering::Relaxed);
    }

    /// Tokens the model has computed for the request so far: its prompt read and its answer
    /// generated. Each costs the model about the same operations, which is what the energy
    /// of overlapping requests is shared by.
    pub fn work(&self) -> u64 {
        self.prefill_done.load(Ordering::Relaxed) + self.decoded.load(Ordering::Relaxed)
    }

    pub fn finish(&self, outcome: RequestOutcome) {
        self.note_decoded(outcome.figures.generated_tokens);
        *self.outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
    }

    pub fn outcome(&self) -> Option<RequestOutcome> {
        *self.outcome.lock().unwrap_or_else(|e| e.into_inner())
    }
}

thread_local! {
    /// The probe of the request whose prompt this thread reads.
    static PREFILL_PROBE: RefCell<Option<Arc<RequestProbe>>> = const { RefCell::new(None) };
    /// The prompt rows read and to read that this thread last reported to its probe.
    static PREFILL_READ: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Binds `probe` to this thread until the guard drops: the chunked prefill run on the thread
/// reports its progress to it.
pub fn bind_prefill(probe: Option<Arc<RequestProbe>>) -> PrefillBinding {
    let previous = PREFILL_PROBE.with(|p| p.replace(probe));
    let previous_read = PREFILL_READ.replace((0, 0));
    PrefillBinding {
        previous,
        previous_read,
    }
}

/// Restores the thread's previous probe when dropped.
pub struct PrefillBinding {
    previous: Option<Arc<RequestProbe>>,
    previous_read: (u64, u64),
}

impl Drop for PrefillBinding {
    fn drop(&mut self) {
        let previous = self.previous.take();
        PREFILL_PROBE.with(|p| *p.borrow_mut() = previous);
        PREFILL_READ.set(self.previous_read);
    }
}

/// Milliseconds a prefill reporting to a bound probe pauses after each report short of the
/// whole prompt: a slow prefill for the tests of the routes that report it.
#[cfg(test)]
pub static PREFILL_PAUSE_MS: AtomicU64 = AtomicU64::new(0);

/// Milliseconds a forward reporting to a bound probe pauses after each layer short of its last:
/// a slow forward for the tests of the reading inside a forward.
#[cfg(test)]
pub static LAYER_PAUSE_MS: AtomicU64 = AtomicU64::new(0);

/// Held by each test that sets [`PREFILL_PAUSE_MS`] or [`LAYER_PAUSE_MS`], or that a pause
/// would fail.
#[cfg(test)]
pub static PREFILL_PAUSE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Reports `done` of `total` prompt tokens read to the probe bound to this thread, if any.
pub fn note_prefill(done: usize, total: usize) {
    PREFILL_PROBE.with(|p| {
        if let Some(probe) = p.borrow().as_ref() {
            probe.note_prefill(done as u64, total as u64);
            PREFILL_READ.set((done as u64, total as u64));
            #[cfg(test)]
            if done < total {
                std::thread::sleep(Duration::from_millis(
                    PREFILL_PAUSE_MS.load(Ordering::Relaxed),
                ));
            }
        }
    });
}

/// Reports to the probe bound to this thread that `layers_done` of the `layers` layers of a
/// forward over `rows` prompt rows have run.
///
/// The rows read are those the thread last reported and the share of the forward's rows that
/// its layers done stand for.
/// A forward of 1 row reports nothing.
/// Neither does a forward whose rows run past the prompt the thread last reported.
pub fn note_layers(rows: usize, layers_done: usize, layers: usize) {
    if rows < 2 || layers_done == 0 || layers_done >= layers {
        return;
    }
    PREFILL_PROBE.with(|p| {
        let Some(probe) = p.borrow().as_ref().cloned() else {
            return;
        };
        let (done, total) = PREFILL_READ.get();
        let rows = rows as u64;
        if done.saturating_add(rows) > total {
            return;
        }
        probe.note_prefill(done + rows * layers_done as u64 / layers as u64, total);
        #[cfg(test)]
        std::thread::sleep(Duration::from_millis(
            LAYER_PAUSE_MS.load(Ordering::Relaxed),
        ));
    });
}

/// Reports `done` of `total` prompt tokens that another instance has read for the request to the
/// probe bound to this thread, if any.
pub fn relay_prefill(done: usize, total: usize) {
    PREFILL_PROBE.with(|p| {
        if let Some(probe) = p.borrow().as_ref() {
            probe.note_prefill(done as u64, total as u64);
        }
    });
}

/// Reports the tokens generated so far to the probe bound to this thread, if any.
pub fn note_decoded(generated: usize) {
    PREFILL_PROBE.with(|p| {
        if let Some(probe) = p.borrow().as_ref() {
            probe.note_decoded(generated as u64);
        }
    });
}

/// Takes in one finished request of `model` and hands its outcome to the request's probe.
pub fn record_outcome(model: &str, probe: Option<&RequestProbe>, outcome: RequestOutcome) {
    record(model, &outcome.figures);
    if let Some(p) = probe {
        p.finish(outcome);
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

    /// The 3 times of a batched request are the gaps between its 4 instants.
    #[test]
    fn a_batched_request_splits_its_time_at_admission_and_first_token() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let timeline = BatchedTimeline {
            submitted: t0,
            admitted: at(20),
            first_token: Some(at(120)),
            done: at(620),
        };
        let f = RequestFigures::batched(&timeline, 40, 16, 7);
        assert!((f.queue_s - 0.020).abs() < 1e-9);
        assert!((f.prefill_s - 0.100).abs() < 1e-9);
        assert!((f.decode_s - 0.500).abs() < 1e-9);
        assert_eq!(
            (f.prompt_tokens, f.cached_prompt_tokens, f.generated_tokens),
            (40, 16, 7)
        );
    }

    /// An answer with no token spends its time reading the prompt and decodes for none.
    #[test]
    fn a_batched_request_without_a_token_has_no_decode_time() {
        let t0 = Instant::now();
        let timeline = BatchedTimeline {
            submitted: t0,
            admitted: t0 + Duration::from_millis(5),
            first_token: None,
            done: t0 + Duration::from_millis(30),
        };
        let f = RequestFigures::batched(&timeline, 8, 0, 0);
        assert!((f.prefill_s - 0.025).abs() < 1e-9);
        assert_eq!(f.decode_s, 0.0);
    }

    /// An instant that precedes the previous one yields a zero gap, never a negative one.
    #[test]
    fn a_batched_request_never_reports_a_negative_time() {
        let t0 = Instant::now();
        let timeline = BatchedTimeline {
            submitted: t0 + Duration::from_millis(10),
            admitted: t0,
            first_token: Some(t0),
            done: t0,
        };
        let f = RequestFigures::batched(&timeline, 1, 0, 1);
        assert_eq!((f.queue_s, f.prefill_s, f.decode_s), (0.0, 0.0, 0.0));
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

    /// The prefill on a thread reports to the probe bound there, and to none once the binding
    /// drops; the outcome reaches the probe it was recorded with.
    #[test]
    fn a_bound_probe_hears_the_prefill_and_the_outcome() {
        let probe = Arc::new(RequestProbe::default());
        note_prefill(1, 9);
        assert_eq!(probe.prefill(), (0, 0));
        {
            let _bound = bind_prefill(Some(probe.clone()));
            note_prefill(4, 9);
        }
        note_prefill(7, 9);
        assert_eq!(probe.prefill(), (4, 9));
        let outcome = RequestOutcome {
            drafted: 5,
            accepted: 3,
            kv: KvPlace::of(Some(12), Some(64)),
            ..Default::default()
        };
        record_outcome("request-metrics-test-probe", Some(&probe), outcome);
        assert_eq!(probe.outcome(), Some(outcome));
        assert_eq!(KvPlace::of(Some(12), None), None);
    }

    /// The layers of a forward report its rows' share from the rows reported before it.
    /// A forward of 1 row, a forward past the prompt and a thread with no probe report nothing.
    #[test]
    fn the_layers_of_a_forward_report_their_share_of_its_rows() {
        let probe = Arc::new(RequestProbe::default());
        note_layers(8, 1, 4);
        {
            let _bound = bind_prefill(Some(probe.clone()));
            note_layers(8, 1, 4);
            assert_eq!(probe.prefill(), (0, 0));
            note_prefill(4, 20);
            note_layers(8, 1, 4);
            assert_eq!(probe.prefill(), (6, 20));
            note_layers(8, 3, 4);
            assert_eq!(probe.prefill(), (10, 20));
            note_layers(8, 4, 4);
            assert_eq!(probe.prefill(), (10, 20));
            note_layers(1, 1, 2);
            note_layers(17, 1, 2);
            assert_eq!(probe.prefill(), (10, 20));
            note_prefill(20, 20);
            note_layers(3, 1, 2);
            assert_eq!(probe.prefill(), (20, 20));
        }
        note_layers(8, 2, 4);
        assert_eq!(probe.prefill(), (20, 20));
    }
}
