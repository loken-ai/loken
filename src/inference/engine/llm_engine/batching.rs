//! When a model serves requests one at a time and when it batches them.
//!
//! Requests that overlap are served by the batch worker.
//! A lone request takes the path that serves a request like the latest lone one sooner: its
//! prompt at the path's measured prompt rate, then its answer at the path's measured decode
//! rate.
//! The serial path drafts (prompt lookup, a draft head, a drafter model) and takes a prompt
//! from a prefill peer.
//! The batch worker does neither.
//!
//! The serial path takes every lone request until the worker has a measure.
//! A lone request is never sent to an unmeasured worker to measure it.
//! The worker is measured on a request it decodes alone: one alone from its submission, or the
//! last request of a burst once the others have ended.
//! A request that arrives while a lone request decodes serially moves that request into the
//! worker at its next step, so the burst is batched at once.
//! The model keeps the worker's paged pool while it serves serially, so a move takes no new
//! card memory.
//! The wait the arriving requests spend until the move is charged to the lone serial requests.
//!
//! A request that needs the serial path (a grammar, token rules or prompt embeddings) keeps
//! the model serial until the next overlap.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// What an admission should do to the serving mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Switch {
    Stay,
    ToBatched,
    ToSerial,
}

/// The decode of the last lone request served on one path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoneDecode {
    /// Generated tokens.
    pub tokens: usize,
    /// Generated tokens per second, from the first generated token to the last.
    pub rate: f64,
    /// Share of the generated tokens that drafts gave, kept by the verify.
    pub drafted: f64,
    /// Share of the generated tokens that a prompt lookup over the prompt and the answer
    /// would have drafted right.
    pub lookup: f64,
    /// Prompt tokens the request had to read when it arrived: the prompt less what the cache
    /// held before any hand-over.
    pub prompt_tokens: usize,
    /// Those tokens per second, from the start of the prompt to the first generated token.
    /// A hand-over from a prefill peer is inside that time.
    /// `None` when the request read no prompt token.
    pub prompt_rate: Option<f64>,
}

impl LoneDecode {
    /// The prompt rate of `tokens` read in `secs`, whatever their number.
    pub fn prompt_rate(tokens: usize, secs: f64) -> Option<f64> {
        (tokens > 0 && secs > 0.0 && secs.is_finite()).then(|| tokens as f64 / secs)
    }
}

/// The measured decode of lone requests on each path, for the loaded model.
#[derive(Debug, Default, Clone, Copy)]
pub struct PathRates {
    pub serial: Option<LoneDecode>,
    pub batched: Option<LoneDecode>,
    /// Prompt lookup share of the latest lone request, on either path.
    pub latest_lookup: Option<f64>,
    /// Generated tokens of the latest lone request, on either path.
    pub latest_tokens: usize,
    /// Prompt rate of the last lone request on the serial path that read a prompt token.
    pub serial_prompt_rate: Option<f64>,
    /// Prompt rate of the last lone request in the worker that read a prompt token.
    pub batched_prompt_rate: Option<f64>,
    /// Prompt tokens the latest lone request had to read, on either path.
    pub latest_prompt_tokens: usize,
}

/// The path a lone request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LonePath {
    Serial,
    Batched,
}

impl PathRates {
    /// The path that serves the next lone request sooner.
    /// That request reads as many prompt tokens and writes as many tokens as the latest lone
    /// request did.
    /// `serial_charge` seconds are added to the serial path.
    ///
    /// The serial path is taken while either path has no measure.
    /// Each path is charged the prompt at its own prompt rate.
    /// A path without a prompt rate is taken to read at the other path's rate.
    /// The prompt then weighs the same on both paths and the decode rates decide.
    /// The serial rate is turned into a rate per forward with the share its drafts gave.
    /// That share scales with the prompt lookup share of the latest lone request.
    pub fn choose(&self, serial_charge: f64) -> LonePath {
        let Some(serial) = self.serial else {
            return LonePath::Serial;
        };
        let Some(batched) = self.batched else {
            return LonePath::Serial;
        };
        let tokens = self.latest_tokens.max(1) as f64;
        let prompt = self.latest_prompt_tokens as f64;
        let (serial_prompt_s, batched_prompt_s) =
            match (self.serial_prompt_rate, self.batched_prompt_rate) {
                (Some(s), Some(b)) => (prompt / s, prompt / b),
                _ => (0.0, 0.0),
            };
        let serial_s =
            serial_prompt_s + tokens / self.expected_serial_rate(&serial) + serial_charge;
        let batched_s = batched_prompt_s + tokens / batched.rate;
        if serial_s < batched_s {
            LonePath::Serial
        } else {
            LonePath::Batched
        }
    }

    /// The serial rate expected for a request whose prompt lookup share is the latest one.
    fn expected_serial_rate(&self, serial: &LoneDecode) -> f64 {
        let per_forward = serial.rate * (1.0 - serial.drafted);
        let lookup = self.latest_lookup.unwrap_or(serial.lookup);
        // Kept drafts per prompt lookup share, as the serial request measured them.
        // A serial request with nothing to look up says nothing: a lookup share then drafts
        // as it would be kept.
        let drafted = if serial.lookup > 0.0 {
            lookup * serial.drafted / serial.lookup
        } else {
            lookup.max(serial.drafted)
        };
        if drafted >= 1.0 {
            return f64::INFINITY;
        }
        per_forward / (1.0 - drafted)
    }
}

/// What the requests so far say about the serving mode.
#[derive(Debug, Default)]
pub struct BatchingSignal {
    /// A request needed the serial path, so a lone arrival does not move the model back.
    serial_needed: bool,
    /// The model was offered the batch worker and could not take it.
    cannot_batch: bool,
    rates: PathRates,
    /// Lone requests the serial path served to their end.
    serial_done: u64,
    /// Lone requests the serial path moved into the worker when another request arrived.
    handed: u64,
    /// Seconds the arriving requests waited for those moves, summed.
    move_wait: f64,
}

impl BatchingSignal {
    /// Records an arrival that found `others` requests running and says whether the model
    /// should change path, given whether it is in its batch worker now.
    pub fn arrive(&mut self, others: usize, batched: bool) -> Switch {
        if self.cannot_batch {
            return Switch::Stay;
        }
        if others > 0 {
            self.serial_needed = false;
            return if batched {
                Switch::Stay
            } else {
                Switch::ToBatched
            };
        }
        if self.serial_needed {
            return Switch::Stay;
        }
        match (self.rates.choose(self.serial_charge()), batched) {
            (LonePath::Serial, true) => Switch::ToSerial,
            (LonePath::Batched, false) => Switch::ToBatched,
            _ => Switch::Stay,
        }
    }

    pub fn mark_cannot_batch(&mut self) {
        self.cannot_batch = true;
    }

    /// Whether the model was offered the batch worker and could not take it.
    pub fn cannot_batch(&self) -> bool {
        self.cannot_batch
    }

    /// Records a request that runs on the serial path only.
    pub fn mark_serial_needed(&mut self) {
        self.serial_needed = true;
    }

    /// Whether a request that needs the serial path holds the model there.
    pub fn serial_needed(&self) -> bool {
        self.serial_needed
    }

    /// Records a lone request served on one path.
    pub fn record(&mut self, path: LonePath, decode: LoneDecode) {
        let prompt_rate = match path {
            LonePath::Serial => {
                self.rates.serial = Some(decode);
                self.serial_done += 1;
                &mut self.rates.serial_prompt_rate
            }
            LonePath::Batched => {
                self.rates.batched = Some(decode);
                &mut self.rates.batched_prompt_rate
            }
        };
        if decode.prompt_rate.is_some() {
            *prompt_rate = decode.prompt_rate;
        }
        self.rates.latest_lookup = Some(decode.lookup);
        self.rates.latest_tokens = decode.tokens;
        self.rates.latest_prompt_tokens = decode.prompt_tokens;
    }

    /// Records the decode of the last request of a burst, from the moment the others ended.
    ///
    /// It gives the worker's decode rate alone.
    /// The size of the latest lone request and the prompt rates stay as they were.
    pub fn record_worker_tail(&mut self, decode: LoneDecode) {
        self.rates.batched = Some(decode);
    }

    /// Records a lone serial request moved into the worker.
    pub fn record_handover(&mut self) {
        self.handed += 1;
    }

    /// Records `seconds` a request waited for a move into the worker.
    pub fn record_move_wait(&mut self, seconds: f64) {
        self.move_wait += seconds;
    }

    /// The wait a lone serial request is expected to cost the requests that arrive after it:
    /// the waits for moves, over the lone requests the serial path served to their end.
    fn serial_charge(&self) -> f64 {
        self.move_wait / self.serial_done.max(1) as f64
    }
}

/// The serving signal of a model and the wake-up of requests that wait for its worker.
#[derive(Debug, Default)]
pub struct Batching {
    signal: std::sync::Mutex<BatchingSignal>,
    /// Woken when the model moves into its batch worker.
    moved: tokio::sync::Notify,
}

impl Batching {
    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, BatchingSignal>> {
        self.signal.lock()
    }

    /// Wakes the requests waiting for the model to enter its batch worker.
    pub fn note_moved(&self) {
        self.moved.notify_waiters();
    }

    /// Returns once `in_worker` holds, checked now and after each move into the worker.
    pub async fn until<F, Fut>(&self, in_worker: F)
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        loop {
            let moved = self.moved.notified();
            tokio::pin!(moved);
            moved.as_mut().enable();
            if in_worker().await {
                return;
            }
            moved.await;
        }
    }
}

/// Share of `generated` that a prompt lookup over `prompt` and the tokens before would have
/// drafted right, cycle by cycle as the serial decode verifies them.
pub fn lookup_share(prompt: &[u32], generated: &[u32]) -> f64 {
    use crate::inference::serve::prompt_lookup::NgramDraftCache;
    if generated.is_empty() {
        return 0.0;
    }
    let mut cache = NgramDraftCache::new_cascade(
        2,
        4,
        crate::inference::engine::decode_step::DRAFT_MAX,
        prompt.len() + generated.len(),
    );
    cache.push_many(prompt);
    let (mut at, mut kept) = (0usize, 0usize);
    while at < generated.len() {
        // The token a plain step samples.
        cache.push(generated[at]);
        at += 1;
        let draft = cache.lookup();
        let right = draft
            .iter()
            .zip(&generated[at..])
            .take_while(|(d, g)| d == g)
            .count();
        cache.push_many(&generated[at..at + right]);
        at += right;
        kept += right;
    }
    kept as f64 / generated.len() as f64
}

/// One admitted request, counted until it is dropped.
pub struct ActiveRequest {
    count: Arc<AtomicUsize>,
    others: usize,
    /// The arrivals call for the batch worker, which this request moves the model into once
    /// it holds the serial gate.
    batched: bool,
    /// The request is lone and the serial path is the faster one for it.
    serial: bool,
}

/// Requests started on every engine of the process, and those running.
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// The requests started on every engine of the process so far, and those running now.
/// A measure of one model's use of a card is its own only while these show no other request.
pub(crate) fn requests_of_the_process() -> (u64, usize) {
    (
        STARTED.load(Ordering::Acquire),
        RUNNING.load(Ordering::Acquire),
    )
}

impl ActiveRequest {
    pub fn enter(count: &Arc<AtomicUsize>) -> Self {
        STARTED.fetch_add(1, Ordering::AcqRel);
        RUNNING.fetch_add(1, Ordering::AcqRel);
        let others = count.fetch_add(1, Ordering::AcqRel);
        Self {
            count: Arc::clone(count),
            others,
            batched: false,
            serial: false,
        }
    }

    /// Marks this request as the one to move the model into its batch worker.
    pub fn want_batched(&mut self) {
        self.batched = true;
    }

    /// Whether this request moves the model into its batch worker.
    pub fn wants_batched(&self) -> bool {
        self.batched
    }

    /// Marks this request as the one to take the model out of its batch worker.
    pub fn want_serial(&mut self) {
        self.serial = true;
    }

    /// Whether this request takes the model out of its batch worker.
    pub fn wants_serial(&self) -> bool {
        self.serial
    }

    /// Requests that were running when this one was admitted.
    pub fn others(&self) -> usize {
        self.others
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::AcqRel);
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _held = IDLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            IDLE.notify_all();
        }
    }
}

/// Held by a thread that waits for an engine without requests, and by the last request that
/// ends or is answered.
static IDLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Woken when the last running request of an engine ends, and when the server has answered
/// every request it handles.
static IDLE: std::sync::Condvar = std::sync::Condvar::new();

/// Requests the server is handling, from their arrival to their answer, whatever their engine.
static ARRIVED: AtomicUsize = AtomicUsize::new(0);

/// Counts a request the server is handling until it drops.
///
/// A request reads its engine before the engine admits it, so the count covers that part.
pub struct Arrived;

impl Arrived {
    pub fn enter() -> Self {
        ARRIVED.fetch_add(1, Ordering::AcqRel);
        Arrived
    }
}

impl Drop for Arrived {
    fn drop(&mut self) {
        if ARRIVED.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _held = IDLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            IDLE.notify_all();
        }
    }
}

/// Whether a request runs on the engine counting `count`, or the server is handling one.
pub(crate) fn busy(count: &AtomicUsize) -> bool {
    count.load(Ordering::Acquire) > 0 || ARRIVED.load(Ordering::Acquire) > 0
}

/// Waits until no request counted in `count` runs.
/// The server may still be handling requests of other engines.
pub(crate) fn wait_no_request(count: &AtomicUsize) {
    let mut held = IDLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    while count.load(Ordering::Acquire) > 0 {
        held = IDLE.wait(held).unwrap_or_else(|e| e.into_inner());
    }
}

/// Waits until [`busy`] is false.
pub(crate) fn wait_idle(count: &AtomicUsize) {
    let mut held = IDLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    while busy(count) {
        held = IDLE.wait(held).unwrap_or_else(|e| e.into_inner());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(rate: f64, drafted: f64, lookup: f64) -> LoneDecode {
        LoneDecode {
            tokens: 200,
            rate,
            drafted,
            lookup,
            prompt_tokens: 0,
            prompt_rate: None,
        }
    }

    /// A plain decode at `rate` after a prompt of `prompt_tokens` read at `prompt_rate`.
    fn request(rate: f64, prompt_tokens: usize, prompt_rate: f64) -> LoneDecode {
        LoneDecode {
            prompt_tokens,
            prompt_rate: Some(prompt_rate),
            ..decode(rate, 0.0, 0.0)
        }
    }

    /// A serial path that decodes slower but reads the prompt much faster, as with a prefill
    /// peer on a faster card, wins a long prompt and loses a short one.
    #[test]
    fn a_path_that_reads_the_prompt_faster_wins_a_long_prompt() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Batched, request(120.0, 8000, 2000.0));
        s.record(LonePath::Serial, request(100.0, 8000, 20000.0));
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        // A request that read no prompt token keeps the rates and moves only the size.
        s.record(
            LonePath::Serial,
            LoneDecode {
                prompt_tokens: 50,
                ..decode(100.0, 0.0, 0.0)
            },
        );
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
    }

    /// Equal prompt rates give the choice of the decode rates alone, and so do prompt rates
    /// known on one path only.
    #[test]
    fn equal_prompt_rates_leave_the_decode_choice() {
        for (serial, batched) in [(180.0, 100.0), (90.0, 100.0)] {
            let mut by_decode = BatchingSignal::default();
            by_decode.record(LonePath::Batched, decode(batched, 0.0, 0.0));
            by_decode.record(LonePath::Serial, decode(serial, 0.0, 0.0));
            let mut by_request = BatchingSignal::default();
            by_request.record(LonePath::Batched, request(batched, 4000, 3000.0));
            by_request.record(LonePath::Serial, request(serial, 4000, 3000.0));
            let mut one_side = BatchingSignal::default();
            one_side.record(LonePath::Batched, decode(batched, 0.0, 0.0));
            one_side.record(LonePath::Serial, request(serial, 4000, 3000.0));
            let decoded = by_decode.rates.choose(0.0);
            assert_eq!(
                by_request.rates.choose(0.0),
                decoded,
                "{serial} against {batched}"
            );
            assert_eq!(
                one_side.rates.choose(0.0),
                decoded,
                "{serial} against {batched}"
            );
        }
    }

    /// A read of any size gives a prompt rate, and a read of nothing gives none.
    #[test]
    fn any_read_gives_a_prompt_rate() {
        assert_eq!(LoneDecode::prompt_rate(100, 0.5), Some(200.0));
        assert_eq!(LoneDecode::prompt_rate(1024, 2.0), Some(512.0));
        assert_eq!(LoneDecode::prompt_rate(1024, 0.0), None);
        assert_eq!(LoneDecode::prompt_rate(0, 0.1), None);
    }

    /// A worker that decodes faster and a serial path that reads the long prompt slower.
    fn long_prompt(serial_prompt_rate: Option<f64>, batched_prompt_rate: Option<f64>) -> PathRates {
        let with = |rate: f64, prompt_rate: Option<f64>| LoneDecode {
            tokens: 128,
            prompt_tokens: 1008,
            prompt_rate,
            ..decode(rate, 0.0, 0.0)
        };
        PathRates {
            serial: Some(with(122.0, serial_prompt_rate)),
            batched: Some(with(134.0, batched_prompt_rate)),
            latest_lookup: Some(0.0),
            latest_tokens: 128,
            serial_prompt_rate,
            batched_prompt_rate,
            latest_prompt_tokens: 1008,
        }
    }

    /// The worker's prompt rate alone is not charged to the worker alone: the decode rates
    /// decide, and the worker decodes faster.
    #[test]
    fn a_long_prompt_with_the_worker_prompt_rate_alone_goes_by_the_decode() {
        assert_eq!(
            long_prompt(None, Some(1008.0 / 0.18)).choose(0.0),
            LonePath::Batched
        );
    }

    /// With both prompt rates the worker reads and decodes faster, and wins.
    #[test]
    fn a_long_prompt_with_both_prompt_rates_counts_both() {
        assert_eq!(
            long_prompt(Some(1008.0 / 0.212), Some(1008.0 / 0.18)).choose(0.0),
            LonePath::Batched
        );
        // A serial path that reads much faster wins the same prompt.
        assert_eq!(
            long_prompt(Some(1008.0 / 0.01), Some(1008.0 / 0.18)).choose(0.0),
            LonePath::Serial
        );
    }

    /// Without any prompt rate the decode rates decide.
    #[test]
    fn a_long_prompt_without_prompt_rates_goes_by_the_decode() {
        assert_eq!(long_prompt(None, None).choose(0.0), LonePath::Batched);
    }

    /// A lone request never goes to a worker without a measure.
    /// The tail of a burst measures the worker, and then the rates choose.
    #[test]
    fn a_lone_request_waits_for_the_worker_measure_of_a_burst() {
        let mut s = BatchingSignal::default();
        assert_eq!(s.arrive(0, false), Switch::Stay);
        s.record(LonePath::Serial, decode(100.0, 0.0, 0.0));
        for _ in 0..3 {
            assert_eq!(s.arrive(0, false), Switch::Stay);
            s.record(LonePath::Serial, decode(100.0, 0.0, 0.0));
        }
        assert_eq!(s.arrive(2, false), Switch::ToBatched);
        s.record_worker_tail(LoneDecode {
            tokens: 40,
            ..decode(110.0, 0.0, 0.0)
        });
        assert_eq!(
            s.rates.latest_tokens, 200,
            "the tail does not size the next request"
        );
        assert_eq!(s.arrive(0, true), Switch::Stay);
    }

    /// A model a burst left in its worker goes back to the serial path while the worker has
    /// no measure.
    #[test]
    fn an_unmeasured_worker_gives_the_lone_request_back_to_the_serial_path() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Serial, decode(100.0, 0.0, 0.0));
        assert_eq!(s.arrive(1, false), Switch::ToBatched);
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
    }

    /// The first lone request of a model is served serially, where the load leaves it.
    /// A model a burst left in its worker goes back to the serial path for it.
    #[test]
    fn the_first_lone_request_is_served_serially() {
        let s = BatchingSignal::default();
        assert_eq!(s.rates.choose(0.0), LonePath::Serial);
        let mut fresh = BatchingSignal::default();
        assert_eq!(fresh.arrive(0, false), Switch::Stay);
        let mut after_burst = BatchingSignal::default();
        assert_eq!(after_burst.arrive(1, false), Switch::ToBatched);
        assert_eq!(after_burst.arrive(0, true), Switch::ToSerial);
    }

    /// The paths of 3 lone requests in a row, each recorded on the path it took.
    fn three_lone_requests(serial: LoneDecode, batched: LoneDecode) -> Vec<LonePath> {
        let mut s = BatchingSignal::default();
        let mut in_worker = false;
        let mut paths = Vec::new();
        for _ in 0..3 {
            match s.arrive(0, in_worker) {
                Switch::ToBatched => in_worker = true,
                Switch::ToSerial => in_worker = false,
                Switch::Stay => {}
            }
            let path = if in_worker {
                LonePath::Batched
            } else {
                LonePath::Serial
            };
            s.record(
                path,
                match path {
                    LonePath::Serial => serial,
                    LonePath::Batched => batched,
                },
            );
            paths.push(path);
        }
        paths
    }

    /// Lone requests stay serial while no burst has measured the worker, whatever it would do.
    #[test]
    fn lone_requests_stay_serial_without_a_worker_measure() {
        use LonePath::Serial;
        assert_eq!(
            three_lone_requests(request(122.0, 1008, 4750.0), request(134.0, 1008, 5600.0)),
            [Serial, Serial, Serial]
        );
        assert_eq!(
            three_lone_requests(request(640.0, 110, 9000.0), request(340.0, 110, 6000.0)),
            [Serial, Serial, Serial]
        );
    }

    /// Once a burst measured the worker, a worker that serves faster takes the lone requests
    /// and a slower one does not.
    #[test]
    fn a_worker_measured_by_a_burst_takes_the_lone_requests_it_serves_faster() {
        let measured = |serial: LoneDecode, tail: f64| {
            let mut s = BatchingSignal::default();
            s.record(LonePath::Serial, serial);
            s.record_worker_tail(decode(tail, 0.0, 0.0));
            s.rates.choose(0.0)
        };
        assert_eq!(
            measured(request(122.0, 1008, 4750.0), 134.0),
            LonePath::Batched
        );
        assert_eq!(
            measured(request(640.0, 110, 9000.0), 340.0),
            LonePath::Serial
        );
    }

    /// A request that arrives while a lone request is served batches, whatever the rates and
    /// whichever path the lone request took.
    #[test]
    fn a_request_beside_a_lone_one_batches() {
        let mut fresh = BatchingSignal::default();
        assert_eq!(fresh.arrive(0, false), Switch::Stay);
        assert_eq!(fresh.arrive(1, false), Switch::ToBatched);
        let mut measured = BatchingSignal::default();
        measured.record(LonePath::Serial, decode(500.0, 0.0, 0.0));
        assert_eq!(measured.arrive(1, false), Switch::ToBatched);
        assert_eq!(measured.arrive(1, true), Switch::Stay);
    }

    /// A model that cannot batch is offered the worker once, at its first overlap.
    #[test]
    fn a_model_that_cannot_batch_stays_serial_after_one_offer() {
        let mut s = BatchingSignal::default();
        assert_eq!(s.arrive(0, false), Switch::Stay);
        s.record(LonePath::Serial, decode(100.0, 0.0, 0.0));
        assert_eq!(s.arrive(1, false), Switch::ToBatched);
        s.mark_cannot_batch();
        for others in [0, 1, 3] {
            assert_eq!(s.arrive(others, false), Switch::Stay);
        }
    }

    #[test]
    fn a_lone_request_takes_the_faster_path() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Batched, decode(100.0, 0.0, 0.0));
        s.record(LonePath::Serial, decode(180.0, 0.5, 0.6));
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        assert_eq!(s.arrive(0, false), Switch::Stay);
        s.record(LonePath::Serial, decode(90.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
        assert_eq!(s.arrive(0, true), Switch::Stay);
    }

    /// A lone request in the worker whose answer a prompt lookup would draft moves the next
    /// lone request back to the serial path, and a plain one keeps the worker.
    #[test]
    fn the_latest_lookup_share_moves_the_choice() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Serial, decode(200.0, 0.5, 0.5));
        s.record(LonePath::Batched, decode(110.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, true), Switch::Stay);
        s.record(LonePath::Batched, decode(110.0, 0.0, 0.6));
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        s.record(LonePath::Serial, decode(95.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
    }

    #[test]
    fn a_serial_request_with_nothing_to_look_up_expects_the_lookup_share() {
        let rates = PathRates {
            serial: Some(decode(100.0, 0.0, 0.0)),
            batched: Some(decode(150.0, 0.0, 0.5)),
            latest_lookup: Some(0.5),
            latest_tokens: 200,
            ..PathRates::default()
        };
        assert_eq!(rates.choose(0.0), LonePath::Serial);
    }

    /// Lone serial requests that bursts follow cost the burst the move into the worker: the
    /// serial path keeps lone requests only while its gain pays that wait.
    #[test]
    fn the_wait_of_a_move_into_the_worker_is_charged_to_the_serial_path() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Batched, decode(100.0, 0.0, 0.0));
        s.record(LonePath::Serial, decode(125.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        // 200 tokens save 0.4 s serially, and a move made 7 requests wait 0.3 s each.
        s.record_handover();
        for _ in 0..7 {
            s.record_move_wait(0.3);
        }
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
        // Lone serial requests served to their end lower the wait each one is charged.
        for _ in 0..5 {
            s.record(LonePath::Serial, decode(125.0, 0.0, 0.0));
        }
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        // A prompt that drafts well pays the move.
        let mut d = BatchingSignal::default();
        d.record(LonePath::Serial, decode(250.0, 0.6, 0.6));
        d.record_handover();
        d.record_move_wait(0.9);
        d.record(LonePath::Batched, decode(100.0, 0.0, 0.6));
        assert_eq!(d.arrive(0, true), Switch::ToSerial);
    }

    /// Lone requests and bursts that alternate, each burst waiting for a move, end in the
    /// worker when the waits cost more than the lone requests gain.
    #[test]
    fn alternating_lone_requests_and_bursts_end_in_the_worker() {
        let mut s = BatchingSignal::default();
        let mut batched = false;
        let mut moves = 0;
        for _ in 0..6 {
            match s.arrive(0, batched) {
                Switch::ToBatched => batched = true,
                Switch::ToSerial => batched = false,
                Switch::Stay => {}
            }
            if batched {
                s.record(LonePath::Batched, decode(100.0, 0.0, 0.0));
            } else {
                // The lone request saves 0.4 s, then the first request of a burst of 8 starts
                // serially, moves, and the 7 others wait 0.1 s each.
                s.record(LonePath::Serial, decode(125.0, 0.0, 0.0));
                s.record_handover();
                for _ in 0..7 {
                    s.record_move_wait(0.1);
                }
                moves += 1;
            }
            assert_eq!(
                s.arrive(3, batched),
                if batched {
                    Switch::Stay
                } else {
                    Switch::ToBatched
                }
            );
            batched = true;
            // The last request of the burst decodes alone in the worker once the others end.
            s.record_worker_tail(decode(100.0, 0.0, 0.0));
        }
        assert!(moves <= 2, "{moves} moves");
        assert_eq!(
            s.arrive(0, true),
            Switch::Stay,
            "the alternation ends in the worker"
        );
    }

    #[test]
    fn an_overlap_goes_to_the_worker_whatever_the_rates() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Serial, decode(500.0, 0.8, 0.8));
        s.record(LonePath::Batched, decode(100.0, 0.0, 0.8));
        assert_eq!(s.arrive(1, false), Switch::ToBatched);
        assert_eq!(s.arrive(3, true), Switch::Stay);
    }

    #[test]
    fn after_a_serial_request_only_an_overlap_batches_again() {
        let mut s = BatchingSignal::default();
        s.record(LonePath::Serial, decode(50.0, 0.0, 0.0));
        s.record(LonePath::Batched, decode(100.0, 0.0, 0.0));
        s.mark_serial_needed();
        assert_eq!(s.arrive(0, false), Switch::Stay);
        assert_eq!(s.arrive(0, false), Switch::Stay);
        assert_eq!(s.arrive(1, false), Switch::ToBatched);
        s.mark_serial_needed();
        s.arrive(0, false);
        assert_eq!(s.arrive(0, false), Switch::Stay);
    }

    #[test]
    fn a_model_that_cannot_batch_is_not_offered_again() {
        let mut s = BatchingSignal::default();
        s.mark_cannot_batch();
        for _ in 0..10 {
            assert_eq!(s.arrive(3, false), Switch::Stay);
            assert_eq!(s.arrive(0, false), Switch::Stay);
        }
    }

    #[test]
    fn a_repeated_span_is_drafted_and_a_plain_answer_is_not() {
        let prompt: Vec<u32> = (10..60).collect();
        let copied: Vec<u32> = (10..60).collect();
        assert!(lookup_share(&prompt, &copied) > 0.8);
        let plain: Vec<u32> = (100..150).collect();
        assert_eq!(lookup_share(&prompt, &plain), 0.0);
        assert_eq!(lookup_share(&prompt, &[]), 0.0);
    }

    /// A request of any engine shows in the process's counts while it runs, and its start stays
    /// counted: a measure taken across it is not one model's own.
    #[test]
    fn a_request_of_another_engine_shows_in_the_process_counts() {
        let (mine, other) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let own = ActiveRequest::enter(&mine);
        let (started, running) = requests_of_the_process();
        assert!(running >= 1);
        let theirs = ActiveRequest::enter(&other);
        let (started_then, running_then) = requests_of_the_process();
        assert!(started_then > started);
        assert!(running_then >= 2);
        drop(theirs);
        assert!(requests_of_the_process().0 > started);
        drop(own);
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

    /// Only the request the arrivals chose moves the model into its batch worker.
    #[test]
    fn the_request_an_overlap_chose_is_the_one_to_batch() {
        let n = Arc::new(AtomicUsize::new(0));
        let mut chosen = ActiveRequest::enter(&n);
        let other = ActiveRequest::enter(&n);
        chosen.want_batched();
        assert!(chosen.wants_batched() && !other.wants_batched());
    }

    #[tokio::test]
    async fn a_waiter_wakes_when_the_model_moves_into_its_worker() {
        let b = Arc::new(Batching::default());
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (b2, f2) = (Arc::clone(&b), Arc::clone(&flag));
        let waiter = tokio::spawn(async move {
            b2.until(|| {
                let f = Arc::clone(&f2);
                async move { f.load(Ordering::Acquire) }
            })
            .await
        });
        tokio::task::yield_now().await;
        flag.store(true, Ordering::Release);
        b.note_moved();
        tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("woken")
            .unwrap();
    }
}
