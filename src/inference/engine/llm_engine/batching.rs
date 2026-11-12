//! When a model serves requests one at a time and when it batches them.
//!
//! Requests that overlap are served by the batch worker.
//! A lone request takes the path whose measured decode rate is higher.
//! The serial path drafts (prompt lookup, a draft head, a drafter model) and takes a prompt
//! from a prefill peer.
//! The batch worker does neither.
//!
//! A path without a measure for the loaded model takes the next lone request, the worker first,
//! so each path is measured by real requests.
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
}

/// The path a lone request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LonePath {
    Serial,
    Batched,
}

impl PathRates {
    /// The path with the shorter expected decode for the next lone request, which writes as
    /// many tokens as the latest lone request did.
    /// `serial_charge` seconds are added to the serial decode.
    ///
    /// A path without a measure is taken first, the worker before the serial path.
    /// The serial rate is turned into a rate per forward with the share its drafts gave.
    /// That share scales with the prompt lookup share of the latest lone request.
    pub fn choose(&self, serial_charge: f64) -> LonePath {
        let Some(batched) = self.batched else {
            return LonePath::Batched;
        };
        let Some(serial) = self.serial else {
            return LonePath::Serial;
        };
        let tokens = self.latest_tokens.max(1) as f64;
        let serial_s = tokens / self.expected_serial_rate(&serial) + serial_charge;
        if serial_s < tokens / batched.rate {
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

    /// Records the decode of a lone request on one path.
    pub fn record(&mut self, path: LonePath, decode: LoneDecode) {
        match path {
            LonePath::Serial => {
                self.rates.serial = Some(decode);
                self.serial_done += 1;
            }
            LonePath::Batched => self.rates.batched = Some(decode),
        }
        self.rates.latest_lookup = Some(decode.lookup);
        self.rates.latest_tokens = decode.tokens;
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

impl ActiveRequest {
    pub fn enter(count: &Arc<AtomicUsize>) -> Self {
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
        self.count.fetch_sub(1, Ordering::AcqRel);
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
        }
    }

    #[test]
    fn each_path_is_measured_before_the_rates_choose() {
        let mut s = BatchingSignal::default();
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
        s.record(LonePath::Batched, decode(110.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, true), Switch::ToSerial);
        s.record(LonePath::Serial, decode(100.0, 0.0, 0.0));
        assert_eq!(s.arrive(0, false), Switch::ToBatched);
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
