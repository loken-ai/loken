//! What one engine measured about its answers: how fast it decodes, how long its answers
//! run, and how far its running requests have gone.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The measures of one engine.
#[derive(Default)]
pub struct Pace {
    state: Mutex<State>,
    /// For automatic saving: the model id of this pace.
    model_id: Mutex<Option<String>>,
}

#[derive(Default)]
struct State {
    /// Tokens decoded after the first token of each finished answer, and the seconds that took.
    decoded: u64,
    decode_secs: f64,
    /// Tokens decoded and seconds of answers that ran while another engine had a request running
    /// (as `decoded`/`decode_secs`, counted only for those answers).
    beside_decoded: u64,
    beside_secs: f64,
    /// Answers finished, and the tokens they held.
    answers: u64,
    answer_tokens: u64,
    /// The gaps between `begin` calls, for mean gap calculation.
    gaps: u64,
    gap_secs: f64,
    /// Last `begin` time, not saved (used for gap calculation).
    last_start: Option<std::time::Instant>,
    /// The requests running, by the number `begin` gave them.
    running: std::collections::HashMap<u64, Running>,
    next: u64,
    /// The last load whose plan was given a predicted decode rate, until an answer measures it.
    load: Option<Load>,
    /// The decode rate the plan of the last load predicts, `None` when its estimate cannot say.
    planned_tok_s: Option<f64>,
}

/// A load's predicted decode rate, and the decode counts when it ended.
struct Load {
    model: String,
    predicted_tok_s: f64,
    decoded: u64,
    decode_secs: f64,
}

/// The decode rate the answers after a load measured, against the rate its plan predicted.
#[derive(Debug, PartialEq)]
struct LoadMeasure {
    model: String,
    predicted_tok_s: f64,
    measured_tok_s: f64,
}

impl State {
    /// The rate decoded since the marked load, once: the mark is then cleared.
    /// `None` before a decode after the load, or when no load is marked.
    fn since_load(&mut self) -> Option<LoadMeasure> {
        let load = self.load.as_ref()?;
        let decoded = self.decoded.checked_sub(load.decoded)?;
        let secs = self.decode_secs - load.decode_secs;
        if decoded == 0 || secs <= 0.0 {
            return None;
        }
        let load = self.load.take()?;
        Some(LoadMeasure {
            model: load.model,
            predicted_tok_s: load.predicted_tok_s,
            measured_tok_s: decoded as f64 / secs,
        })
    }
}

struct Running {
    generated: u64,
    max_tokens: Option<u64>,
}

impl Pace {
    /// Marks a load of `model` whose plan was predicted to decode at `predicted_tok_s`.
    ///
    /// The first answer after it logs the rate it measured against the prediction.
    /// `None` clears the mark of an earlier load.
    pub fn loaded(&self, model: &str, predicted_tok_s: Option<f64>) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.planned_tok_s = predicted_tok_s;
        s.load = predicted_tok_s.map(|predicted_tok_s| Load {
            model: model.to_string(),
            predicted_tok_s,
            decoded: s.decoded,
            decode_secs: s.decode_secs,
        });
    }

    /// Decode tokens per second, or `None` when no rate has been measured yet.
    pub fn decode_tok_s(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.decoded > 0 && s.decode_secs > 0.0 {
            Some(s.decoded as f64 / s.decode_secs)
        } else {
            None
        }
    }

    /// Beside decode tokens per second (while another engine runs), or `None` when no rate
    /// has been measured yet.
    pub fn beside_tok_s(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.beside_decoded > 0 && s.beside_secs > 0.0 {
            Some(s.beside_decoded as f64 / s.beside_secs)
        } else {
            None
        }
    }

    /// Mean tokens per answer, or `None` when no answer has finished yet.
    pub fn mean_answer_tokens(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.answers > 0 {
            Some(s.answer_tokens as f64 / s.answers as f64)
        } else {
            None
        }
    }

    /// Mean gap between `begin` calls, or `None` when no gap has been measured yet.
    pub fn mean_gap_secs(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.gaps > 0 && s.gap_secs > 0.0 {
            Some(s.gap_secs / s.gaps as f64)
        } else {
            None
        }
    }

    /// Seconds until every running request is expected to end: `Some(0.0)` when none runs.
    ///
    /// A request with a `max_tokens` has that many tokens at most.
    /// One without has the mean length of the answers finished so far.
    /// `None` when no rate is measured yet, or when a request without `max_tokens` runs before
    /// any answer has finished.
    pub fn expected_end_secs(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.running.is_empty() {
            return Some(0.0);
        }
        let rate =
            (s.decoded > 0 && s.decode_secs > 0.0).then(|| s.decoded as f64 / s.decode_secs)?;
        let mean = (s.answers > 0).then(|| s.answer_tokens as f64 / s.answers as f64);
        s.running.values().try_fold(0.0_f64, |longest, r| {
            let left = match r.max_tokens {
                Some(max) => max.saturating_sub(r.generated) as f64,
                None => (mean? - r.generated as f64).max(0.0),
            };
            Some(longest.max(left / rate))
        })
    }

    /// Starts following a request with the given `max_tokens`, and returns its number.
    fn begin(&self, max_tokens: Option<u64>) -> u64 {
        self.begin_at(max_tokens, std::time::Instant::now())
    }

    /// [`Self::begin`] at the instant `now`.
    ///
    /// Each start after the first adds the time since the previous start to the gaps.
    fn begin_at(&self, max_tokens: Option<u64>, now: std::time::Instant) -> u64 {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = s.next;
        s.next += 1;
        if let Some(last) = s.last_start {
            s.gaps += 1;
            s.gap_secs += now.saturating_duration_since(last).as_secs_f64();
        }
        s.last_start = Some(now);
        s.running.insert(
            id,
            Running {
                generated: 0,
                max_tokens,
            },
        );
        id
    }

    /// Record that a token was generated for the given request.
    fn token(&self, id: u64) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = s.running.get_mut(&id) {
            running.generated += 1;
        }
    }

    /// End the request. When `decode_secs` is Some and positive and generated tokens > 1,
    /// adds to the decode statistics. If `beside` is true, updates beside stats instead.
    fn end(&self, id: u64, decode_secs: Option<f64>, beside: bool) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = s.running.remove(&id) {
            if running.generated > 0 {
                s.answer_tokens += running.generated;
                s.answers += 1;
            }
            if let Some(secs) = decode_secs {
                if secs > 0.0 && running.generated > 1 {
                    if beside {
                        s.beside_decoded += running.generated - 1;
                        s.beside_secs += secs;
                    } else {
                        s.decoded += running.generated - 1;
                        s.decode_secs += secs;
                    }
                }
            }
        }
        if let Some(m) = s.since_load() {
            tracing::info!(
                "{}: decode measured at {:.1} tok/s after the load, predicted {:.1} tok/s for its plan ({:.2} of the prediction)",
                m.model,
                m.measured_tok_s,
                m.predicted_tok_s,
                m.measured_tok_s / m.predicted_tok_s
            );
        }
        drop(s);
        // Save pace measures after request ends, if a model_id is set.
        self.save_if_loaded();
    }

    /// Save pace measures if a model_id is set.
    fn save_if_loaded(&self) {
        let model_id = self
            .model_id
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(model_id) = model_id {
            let kept = self.kept();
            Pace::save(&model_id, &kept);
        }
    }
}

/// Ends the request it watches when dropped, so a forwarder that stops early still ends it.
struct Watched {
    pace: Arc<Pace>,
    id: u64,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    beside: bool,
}

impl Drop for Watched {
    fn drop(&mut self) {
        // The decode time runs from the first token to the last.
        let secs = match (self.first, self.last) {
            (Some(first), Some(last)) if last > first => Some((last - first).as_secs_f64()),
            _ => None,
        };
        self.pace.end(self.id, secs, self.beside);
    }
}

#[derive(Clone)]
pub struct Followed {
    pace: Arc<Pace>,
    id: u64,
    decode_secs: Option<f64>,
    beside: bool,
}

impl Followed {
    /// The answer held `tokens` tokens, decoded in `secs` seconds.
    pub fn finished(&mut self, tokens: u64, secs: f64) {
        let mut s = self.pace.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = s.running.get_mut(&self.id) {
            running.generated = tokens;
        }
        // The rate counts the tokens after the first, as a streamed answer does.
        if tokens > 1 {
            self.decode_secs = Some(secs * (tokens - 1) as f64 / tokens as f64);
        }
    }
}

impl Drop for Followed {
    fn drop(&mut self) {
        self.pace.end(self.id, self.decode_secs, self.beside);
    }
}

impl Pace {
    /// Follows a request answered whole, whose tokens come at its end.
    ///
    /// Until then it counts as having generated nothing.
    /// If `beside` is true, the stats will be recorded as beside stats.
    pub fn follow(self: &Arc<Self>, max_tokens: Option<u64>, beside: bool) -> Followed {
        Followed {
            pace: self.clone(),
            id: self.begin(max_tokens),
            decode_secs: None,
            beside,
        }
    }

    /// The messages of `rx`, forwarded unchanged, each `Ok` one counted as a token of a request
    /// this pace follows until `rx` ends or the returned receiver is dropped.
    /// If `beside` is true, the stats will be recorded as beside stats.
    pub fn watch(
        self: &Arc<Self>,
        mut rx: tokio::sync::mpsc::Receiver<Result<String, String>>,
        max_tokens: Option<u64>,
        beside: bool,
    ) -> tokio::sync::mpsc::Receiver<Result<String, String>> {
        let (tx, out) = tokio::sync::mpsc::channel(rx.max_capacity());
        let mut watched = Watched {
            pace: self.clone(),
            id: self.begin(max_tokens),
            first: None,
            last: None,
            beside,
        };
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if msg.is_ok() {
                    watched.pace.token(watched.id);
                    let now = std::time::Instant::now();
                    watched.first.get_or_insert(now);
                    watched.last = Some(now);
                }
                // A receiver that went away ends the request: dropping `rx` stops its producer.
                if tx.send(msg).await.is_err() {
                    break;
                }
            }
        });
        out
    }
}

/// What is persisted of a [`Pace`] for a model, for reuse after a reload.
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct Kept {
    pub decoded: u64,
    pub decode_secs: f64,
    pub beside_decoded: u64,
    pub beside_secs: f64,
    pub answers: u64,
    pub answer_tokens: u64,
    pub gaps: u64,
    pub gap_secs: f64,
    /// The decode rate the plan of the last load predicts.
    #[serde(default)]
    pub planned_tok_s: Option<f64>,
}

impl Kept {
    /// Decode tokens per second of the answers that ran alone, `None` before one was measured.
    pub fn decode_tok_s(&self) -> Option<f64> {
        (self.decoded > 0 && self.decode_secs > 0.0).then(|| self.decoded as f64 / self.decode_secs)
    }

    /// Mean tokens per answer, `None` before an answer finished.
    pub fn mean_answer_tokens(&self) -> Option<f64> {
        (self.answers > 0).then(|| self.answer_tokens as f64 / self.answers as f64)
    }

    /// The decode rate of a plan whose estimate predicts `predicted_tok_s`: the estimate scaled
    /// by the rate measured against the rate the plan of the last load predicted.
    /// `None` when a rate is not known.
    pub fn rate_on(&self, predicted_tok_s: f64) -> Option<f64> {
        let secs = crate::inference::place::decode_time::calibrated(
            1.0 / predicted_tok_s,
            1.0 / self.planned_tok_s?,
            self.decode_tok_s()?,
        )?;
        Some(1.0 / secs)
    }
}

impl Pace {
    /// Convert the current state to a persisted form.
    pub fn kept(&self) -> Kept {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        Kept {
            decoded: s.decoded,
            decode_secs: s.decode_secs,
            beside_decoded: s.beside_decoded,
            beside_secs: s.beside_secs,
            answers: s.answers,
            answer_tokens: s.answer_tokens,
            gaps: s.gaps,
            gap_secs: s.gap_secs,
            planned_tok_s: s.planned_tok_s,
        }
    }

    /// Restore a persisted state, keeping the running requests intact.
    pub fn restore(&self, kept: &Kept) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.decoded = kept.decoded;
        s.decode_secs = kept.decode_secs;
        s.beside_decoded = kept.beside_decoded;
        s.beside_secs = kept.beside_secs;
        s.answers = kept.answers;
        s.answer_tokens = kept.answer_tokens;
        s.gaps = kept.gaps;
        s.gap_secs = kept.gap_secs;
        // last_start is not persisted, will be initialized on next begin
        //
        // A load mark counts from the restored measures, so the first answer measures only itself.
        let (decoded, decode_secs) = (s.decoded, s.decode_secs);
        if let Some(load) = s.load.as_mut() {
            load.decoded = decoded;
            load.decode_secs = decode_secs;
        }
    }

    /// Unique key for persisting pace measures of `model_id`.
    pub fn key(model_id: &str) -> String {
        let version = env!("LOKEN_VERSION");
        let machine = crate::inference::place::device_probe::probe_cuda_gpus()
            .into_iter()
            .map(|g| format!("{}:{}", g.total, g.num_cores))
            .collect::<Vec<_>>()
            .join(",");
        format!("{}|{}|{}", model_id, version, machine)
    }

    /// Load persisted pace measures for `model_id`, from the cache directory.
    pub fn load(model_id: &str) -> Option<Kept> {
        let dir = crate::inference::generic_transformer::routing_calibration::cache_dir();
        Self::load_from(model_id, dir.as_deref())
    }

    fn load_from(model_id: &str, dir: Option<&Path>) -> Option<Kept> {
        let path = dir?.join("pace.json");
        let text = std::fs::read_to_string(path).ok()?;
        let map: std::collections::HashMap<String, Kept> = serde_json::from_str(&text).ok()?;
        map.get(&Self::key(model_id)).cloned()
    }

    /// Save persisted pace measures for `model_id`, to the cache directory.
    pub fn save(model_id: &str, kept: &Kept) {
        let dir = crate::inference::generic_transformer::routing_calibration::cache_dir();
        Self::save_to(model_id, kept, dir.as_deref());
    }

    fn save_to(model_id: &str, kept: &Kept, dir: Option<&Path>) {
        let Some(dir) = dir else { return };
        let path = dir.join("pace.json");
        let mut map: std::collections::HashMap<String, Kept> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        map.insert(Self::key(model_id), kept.clone());
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(
            path,
            serde_json::to_string_pretty(&map).ok().unwrap_or_default(),
        );
    }
}

impl Pace {
    /// Set the model id for automatic saving.
    pub fn set_model_id(&self, model_id: Option<String>) {
        *self.model_id.lock().unwrap_or_else(|e| e.into_inner()) = model_id;
    }

    /// The model id, for automatic saving.
    pub fn model_id(&self) -> Option<String> {
        self.model_id
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn idle_pace_expects_zero() {
        let pace = Pace::default();
        assert_eq!(pace.expected_end_secs(), Some(0.0));
    }

    #[tokio::test]
    async fn expected_end_none_after_begin_and_tokens_no_rate() {
        let pace = Pace::default();
        let id = pace.begin(Some(100));
        for _ in 0..10 {
            pace.token(id);
        }
        assert_eq!(pace.expected_end_secs(), None);
    }

    #[tokio::test]
    async fn rate_and_answer_tokens_after_finish() {
        let pace = Pace::default();
        let id = pace.begin(Some(100));
        for _ in 0..51 {
            pace.token(id);
        }
        pace.end(id, Some(5.0), false);
        assert_eq!(pace.decode_tok_s(), Some(10.0));
        assert_eq!(pace.mean_answer_tokens(), Some(51.0));
    }

    #[tokio::test]
    async fn expected_end_for_request_with_max_tokens() {
        let pace = Pace::default();
        // Finish an answer first to get a rate
        let id1 = pace.begin(Some(100));
        for _ in 0..51 {
            pace.token(id1);
        }
        pace.end(id1, Some(5.0), false);

        // Start a new request
        let id2 = pace.begin(Some(100));
        for _ in 0..40 {
            pace.token(id2);
        }

        assert_eq!(pace.expected_end_secs(), Some(6.0));
    }

    #[tokio::test]
    async fn expected_end_for_request_without_max_tokens() {
        let pace = Pace::default();
        // Finish an answer first to get a rate and mean
        let id1 = pace.begin(Some(100));
        for _ in 0..51 {
            pace.token(id1);
        }
        pace.end(id1, Some(5.0), false);

        // Start a new request without max_tokens
        let id2 = pace.begin(None);
        for _ in 0..11 {
            pace.token(id2);
        }

        assert_eq!(pace.expected_end_secs(), Some(4.0));
    }

    #[tokio::test]
    async fn zero_tokens_generated_does_not_count_as_answer() {
        let pace = Pace::default();
        let id = pace.begin(Some(100));
        // Don't call token, just end
        pace.end(id, Some(5.0), false);

        assert_eq!(pace.mean_answer_tokens(), None);
    }

    /// The first answer after a marked load measures the rate decoded since that load, once.
    #[test]
    fn the_first_answer_after_a_load_measures_its_rate_once() {
        let pace = Pace::default();
        let id = pace.begin(None);
        for _ in 0..11 {
            pace.token(id);
        }
        pace.end(id, Some(1.0), false);
        pace.loaded("m", Some(25.0));
        let mut s = pace.state.lock().unwrap();
        assert_eq!(s.since_load(), None);
        s.decoded += 40;
        s.decode_secs += 2.0;
        assert_eq!(
            s.since_load(),
            Some(LoadMeasure {
                model: "m".to_string(),
                predicted_tok_s: 25.0,
                measured_tok_s: 20.0,
            })
        );
        assert_eq!(s.since_load(), None);
        drop(s);
        // A load with no prediction clears the mark.
        pace.loaded("m", Some(25.0));
        pace.loaded("m", None);
        let id = pace.begin(None);
        pace.token(id);
        pace.token(id);
        pace.end(id, Some(1.0), false);
        assert!(pace.state.lock().unwrap().load.is_none());
    }

    /// The rate of another plan scales the measured rate by the plan's estimate against the
    /// estimate of the plan the rate was measured on, and a load keeps its plan's estimate.
    #[test]
    fn the_rate_of_another_plan_follows_its_estimate() {
        let pace = Pace::default();
        pace.loaded("m", Some(20.0));
        pace.restore(&Kept {
            decoded: 100,
            decode_secs: 10.0,
            ..Default::default()
        });
        let kept = pace.kept();
        assert_eq!(kept.planned_tok_s, Some(20.0));
        assert_eq!(kept.decode_tok_s(), Some(10.0));
        assert!((kept.rate_on(10.0).unwrap() - 5.0).abs() < 1e-9);
        let unplanned = Kept {
            planned_tok_s: None,
            ..kept
        };
        assert_eq!(unplanned.rate_on(10.0), None);
    }

    /// Measures restored after a load leave the load mark measuring only the answers after it.
    #[test]
    fn a_restore_after_a_load_keeps_the_mark_on_the_new_answers() {
        let pace = Pace::default();
        pace.loaded("m", Some(10.0));
        pace.restore(&Kept {
            decoded: 1000,
            decode_secs: 100.0,
            ..Default::default()
        });
        let mut s = pace.state.lock().unwrap();
        s.decoded += 20;
        s.decode_secs += 1.0;
        assert_eq!(s.since_load().map(|m| m.measured_tok_s), Some(20.0));
    }

    /// A request answered whole counts while it runs and gives its rate at its end.
    #[test]
    fn a_followed_request_runs_then_gives_its_rate() {
        let pace = Arc::new(Pace::default());
        let mut followed = pace.follow(Some(50), false);
        assert_eq!(pace.state.lock().unwrap().running.len(), 1);
        followed.finished(11, 1.1);
        drop(followed);
        assert_eq!(pace.mean_answer_tokens(), Some(11.0));
        assert_eq!(pace.decode_tok_s(), Some(10.0));
        assert_eq!(pace.expected_end_secs(), Some(0.0));
    }

    #[tokio::test]
    async fn watch_forwards_and_counts_tokens() {
        let pace = Arc::new(Pace::default());
        let (tx, rx) = mpsc::channel::<Result<String, String>>(4);

        let mut rx = pace.watch(rx, Some(100), false);

        // Send 3 Ok and 1 Err
        tx.send(Ok("a".to_string())).await.unwrap();
        tx.send(Ok("b".to_string())).await.unwrap();
        tx.send(Ok("c".to_string())).await.unwrap();
        tx.send(Err("error".to_string())).await.unwrap();
        drop(tx);

        // Read until end to ensure the forwarder completes
        let mut count = 0;
        while rx.recv().await.is_some() {
            count += 1;
        }

        assert_eq!(count, 4);
        assert_eq!(pace.mean_answer_tokens(), Some(3.0));
    }

    /// Test that beside stats are tracked separately from plain stats.
    #[test]
    fn beside_stats_separate_from_plain_stats() {
        let pace = Pace::default();
        // Plain request
        let id1 = pace.begin(Some(100));
        for _ in 0..51 {
            pace.token(id1);
        }
        pace.end(id1, Some(5.0), false);

        // Beside request
        let id2 = pace.begin(Some(100));
        for _ in 0..51 {
            pace.token(id2);
        }
        pace.end(id2, Some(5.0), true);

        // Plain stats should have 50 decoded tokens at 10 tok/s
        assert_eq!(pace.decode_tok_s(), Some(10.0));
        // Beside stats should also have 50 decoded tokens at 10 tok/s
        assert_eq!(pace.beside_tok_s(), Some(10.0));
        // Plain answer tokens should be 51
        assert_eq!(pace.mean_answer_tokens(), Some(51.0));
    }

    /// Test that mean_gap_secs computes correctly from 3 begin calls.
    #[test]
    fn mean_gap_secs_computes_correctly() {
        let now = std::time::Instant::now();
        let pace = Pace::default();
        // We can't actually wait, so we use begin_at which lets us set a fake now
        let id1 = pace.begin_at(Some(100), now);
        let id2 = pace.begin_at(Some(100), now + std::time::Duration::from_secs(2));
        let id3 = pace.begin_at(Some(100), now + std::time::Duration::from_secs(5));

        // We have 2 gaps: 2 seconds and 3 seconds, mean = 2.5
        assert_eq!(pace.mean_gap_secs(), Some(2.5));
    }

    /// Test that begin with beside=true tracks gaps.
    #[test]
    fn every_start_after_the_first_adds_a_gap() {
        let now = std::time::Instant::now();
        let pace = Pace::default();
        // The first start has no gap before it.
        let _id1 = pace.begin_at(Some(100), now);
        let _id2 = pace.begin_at(Some(100), now + std::time::Duration::from_secs(3));
        let _id3 = pace.begin_at(Some(100), now + std::time::Duration::from_secs(6));

        // 2 gaps of 3 seconds each, mean = 3
        assert_eq!(pace.mean_gap_secs(), Some(3.0));
    }

    /// Test kept/restore round-trip preserves running requests.
    #[test]
    fn kept_restore_roundtrip_preserves_running() {
        let pace = Arc::new(Pace::default());
        // Start a request
        let id = pace.begin(Some(100));
        for _ in 0..10 {
            pace.token(id);
        }
        // Get state before end
        let before = pace
            .state
            .lock()
            .unwrap()
            .running
            .get(&id)
            .unwrap()
            .generated;
        assert_eq!(before, 10);

        // Finish and create a kept state
        pace.end(id, Some(1.0), false);
        let kept = pace.kept();

        // Restore should keep the same stats
        pace.restore(&kept);
        assert_eq!(pace.mean_answer_tokens(), Some(10.0));
        assert_eq!(pace.decode_tok_s(), Some(9.0));
    }

    /// Test that key includes model id and version.
    #[test]
    fn key_includes_model_id_and_version() {
        let key = Pace::key("test-model");
        assert!(key.starts_with("test-model|"));
        assert!(key.contains("|"));
    }

    /// Test save and load through private helpers in a temp dir.
    #[test]
    fn save_and_load_in_temp_dir() {
        use tempfile::tempdir;
        let temp = tempdir().unwrap();
        let dir = temp.path();

        // Save one model's pace
        let kept1 = Kept {
            decoded: 100,
            decode_secs: 10.0,
            beside_decoded: 0,
            beside_secs: 0.0,
            answers: 10,
            answer_tokens: 100,
            gaps: 0,
            gap_secs: 0.0,
            planned_tok_s: None,
        };
        Pace::save_to("model1", &kept1, Some(dir));

        // Load it back
        let loaded1 = Pace::load_from("model1", Some(dir)).unwrap();
        assert_eq!(loaded1, kept1);

        // Save another model's pace
        let kept2 = Kept {
            decoded: 200,
            decode_secs: 20.0,
            beside_decoded: 50,
            beside_secs: 5.0,
            answers: 20,
            answer_tokens: 200,
            gaps: 1,
            gap_secs: 2.0,
            planned_tok_s: Some(30.0),
        };
        Pace::save_to("model2", &kept2, Some(dir));

        // Both models should be in the file
        let loaded1_again = Pace::load_from("model1", Some(dir)).unwrap();
        let loaded2 = Pace::load_from("model2", Some(dir)).unwrap();
        assert_eq!(loaded1_again, kept1);
        assert_eq!(loaded2, kept2);

        // Missing model returns None
        assert!(Pace::load_from("model3", Some(dir)).is_none());
    }
}
