//! What one engine measured about its answers: how fast it decodes, how long its answers
//! run, and how far its running requests have gone.

use std::sync::{Arc, Mutex};

/// The measures of one engine.
#[derive(Default)]
pub struct Pace {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Tokens decoded after the first token of each finished answer, and the seconds that took.
    decoded: u64,
    decode_secs: f64,
    /// Answers finished, and the tokens they held.
    answers: u64,
    answer_tokens: u64,
    /// The requests running, by the number `begin` gave them.
    running: std::collections::HashMap<u64, Running>,
    next: u64,
    /// The last load whose plan was given a predicted decode rate, until an answer measures it.
    load: Option<Load>,
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

    /// Mean tokens per answer, or `None` when no answer has finished yet.
    pub fn mean_answer_tokens(&self) -> Option<f64> {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.answers > 0 {
            Some(s.answer_tokens as f64 / s.answers as f64)
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

    /// Start tracking a request with the given max_tokens. Returns the request id.
    fn begin(&self, max_tokens: Option<u64>) -> u64 {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = s.next;
        s.next += 1;
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
    /// adds to the decode statistics.
    fn end(&self, id: u64, decode_secs: Option<f64>) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(running) = s.running.remove(&id) {
            if running.generated > 0 {
                s.answer_tokens += running.generated;
                s.answers += 1;
            }
            if let Some(secs) = decode_secs {
                if secs > 0.0 && running.generated > 1 {
                    s.decoded += running.generated - 1;
                    s.decode_secs += secs;
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
    }
}

/// Ends the request it watches when dropped, so a forwarder that stops early still ends it.
struct Watched {
    pace: Arc<Pace>,
    id: u64,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
}

impl Drop for Watched {
    fn drop(&mut self) {
        // The decode time runs from the first token to the last.
        let secs = match (self.first, self.last) {
            (Some(first), Some(last)) if last > first => Some((last - first).as_secs_f64()),
            _ => None,
        };
        self.pace.end(self.id, secs);
    }
}

/// A request answered whole, which this pace follows until it drops.
pub struct Followed {
    pace: Arc<Pace>,
    id: u64,
    decode_secs: Option<f64>,
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
        self.pace.end(self.id, self.decode_secs);
    }
}

impl Pace {
    /// Follows a request answered whole, whose tokens come at its end.
    ///
    /// Until then it counts as having generated nothing.
    pub fn follow(self: &Arc<Self>, max_tokens: Option<u64>) -> Followed {
        Followed {
            pace: self.clone(),
            id: self.begin(max_tokens),
            decode_secs: None,
        }
    }

    /// The messages of `rx`, forwarded unchanged, each `Ok` one counted as a token of a request
    /// this pace follows until `rx` ends or the returned receiver is dropped.
    pub fn watch(
        self: &Arc<Self>,
        mut rx: tokio::sync::mpsc::Receiver<Result<String, String>>,
        max_tokens: Option<u64>,
    ) -> tokio::sync::mpsc::Receiver<Result<String, String>> {
        let (tx, out) = tokio::sync::mpsc::channel(rx.max_capacity());
        let mut watched = Watched {
            pace: self.clone(),
            id: self.begin(max_tokens),
            first: None,
            last: None,
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
        pace.end(id, Some(5.0));
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
        pace.end(id1, Some(5.0));

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
        pace.end(id1, Some(5.0));

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
        pace.end(id, Some(5.0));

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
        pace.end(id, Some(1.0));
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
        pace.end(id, Some(1.0));
        assert!(pace.state.lock().unwrap().load.is_none());
    }

    /// A request answered whole counts while it runs and gives its rate at its end.
    #[test]
    fn a_followed_request_runs_then_gives_its_rate() {
        let pace = Arc::new(Pace::default());
        let mut followed = pace.follow(Some(50));
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

        let mut rx = pace.watch(rx, Some(100));

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
}
