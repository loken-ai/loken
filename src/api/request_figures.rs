//! What a chat answer reports about itself: the `loken` object in its usage, and the comment
//! lines a streamed answer sends while its prompt is read.

use crate::energy_report::EnergyShare;
use crate::request_metrics::{RequestOutcome, RequestProbe};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

/// How often a streamed answer reports the reading of its prompt, and how long the reading
/// runs before the first report.
pub const PREFILL_REPORT_PERIOD: Duration = Duration::from_millis(250);

/// How often the reading is looked at between two reports.
const PREFILL_POLLS_PER_PERIOD: u32 = 5;

/// One request's own figures. A field loken did not measure is absent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LokenUsage {
    /// Milliseconds from its arrival at the model to the start of its prompt.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub queue_ms: Option<f64>,
    /// Milliseconds reading its prompt, to its first token.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prefill_ms: Option<f64>,
    /// Milliseconds writing its answer after the first token.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub energy: Option<LokenEnergy>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kv: Option<LokenKv>,
}

/// The request's share of the energy drawn while it ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LokenEnergy {
    /// The domains read, summed.
    pub joules: f64,
    /// Absent when no GPU counter is read.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub gpu_joules: Option<f64>,
    /// CPU package and DRAM; `null` when nothing reads them.
    pub cpu_joules: Option<f64>,
    /// The CPU part is not read from RAPL.
    pub estimated: bool,
    pub share: ShareRule,
}

/// How the energy of the samples the request was active in was attributed to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShareRule {
    /// No other request was active: every sample is the request's.
    Alone,
    /// Others were: each sample is split among the requests active during it by the tokens
    /// each computed over it, evenly when none computed any.
    Work,
}

/// The request's sequence in the KV cache and the cache's capacity, in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LokenKv {
    pub used_tokens: u64,
    pub capacity_tokens: u64,
}

impl LokenUsage {
    /// From what the decode path reported and the request's energy share, either absent when
    /// it was not measured.
    pub fn new(outcome: Option<&RequestOutcome>, energy: Option<&EnergyShare>) -> Self {
        let ms = |s: f64| s.max(0.0) * 1e3;
        Self {
            queue_ms: outcome.map(|o| ms(o.figures.queue_s)),
            prefill_ms: outcome.map(|o| ms(o.figures.prefill_s)),
            decode_ms: outcome.map(|o| ms(o.figures.decode_s)),
            energy: energy.and_then(|e| {
                Some(LokenEnergy {
                    joules: e.joules()?,
                    gpu_joules: e.gpu_j,
                    cpu_joules: e.cpu_j,
                    estimated: e.cpu_estimated,
                    share: if e.shared {
                        ShareRule::Work
                    } else {
                        ShareRule::Alone
                    },
                })
            }),
            kv: outcome.and_then(|o| o.kv).map(|kv| LokenKv {
                used_tokens: kv.used_tokens,
                capacity_tokens: kv.capacity_tokens,
            }),
        }
    }

    /// The object as JSON, for the routes that build their usage as a value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

/// Proposed tokens the model kept and proposed tokens it refused.
pub fn prediction_tokens(outcome: Option<&RequestOutcome>) -> (u64, u64) {
    outcome.map_or((0, 0), |o| {
        (o.accepted, o.drafted.saturating_sub(o.accepted))
    })
}

/// The text of the SSE comment that reports `done` of `total` prompt tokens read.
pub fn prefill_comment(done: u64, total: u64) -> String {
    format!("loken-prefill {{\"done\": {done}, \"total\": {total}}}")
}

/// A stream's start after the first period: opened, or still reading its prompt.
pub enum Opening<T, F> {
    Open(T),
    Reading(F),
}

/// Waits one [`PREFILL_REPORT_PERIOD`] for `start`: a start that failed by then fails the
/// request as before any answer was sent; one still running is handed back to be driven by
/// [`read_prompt`] inside the answer.
pub async fn open_stream<T, E, F>(
    mut start: std::pin::Pin<Box<F>>,
) -> Result<Opening<T, std::pin::Pin<Box<F>>>, E>
where
    F: Future<Output = Result<T, E>>,
{
    match tokio::time::timeout(PREFILL_REPORT_PERIOD, &mut start).await {
        Ok(Ok(opened)) => Ok(Opening::Open(opened)),
        Ok(Err(e)) => Err(e),
        Err(_) => Ok(Opening::Reading(start)),
    }
}

/// What driving a stream's start yields: the reading of the prompt while it lasts, then what
/// the start returned.
#[derive(Debug)]
pub enum Reading<T> {
    Progress { done: u64, total: u64 },
    Started(T),
}

/// Drives `start` to its end, reporting how far `probe` says the prompt is read about every
/// [`PREFILL_REPORT_PERIOD`], once the reading has run that long and while it is not done.
pub fn read_prompt<T, F>(
    start: F,
    probe: Arc<RequestProbe>,
) -> impl futures::Stream<Item = Reading<T>> + Send
where
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
{
    async_stream::stream! {
        let mut start = Box::pin(start);
        let poll = PREFILL_REPORT_PERIOD / PREFILL_POLLS_PER_PERIOD;
        let mut last_report: Option<std::time::Instant> = None;
        loop {
            let started = tokio::select! {
                r = &mut start => Some(r),
                _ = tokio::time::sleep(poll) => None,
            };
            if let Some(r) = started {
                yield Reading::Started(r);
                break;
            }
            let (done, total) = probe.prefill();
            let reading_long = probe
                .prefill_elapsed()
                .is_some_and(|e| e >= PREFILL_REPORT_PERIOD);
            let due = last_report.is_none_or(|t| t.elapsed() >= PREFILL_REPORT_PERIOD);
            if reading_long && due && done < total {
                last_report = Some(std::time::Instant::now());
                yield Reading::Progress { done, total };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_metrics::{KvPlace, RequestFigures};
    use futures::StreamExt;

    /// The object carries the times in milliseconds, the energy with its share rule and the
    /// cache place; a field not measured is absent, and an unread CPU part is `null`.
    #[test]
    fn the_object_reports_what_was_measured_and_nothing_else() {
        let outcome = RequestOutcome {
            figures: RequestFigures {
                queue_s: 0.002,
                prefill_s: 0.5,
                decode_s: 1.25,
                ..Default::default()
            },
            drafted: 10,
            accepted: 7,
            kv: KvPlace::of(Some(40), Some(4096)),
        };
        let energy = EnergyShare {
            gpu_j: Some(3.0),
            cpu_j: None,
            cpu_estimated: true,
            shared: true,
            duration_s: 1.8,
        };
        let v = LokenUsage::new(Some(&outcome), Some(&energy)).to_json();
        assert_eq!(v["queue_ms"], 2.0);
        assert_eq!(v["prefill_ms"], 500.0);
        assert_eq!(v["decode_ms"], 1250.0);
        assert_eq!(v["energy"]["joules"], 3.0);
        assert_eq!(v["energy"]["gpu_joules"], 3.0);
        assert!(v["energy"]["cpu_joules"].is_null());
        assert!(v["energy"].get("cpu_joules").is_some());
        assert_eq!(v["energy"]["estimated"], true);
        assert_eq!(v["energy"]["share"], "work");
        assert_eq!(v["kv"]["used_tokens"], 40);
        assert_eq!(v["kv"]["capacity_tokens"], 4096);
        assert_eq!(prediction_tokens(Some(&outcome)), (7, 3));

        let unread = EnergyShare::default();
        let v = LokenUsage::new(None, Some(&unread)).to_json();
        assert_eq!(v, serde_json::json!({}));
        let alone = EnergyShare {
            cpu_j: Some(2.0),
            ..Default::default()
        };
        let v = LokenUsage::new(None, Some(&alone)).to_json();
        assert_eq!(v["energy"]["share"], "alone");
        assert!(v["energy"].get("gpu_joules").is_none());
    }

    /// The comment line is the agreed one.
    #[test]
    fn the_comment_line_is_the_agreed_one() {
        assert_eq!(
            prefill_comment(512, 2048),
            "loken-prefill {\"done\": 512, \"total\": 2048}"
        );
    }

    /// A slow prefill is reported about every period while it runs, with what it has read; a
    /// fast one is not reported at all.
    #[tokio::test]
    async fn a_slow_prefill_is_reported_as_it_goes() {
        const TOTAL: u64 = 4096;
        const CHUNKS: u64 = 8;
        let probe = Arc::new(RequestProbe::default());
        let reader = probe.clone();
        let slow = async move {
            for c in 0..CHUNKS {
                reader.note_prefill(c * TOTAL / CHUNKS, TOTAL);
                tokio::time::sleep(PREFILL_REPORT_PERIOD / 2).await;
            }
            reader.note_prefill(TOTAL, TOTAL);
            "first token"
        };
        let items: Vec<_> = read_prompt(slow, probe).collect().await;
        let progress: Vec<(u64, u64)> = items
            .iter()
            .filter_map(|r| match r {
                Reading::Progress { done, total } => Some((*done, *total)),
                Reading::Started(_) => None,
            })
            .collect();
        assert!(progress.len() >= 2, "{progress:?}");
        assert!(progress.len() <= (CHUNKS / 2 + 1) as usize, "{progress:?}");
        assert!(
            progress.windows(2).all(|w| w[0].0 <= w[1].0),
            "{progress:?}"
        );
        assert!(progress.iter().all(|&(d, t)| t == TOTAL && d < TOTAL));
        assert!(matches!(
            items.last(),
            Some(Reading::Started("first token"))
        ));

        let probe = Arc::new(RequestProbe::default());
        let reader = probe.clone();
        let fast = async move {
            reader.note_prefill(0, TOTAL);
            tokio::time::sleep(PREFILL_REPORT_PERIOD / 4).await;
            reader.note_prefill(TOTAL, TOTAL);
        };
        let items: Vec<_> = read_prompt(fast, probe).collect().await;
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Reading::Started(())));
    }
}
