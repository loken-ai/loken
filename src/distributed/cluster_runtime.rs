//! The loops that keep the cluster's picture of itself current, and the hand-over itself.
//!
//! Everything beside this is a decision waiting for facts. This is what supplies them: an
//! announcement so peers can find this node, a poll so this node learns theirs, and the
//! round-trip time measured by that very poll rather than assumed.
//!
//! Measuring the RTT here matters more than it looks. The router compares
//! `predicted(peer) + rtt` against serving locally, so a wrong RTT does not degrade the
//! decision gracefully - it inverts it. Taking it from the request we are already making costs
//! nothing and is the only figure that reflects the network as it is now.
//!
//! One rule the hand-over must not break: the relayed answer is the peer's answer, byte for
//! byte, with its status. A proxy that reinterprets a response turns a peer's clean error into
//! a local one and hides which node actually failed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::cluster::{Cluster, FORWARDED_HEADER};
use super::discovery::{self, Announcement, PeerBook};
use super::membership::NodeState;
use super::routing::NodeRates;

/// What a peer answers on the state endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerReport {
    pub node_id: String,
    pub models: Vec<String>,
    /// What this node could load, as opposed to what it holds. Absent from the wire when the
    /// peer runs a build that has no catalogue, which is why it is an Option rather than a
    /// defaulted list: "did not say" must not be read as "serves nothing".
    #[serde(default)]
    pub serves: Option<Vec<String>>,
    pub load: f32,
    /// In-flight plus queued, raw. A fraction cannot express a queue.
    #[serde(default)]
    pub busy: u32,
    /// The admission gate's width. Zero = an older build that never said.
    #[serde(default)]
    pub lanes: u32,
    pub devices: Vec<String>,
    pub prefix_blocks: Vec<u64>,
    /// What each card admits as a whole load, bytes; absent from an older build.
    #[serde(default)]
    pub cards: Vec<u64>,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub model_load_s: f64,
    /// What this node measured PER MODEL. A node has no single speed: the flat fields above
    /// describe whatever it happened to run last, which is the wrong number to place a
    /// different model with. Defaulted so a peer that publishes none still parses.
    #[serde(default)]
    pub rates_by_model: HashMap<String, (f64, f64, f64, f64)>,
}

impl PeerReport {
    pub fn split(self) -> (String, NodeState, NodeRates) {
        // A published zero means UNKNOWN, not "infinitely slow". Taken literally it divides
        // to an infinite completion, and when both nodes report zero every candidate ties at
        // infinity and the choice becomes arbitrary - which is what a two-process run showed
        // the first time it was tried. An unknown peer falls back to the pessimistic default
        // instead: it has to prove itself rather than win, or lose, by accident.
        let d = NodeRates::default();
        let or_unknown = |v: f64, fallback: f64| if v > 0.0 { v } else { fallback };
        let rates = NodeRates {
            prefill_tok_per_s: or_unknown(self.prefill_tok_per_s, d.prefill_tok_per_s),
            decode_tok_per_s: or_unknown(self.decode_tok_per_s, d.decode_tok_per_s),
            model_load_s: or_unknown(self.model_load_s, d.model_load_s),
            // Zero stays zero here: an unmeasured aggregate must NOT be defaulted, it is the
            // signal that sends the estimate to its fallback models.
            agg_tok_per_s: 0.0,
        };
        let state = NodeState {
            devices: self.devices,
            load: self.load,
            busy: self.busy,
            lanes: self.lanes,
            models: self.models,
            serves: self.serves,
            link_gbps: Default::default(),
            prefix_blocks: self.prefix_blocks,
            cards: self.cards,
        };
        (self.node_id, state, rates)
    }
}

/// Milliseconds since an arbitrary fixed point, for the detector.
///
/// A monotonic source, not the wall clock: a clock stepped backwards by NTP would make every
/// peer look silent at once and empty the routing table.
pub fn now_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

/// Ask one peer how it is, and time the asking.
pub async fn poll_peer(client: &reqwest::Client, url: &str) -> Result<(PeerReport, f64), String> {
    let t0 = Instant::now();
    let resp = client
        .get(format!("{}/api/cluster/state", url.trim_end_matches('/')))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map_err(|e| format!("{url}: {e}"))?;
    let rtt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !resp.status().is_success() {
        return Err(format!("{url}: {}", resp.status()));
    }
    let report: PeerReport = resp.json().await.map_err(|e| format!("{url}: {e}"))?;
    Ok((report, rtt_ms))
}

/// Announce this node and take in what others announce.
///
/// Runs on a blocking thread rather than the async runtime: the socket read has a timeout and
/// spends its life waiting, which is exactly what a runtime worker must not do.
pub fn spawn_discovery(
    cluster_name: String,
    node_id: String,
    endpoint: String,
    book: Arc<std::sync::Mutex<PeerBook>>,
    interval: Duration,
) {
    std::thread::spawn(move || {
        let sock = match discovery::bind_discovery(std::net::Ipv4Addr::UNSPECIFIED) {
            Ok(s) => s,
            Err(e) => {
                // A network without multicast is a normal deployment, not a failure: seeds
                // still work. Say so once and stop, rather than retrying forever in a log.
                tracing::info!("cluster: no multicast discovery ({e}); seeds only");
                return;
            }
        };
        let _ = sock.set_multicast_loop_v4(true);
        let me = Announcement {
            cluster: cluster_name.clone(),
            node_id: node_id.clone(),
            endpoint,
        };
        let mut last_announce = Instant::now() - interval;
        loop {
            if last_announce.elapsed() >= interval {
                let _ = discovery::announce(&sock, &me);
                last_announce = Instant::now();
            }
            // Returns on its own timeout, so the announcement above keeps its cadence even
            // when nobody is talking.
            if let Some(heard) = discovery::receive(&sock) {
                let mut b = book.lock().unwrap_or_else(|e| e.into_inner());
                if b.learn(&cluster_name, &node_id, &heard) {
                    tracing::info!("cluster: {} joined at {}", heard.node_id, heard.endpoint);
                }
            }
        }
    });
}

/// Poll every known peer on a timer.
pub fn spawn_gossip(
    cluster: Arc<Cluster>,
    book: Arc<std::sync::Mutex<PeerBook>>,
    started: Instant,
) {
    let interval = Duration::from_millis(cluster.config.gossip_interval_ms);
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            tokio::time::sleep(interval).await;
            let urls = book.lock().unwrap_or_else(|e| e.into_inner()).endpoints();
            for url in urls {
                match poll_peer(&client, &url).await {
                    Ok((report, rtt)) => {
                        let per_model: std::collections::HashMap<String, NodeRates> = report
                            .rates_by_model
                            .iter()
                            .map(|(m, &(p, d, l, a))| {
                                let dflt = NodeRates::default();
                                let or_unknown = |v: f64, fb: f64| if v > 0.0 { v } else { fb };
                                (
                                    m.clone(),
                                    NodeRates {
                                        prefill_tok_per_s: or_unknown(p, dflt.prefill_tok_per_s),
                                        decode_tok_per_s: or_unknown(d, dflt.decode_tok_per_s),
                                        model_load_s: or_unknown(l, dflt.model_load_s),
                                        agg_tok_per_s: a,
                                    },
                                )
                            })
                            .collect();
                        let (id, state, rates) = report.split();
                        // Our own endpoint can appear in the seed list; adopting it would make
                        // the node a peer of itself.
                        if id == cluster.config.node_id {
                            continue;
                        }
                        cluster.observe_peer(
                            &id,
                            &url,
                            now_ms(started),
                            state,
                            rates,
                            per_model,
                            rtt,
                        );
                    }
                    // Not an error to report loudly: a peer that is down is exactly what the
                    // detector is for, and it will notice through the silence.
                    Err(e) => tracing::debug!("cluster: {e}"),
                }
            }
        }
    });
}

/// Hand a request to a peer and return its answer unchanged.
///
/// The forwarded marker is what stops a request bouncing: the peer's own router sees it and
/// serves locally whatever its arithmetic says.
pub async fn forward_generate(
    url: &str,
    body: &serde_json::Value,
) -> Result<
    (
        reqwest::StatusCode,
        reqwest::header::HeaderMap,
        reqwest::Response,
    ),
    String,
> {
    forward_request(url, "/api/generate", body).await
}

/// Post a request body to `path` on a peer, marked as forwarded so the peer serves it
/// rather than handing it on again.
pub async fn forward_request(
    url: &str,
    path: &str,
    body: &serde_json::Value,
) -> Result<
    (
        reqwest::StatusCode,
        reqwest::header::HeaderMap,
        reqwest::Response,
    ),
    String,
> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}{path}", url.trim_end_matches('/')))
        .header(FORWARDED_HEADER, "1")
        .json(body)
        .send()
        .await
        .map_err(|e| format!("forward to {url}: {e}"))?;
    Ok((resp.status(), resp.headers().clone(), resp))
}

/// Post a body as it arrived, with its content type: the shape a multipart upload keeps.
pub async fn forward_request_bytes(
    url: &str,
    path: &str,
    content_type: &str,
    body: axum::body::Bytes,
) -> Result<
    (
        reqwest::StatusCode,
        reqwest::header::HeaderMap,
        reqwest::Response,
    ),
    String,
> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}{path}", url.trim_end_matches('/')))
        .header(FORWARDED_HEADER, "1")
        .header(reqwest::header::CONTENT_TYPE, content_type)
        .body(body)
        .send()
        .await
        .map_err(|e| format!("forward to {url}: {e}"))?;
    Ok((resp.status(), resp.headers().clone(), resp))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rate of zero is unknown, not infinitely slow. Read literally it makes every
    /// candidate tie at an infinite completion and the choice becomes arbitrary.
    #[test]
    fn a_published_zero_rate_reads_as_unknown_rather_than_as_zero() {
        let r = PeerReport {
            node_id: "a".into(),
            models: vec![],
            serves: None,
            load: 0.0,
            busy: 0,
            lanes: 0,
            devices: vec![],
            prefix_blocks: vec![],
            cards: Vec::new(),
            prefill_tok_per_s: 0.0,
            decode_tok_per_s: 0.0,
            model_load_s: 0.0,
            rates_by_model: HashMap::new(),
        };
        let (_, _, rates) = r.split();
        assert!(rates.prefill_tok_per_s > 0.0);
        assert!(rates.decode_tok_per_s > 0.0);
        assert!(
            rates.model_load_s > 0.0,
            "a load that costs nothing would be free to move to"
        );
    }

    #[test]
    fn a_peer_report_splits_into_what_the_router_needs() {
        let r = PeerReport {
            node_id: "gpu-b".into(),
            models: vec!["qwen3:8b".into()],
            serves: None,
            load: 0.25,
            busy: 0,
            lanes: 0,
            devices: vec!["cuda:0".into()],
            prefix_blocks: vec![1, 2, 3],
            cards: Vec::new(),
            prefill_tok_per_s: 1800.0,
            decode_tok_per_s: 72.0,
            model_load_s: 18.0,
            rates_by_model: HashMap::new(),
        };
        let (id, state, rates) = r.split();
        assert_eq!(id, "gpu-b");
        assert_eq!(state.models, vec!["qwen3:8b".to_string()]);
        assert_eq!(state.prefix_blocks, vec![1, 2, 3]);
        assert!((rates.decode_tok_per_s - 72.0).abs() < 1e-9);
    }

    /// The detector must not be fed a clock that can go backwards: an NTP step would make
    /// every peer look silent at once and empty the routing table in one round.
    #[test]
    fn the_clock_the_detector_sees_only_moves_forward() {
        let started = Instant::now();
        let a = now_ms(started);
        std::thread::sleep(Duration::from_millis(5));
        let b = now_ms(started);
        assert!(b >= a, "{b} < {a}");
    }

    /// A peer report has to survive the wire as the endpoint will send it - the router reads
    /// these fields and a rename would silently zero a rate rather than fail.
    #[test]
    fn a_peer_report_round_trips_through_json() {
        let r = PeerReport {
            node_id: "a".into(),
            models: vec![],
            serves: None,
            load: 0.5,
            busy: 0,
            lanes: 0,
            devices: vec![],
            prefix_blocks: vec![],
            cards: Vec::new(),
            prefill_tok_per_s: 1.0,
            decode_tok_per_s: 2.0,
            model_load_s: 3.0,
            rates_by_model: HashMap::new(),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: PeerReport = serde_json::from_str(&s).unwrap();
        assert_eq!(back.node_id, "a");
        assert!((back.decode_tok_per_s - 2.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod card_wire_tests {
    use super::PeerReport;

    #[test]
    fn a_report_from_an_older_build_carries_no_cards() {
        let old = r#"{"node_id":"a","models":[],"serves":null,"load":0.0,"busy":0,"lanes":1,
            "devices":[],"prefix_blocks":[],"prefill_tok_per_s":0.0,"decode_tok_per_s":0.0,
            "model_load_s":0.0,"rates_by_model":{}}"#;
        let report: PeerReport = serde_json::from_str(old).unwrap();
        let (_, state, _) = report.split();
        assert!(state.cards.is_empty());
        let new = r#"{"node_id":"a","models":[],"serves":null,"load":0.0,"busy":0,"lanes":1,
            "devices":[],"prefix_blocks":[],"cards":[17000000000],"prefill_tok_per_s":0.0,
            "decode_tok_per_s":0.0,"model_load_s":0.0,"rates_by_model":{}}"#;
        let report: PeerReport = serde_json::from_str(new).unwrap();
        assert_eq!(report.split().1.cards, vec![17_000_000_000]);
    }
}
