//! Priority-aware request gate.
//!
//! Concurrent inference requests serialize on the model lock anyway - only
//! one decode can run at a time per model. Without explicit ordering, the
//! request that grabbed the lock first wins, regardless of latency
//! sensitivity. This gate sits in front of `engine.generate*()` and orders
//! waiters so a high-priority request (FIM editor completions) skips ahead
//! of queued lower-priority requests. It does *not* preempt: a generation
//! already in flight runs to completion, then the highest-priority waiter
//! is woken next.
//!
//! Single-tenant impact: small but real - a chat that's mid-decode still
//! finishes before a freshly-arrived FIM, but the FIM jumps ahead of any
//! other queued chats.

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::sync::Mutex;

/// Request priority. Higher discriminant = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Priority {
    /// Background agent / batch fill - yields to interactive.
    Batch = 0,
    /// Default chat / generate.
    Interactive = 1,
    /// Editor completions - latency-critical, jump the queue.
    Fim = 2,
}

impl Priority {
    /// The priority a request declares in its headers, if it declares one.
    ///
    /// The OpenAI-compatible body has no field for this and adding one would put a
    /// private extension in a shape other servers parse. A header carries a transport
    /// hint without touching the schema, and a client that does not send it is
    /// interactive, which is what a client that has not thought about it should be.
    pub const HEADER: &'static str = "x-loken-priority";

    pub fn from_headers(headers: &axum::http::HeaderMap) -> Option<Self> {
        headers
            .get(Self::HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(Self::parse)
    }

    /// Parse from an Ollama `options.priority` string. Unknown values map
    /// to None so the caller can fall back to a default.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fim" | "high" | "critical" => Some(Self::Fim),
            "interactive" | "normal" => Some(Self::Interactive),
            "batch" | "low" => Some(Self::Batch),
            _ => None,
        }
    }
}

/// The context window a request asks the model to be served with.
///
/// The Ollama surface says this in `options.num_ctx`; the OpenAI-compatible body has
/// nowhere to say it, and adding a field would put a private extension in a shape other
/// servers parse. So it travels as a header, like the priority beside it. A client that
/// does not send one is served with the configured window.
///
/// Without it an OpenAI client cannot reach past the server-wide default, whatever the
/// model can do: an agent talking to a node configured for a small model was reading
/// files a few hundred characters at a time and had no way to say otherwise.
pub struct Window;

impl Window {
    pub const HEADER: &'static str = "x-loken-num-ctx";

    /// The window a request declares in its headers, if it declares one.
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Option<usize> {
        headers
            .get(Self::HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
    }
}

/// FIFO-within-priority waiter on the gate.
struct Waiter {
    priority: Priority,
    seq: u64,
    tx: oneshot::Sender<()>,
    req_id: u64,
}

// BinaryHeap is a max-heap; we want highest priority at the top, then
// lowest seq within the same priority (FIFO).
impl PartialEq for Waiter {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}
impl Eq for Waiter {}
impl PartialOrd for Waiter {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Waiter {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first. Within same priority, lower seq first
        // (FIFO) - invert the seq comparison.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Default)]
struct GateState {
    in_flight: usize,
    waiters: BinaryHeap<Waiter>,
    // Per-request metadata carried by the single-actor scheduler, so
    // /api/inflight can report what's actually running and queued, not
    // just counts. Keyed by `req_id`. Removed on guard drop.
    tracked: HashMap<u64, TrackedRequest>,
}

/// Snapshot of one tracked request - what we expose for observability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackedRequest {
    pub req_id: u64,
    pub priority: Priority,
    pub model: String,
    pub endpoint: &'static str,     // "/api/generate", "/api/chat", etc.
    pub queued_at_ms: i64,          // unix-ms, when the gate was acquired-or-queued
    pub started_at_ms: Option<i64>, // unix-ms, when the permit was actually granted
    pub state: RequestState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestState {
    Queued,
    Running,
}

/// Priority-ordered single-permit gate. Wraps a virtual "permit" - only
/// `max_in_flight` callers can hold a guard simultaneously; the rest wait,
/// and the next wake is the highest-priority waiter.
pub struct RequestGate {
    state: Mutex<GateState>,
    max_in_flight: usize,
    seq_counter: AtomicU64,
    req_id_counter: AtomicU64,
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl RequestGate {
    pub fn new(max_in_flight: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState::default()),
            max_in_flight: max_in_flight.max(1),
            seq_counter: AtomicU64::new(0),
            req_id_counter: AtomicU64::new(0),
        })
    }

    /// Like [`acquire_with_info`] but gives up after `wait`: returns `None` so the
    /// caller can answer 503 "busy" instead of queueing forever behind a wedged
    /// generation. On timeout the waiter is removed under the lock; the race where
    /// the permit was already transferred to us (release popped our waiter and the
    /// send landed just as the deadline fired) is handled by draining the channel
    /// and handing the permit onward - no slot is ever leaked.
    pub async fn acquire_with_info_timeout(
        self: &Arc<Self>,
        priority: Priority,
        model: String,
        endpoint: &'static str,
        wait: std::time::Duration,
    ) -> Option<RequestGuard> {
        let req_id = self.req_id_counter.fetch_add(1, Ordering::Relaxed);
        let queued_at_ms = now_unix_ms();
        let rx_opt = {
            let mut g = self.state.lock().await;
            if g.in_flight < self.max_in_flight {
                g.in_flight += 1;
                g.tracked.insert(
                    req_id,
                    TrackedRequest {
                        req_id,
                        priority,
                        model: model.clone(),
                        endpoint,
                        queued_at_ms,
                        started_at_ms: Some(queued_at_ms),
                        state: RequestState::Running,
                    },
                );
                None
            } else {
                let seq = self.seq_counter.fetch_add(1, Ordering::Relaxed);
                let (tx, rx) = oneshot::channel();
                g.waiters.push(Waiter {
                    priority,
                    seq,
                    tx,
                    req_id,
                });
                g.tracked.insert(
                    req_id,
                    TrackedRequest {
                        req_id,
                        priority,
                        model: model.clone(),
                        endpoint,
                        queued_at_ms,
                        started_at_ms: None,
                        state: RequestState::Queued,
                    },
                );
                tracing::info!(
                    "RequestGate: QUEUED req_id={} priority={:?} seq={} queue_depth={} (timeout {:?})",
                    req_id, priority, seq, g.waiters.len(), wait
                );
                Some(rx)
            }
        };
        if let Some(mut rx) = rx_opt {
            let sleep = tokio::time::sleep(wait);
            tokio::pin!(sleep);
            let timed_out = tokio::select! {
                _ = &mut rx => false,
                _ = &mut sleep => true,
            };
            if timed_out {
                let mut g = self.state.lock().await;
                let before = g.waiters.len();
                // BinaryHeap::retain - drop our waiter if still queued.
                g.waiters.retain(|w| w.req_id != req_id);
                let was_queued = g.waiters.len() < before;
                g.tracked.remove(&req_id);
                if !was_queued {
                    // release() already popped us; if its send landed we own a
                    // permit - pass it to the next waiter / free the slot.
                    if rx.try_recv().is_ok() {
                        self.release(&mut g);
                    }
                }
                tracing::warn!(
                    "RequestGate: TIMEOUT req_id={} after {:?} (endpoint={endpoint})",
                    req_id,
                    wait
                );
                return None;
            }
            let mut g = self.state.lock().await;
            if let Some(entry) = g.tracked.get_mut(&req_id) {
                entry.started_at_ms = Some(now_unix_ms());
                entry.state = RequestState::Running;
            }
        }
        Some(RequestGuard {
            gate: self.clone(),
            req_id,
        })
    }

    fn release(&self, state: &mut GateState) {
        // Loop until the permit actually lands: a waiter whose receiver is gone
        // (client disconnected while queued, or acquire timed out) must NOT
        // swallow the permit - a silently-failing `tx.send` here used to leak
        // the in-flight slot permanently, wedging the whole gate.
        while let Some(next) = state.waiters.pop() {
            if next.tx.send(()).is_ok() {
                tracing::info!(
                    "RequestGate: WAKING (priority={:?}, seq={}, remaining={})",
                    next.priority,
                    next.seq,
                    state.waiters.len()
                );
                // in_flight stays at max_in_flight: we transfer the permit.
                return;
            }
            // Dead waiter: drop its tracked entry and hand the permit onward.
            state.tracked.remove(&next.req_id);
            tracing::info!(
                "RequestGate: skipping dead waiter req_id={} (receiver gone)",
                next.req_id
            );
        }
        state.in_flight = state.in_flight.saturating_sub(1);
    }

    /// Snapshot of the current gate state for observability.
    /// How many requests may run at once. Published so a peer can read this node's load as a
    /// fraction of what it can actually take, rather than as a raw count that means nothing
    /// without the capacity beside it.
    pub fn capacity(&self) -> usize {
        self.max_in_flight
    }

    pub async fn snapshot(&self) -> GateSnapshot {
        let g = self.state.lock().await;
        let mut by_priority = [0usize; 3]; // [Batch, Interactive, Fim]
        for w in g.waiters.iter() {
            let idx = match w.priority {
                Priority::Batch => 0,
                Priority::Interactive => 1,
                Priority::Fim => 2,
            };
            by_priority[idx] += 1;
        }
        // Per-request detail: collect tracked entries, sorted so running ones
        // come first, then queued by (priority desc, queued_at asc).
        let mut requests: Vec<TrackedRequest> = g.tracked.values().cloned().collect();
        requests.sort_by(|a, b| {
            // Running before Queued
            (b.state == RequestState::Running)
                .cmp(&(a.state == RequestState::Running))
                // Higher priority first
                .then(b.priority.cmp(&a.priority))
                // Earlier queued first
                .then(a.queued_at_ms.cmp(&b.queued_at_ms))
        });
        GateSnapshot {
            in_flight: g.in_flight,
            queue_depth: g.waiters.len(),
            queued_batch: by_priority[0],
            queued_interactive: by_priority[1],
            queued_fim: by_priority[2],
            requests,
        }
    }
}

/// Read-only view of the gate's queueing state. Used by /api/inflight.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GateSnapshot {
    pub in_flight: usize,
    pub queue_depth: usize,
    pub queued_batch: usize,
    pub queued_interactive: usize,
    pub queued_fim: usize,
    /// Per-request detail: running and queued. Sorted with running first,
    /// then queued by priority desc, queued_at asc.
    pub requests: Vec<TrackedRequest>,
}

pub struct RequestGuard {
    gate: Arc<RequestGate>,
    req_id: u64,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let gate = self.gate.clone();
        let req_id = self.req_id;
        if let Ok(mut g) = gate.state.try_lock() {
            g.tracked.remove(&req_id);
            gate.release(&mut g);
            return;
        }
        tokio::spawn(async move {
            let mut g = gate.state.lock().await;
            g.tracked.remove(&req_id);
            gate.release(&mut g);
        });
    }
}

#[cfg(test)]
mod tests {
    /// The window a client asks for, and what an absent or unusable header means.
    #[test]
    fn the_window_is_read_from_the_header_it_travels_in() {
        let mut h = axum::http::HeaderMap::new();
        assert_eq!(Window::from_headers(&h), None);
        h.insert(Window::HEADER, "32768".parse().unwrap());
        assert_eq!(Window::from_headers(&h), Some(32768));
        // A window of zero is not a window; the configured one stands.
        h.insert(Window::HEADER, "0".parse().unwrap());
        assert_eq!(Window::from_headers(&h), None);
        h.insert(Window::HEADER, "lots".parse().unwrap());
        assert_eq!(Window::from_headers(&h), None);
    }

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn priority_parse_maps_documented_aliases_case_insensitively() {
        // Pin the Ollama options.priority alias map. Drift would
        // silently route requests into the wrong scheduling tier.
        assert_eq!(Priority::parse("fim"), Some(Priority::Fim));
        assert_eq!(Priority::parse("FIM"), Some(Priority::Fim));
        assert_eq!(Priority::parse("high"), Some(Priority::Fim));
        assert_eq!(Priority::parse("critical"), Some(Priority::Fim));
        assert_eq!(Priority::parse("interactive"), Some(Priority::Interactive));
        assert_eq!(Priority::parse("Normal"), Some(Priority::Interactive));
        assert_eq!(Priority::parse("batch"), Some(Priority::Batch));
        assert_eq!(Priority::parse("low"), Some(Priority::Batch));
        // Unknown -> None so the caller can fall back to its own default
        // (typically Interactive). Silent mapping to a default tier
        // would mask client typos.
        assert_eq!(Priority::parse(""), None);
        assert_eq!(Priority::parse("urgent"), None);
        assert_eq!(Priority::parse("medium"), None);
    }

    #[test]
    fn priority_ord_is_fim_gt_interactive_gt_batch() {
        // BinaryHeap is a max-heap; the heap pops the *greatest*
        // priority first. Pin the natural ordering so a future
        // discriminant renumber doesn't silently demote Fim.
        assert!(Priority::Fim > Priority::Interactive);
        assert!(Priority::Interactive > Priority::Batch);
        assert!(Priority::Fim > Priority::Batch);
    }

    #[test]
    fn waiter_ord_higher_priority_pops_first() {
        // Construct waiters by-hand and pop a BinaryHeap. Higher
        // priority wins regardless of arrival order. Pure-data test
        // - no tokio needed.
        use std::collections::BinaryHeap;
        let (tx_b, _rx_b) = oneshot::channel();
        let (tx_i, _rx_i) = oneshot::channel();
        let (tx_f, _rx_f) = oneshot::channel();
        let mut heap: BinaryHeap<Waiter> = BinaryHeap::new();
        // Insert lowest priority first (seq 0), highest priority last
        // (seq 2). A naive FIFO would pop them in [0,1,2]; a priority
        // heap pops them in [Fim, Interactive, Batch] = [seq 2, 1, 0].
        heap.push(Waiter {
            priority: Priority::Batch,
            seq: 0,
            tx: tx_b,
            req_id: 100,
        });
        heap.push(Waiter {
            priority: Priority::Interactive,
            seq: 1,
            tx: tx_i,
            req_id: 101,
        });
        heap.push(Waiter {
            priority: Priority::Fim,
            seq: 2,
            tx: tx_f,
            req_id: 102,
        });
        assert_eq!(heap.pop().unwrap().req_id, 102, "Fim should pop first");
        assert_eq!(heap.pop().unwrap().req_id, 101, "Interactive second");
        assert_eq!(heap.pop().unwrap().req_id, 100, "Batch last");
    }

    #[tokio::test]
    async fn snapshot_orders_running_first_then_priority_then_queued_at() {
        // /api/inflight presents per-request rows to the Hardware tab.
        // The contract:
        //   1. Running requests come BEFORE queued (active work surfaced
        //      at the top regardless of priority).
        //   2. Within the same state, higher priority first (Fim >
        //      Interactive > Batch).
        //   3. Within the same priority, older queued_at first (FIFO
        //      shows next-to-run-first).
        //
        // Build a gate, hand-populate `tracked` (the snapshot path reads
        // from there directly), and verify the sort.
        let gate = RequestGate::new(1);
        // Drop the test's own state lock pattern - we just need to
        // poke entries in. Acquire the lock, insert, drop.
        {
            let mut g = gate.state.lock().await;
            // Two running (A=Batch, B=Fim), three queued
            // (C=Interactive @ t=200, D=Fim @ t=100, E=Interactive @ t=50).
            // Expected order: B (running Fim), A (running Batch),
            //                 D (queued Fim),
            //                 E (queued Interactive earlier),
            //                 C (queued Interactive later).
            for (req_id, prio, state, queued_at) in [
                (1u64, Priority::Batch, RequestState::Running, 10),
                (2, Priority::Fim, RequestState::Running, 20),
                (3, Priority::Interactive, RequestState::Queued, 200),
                (4, Priority::Fim, RequestState::Queued, 100),
                (5, Priority::Interactive, RequestState::Queued, 50),
            ] {
                g.tracked.insert(
                    req_id,
                    TrackedRequest {
                        req_id,
                        priority: prio,
                        model: "test".to_string(),
                        endpoint: "/api/chat",
                        queued_at_ms: queued_at,
                        started_at_ms: if state == RequestState::Running {
                            Some(queued_at + 1)
                        } else {
                            None
                        },
                        state,
                    },
                );
            }
        }
        let snap = gate.snapshot().await;
        let ids: Vec<u64> = snap.requests.iter().map(|r| r.req_id).collect();
        assert_eq!(ids, vec![2, 1, 4, 5, 3], "got: {ids:?}");
        // The aggregate counters must also be right.
        assert_eq!(snap.in_flight, 0, "no real permits taken in this test");
        // queue_depth comes from the BinaryHeap, not from tracked  -
        // tests above already pin that; here we just verify the
        // bucket counters match the queued tracked entries' priorities.
        // (queued_* fields read from waiters, not tracked - empty here.)
        assert_eq!(snap.queued_batch, 0);
        assert_eq!(snap.queued_interactive, 0);
        assert_eq!(snap.queued_fim, 0);
    }

    #[test]
    fn waiter_ord_fifo_within_same_priority() {
        // Same priority -> lower seq pops first (FIFO). The cmp impl
        // inverts the seq comparison to make the heap behave FIFO
        // within each tier - pin so a future refactor doesn't
        // accidentally LIFO.
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Waiter> = BinaryHeap::new();
        // Insert seq 5, 1, 3 in that arrival order. Expected pop
        // order: 1, 3, 5 (lowest seq = earliest arrival = first out).
        for seq in [5_u64, 1, 3] {
            let (tx, _rx) = oneshot::channel();
            heap.push(Waiter {
                priority: Priority::Interactive,
                seq,
                tx,
                req_id: seq + 1000,
            });
        }
        assert_eq!(
            heap.pop().unwrap().seq,
            1,
            "seq 1 (first arrival) pops first"
        );
        assert_eq!(heap.pop().unwrap().seq, 3);
        assert_eq!(heap.pop().unwrap().seq, 5);
    }

    /// The tests go through the entry point the server uses. There was an untimed
    /// `acquire` beside it, duplicating the whole enqueue-and-wake body for the sake of a
    /// shorter call here - so the queueing the tests pinned was a second copy of the
    /// queueing that runs.
    async fn acquire(gate: &Arc<RequestGate>, priority: Priority) -> RequestGuard {
        gate.acquire_with_info_timeout(
            priority,
            String::new(),
            "test",
            std::time::Duration::from_secs(30),
        )
        .await
        .expect("the gate should admit within 30s in a test")
    }

    #[tokio::test]
    async fn solo_acquire_is_immediate() {
        let gate = RequestGate::new(1);
        let _g = acquire(&gate, Priority::Interactive).await;
        // No other waiters; we should hold the permit.
    }

    #[tokio::test]
    async fn higher_priority_skips_ahead_of_queued_lower() {
        let gate = RequestGate::new(1);
        let g0 = acquire(&gate, Priority::Interactive).await; // holds permit

        let order = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));

        // Two waiters arrive in this order: low-priority first, then FIM.
        // Despite arriving second, FIM should run before low.
        let order1 = order.clone();
        let gate1 = gate.clone();
        let h_low = tokio::spawn(async move {
            let _g = acquire(&gate1, Priority::Batch).await;
            order1.lock().unwrap().push("low");
        });
        // Give low a moment to enqueue.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let order2 = order.clone();
        let gate2 = gate.clone();
        let h_fim = tokio::spawn(async move {
            let _g = acquire(&gate2, Priority::Fim).await;
            order2.lock().unwrap().push("fim");
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        drop(g0); // release permit
        let _ = h_low.await;
        let _ = h_fim.await;

        let final_order = order.lock().unwrap().clone();
        assert_eq!(final_order, vec!["fim", "low"]);
    }

    #[tokio::test]
    async fn fifo_within_same_priority() {
        let gate = RequestGate::new(1);
        let g0 = acquire(&gate, Priority::Interactive).await;

        let counter = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));

        let mut handles = Vec::new();
        for i in 0..5 {
            let gate = gate.clone();
            let counter = counter.clone();
            let order = order.clone();
            handles.push(tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(i as u64 * 5)).await;
                let _g = acquire(&gate, Priority::Interactive).await;
                let n = counter.fetch_add(1, Ordering::SeqCst);
                order.lock().unwrap().push(i);
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                let _ = n;
            }));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(g0);
        for h in handles {
            let _ = h.await;
        }
        assert_eq!(order.lock().unwrap().clone(), vec![0, 1, 2, 3, 4]);
    }
}

#[cfg(test)]
mod header_tests {
    use super::Priority;

    /// A client with background work says so in a header, because the body it sends is a
    /// shape other servers parse and a private field there would not be understood.
    #[test]
    fn the_header_says_what_the_body_cannot() {
        let mut headers = axum::http::HeaderMap::new();
        assert_eq!(Priority::from_headers(&headers), None);

        headers.insert(Priority::HEADER, "batch".parse().unwrap());
        assert_eq!(Priority::from_headers(&headers), Some(Priority::Batch));

        headers.insert(Priority::HEADER, "LOW".parse().unwrap());
        assert_eq!(Priority::from_headers(&headers), Some(Priority::Batch));

        // A value nobody defined is not a priority, and answering None lets the caller
        // fall back rather than take a guess as an instruction.
        headers.insert(Priority::HEADER, "whenever".parse().unwrap());
        assert_eq!(Priority::from_headers(&headers), None);
    }
}
