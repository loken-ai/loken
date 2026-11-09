//! The experts a host tier misses on a recorded routing: the trace a served model's host tier
//! writes to the stage report (`host_tier_trace`), replayed through each ranking of the kept
//! experts at each size.
//!
//! No model, no disk and no card is touched.
//! Every expert a forward computes on the host is admitted, then the ranking lets the lowest go
//! until the layer holds its bound, never one of a decoded row's own.
//! A decoded row's experts the tier did not hold are its misses, read from the disk.
//! The prompts' misses are counted apart: they are read whatever the tier holds.

use super::{Context, Measured};
use crate::inference::engine::decode_step::VERIFY_ROWS;
use crate::inference::expert_counters::{Found, TierEvent};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// How the kept experts are ranked: the lowest gives way first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rank {
    /// The last row that routed to it, a prompt's rows ticking in order.
    Recency,
    /// The rows routed to it since the trace began, then recency.
    Frequency,
    /// The decoded rows routed to it since the trace began, then recency.
    DecodedFrequency,
    /// An expert a prompt used last gives way before one a decoded row used last, then recency.
    DecodedFirst,
    /// The use furthest ahead, which needs the future: the floor no ranking passes.
    Farthest,
}

impl Rank {
    const ALL: [Rank; 5] = [
        Rank::Recency,
        Rank::Frequency,
        Rank::DecodedFrequency,
        Rank::DecodedFirst,
        Rank::Farthest,
    ];

    fn name(self) -> &'static str {
        match self {
            Rank::Recency => "recency",
            Rank::Frequency => "frequency",
            Rank::DecodedFrequency => "decoded frequency",
            Rank::DecodedFirst => "decoded first",
            Rank::Farthest => "farthest next use",
        }
    }
}

/// 1 forward of 1 layer, as the trace recorded it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Step {
    pub layer: usize,
    pub rows: usize,
    /// Per expert the host computed: its id, the last row routed to it, the rows routed to it.
    pub host: Vec<(usize, usize, usize)>,
    /// What the tier held of each, and its bytes not in memory, as recorded.
    pub found: Vec<(Found, u64)>,
    /// The request the forward belongs to, from 0.
    pub request: usize,
}

impl Step {
    pub(crate) fn decoded(&self) -> bool {
        self.rows <= VERIFY_ROWS
    }
}

/// A recorded trace: per layer what it kept when the trace began and its bound, and the
/// forwards in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Recorded {
    pub start: HashMap<usize, (Vec<usize>, usize)>,
    pub steps: Vec<Step>,
}

impl Recorded {
    /// The trace of `events`.
    ///
    /// A request starts at the first wide layer's prompt forward that follows a decoded one.
    pub(crate) fn of(events: Vec<TierEvent>) -> Self {
        let mut r = Recorded::default();
        let mut request = 0usize;
        let mut decoded_since = false;
        for ev in events {
            match ev {
                TierEvent::Kept {
                    layer,
                    experts,
                    capacity,
                } => {
                    r.start.entry(layer).or_insert((experts, capacity));
                }
                TierEvent::Forward {
                    layer, rows, host, ..
                } => {
                    let decoded = rows <= VERIFY_ROWS;
                    if layer == 0 && !decoded && decoded_since {
                        request += 1;
                        decoded_since = false;
                    }
                    decoded_since |= decoded;
                    r.steps.push(Step {
                        layer,
                        rows,
                        host: host.iter().map(|h| (h.expert, h.last, h.rows)).collect(),
                        found: host.iter().map(|h| (h.found, h.absent)).collect(),
                        request,
                    });
                }
            }
        }
        r
    }

    /// The requests the trace holds.
    pub(crate) fn requests(&self) -> usize {
        self.steps.last().map_or(0, |s| s.request + 1)
    }

    /// The bound the tier had: the largest any layer started with.
    pub(crate) fn capacity(&self) -> usize {
        self.start.values().map(|s| s.1).max().unwrap_or(0)
    }

    /// Decoded tokens per request: the decoded rows of the first wide layer.
    pub(crate) fn tokens(&self) -> Vec<u64> {
        let mut t = vec![0u64; self.requests()];
        for s in self.steps.iter().filter(|s| s.layer == 0 && s.decoded()) {
            t[s.request] += s.rows as u64;
        }
        t
    }
}

/// What a replay counted, per request.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Missed {
    /// A decoded row's experts the tier did not hold.
    pub decoded: Vec<u64>,
    /// A prompt's experts the tier did not hold.
    pub prompt: Vec<u64>,
}

/// 1 layer's tier in a replay.
struct Layer {
    kept: HashSet<usize>,
    clock: u64,
    last: HashMap<usize, u64>,
    rows: HashMap<usize, u64>,
    decoded: HashMap<usize, u64>,
    by_prompt: HashSet<usize>,
    /// Per expert, the index of the next step using it, for the farthest-next-use ranking.
    next: HashMap<usize, u64>,
}

impl Layer {
    /// A layer keeping `start`, the least recent first, whose first use of an expert is
    /// `first(e)`.
    fn new(start: &[usize], first: impl Fn(usize) -> u64) -> Self {
        let mut l = Layer {
            kept: HashSet::new(),
            clock: 0,
            last: HashMap::new(),
            rows: HashMap::new(),
            decoded: HashMap::new(),
            by_prompt: HashSet::new(),
            next: HashMap::new(),
        };
        for &e in start {
            l.clock += 1;
            l.kept.insert(e);
            l.last.insert(e, l.clock);
            l.next.insert(e, first(e));
        }
        l
    }
}

/// Replays `rec` with each layer bound to `capacity` experts, ranked by `rank`.
pub(crate) fn replay(rec: &Recorded, capacity: usize, rank: Rank) -> Missed {
    let requests = rec.requests();
    let mut missed = Missed {
        decoded: vec![0; requests],
        prompt: vec![0; requests],
    };
    let (next, first) = match rank {
        Rank::Farthest => next_uses(rec),
        _ => Default::default(),
    };
    let mut layers: HashMap<usize, Layer> = HashMap::new();
    for (i, s) in rec.steps.iter().enumerate() {
        let l = layers.entry(s.layer).or_insert_with(|| {
            let start = rec.start.get(&s.layer).map_or(&[][..], |v| &v.0[..]);
            let skip = start.len().saturating_sub(capacity);
            let at = |e: usize| first.get(&(s.layer, e)).copied().unwrap_or(u64::MAX);
            Layer::new(&start[skip..], at)
        });
        if let Some(uses) = next.get(i) {
            for h in &s.host {
                l.next
                    .insert(h.0, uses.get(&h.0).copied().unwrap_or(u64::MAX));
            }
        }
        let decoded = s.decoded();
        let misses = s.host.iter().filter(|h| !l.kept.contains(&h.0)).count() as u64;
        match decoded {
            true => missed.decoded[s.request] += misses,
            false => missed.prompt[s.request] += misses,
        }
        let base = l.clock;
        for &(e, last, rows) in &s.host {
            l.kept.insert(e);
            *l.rows.entry(e).or_insert(0) += rows as u64;
            match decoded {
                true => {
                    l.last.insert(e, base + 1);
                    *l.decoded.entry(e).or_insert(0) += rows as u64;
                    l.by_prompt.remove(&e);
                }
                false => {
                    l.last.insert(e, base + last as u64 + 1);
                    l.by_prompt.insert(e);
                }
            }
        }
        l.clock += match decoded {
            true => 1,
            false => s.rows as u64,
        };
        if l.kept.len() <= capacity {
            continue;
        }
        let spare: HashSet<usize> = match decoded {
            true => s.host.iter().map(|h| h.0).collect(),
            false => HashSet::new(),
        };
        let mut order: Vec<((u64, u64), usize)> = l
            .kept
            .iter()
            .copied()
            .filter(|e| !spare.contains(e))
            .map(|e| {
                let last = l.last.get(&e).copied().unwrap_or(0);
                let key = match rank {
                    Rank::Recency => (0, last),
                    Rank::Frequency => (l.rows.get(&e).copied().unwrap_or(0), last),
                    Rank::DecodedFrequency => (l.decoded.get(&e).copied().unwrap_or(0), last),
                    Rank::DecodedFirst => (u64::from(!l.by_prompt.contains(&e)), last),
                    Rank::Farthest => {
                        (u64::MAX - l.next.get(&e).copied().unwrap_or(u64::MAX), last)
                    }
                };
                (key, e)
            })
            .collect();
        order.sort_unstable();
        let over = l.kept.len() - capacity;
        for (_, e) in order.into_iter().take(over) {
            l.kept.remove(&e);
        }
    }
    missed
}

/// Per step, per host expert it used, the index of the next step of the same layer using it,
/// and per layer and expert the index of the first step using it.
type NextUses = (Vec<HashMap<usize, u64>>, HashMap<(usize, usize), u64>);

fn next_uses(rec: &Recorded) -> NextUses {
    let mut out: Vec<HashMap<usize, u64>> = vec![HashMap::new(); rec.steps.len()];
    let mut seen: HashMap<(usize, usize), u64> = HashMap::new();
    for (i, s) in rec.steps.iter().enumerate().rev() {
        for h in &s.host {
            if let Some(&n) = seen.get(&(s.layer, h.0)) {
                out[i].insert(h.0, n);
            }
            seen.insert((s.layer, h.0), i as u64);
        }
    }
    (out, seen)
}

/// What the trace recorded per request: a decoded row's experts the tier did not hold, and the
/// bytes of a decoded row's experts not in memory when asked for.
pub(crate) fn recorded(rec: &Recorded) -> (Vec<u64>, Vec<u64>) {
    let n = rec.requests();
    let (mut missed, mut absent) = (vec![0u64; n], vec![0u64; n]);
    for s in rec.steps.iter().filter(|s| s.decoded()) {
        for &(found, bytes) in &s.found {
            missed[s.request] += u64::from(found != Found::Kept);
            absent[s.request] += bytes;
        }
    }
    (missed, absent)
}

pub fn run(cx: &Context, path: &Path) -> Result<Vec<Measured>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let events: Vec<TierEvent> =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let rec = Recorded::of(events);
    let tokens = rec.tokens();
    if tokens.iter().sum::<u64>() == 0 {
        return Err(format!("{}: no decoded row in the trace", path.display()));
    }
    let expert_bytes = cx.shapes.expert_bytes() as f64;
    let disk = cx.options.disk_gbs * 1e9;
    let per_token = |counts: &[u64]| -> Vec<f64> {
        counts
            .iter()
            .zip(&tokens)
            .map(|(&c, &t)| c as f64 / t.max(1) as f64)
            .collect()
    };
    let show = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut rows = Vec::new();
    let (missed, absent) = recorded(&rec);
    let absent_mb: Vec<f64> = per_token(&absent).iter().map(|b| b / 1e6).collect();
    rows.push(Measured {
        variant: format!("recorded, {} slots", rec.capacity()),
        bytes: None,
        samples_us: per_token(&absent).iter().map(|b| b / disk * 1e6).collect(),
        note: Some(format!(
            "per request: {} decoded tokens, misses a token {}, MB a token not in memory {}",
            tokens
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            show(&per_token(&missed)),
            show(&absent_mb)
        )),
    });
    let mut sizes = vec![rec.capacity()];
    sizes.extend(cx.options.slots.iter().copied());
    sizes.dedup();
    for &slots in &sizes {
        for rank in Rank::ALL {
            let m = replay(&rec, slots, rank);
            let misses = per_token(&m.decoded);
            let ms: Vec<f64> = misses
                .iter()
                .map(|x| x * expert_bytes / disk * 1e3)
                .collect();
            rows.push(Measured {
                variant: format!("{slots} slots, {}", rank.name()),
                bytes: Some((misses.iter().sum::<f64>() / misses.len() as f64 * expert_bytes) as u64),
                samples_us: ms.iter().map(|x| x * 1e3).collect(),
                note: Some(format!(
                    "per request: decoded misses a token {}, ms a token at {} GB/s {}, prompt misses {}",
                    show(&misses),
                    cx.options.disk_gbs,
                    show(&ms),
                    m.prompt
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::expert_counters::HostExpert;

    fn forward(layer: usize, rows: usize, host: &[(usize, usize)]) -> TierEvent {
        TierEvent::Forward {
            layer,
            rows,
            card: Vec::new(),
            host: host
                .iter()
                .map(|&(expert, last)| HostExpert {
                    expert,
                    last,
                    rows: 1,
                    found: Found::Kept,
                    absent: 0,
                    ahead: false,
                })
                .collect(),
        }
    }

    /// A prompt that follows a decoded row starts a request, and the first layer's decoded rows
    /// count the tokens.
    #[test]
    fn requests_start_at_a_prompt_after_a_decode() {
        let p = VERIFY_ROWS + 1;
        let rec = Recorded::of(vec![
            forward(0, p, &[(1, 0)]),
            forward(1, p, &[(1, 0)]),
            forward(0, 1, &[(1, 0)]),
            forward(1, 1, &[(1, 0)]),
            forward(0, 1, &[(2, 0)]),
            forward(0, p, &[(3, 0)]),
            forward(0, 1, &[(3, 0)]),
        ]);
        assert_eq!(rec.requests(), 2);
        assert_eq!(rec.tokens(), vec![2, 1]);
    }

    /// The replay of recency reproduces the tier: a decoded row's experts stay, the least
    /// recent goes, a prompt keeps its newest.
    #[test]
    fn recency_lets_the_least_recent_go() {
        let p = VERIFY_ROWS + 1;
        let mut events = vec![TierEvent::Kept {
            layer: 0,
            experts: vec![7, 8],
            capacity: 2,
        }];
        events.push(forward(0, 1, &[(8, 0)]));
        events.push(forward(0, 1, &[(9, 0)]));
        events.push(forward(0, 1, &[(7, 0)]));
        events.push(forward(0, p, &[(1, 0), (2, 1)]));
        events.push(forward(0, 1, &[(2, 0)]));
        let rec = Recorded::of(events);
        let m = replay(&rec, 2, Rank::Recency);
        // 8 hits, 9 misses and lets 7 go, 7 misses, the prompt misses 1 and 2, 2 hits.
        assert_eq!(m.decoded, vec![2, 0]);
        assert_eq!(m.prompt, vec![0, 2]);
    }

    /// A prompt's experts give way before a decoded row's under the decoded-first ranking, and
    /// the farthest next use misses no more than any ranking.
    #[test]
    fn decoded_first_keeps_the_decode_through_a_prompt() {
        let p = VERIFY_ROWS + 1;
        let events = vec![
            forward(0, 1, &[(1, 0)]),
            forward(0, 1, &[(2, 0)]),
            forward(0, p, &[(5, 0), (6, 1)]),
            forward(0, 1, &[(1, 0)]),
            forward(0, 1, &[(2, 0)]),
        ];
        let rec = Recorded::of(events);
        let recency = replay(&rec, 2, Rank::Recency);
        let first = replay(&rec, 2, Rank::DecodedFirst);
        assert_eq!(recency.decoded, vec![2, 2]);
        assert_eq!(first.decoded, vec![2, 0]);
        for rank in Rank::ALL {
            let m = replay(&rec, 2, rank);
            let far = replay(&rec, 2, Rank::Farthest);
            assert!(far.decoded.iter().sum::<u64>() <= m.decoded.iter().sum::<u64>());
        }
    }
}
