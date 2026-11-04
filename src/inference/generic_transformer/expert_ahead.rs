//! The bookkeeping of experts fetched one layer ahead: what a token shares with the token
//! before it, the guess of the next layer's missing experts, what sits in a transit buffer, and
//! the decision to queue the guess.
//!
//! Nothing here touches a card. The tier that owns the state does the copies.

use std::time::Instant;

/// Experts in both `a` and `b`, each counted once.
pub(crate) fn shared(a: &[u32], b: &[u32]) -> usize {
    let mut seen: Vec<u32> = Vec::with_capacity(a.len());
    for e in a {
        if b.contains(e) && !seen.contains(e) {
            seen.push(*e);
        }
    }
    seen.len()
}

/// Place of an empty entry in a [`Held`] record.
const EMPTY: u32 = u32::MAX;

/// What sits in one transit buffer: the layer it was filled for and the expert in each place.
#[derive(Default)]
pub(crate) struct Held {
    owner: usize,
    experts: Vec<u32>,
}

impl Held {
    /// Forget the contents: something else wrote the buffer.
    pub(crate) fn clear(&mut self) {
        self.experts.clear();
    }

    /// The buffer holds exactly `puts` (expert, place) for layer `owner`.
    pub(crate) fn set(&mut self, owner: usize, puts: &[(u32, usize)]) {
        self.owner = owner;
        self.experts.clear();
        for &(e, place) in puts {
            if self.experts.len() <= place {
                self.experts.resize(place + 1, EMPTY);
            }
            self.experts[place] = e;
        }
    }

    /// The place of `expert` for layer `owner`, if the buffer holds it for that layer.
    pub(crate) fn place_of(&self, owner: usize, expert: u32) -> Option<usize> {
        if self.owner != owner {
            return None;
        }
        self.experts.iter().position(|&e| e == expert)
    }
}

/// How a request of layer `owner` uses a buffer of `room` places that may already hold its
/// experts.
pub(crate) struct Plan {
    /// Missing experts already in the buffer, with their places.
    pub(crate) hits: Vec<(u32, usize)>,
    /// Places the other missing experts may be copied to, ascending.
    pub(crate) free: Vec<usize>,
}

/// Which of `missing` the buffer holds for layer `owner`, and which places stay free.
pub(crate) fn plan(held: &Held, owner: usize, missing: &[u32], room: usize) -> Plan {
    let hits: Vec<(u32, usize)> = missing
        .iter()
        .filter_map(|&e| Some((e, held.place_of(owner, e)?)))
        .filter(|&(_, place)| place < room)
        .collect();
    let free = (0..room)
        .filter(|p| !hits.iter().any(|&(_, at)| at == *p))
        .collect();
    Plan { hits, free }
}

/// The experts of `rest` given the first free places, in order.
pub(crate) fn assign(free: &[usize], rest: &[u32]) -> Vec<(u32, usize)> {
    rest.iter().copied().zip(free.iter().copied()).collect()
}

/// The share of a guess a layer must find right for the guess to pay.
///
/// The guess of `guess` experts takes `guess * copy_us` on the copy stream and has `lead_us`
/// before the layer asks. What is still copying when the layer asks delays its own copies, and
/// what is right spares them: the layer gains when the right share exceeds the part of the
/// guess's copy time that the lead does not cover. `None` when a time is unknown.
pub(crate) fn break_even(guess: usize, copy_us: f64, lead_us: f64) -> Option<f64> {
    if guess == 0 || copy_us <= 0.0 || lead_us <= 0.0 {
        return None;
    }
    Some((1.0 - lead_us / (guess as f64 * copy_us)).max(0.0))
}

/// Whether to queue a guess: its measured hit rate (`None` before one is measured) is above
/// the break-even.
pub(crate) fn worth_queueing(hit: Option<f64>, guess: usize, copy_us: f64, lead_us: f64) -> bool {
    match (hit, break_even(guess, copy_us, lead_us)) {
        (Some(hit), Some(even)) => hit > even,
        _ => false,
    }
}

/// Fold `sample` into `cell`, which a first sample sets: an eighth of the way each time.
fn blend(cell: &mut Option<f64>, sample: f64) {
    *cell = Some(match *cell {
        None => sample,
        Some(old) => old + (sample - old) * 0.125,
    });
}

/// What a request told, for the counters.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Seen {
    /// Routed experts of the request, and those the request before also routed to.
    pub(crate) routed: usize,
    pub(crate) shared: usize,
    /// Experts of the guess, and those of them that were missing.
    pub(crate) guessed: usize,
    pub(crate) right: usize,
}

/// One layer's history of one-row requests, and the pace of its guesses.
#[derive(Default)]
pub(crate) struct Ahead {
    last_ids: Vec<u32>,
    last_missing: Vec<u32>,
    /// The share of the guess that was missing, smoothed.
    hit: Option<f64>,
    /// Microseconds from the end of the layer before's copies to this layer's request, smoothed.
    lead_us: Option<f64>,
    lead_from: Option<Instant>,
}

impl Ahead {
    /// The guess for this layer's next request: the experts it found missing last time that
    /// `on_card` does not hold now, at most `room` of them.
    pub(crate) fn guess(&self, on_card: impl Fn(u32) -> bool, room: usize) -> Vec<u32> {
        self.last_missing
            .iter()
            .copied()
            .filter(|&e| !on_card(e))
            .take(room)
            .collect()
    }

    /// A request of one row routed to `ids` and found `missing` of them missing, `guess` having
    /// been the guess for it.
    pub(crate) fn note_request(&mut self, ids: &[u32], missing: &[u32], guess: &[u32]) -> Seen {
        let seen = Seen {
            routed: ids.len(),
            shared: shared(&self.last_ids, ids),
            guessed: guess.len(),
            right: guess.iter().filter(|e| missing.contains(e)).count(),
        };
        if seen.guessed > 0 {
            blend(&mut self.hit, seen.right as f64 / seen.guessed as f64);
        }
        self.last_ids = ids.to_vec();
        self.last_missing = missing.to_vec();
        seen
    }

    /// A request of several rows: the history of single tokens no longer follows.
    pub(crate) fn forget(&mut self) {
        self.last_ids.clear();
        self.last_missing.clear();
        self.lead_from = None;
    }

    /// The copy stream is free of the layer before's copies at `at`: the lead starts.
    pub(crate) fn lead_starts(&mut self, at: Instant) {
        self.lead_from = Some(at);
    }

    /// The layer's request arrived at `at`.
    pub(crate) fn lead_ends(&mut self, at: Instant) {
        if let Some(from) = self.lead_from.take() {
            blend(
                &mut self.lead_us,
                at.saturating_duration_since(from).as_secs_f64() * 1e6,
            );
        }
    }

    /// The measured hit rate, and the lead in microseconds.
    pub(crate) fn pace(&self) -> (Option<f64>, Option<f64>) {
        (self.hit, self.lead_us)
    }

    /// Whether to queue `guess` now, `copy_us` being the time to copy one expert.
    pub(crate) fn worth(&self, guess: usize, copy_us: f64) -> bool {
        worth_queueing(self.hit, guess, copy_us, self.lead_us.unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_counts_each_expert_once() {
        assert_eq!(shared(&[1, 2, 3, 4], &[3, 4, 5, 6]), 2);
        assert_eq!(shared(&[1, 1, 2], &[1, 2, 2]), 2);
        assert_eq!(shared(&[], &[1]), 0);
    }

    #[test]
    fn the_break_even_is_what_the_lead_leaves_uncovered() {
        // 4 experts of 100 us each: 400 us of copies.
        assert_eq!(break_even(4, 100.0, 200.0), Some(0.5));
        assert_eq!(break_even(4, 100.0, 400.0), Some(0.0));
        assert_eq!(break_even(4, 100.0, 800.0), Some(0.0));
        assert_eq!(break_even(4, 100.0, 100.0), Some(0.75));
        assert_eq!(break_even(0, 100.0, 100.0), None);
        assert_eq!(break_even(4, 0.0, 100.0), None);
        assert_eq!(break_even(4, 100.0, 0.0), None);
    }

    #[test]
    fn a_guess_is_queued_only_above_the_break_even() {
        assert!(worth_queueing(Some(0.6), 4, 100.0, 200.0));
        assert!(!worth_queueing(Some(0.5), 4, 100.0, 200.0));
        assert!(!worth_queueing(Some(0.4), 4, 100.0, 200.0));
        // A lead that covers the copies pays at any hit above 0, never at 0.
        assert!(worth_queueing(Some(0.01), 4, 100.0, 400.0));
        assert!(!worth_queueing(Some(0.0), 4, 100.0, 400.0));
        // Unmeasured: never.
        assert!(!worth_queueing(None, 4, 100.0, 400.0));
        assert!(!worth_queueing(Some(1.0), 4, 100.0, 0.0));
        assert!(!worth_queueing(Some(1.0), 4, 0.0, 100.0));
    }

    #[test]
    fn a_buffer_names_what_it_holds_for_one_layer() {
        let mut held = Held::default();
        held.set(7, &[(10, 0), (11, 1), (12, 3)]);
        assert_eq!(held.place_of(7, 11), Some(1));
        assert_eq!(held.place_of(7, 12), Some(3));
        assert_eq!(held.place_of(7, 13), None);
        assert_eq!(held.place_of(8, 11), None);
        held.clear();
        assert_eq!(held.place_of(7, 11), None);
    }

    #[test]
    fn late_pairs_take_the_places_of_experts_already_there() {
        let mut held = Held::default();
        held.set(3, &[(10, 0), (11, 1), (12, 2)]);
        // 11 and 12 are missing and held, 20 and 21 are missing and not.
        let p = plan(&held, 3, &[20, 11, 21, 12], 4);
        assert_eq!(p.hits, vec![(11, 1), (12, 2)]);
        assert_eq!(p.free, vec![0, 3]);
        // The room the transit share may use shrinks by the hits.
        assert_eq!(p.free.len(), 4 - p.hits.len());
        assert_eq!(assign(&p.free, &[20, 21]), vec![(20, 0), (21, 3)]);
        // Fewer places than experts: the rest go unplaced, the host's.
        assert_eq!(assign(&p.free[..1], &[20, 21]), vec![(20, 0)]);
        // Another layer finds nothing.
        let q = plan(&held, 4, &[11], 4);
        assert!(q.hits.is_empty());
        assert_eq!(q.free, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_hit_outside_the_room_is_not_used() {
        let mut held = Held::default();
        held.set(1, &[(5, 6)]);
        let p = plan(&held, 1, &[5], 4);
        assert!(p.hits.is_empty());
    }

    #[test]
    fn requests_tell_what_a_token_shares_and_what_the_guess_got_right() {
        let mut ahead = Ahead::default();
        let none = |_: u32| false;
        let g = ahead.guess(none, 8);
        assert!(g.is_empty());
        let s = ahead.note_request(&[1, 2, 3, 4], &[3, 4], &g);
        assert_eq!(
            s,
            Seen {
                routed: 4,
                shared: 0,
                guessed: 0,
                right: 0
            }
        );
        // The next token: 2 of 4 shared, the guess is last time's missing set.
        let g = ahead.guess(none, 8);
        assert_eq!(g, vec![3, 4]);
        let s = ahead.note_request(&[3, 4, 5, 6], &[4, 5, 6], &g);
        assert_eq!(
            s,
            Seen {
                routed: 4,
                shared: 2,
                guessed: 2,
                right: 1
            }
        );
        assert_eq!(ahead.pace().0, Some(0.5));
        // An expert the card holds now is not guessed, and the room caps the guess.
        assert_eq!(ahead.guess(|e| e == 5, 8), vec![4, 6]);
        assert_eq!(ahead.guess(none, 2), vec![4, 5]);
        ahead.forget();
        assert!(ahead.guess(none, 8).is_empty());
    }

    #[test]
    fn the_hit_rate_follows_the_requests() {
        let mut ahead = Ahead::default();
        for _ in 0..40 {
            ahead.note_request(&[1, 2], &[1, 2], &[1, 2]);
        }
        assert!(ahead.pace().0.unwrap() > 0.99);
        for _ in 0..80 {
            ahead.note_request(&[1, 2], &[1, 2], &[7, 8]);
        }
        assert!(ahead.pace().0.unwrap() < 0.01);
    }

    #[test]
    fn the_lead_is_measured_between_the_free_stream_and_the_request() {
        let mut ahead = Ahead::default();
        assert!(!ahead.worth(2, 100.0));
        let t = Instant::now();
        ahead.lead_starts(t);
        ahead.lead_ends(t + std::time::Duration::from_micros(300));
        let (_, lead) = ahead.pace();
        assert!((lead.unwrap() - 300.0).abs() < 1.0);
        // A lead is read once.
        ahead.lead_ends(t + std::time::Duration::from_micros(900));
        assert!((ahead.pace().1.unwrap() - 300.0).abs() < 1.0);
        // Hit rate unmeasured: still no.
        assert!(!ahead.worth(2, 100.0));
        for _ in 0..20 {
            ahead.note_request(&[1], &[1], &[1]);
        }
        assert!(ahead.worth(2, 100.0));
    }
}
