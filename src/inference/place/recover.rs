//! Recovery after a component has been placed: a size that shrinks until it fits, and a
//! request that ran out of card memory freeing what can be freed before it runs again.
//!
//! A request walks the same pressure rungs as the load ladder: each re-plan moves the
//! component further off the cards, the last one onto the host. Every bound here comes from
//! those rungs or from the sizes the caller offers.

use super::ladder::{is_out_of_memory, PRESSURE_RUNGS};

/// Runs `attempt` at `first`, and each time it runs out of card memory at the next size
/// `smaller` offers, until it offers none. Once the sizes are spent, `make_room` may free
/// memory held elsewhere, once, and the last size runs again when it did. Returns the
/// outcome with the size it ran at; any other failure ends the walk at once.
pub fn shrink_on_oom<T, E: std::fmt::Display>(
    what: &str,
    first: usize,
    mut smaller: impl FnMut(usize) -> Option<usize>,
    mut make_room: impl FnMut(usize) -> bool,
    mut attempt: impl FnMut(usize) -> Result<T, E>,
) -> Result<(T, usize), E> {
    let mut size = first;
    let mut asked_for_room = false;
    loop {
        match attempt(size) {
            Ok(v) => return Ok((v, size)),
            Err(e) if !is_out_of_memory(&e) => return Err(e),
            Err(e) => match smaller(size) {
                Some(next) if next < size => {
                    tracing::warn!("{what}: out of memory at {size} ({e}); retrying at {next}");
                    size = next;
                }
                _ if !asked_for_room && make_room(size) => {
                    asked_for_room = true;
                    tracing::warn!("{what}: out of memory at {size} ({e}); freed room, retrying");
                }
                _ => return Err(e),
            },
        }
    }
}

/// One request's escalation of the placement pressure, at most [`PRESSURE_RUNGS`] rungs.
/// Dropping it takes back the rungs it raised.
#[derive(Debug, Default)]
pub struct Pressure {
    raised: u64,
}

impl Pressure {
    pub fn new() -> Self {
        Self::default()
    }

    /// Raises the pressure one rung for `what`, which ran out of memory with `err`, and
    /// returns the level now in force; `None` once this request has reached the host rung.
    pub fn raise(&mut self, what: &str, err: &dyn std::fmt::Display) -> Option<u64> {
        if self.spent() {
            return None;
        }
        self.raised += 1;
        let level = super::vram_manager::vram_degrade();
        tracing::warn!(
            "{what}: out of memory ({err}); re-planning further off the cards at pressure {level}"
        );
        Some(level)
    }

    /// Whether this request has walked every rung.
    pub fn spent(&self) -> bool {
        self.raised >= PRESSURE_RUNGS
    }
}

impl Drop for Pressure {
    fn drop(&mut self) {
        super::vram_manager::vram_degrade_lower(self.raised);
    }
}

/// Runs a request, and each time it runs out of card memory frees what can be freed and
/// runs it again: the pools first, then idle residents one at a time, then the request's
/// own component re-placed one pressure rung further off the cards, down to the host.
///
/// `oom` names the out-of-memory message of an outcome, `release` hands the pools back,
/// `evict_idle` unloads one idle resident and returns the bytes it freed, and `replan`
/// re-places the component at the given level, returning the outcome to stop with when it
/// cannot. The outcome of the last rung is returned whatever it is.
///
/// A flagged outcome is dropped before the next await: one that carries a `Box<dyn Error>`
/// is not `Send`, and holding it across the await would make the caller's future non-`Send`.
pub async fn retry_on_oom<T, A, AF, O, R, E, EF, P, PF>(
    what: &str,
    mut attempt: A,
    oom: O,
    release: R,
    mut evict_idle: E,
    mut replan: P,
) -> T
where
    A: FnMut() -> AF,
    AF: std::future::Future<Output = T>,
    O: Fn(&T) -> Option<String>,
    R: Fn(),
    E: FnMut() -> EF,
    EF: std::future::Future<Output = u64>,
    P: FnMut(u64, String) -> PF,
    PF: std::future::Future<Output = Option<T>>,
{
    let mut message = {
        let outcome = attempt().await;
        match oom(&outcome) {
            None => return outcome,
            Some(message) => message,
        }
    };
    tracing::warn!("{what}: out of memory ({message}); returning the pools and retrying");
    release();
    message = {
        let outcome = attempt().await;
        match oom(&outcome) {
            None => return outcome,
            Some(message) => message,
        }
    };
    while evict_idle().await > 0 {
        release();
        message = {
            let outcome = attempt().await;
            match oom(&outcome) {
                None => return outcome,
                Some(message) => message,
            }
        };
        tracing::warn!("{what}: still out of memory after unloading an idle resident");
    }
    let mut pressure = Pressure::new();
    while let Some(level) = pressure.raise(what, &message) {
        release();
        if let Some(stop) = replan(level, message.clone()).await {
            return stop;
        }
        message = {
            let outcome = attempt().await;
            match oom(&outcome) {
                Some(message) if !pressure.spent() => message,
                _ => return outcome,
            }
        };
    }
    release();
    attempt().await
}

/// [`retry_on_oom`] for a request whose outcome is a `Result`: its error is the message.
pub fn oom_message<T, E: std::fmt::Display>(outcome: &Result<T, E>) -> Option<String> {
    match outcome {
        Err(e) if is_out_of_memory(e) => Some(e.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn a_size_halves_until_it_fits() {
        let mut tried = Vec::new();
        let got: Result<(usize, usize), String> = shrink_on_oom(
            "t",
            512,
            |s| (s > 32).then(|| (s / 2).max(32)),
            |_| false,
            |s| {
                tried.push(s);
                if s <= 128 {
                    Ok(s)
                } else {
                    Err("[oom] cuda alloc".to_string())
                }
            },
        );
        assert_eq!(got.unwrap(), (128, 128));
        assert_eq!(tried, vec![512, 256, 128]);
    }

    #[test]
    fn room_is_asked_for_once_at_the_floor() {
        let asked = Cell::new(0);
        let mut tries = 0;
        let got: Result<((), usize), String> = shrink_on_oom(
            "t",
            64,
            |s| (s > 32).then_some(32),
            |_| {
                asked.set(asked.get() + 1);
                true
            },
            |_| {
                tries += 1;
                Err("out of memory".to_string())
            },
        );
        assert!(got.is_err());
        assert_eq!(asked.get(), 1);
        assert_eq!(tries, 3);
    }

    #[test]
    fn another_failure_does_not_shrink() {
        let mut tries = 0;
        let got: Result<((), usize), String> = shrink_on_oom(
            "t",
            64,
            |s| Some(s / 2),
            |_| true,
            |_| {
                tries += 1;
                Err("shape mismatch".to_string())
            },
        );
        assert_eq!(got.unwrap_err(), "shape mismatch");
        assert_eq!(tries, 1);
    }

    /// Replays `outcomes` in order, `true` standing for an out-of-memory attempt, with `idle`
    /// residents to evict and re-plans that succeed while `replans_ok` lasts; returns the
    /// attempt that ended the walk, the evictions asked for and the re-plans run.
    fn run(outcomes: &[bool], idle: u64, replans_ok: u64) -> (usize, u64, u64) {
        let attempts = Cell::new(0usize);
        let evictions = Cell::new(0u64);
        let replans = Cell::new(0u64);
        let ended = futures::executor::block_on(retry_on_oom(
            "t",
            || {
                let n = attempts.get();
                attempts.set(n + 1);
                std::future::ready((n, outcomes.get(n).copied().unwrap_or(true)))
            },
            |&(_, is_oom)| is_oom.then(|| "out of memory".to_string()),
            || {},
            || {
                let n = evictions.get();
                evictions.set(n + 1);
                std::future::ready(u64::from(n < idle))
            },
            |_, _| {
                replans.set(replans.get() + 1);
                std::future::ready((replans.get() > replans_ok).then_some((usize::MAX, true)))
            },
        ));
        (ended.0, evictions.get(), replans.get())
    }

    #[test]
    fn a_request_that_fits_runs_once() {
        assert_eq!(run(&[false], 3, 9), (0, 0, 0));
    }

    #[test]
    fn the_pools_come_back_before_anything_is_evicted() {
        assert_eq!(run(&[true, false], 3, 9), (1, 0, 0));
    }

    #[test]
    fn idle_residents_go_before_the_request_is_replanned() {
        assert_eq!(run(&[true, true, false], 3, 9), (2, 1, 0));
        assert_eq!(run(&[true, true, true, false], 3, 9), (3, 2, 0));
    }

    #[test]
    fn the_replan_walks_every_rung_and_stops_at_the_host() {
        let (ended, evictions, replans) = run(&[], 0, 9);
        assert_eq!(evictions, 1);
        assert_eq!(replans, PRESSURE_RUNGS);
        assert_eq!(ended, 2 + PRESSURE_RUNGS as usize - 1);
    }

    #[test]
    fn a_replan_that_fails_ends_the_walk() {
        assert_eq!(run(&[], 0, 0).0, usize::MAX);
    }
}
