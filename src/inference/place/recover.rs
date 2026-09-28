//! Recovery after a component has been placed: a size that shrinks until it fits, and a
//! request that ran out of card memory freeing what can be freed before it runs again.
//!
//! A request walks the same pressure rungs as the load ladder: each re-plan moves the
//! component further off the cards, the last one onto the host. Every bound here comes from
//! those rungs or from the sizes the caller offers.

use super::ladder::is_out_of_memory;

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
}
