//! What a card may hold for the streamed placement: the bytes taken, the ceiling, and what is
//! set aside for the weights every token reads. One accounting per device, shared by every
//! offload that points at the card, so two of them cannot fill it together and past it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub struct Room {
    /// Bytes kept on the card by the placement.
    taken: usize,
    /// The most the placement may keep: the card's memory less its working room.
    ceiling: usize,
    /// What the card's own transient work needs beside the weights it holds: the largest step
    /// the model declared.
    working: usize,
    /// Still set aside for always-read weights not yet kept.
    reserved: usize,
}

impl Room {
    pub fn taken(&self) -> usize {
        self.taken
    }

    pub fn ceiling(&self) -> usize {
        self.ceiling
    }

    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Set aside up to `bytes` for the weights every token reads, within what is left of the
    /// ceiling; answers what could not be set aside here, for the next card.
    pub fn reserve(&mut self, bytes: usize) -> usize {
        let here = bytes.min(self.ceiling.saturating_sub(self.taken + self.reserved));
        self.reserved += here;
        bytes - here
    }

    /// Take `bytes`, or refuse. A weight every token reads is never refused by this accounting:
    /// the reserve for its kind is an estimate, and the working room absorbs what the estimate
    /// missed; only an allocation that fails refuses it. A routed expert is held to what is
    /// left once the reserve is out, so the experts a prompt admits never leave the weights
    /// every token reads without a place.
    pub fn take(&mut self, bytes: usize, always_read: bool) -> bool {
        if always_read {
            self.taken += bytes;
            self.reserved = self.reserved.saturating_sub(bytes);
            return true;
        }
        if self.taken + bytes > self.ceiling.saturating_sub(self.reserved) {
            return false;
        }
        self.taken += bytes;
        true
    }

    pub fn give(&mut self, bytes: usize) {
        self.taken = self.taken.saturating_sub(bytes);
    }

    /// Whether `free` bytes on the card right now leave `bytes` beyond the working room and
    /// the reserve: the accounting knows what is kept, not what a step in flight holds.
    pub fn leaves(&self, free: usize, bytes: usize) -> bool {
        free > bytes + self.working + self.reserved
    }

    pub fn summary(&self) -> String {
        format!(
            "{} MB taken of a {} MB ceiling, {} MB still reserved, {} MB working room",
            self.taken / 1_000_000,
            self.ceiling / 1_000_000,
            self.reserved / 1_000_000,
            self.working / 1_000_000
        )
    }
}

fn rooms() -> &'static Mutex<HashMap<usize, &'static Mutex<Room>>> {
    static ROOMS: OnceLock<Mutex<HashMap<usize, &'static Mutex<Room>>>> = OnceLock::new();
    ROOMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The room of device `ordinal`, made on its first opening from the card's `total` bytes and
/// the `working` room the placement's steps need; an opening after the first answers the same
/// room, whatever figures it brings.
pub fn open(ordinal: usize, total: usize, working: usize) -> &'static Mutex<Room> {
    let slot = *rooms().lock().unwrap().entry(ordinal).or_insert_with(|| {
        Box::leak(Box::new(Mutex::new(Room {
            taken: 0,
            ceiling: 0,
            working: 0,
            reserved: 0,
        })))
    });
    // A fresh placement supersedes any earlier one on this device: its cards were dropped and
    // their memory freed, so the accounting starts over. Kept, it would still count weights that
    // no longer sit there and push a re-opened placement's always-read path onto the host. Called
    // once per device before any weight is taken, so the reset never lands mid-placement.
    *slot.lock().unwrap() = Room {
        taken: 0,
        ceiling: total.saturating_sub(working),
        working,
        reserved: 0,
    };
    slot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_read_weights_are_never_refused_and_experts_are_held_to_the_rest() {
        let mut r = Room {
            taken: 0,
            ceiling: 100,
            working: 20,
            reserved: 0,
        };
        assert_eq!(r.reserve(60), 0);
        assert_eq!(
            r.reserve(80),
            40,
            "only what is left of the ceiling is set aside"
        );
        assert!(
            !r.take(10, false),
            "the reserve is out of an expert's reach"
        );
        assert!(
            r.take(70, true),
            "an always-read weight past its reserve is still taken"
        );
        assert_eq!(r.reserved(), 30);
        assert!(r.leaves(200, 100));
        assert!(
            !r.leaves(150, 100),
            "the working room and the reserve come first"
        );
    }
}
