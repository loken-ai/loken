//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

mod imma;
pub use imma::*;
mod attn;
pub use attn::*;
mod gate;
pub use gate::*;
mod shortconv;
pub use shortconv::*;
mod rmsnorm;
pub use rmsnorm::*;
mod hyper;
pub use hyper::*;
mod indexer;
pub use indexer::*;
mod gdn;
pub use gdn::*;
mod qk_prep;
pub use qk_prep::*;

thread_local! {
    static IN_PLACE_STATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While the returned guard lives, the recurrent-state launches on this thread advance the
/// state they are given where it lies, and hand it back as the new state, instead of
/// computing the next state into a buffer of its own. A decode step that does so leaves every
/// state at the address it had, which is what lets a recorded step run again and again.
pub fn in_place_state(on: bool) -> InPlaceState {
    InPlaceState(IN_PLACE_STATE.with(|f| f.replace(on)))
}

/// Restores the previous setting on drop; see [`in_place_state`].
pub struct InPlaceState(bool);

impl Drop for InPlaceState {
    fn drop(&mut self) {
        IN_PLACE_STATE.with(|f| f.set(self.0));
    }
}

/// Whether the state launches on this thread advance their state in place.
pub(crate) fn in_place_state_on() -> bool {
    IN_PLACE_STATE.with(|f| f.get())
}
