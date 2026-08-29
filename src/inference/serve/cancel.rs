//! Cooperative cancellation for long-running generation loops.
//!
//! Every media generation runs inside `spawn_blocking`; when the HTTP client disconnects,
//! axum drops the handler future but the blocking render keeps going to completion - orphaned
//! renders stack up and burn CPU/GPU for minutes ("the server sits at 1000%"). Tokio cannot
//! kill a blocking task, so cancellation is cooperative: the handler holds a [`CancelGuard`]
//! whose Drop fires when the request future is dropped FOR ANY reason - client gone, timeout -
//! and the denoise loops poll [`CancelToken::is_cancelled`] once per step (cheap: one relaxed
//! atomic load per step) and bail out.
//!
//! The guard is DISARMED on normal completion so a finished render does not mark its token.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared cancellation flag polled by generation loops.
#[derive(Clone, Default, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Standard early-return for generation loops: `token.bail()?` once per step.
    pub fn bail(&self) -> Result<(), crate::tensor::Error> {
        if self.is_cancelled() {
            Err(crate::tensor::Error(
                "generation cancelled (client disconnected)".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// RAII armed guard: cancels the token on Drop unless [`CancelGuard::disarm`] ran first.
/// Hold it across the `await` on the blocking render inside the request handler - if the
/// client disconnects, axum drops the future, the guard drops, the render stops at its next
/// step check.
pub struct CancelGuard {
    token: CancelToken,
    armed: bool,
}

impl CancelGuard {
    pub fn new(token: CancelToken) -> Self {
        Self { token, armed: true }
    }

    /// The render finished (or errored) normally - do not cancel on drop.
    pub fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

/// Request-scoped cancellation for loops that cannot take a token parameter.
///
/// The autoregressive TTS decoders are reached through several engine layers whose
/// signatures are shared with parity binaries; threading a token to every step loop
/// would churn all of them. Their generation runs on ONE `spawn_blocking` thread, so
/// the thread that starts the render can publish its token for the duration and the
/// loops consult it with [`scoped_bail`]. Outside a scope there is no token and
/// nothing ever cancels, so a binary or a test behaves exactly as before.
///
/// Weight loading is the second user, for the same reason and with a stronger one on
/// top: every safetensors tensor in the process is read through a single function, so
/// publishing the token for a load makes the whole checkpoint - encoder, transformer,
/// VAE - stop within ONE tensor, without a single model file having to learn that a
/// request exists.
///
/// This is deliberately NOT a general mechanism: work that fans out to a thread pool
/// must take a token explicitly, since the pool threads cannot see this one. The fp8
/// decode is exactly that case and does take one.
pub mod scoped {
    use super::CancelToken;
    use std::cell::RefCell;

    thread_local! {
        static CURRENT: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
    }

    /// Restores the previous token when dropped, INCLUDING on an unwind.
    ///
    /// These scopes are published on pooled blocking threads, which are handed to the next
    /// request when the work ends. A token left behind by a load that failed would make
    /// that next request bail on a cancellation that was never its own.
    pub struct Scope {
        prev: Option<CancelToken>,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            CURRENT.with(|c| *c.borrow_mut() = self.prev.take());
        }
    }

    /// Publish `token` on this thread until the returned guard drops. For work whose body
    /// is a long function that cannot be wrapped in a closure without reindenting all of
    /// it - a model load, in practice.
    #[must_use = "the token is published only while the guard is alive"]
    pub fn publish(token: &CancelToken) -> Scope {
        Scope {
            prev: CURRENT.with(|c| c.borrow_mut().replace(token.clone())),
        }
    }

    /// Publish `token` on this thread for the duration of `f`, restoring whatever was
    /// there before (so nested scopes compose).
    pub fn with<T>(token: &CancelToken, f: impl FnOnce() -> T) -> T {
        let _scope = publish(token);
        f()
    }

    /// `token.bail()?` against the thread's current token; Ok when there is none.
    pub fn bail() -> Result<(), crate::tensor::Error> {
        CURRENT.with(|c| match c.borrow().as_ref() {
            Some(t) => t.bail(),
            None => Ok(()),
        })
    }
}

#[cfg(test)]
mod scoped_tests {
    use super::{scoped, CancelGuard, CancelToken};

    #[test]
    fn outside_a_scope_nothing_ever_cancels() {
        assert!(scoped::bail().is_ok());
    }

    #[test]
    fn a_scoped_token_is_visible_to_the_work_it_wraps() {
        let t = CancelToken::new();
        scoped::with(&t, || {
            assert!(scoped::bail().is_ok());
            t.cancel();
            assert!(scoped::bail().is_err(), "a cancelled scope must bail");
        });
        // And the scope is torn down afterwards, even after a cancel.
        assert!(scoped::bail().is_ok());
    }

    #[test]
    fn scopes_nest_and_restore_the_outer_token() {
        let outer = CancelToken::new();
        let inner = CancelToken::new();
        scoped::with(&outer, || {
            scoped::with(&inner, || {
                inner.cancel();
                assert!(scoped::bail().is_err());
            });
            // Back on the outer token, which was never cancelled.
            assert!(scoped::bail().is_ok());
        });
    }

    /// A load that fails - or panics - must not leave its token published: these scopes
    /// live on pooled blocking threads that the next request inherits.
    #[test]
    fn a_published_scope_is_torn_down_by_an_unwind() {
        let t = CancelToken::new();
        t.cancel();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _s = scoped::publish(&t);
            assert!(scoped::bail().is_err());
            panic!("load failed");
        }));
        assert!(r.is_err());
        assert!(
            scoped::bail().is_ok(),
            "the token must not outlive the work that published it"
        );
    }

    #[test]
    fn dropping_the_guard_is_what_cancels_the_scope() {
        // Mirrors the engine: the guard lives in the async frame, the token is what
        // the blocking work sees.
        let t = CancelToken::new();
        {
            let _g = CancelGuard::new(t.clone());
            assert!(!t.is_cancelled());
        }
        assert!(t.is_cancelled(), "a dropped, armed guard must cancel");
        scoped::with(&t, || assert!(scoped::bail().is_err()));
    }
}

/// Renders in flight, so anyone holding the identifier can stop one.
///
/// The transport dropping is not a stop signal: the web framework does not abandon a handler
/// whose response has not started, so a client that goes away leaves the work running to
/// completion for nobody - a full clip, and tens of gigabytes of host memory while the model
/// sits there. The only dependable signal is one the CLIENT sends, which is what this holds
/// the identifiers for.
///
/// It also gives an interface something it never had: closing a window can now actually
/// cancel, rather than hoping the connection notices.
pub mod registry {
    use super::CancelToken;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn table() -> &'static Mutex<HashMap<String, CancelToken>> {
        static T: OnceLock<Mutex<HashMap<String, CancelToken>>> = OnceLock::new();
        T.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Publish `token` under `id` until [`forget`] runs. Replacing an existing id cancels
    /// what was there: two renders cannot share an identifier, and silently losing the
    /// first one's handle would leave it unstoppable.
    pub fn register(id: &str, token: &CancelToken) {
        let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = t.insert(id.to_string(), token.clone()) {
            old.cancel();
        }
    }

    /// Drop the entry. Always called when a render ends, however it ends.
    pub fn forget(id: &str) {
        table().lock().unwrap_or_else(|e| e.into_inner()).remove(id);
    }

    /// Cancel by id. False when nothing is registered under it - already finished, or never
    /// existed; the caller cannot tell the difference and does not need to.
    pub fn cancel(id: &str) -> bool {
        let t = table().lock().unwrap_or_else(|e| e.into_inner());
        match t.get(id) {
            Some(tok) => {
                tok.cancel();
                true
            }
            None => false,
        }
    }

    /// The identifiers currently in flight.
    pub fn in_flight() -> Vec<String> {
        table()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Ties an entry to a scope: registered on creation, removed on drop, so a render that
    /// ends by error or by panic does not leave a handle behind that would cancel whatever
    /// reuses the identifier next.
    pub struct Entry(String);

    impl Entry {
        pub fn new(id: &str, token: &CancelToken) -> Self {
            register(id, token);
            Self(id.to_string())
        }
    }

    impl Drop for Entry {
        fn drop(&mut self) {
            forget(&self.0);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn an_entry_lives_and_dies_with_its_scope() {
            let tok = CancelToken::new();
            {
                let _e = Entry::new("render-a", &tok);
                assert!(in_flight().contains(&"render-a".to_string()));
                assert!(
                    cancel("render-a"),
                    "a registered render must be cancellable"
                );
                assert!(tok.is_cancelled());
            }
            assert!(!in_flight().contains(&"render-a".to_string()));
            assert!(!cancel("render-a"), "an ended render is not cancellable");
        }

        /// Reusing an identifier must not orphan the render already under it - that one
        /// would then be unstoppable, which is the defect this whole registry exists for.
        #[test]
        fn reusing_an_id_cancels_what_was_there() {
            let (first, second) = (CancelToken::new(), CancelToken::new());
            let _a = Entry::new("render-b", &first);
            register("render-b", &second);
            assert!(
                first.is_cancelled(),
                "the displaced render must be cancelled"
            );
            assert!(!second.is_cancelled());
            forget("render-b");
        }
    }
}
