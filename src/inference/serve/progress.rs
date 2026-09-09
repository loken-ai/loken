//! What a long request is DOING, not just how long it has been doing it.
//!
//! A render is several phases and only one of them counts steps. Reporting the denoise
//! alone left every other phase as a spinner: measured on this machine, a video spent
//! 154 s loading its text encoder before the first step existed to report, and the client
//! had nothing to show but a rising number of seconds. Someone watching that cannot tell a
//! slow load from a wedged one, and neither can whoever they ask about it.
//!
//! An engine calls ONE function, so there is one thing to thread through a pipeline:
//!
//! ```ignore
//! report(phase::LOAD_ENCODER, 0, 0);        // beginning, nothing to count
//! report(phase::DENOISE, step + 1, steps);  // counted
//! ```
//!
//! `total == 0` means the phase has no step count and is simply starting: a client shows
//! the name, and with a total it can also show a bar. Phases travel as plain strings, so a
//! phase added here needs no client change to be DISPLAYED - only to be translated.
//!
//! This file previously held a `LoadingStage` enum with a callback type, written for model
//! loading alone. It had no call sites anywhere and was not declared in `mod.rs`, so none
//! of it was ever compiled. It is replaced rather than extended because the shape it chose
//! - stages of a load, with a percentage - does not describe a render, where loading is one
//! phase among several and the only counted one is the sampling loop.

/// `(phase, done, total)`. `total == 0` = a phase that is starting, with nothing to count.
///
/// Deliberately not `Send + Sync`: a reporter usually writes to a channel owned by the
/// request, and forcing it across threads would push every caller into an `Arc` it does
/// not need. Engines that spawn call it from the thread that owns the work.
pub type ProgressFn<'a> = dyn Fn(&str, usize, usize) + 'a;

/// The phase names, so the server and the client agree on the spelling.
///
/// Deliberately coarse: a caller wants to know whether it is waiting on a disk, a GPU or a
/// queue, not which layer is loading. Anything finer belongs in the log.
pub mod phase {
    /// Waiting for a card - reclaiming a resident model, or queued behind another render.
    pub const ADMIT: &str = "admit";
    /// Reading a text encoder off disk and onto its device. Often the longest phase, and
    /// the one that used to be invisible.
    pub const LOAD_ENCODER: &str = "load-encoder";
    /// Turning the prompt into conditioning.
    pub const ENCODE: &str = "encode";
    /// Reading the denoiser onto its device.
    pub const LOAD_MODEL: &str = "load-model";
    /// The sampling loop - the only phase that has ever had a step count.
    pub const DENOISE: &str = "denoise";
    /// The audio codes a language model decodes before a denoiser renders them.
    pub const CODES: &str = "codes";
    /// Turning text into speech, counted in the units the caller can see: a chunk of the
    /// text it sent, not a decoder step it has no way to relate to what it asked for.
    pub const SYNTHESIZE: &str = "synthesize";
    /// Latents to pixels.
    pub const DECODE: &str = "decode";
    /// Per-frame work after the clip exists: the reference face, an upscale.
    pub const FRAMES: &str = "frames";
    /// Muxing and encoding the artefact that goes back over the wire.
    pub const ENCODE_OUTPUT: &str = "encode-output";
}

/// A human-readable label, for a client with no opinion of its own.
///
/// Falls back to the raw name, so a phase added on the server still reads as SOMETHING in
/// an older client rather than disappearing. A client too old to know a phase is not a
/// client that should show nothing.
pub fn label(name: &str) -> &str {
    match name {
        phase::ADMIT => "Waiting for a GPU",
        phase::LOAD_ENCODER => "Loading the text encoder",
        phase::ENCODE => "Encoding the prompt",
        phase::LOAD_MODEL => "Loading the model",
        phase::DENOISE => "Rendering",
        phase::SYNTHESIZE => "Synthesising the speech",
        phase::DECODE => "Decoding",
        phase::FRAMES => "Finishing the frames",
        phase::ENCODE_OUTPUT => "Encoding the output",
        other => other,
    }
}

/// A reporter that can be PUBLISHED for the duration of a call instead of threaded through
/// it - see [`scoped`]. Owned and thread-safe, so a load that fans out to a pool can hand a
/// clone to its workers.
pub type SharedProgressFn = std::sync::Arc<dyn Fn(&str, usize, usize) + Send + Sync>;

/// A reporter published for the work a call performs, for phases nobody can thread a
/// parameter into.
///
/// Loading is the longest silent phase of a render and the one no engine reported: a client
/// saw a stage name that did not move for tens of seconds, which reads exactly like a wedged
/// server. The weights are read by a handful of loaders shared by every engine, and those
/// loaders sit ten call levels below the request - reaching them with a parameter would mean
/// changing every model file, and the NEXT model added would arrive silent again.
///
/// So the thread that starts a load publishes a reporter for its duration, and the loaders
/// count what they read into it. Outside a scope nothing is published and a loader emits
/// nothing, so a binary, a test or a parity harness behaves exactly as before.
///
/// Two rules come with it:
///
/// - A phase must have ONE source of counts. An engine that already counts its own blocks
///   into a threaded reporter must not also run inside a published scope, or the bar would
///   flip between two different totals for the same phase name.
/// - Work that fans out to a thread pool must carry the reporter EXPLICITLY: a pool thread
///   cannot see this one. Take a clone with [`scoped::current`] before the fan-out.
///
/// Mirrors `cancel::scoped`, which publishes a cancellation token the same way and for the
/// same reason.
pub mod scoped {
    use super::SharedProgressFn;
    use std::cell::RefCell;

    thread_local! {
        static CURRENT: RefCell<Option<SharedProgressFn>> = const { RefCell::new(None) };
    }

    /// Restores the previous reporter when dropped, INCLUDING on an unwind.
    ///
    /// Restoring at the end of a block instead would leave a reporter published on a load
    /// that failed - and these run on pooled blocking threads, which are handed to the next
    /// request, so the leak would send one request's counts into another request's channel.
    pub struct Scope {
        prev: Option<SharedProgressFn>,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            CURRENT.with(|c| *c.borrow_mut() = self.prev.take());
        }
    }

    /// Publish `report` until the returned guard drops. For a load whose body is a long
    /// function that cannot be wrapped in a closure without reindenting all of it.
    #[must_use = "the reporter is published only while the guard is alive"]
    pub fn publish(report: SharedProgressFn) -> Scope {
        Scope {
            prev: CURRENT.with(|c| c.borrow_mut().replace(report)),
        }
    }

    /// Publish `report` for the duration of `f`, restoring whatever was there before (so
    /// nested scopes compose).
    pub fn with<T>(report: SharedProgressFn, f: impl FnOnce() -> T) -> T {
        let _scope = publish(report);
        f()
    }

    /// A clone of the reporter in force, for work about to leave this thread.
    pub fn current() -> Option<SharedProgressFn> {
        CURRENT.with(|c| c.borrow().clone())
    }

    /// [`super::note`] against the thread's current reporter; nothing when there is none.
    ///
    /// The reporter is CLONED out before it is called, so it may itself publish a scope or
    /// report in turn without deadlocking on the cell it lives in.
    pub fn note(phase: &str, done: usize, total: usize) {
        let Some(f) = current() else { return };
        let f = move |p: &str, d: usize, t: usize| f(p, d, t);
        super::note(Some(&f), phase, done, total);
    }
}

/// A reporter that can also STOP the work: an error returned from it cancels the render.
///
/// Separate from [`ProgressFn`] rather than folded into it, because most engines have
/// nothing to say back and threading a `Result` through them would add an error path that
/// can never fire. Engines whose loop already checks for cancellation use this one.
pub type ProgressTryFn<'a> = dyn Fn(&str, usize, usize) -> crate::tensor::Result<()> + 'a;

/// [`note`], for a reporter that can cancel. The error is the caller's to propagate.
pub fn try_note(
    report: Option<&ProgressTryFn<'_>>,
    phase: &str,
    done: usize,
    total: usize,
) -> crate::tensor::Result<()> {
    match report {
        Some(f) => f(phase, clamp_done(done, total), total),
        None => Ok(()),
    }
}

/// Call `report` if there is one. Saves every engine an `if let` around every phase.
///
/// The count is CLAMPED to its total on the way through. A phase that miscounts is a bug and
/// should be fixed at the source, but a bar that reads 8 of 4 is worse than one that reads
/// 4 of 4: it tells someone the estimate is nonsense, and there is nothing they can do with
/// that. One such miscount shipped and travelled the whole chain without anything objecting,
/// which is why the invariant lives here now rather than in each caller's head.
///
/// A total of zero means "no count for this phase" and passes through untouched - that is
/// how a phase says it has started and has nothing to report yet.
pub fn note(report: Option<&ProgressFn<'_>>, phase: &str, done: usize, total: usize) {
    if let Some(f) = report {
        f(phase, clamp_done(done, total), total);
    }
}

/// Wrap a reporter so a phase reports again only when its PERCENTAGE changes.
///
/// A load counts TENSORS, of which a checkpoint has thousands, and every count travels the
/// whole chain to the client. Nobody can read more than a hundred distinct positions on a
/// bar, so the rest is traffic that can only slow the work down - the reason engines used to
/// report nothing at all rather than report per item. Phases are kept apart: a phase that
/// starts at the percentage another one ended on still announces itself.
///
/// Uncounted notifications (`total == 0`) always pass: they are how a phase says it has begun,
/// and there is nothing there to be redundant about.
pub fn per_percent(inner: SharedProgressFn) -> SharedProgressFn {
    let last = std::sync::Mutex::new((String::new(), usize::MAX));
    std::sync::Arc::new(move |phase: &str, done: usize, total: usize| {
        if total > 0 {
            let pct = done * 100 / total;
            let mut l = last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if l.0 == phase && l.1 == pct {
                return;
            }
            phase.clone_into(&mut l.0);
            l.1 = pct;
        }
        inner(phase, done, total);
    })
}

/// A count may never exceed its total. Zero total means the phase is not counting.
fn clamp_done(done: usize, total: usize) -> usize {
    if total == 0 {
        done
    } else {
        done.min(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bar that reads 8 of 4 tells someone the estimate is nonsense, and there is nothing
    /// they can do with that. Exactly that shipped: a route counted NOTIFICATIONS rather
    /// than work - the one announcing the start included - and the number travelled the
    /// whole chain without anything objecting.
    #[test]
    fn a_count_can_never_exceed_its_total() {
        let seen = std::cell::RefCell::new(Vec::new());
        let f = |p: &str, d: usize, t: usize| {
            seen.borrow_mut().push((p.to_string(), d, t));
        };
        let f: &ProgressFn<'_> = &f;
        note(Some(f), phase::DENOISE, 8, 4);
        note(Some(f), phase::DECODE, 3, 9);
        // A total of zero is a phase saying it has started with nothing to count, and must
        // pass through rather than be clamped away.
        note(Some(f), phase::LOAD_MODEL, 0, 0);
        let got = seen.borrow().clone();
        assert_eq!(
            got[0].1, 4,
            "a count past its total must be clamped, got {:?}",
            got[0]
        );
        assert_eq!(got[1].1, 3, "a count inside its total must pass untouched");
        assert_eq!(
            (got[2].1, got[2].2),
            (0, 0),
            "an uncounted phase must stay uncounted"
        );
    }

    #[test]
    fn every_named_phase_has_a_label_of_its_own() {
        let names = [
            phase::ADMIT,
            phase::LOAD_ENCODER,
            phase::ENCODE,
            phase::LOAD_MODEL,
            phase::DENOISE,
            phase::SYNTHESIZE,
            phase::DECODE,
            phase::FRAMES,
            phase::ENCODE_OUTPUT,
        ];
        for n in names {
            assert_ne!(label(n), n, "'{n}' falls through to its raw name");
        }
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "two phases share a name");
    }

    /// A phase the client has never heard of must still say something.
    #[test]
    fn an_unknown_phase_reads_as_itself() {
        assert_eq!(label("upscale"), "upscale");
        assert_eq!(label(""), "");
    }

    /// A recording reporter, shaped like the ones engines publish.
    fn recorder() -> (
        SharedProgressFn,
        std::sync::Arc<std::sync::Mutex<Vec<(String, usize, usize)>>>,
    ) {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let f: SharedProgressFn = std::sync::Arc::new(move |p: &str, d: usize, t: usize| {
            sink.lock().unwrap().push((p.to_string(), d, t));
        });
        (f, seen)
    }

    #[test]
    fn outside_a_scope_nothing_is_reported() {
        // A loader outside a render - a parity binary, a test - must stay silent rather
        // than report into whatever ran last on this thread.
        assert!(scoped::current().is_none());
        scoped::note(phase::LOAD_MODEL, 3, 10); // must not panic, must reach nobody
    }

    #[test]
    fn a_published_reporter_receives_the_counts_of_the_work_it_wraps() {
        let (f, seen) = recorder();
        scoped::with(f, || {
            scoped::note(phase::LOAD_MODEL, 1, 4);
            // Clamped exactly like the threaded path: one invariant, not two.
            scoped::note(phase::LOAD_MODEL, 9, 4);
        });
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[
                (phase::LOAD_MODEL.to_string(), 1, 4),
                (phase::LOAD_MODEL.to_string(), 4, 4),
            ]
        );
        // And the scope is torn down: a pooled thread must not carry it to the next request.
        assert!(scoped::current().is_none());
        scoped::note(phase::LOAD_MODEL, 2, 4);
        assert_eq!(
            seen.lock().unwrap().len(),
            2,
            "a torn-down scope must not report"
        );
    }

    #[test]
    fn scopes_nest_and_restore_the_outer_reporter() {
        let (outer, outer_seen) = recorder();
        let (inner, inner_seen) = recorder();
        scoped::with(outer, || {
            scoped::with(inner, || scoped::note(phase::LOAD_ENCODER, 1, 2));
            scoped::note(phase::LOAD_MODEL, 1, 2);
        });
        assert_eq!(
            inner_seen.lock().unwrap().len(),
            1,
            "the inner scope takes over"
        );
        assert_eq!(
            outer_seen.lock().unwrap().as_slice(),
            &[(phase::LOAD_MODEL.to_string(), 1, 2)],
            "the outer reporter must be restored, and must not see the inner scope's counts"
        );
    }

    /// The published reporter must survive an unwind: these run on pooled blocking threads,
    /// and a load that fails leaving its reporter behind would send the NEXT request's counts
    /// into a channel that belongs to nobody.
    #[test]
    fn a_panicking_load_leaves_no_reporter_behind() {
        let (f, _seen) = recorder();
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scoped::with(f, || panic!("load failed"));
        }));
        assert!(hit.is_err());
        assert!(
            scoped::current().is_none(),
            "an unwound scope must be torn down"
        );
    }

    /// Work that leaves the thread has to carry the reporter itself - a pool thread cannot
    /// see the scope. This is the contract `fp8_scaled`'s parallel decode relies on.
    #[test]
    fn a_reporter_taken_before_a_fan_out_still_reports_from_another_thread() {
        let (f, seen) = recorder();
        scoped::with(f, || {
            let carried = scoped::current().expect("a scope is in force");
            std::thread::spawn(move || {
                assert!(
                    scoped::current().is_none(),
                    "the scope is this thread's alone"
                );
                carried(phase::LOAD_MODEL, 7, 9);
            })
            .join()
            .unwrap();
        });
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[(phase::LOAD_MODEL.to_string(), 7, 9)]
        );
    }

    #[test]
    fn a_percent_filter_keeps_the_positions_a_bar_can_show() {
        let (f, seen) = recorder();
        let throttled = per_percent(f);
        // 400 tensors, every one counted: at most one message per percent.
        for i in 1..=400 {
            throttled(phase::LOAD_MODEL, i, 400);
        }
        // 0% through 100% inclusive: the positions a bar can actually show, and nothing else.
        assert_eq!(
            seen.lock().unwrap().len(),
            101,
            "one message per percent, no more"
        );
        // A different phase at the same percentage still announces itself.
        throttled(phase::LOAD_ENCODER, 400, 400);
        // And an uncounted phase always passes - it is how a phase says it has begun.
        throttled(phase::DECODE, 0, 0);
        throttled(phase::DECODE, 0, 0);
        let got = seen.lock().unwrap();
        assert_eq!(got.len(), 104);
        assert_eq!(got[101].0, phase::LOAD_ENCODER);
        assert_eq!(got[102].0, phase::DECODE);
    }

    #[test]
    fn note_is_a_no_op_without_a_reporter() {
        note(None, phase::DENOISE, 1, 10); // must not panic
        let seen = std::cell::RefCell::new(Vec::new());
        let f = |p: &str, d: usize, t: usize| seen.borrow_mut().push((p.to_string(), d, t));
        note(Some(&f), phase::DECODE, 0, 0);
        assert_eq!(
            seen.borrow().as_slice(),
            &[(phase::DECODE.to_string(), 0, 0)]
        );
    }
}
