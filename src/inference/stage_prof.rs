//! A profile of the named stages a forward runs through.
//!
//! A stage is recorded by name with its time and its call count.
//! The profile sums the records of every thread.
//! It is off by default: each record takes a lock.
//! A stage recorded into a graph is timed by 2 events the card stamps at each run of the graph.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Each stage's total nanoseconds and call count, summed over every thread that records.
static STAGE_PROF: Mutex<BTreeMap<&'static str, (u64, u64)>> = Mutex::new(BTreeMap::new());
static STAGE_PROF_ON: AtomicBool = AtomicBool::new(false);

/// Switch the stage profile on (zeroing it) or off.
pub fn stage_prof_enable(on: bool) {
    if on {
        STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    STAGE_PROF_ON.store(on, Ordering::Relaxed);
}

/// Whether the profile records.
pub fn stage_prof_enabled() -> bool {
    STAGE_PROF_ON.load(Ordering::Relaxed)
}

/// Add `ns` and 1 call to stage `stage`.
/// A no-op when the profile is off.
pub fn prof_record(stage: &'static str, ns: u64) {
    if STAGE_PROF_ON.load(Ordering::Relaxed) {
        let mut m = STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(stage).or_default();
        e.0 += ns;
        e.1 += 1;
    }
}

/// The name stage `stage` of card `ordinal` is reported under.
pub fn on_card(ordinal: usize, stage: &'static str) -> &'static str {
    static NAMES: Mutex<BTreeMap<(usize, &'static str), &'static str>> =
        Mutex::new(BTreeMap::new());
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names
        .entry((ordinal, stage))
        .or_insert_with(|| Box::leak(format!("card {ordinal}: {stage}").into_boxed_str()))
}

/// Add `count` to stage `stage` as thousandths of a microsecond, so its total in microseconds
/// reads as the count.
pub fn prof_count(stage: &'static str, count: u64) {
    prof_record(stage, count.saturating_mul(1000));
}

/// A stage recorded into a graph: the events at its two ends, which the card stamps each time
/// the graph runs.
#[cfg(feature = "gpu")]
pub struct StageMark {
    name: &'static str,
    start: crate::tensor::gpu_runtime::driver::CudaEvent,
    end: crate::tensor::gpu_runtime::driver::CudaEvent,
}

#[cfg(feature = "gpu")]
thread_local! {
    static CAPTURE_MARKS: std::cell::RefCell<Vec<StageMark>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The name a stage is reported under when it ran inside a graph.
#[cfg(feature = "gpu")]
fn in_graph_name(name: &'static str) -> &'static str {
    static NAMES: Mutex<BTreeMap<&'static str, &'static str>> = Mutex::new(BTreeMap::new());
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names
        .entry(name)
        .or_insert_with(|| Box::leak(format!("in graph: {name}").into_boxed_str()))
}

/// Run `f`, a stage being recorded into a graph on `stream`, between two events when the stage
/// profile is on. The marks are taken by [`take_capture_marks`] once the capture ends.
#[cfg(feature = "gpu")]
pub fn mark_stage_in_capture<R>(
    name: &'static str,
    stream: &std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    f: impl FnOnce() -> R,
) -> R {
    let ctx = stream.context();
    let flags = Some(crate::tensor::gpu_runtime::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let (Ok(start), Ok(end)) = (ctx.new_event(flags), ctx.new_event(flags)) else {
        return f();
    };
    // An external record keeps the event a node of the graph that the card stamps at each run.
    let record = |e: &crate::tensor::gpu_runtime::driver::CudaEvent| unsafe {
        crate::tensor::gpu_runtime::driver::sys::cuEventRecordWithFlags(
            e.cu_event(),
            stream.cu_stream(),
            crate::tensor::gpu_runtime::driver::sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL
                as u32,
        )
        .result()
    };
    if record(&start).is_err() {
        return f();
    }
    let out = f();
    if record(&end).is_ok() {
        CAPTURE_MARKS.with(|m| m.borrow_mut().push(StageMark { name, start, end }));
    }
    out
}

/// The marks the capture just ended recorded, in the order they were made.
#[cfg(feature = "gpu")]
pub fn take_capture_marks() -> Vec<StageMark> {
    CAPTURE_MARKS.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Add what the card stamped on each mark of a graph that has just run to the profile.
#[cfg(feature = "gpu")]
pub fn record_stage_marks(marks: &[StageMark]) {
    for m in marks {
        match m.start.elapsed_ms(&m.end) {
            Ok(ms) => prof_record(in_graph_name(m.name), (ms as f64 * 1e6) as u64),
            Err(e) => {
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!("a stage mark of a graph cannot be read ({}): {e:?}", m.name);
                }
            }
        }
    }
}

/// Each stage's total nanoseconds and call count since it was switched on.
pub fn stage_prof_snapshot() -> Vec<(&'static str, u64, u64)> {
    STAGE_PROF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(k, v)| (*k, v.0, v.1))
        .collect()
}
