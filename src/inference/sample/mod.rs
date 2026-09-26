//! Drawing a value: the token sampler for text, and the flow-matching solver the
//! image and audio stacks step along.
//!
//! `crate::inference::sample::<part>`

pub mod flow_unipc;
pub mod token_sampling;

/// Receives `(completed, total)` as a denoise advances: `(0, total)` before the first
/// step, then once after every step. `total` is the number of steps this run takes,
/// after any truncation of the schedule.
pub type StepSink<'a> = &'a dyn Fn(usize, usize);

/// A [`StepSink`] for callers with nobody to report to.
pub fn no_progress(_completed: usize, _total: usize) {}
