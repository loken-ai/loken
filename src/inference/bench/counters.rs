//! What a decode step moves and waits on, counted where the model hands its steps to the cards
//! and read from the kernel's accounting of the process.
//!
//! The crossings are counted at the `Offload` seam: every step a layer offers a card is one call,
//! its arguments the bytes handed over and its answer the bytes read back. A step the card takes
//! returns its answer to the host before the layer goes on, so each taken step is also one wait
//! on the card's stream. A step the card declines runs on the host and moves nothing.

use crate::inference::offload::projection::Projection;
use crate::inference::offload::{AttnBlock, HcStream, Offload, PrefillAttn};
use crate::tensor::Result;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// One kind of step, as counted at the seam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StepCount {
    /// Calls the card took.
    pub taken: u64,
    /// Calls the card declined, run on the host.
    pub declined: u64,
    /// Bytes handed to the card over the taken calls.
    pub bytes_out: u64,
    /// Bytes read back from the card over the taken calls.
    pub bytes_back: u64,
}

/// The counts of every kind of step since the counting began.
#[derive(Default)]
pub struct Crossings {
    steps: Mutex<BTreeMap<&'static str, StepCount>>,
}

impl Crossings {
    /// Note one call of `step` that handed `out` bytes and got `back` bytes, or was declined.
    pub fn note(&self, step: &'static str, taken: bool, out: u64, back: u64) {
        let mut m = self.steps.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(step).or_default();
        if taken {
            e.taken += 1;
            e.bytes_out += out;
            e.bytes_back += back;
        } else {
            e.declined += 1;
        }
    }

    /// Every kind of step and its counts, by name.
    pub fn snapshot(&self) -> Vec<(&'static str, StepCount)> {
        self.steps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// Zero every count.
    pub fn reset(&self) {
        self.steps.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Bytes of a slice of `f32`.
fn f32_bytes(xs: &[f32]) -> u64 {
    std::mem::size_of_val(xs) as u64
}

/// Bytes of a slice of `i32`.
fn i32_bytes(xs: &[i32]) -> u64 {
    std::mem::size_of_val(xs) as u64
}

/// The bytes an answer brings back: a vector of `f32`, or nothing for an error.
fn back_bytes(r: &Result<Vec<f32>>) -> u64 {
    r.as_ref().map_or(0, |v| f32_bytes(v))
}

/// An offload that counts every step through `crossings` and hands it to `inner`. Without an
/// inner offload every step is declined, which is what the host alone answers: the counts then
/// say how many steps a card would have been offered.
pub struct Counting {
    inner: Option<Arc<dyn Offload>>,
    pub crossings: Arc<Crossings>,
}

impl Counting {
    pub fn new(inner: Option<Arc<dyn Offload>>) -> Self {
        Self {
            inner,
            crossings: Arc::new(Crossings::default()),
        }
    }
}

impl Offload for Counting {
    fn prefill_attention(&self, a: &PrefillAttn) -> Option<Result<Vec<f32>>> {
        let r = self.inner.as_ref().and_then(|o| o.prefill_attention(a));
        let out = f32_bytes(a.qr) + f32_bytes(a.kv) + f32_bytes(a.sink) + i32_bytes(a.idxs);
        self.crossings.note(
            "prefill attention",
            r.is_some(),
            out,
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn hc_stream(&self, x: &[f32], hc: usize, d: usize) -> Option<Box<dyn HcStream>> {
        let r = self.inner.as_ref().and_then(|o| o.hc_stream(x, hc, d));
        self.crossings
            .note("hc stream", r.is_some(), f32_bytes(x), 0);
        r
    }

    fn projection(&self, p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>> {
        let r = self.inner.as_ref().and_then(|o| o.projection(p, xs));
        self.crossings.note(
            "projection",
            r.is_some(),
            f32_bytes(xs),
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn sparse_attention(
        &self,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        idxs: &[i32],
        dims: (usize, usize, usize, usize),
        scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        let r = self
            .inner
            .as_ref()
            .and_then(|o| o.sparse_attention(q, kv, sink, idxs, dims, scale));
        let out = f32_bytes(q) + f32_bytes(kv) + f32_bytes(sink) + i32_bytes(idxs);
        self.crossings.note(
            "sparse attention",
            r.is_some(),
            out,
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn index_scores(
        &self,
        q: &[f32],
        k: &[f32],
        weights: &[f32],
        dims: (usize, usize, usize, usize),
        ratio: usize,
        scale: f32,
        q_off: usize,
    ) -> Option<Result<Vec<f32>>> {
        let r = self
            .inner
            .as_ref()
            .and_then(|o| o.index_scores(q, k, weights, dims, ratio, scale, q_off));
        let out = f32_bytes(q) + f32_bytes(k) + f32_bytes(weights);
        self.crossings.note(
            "index scores",
            r.is_some(),
            out,
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn attn_block(&self, p: &AttnBlock) -> Option<Result<(Vec<f32>, Vec<f32>)>> {
        let r = self.inner.as_ref().and_then(|o| o.attn_block(p));
        let out = f32_bytes(p.x) + f32_bytes(p.cos) + f32_bytes(p.sin) + f32_bytes(p.window);
        let back = r.as_ref().map_or(0, |r| {
            r.as_ref().map_or(0, |(o, kv)| f32_bytes(o) + f32_bytes(kv))
        });
        self.crossings
            .note("attention block", r.is_some(), out, back);
        r
    }

    fn expert_dev(
        &self,
        w1: &Projection,
        w3: &Projection,
        w2: &Projection,
        x: &[f32],
        limit: f32,
    ) -> Option<Result<Vec<f32>>> {
        let r = self
            .inner
            .as_ref()
            .and_then(|o| o.expert_dev(w1, w3, w2, x, limit));
        self.crossings.note(
            "shared expert",
            r.is_some(),
            f32_bytes(x),
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn attn_out(
        &self,
        wo_a: &Projection,
        wo_b: &Projection,
        o: &[f32],
        o_groups: usize,
        p: usize,
        o_lora: usize,
        dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        let r = self
            .inner
            .as_ref()
            .and_then(|x| x.attn_out(wo_a, wo_b, o, o_groups, p, o_lora, dim));
        self.crossings.note(
            "attention out",
            r.is_some(),
            f32_bytes(o),
            r.as_ref().map_or(0, back_bytes),
        );
        r
    }

    fn record(&self, stage: &'static str, nanos: u64) {
        if let Some(o) = &self.inner {
            o.record(stage, nanos);
        }
    }
}

/// What the kernel counts for this process: the page faults that read from the disk, and the
/// bytes the process had read from storage. Zero on a system that does not say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessCounters {
    pub major_faults: u64,
    pub read_bytes: u64,
}

impl ProcessCounters {
    /// The counters now.
    pub fn now() -> Self {
        Self {
            major_faults: major_faults().unwrap_or(0),
            read_bytes: read_bytes().unwrap_or(0),
        }
    }

    /// What was counted between `earlier` and this.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            major_faults: self.major_faults.saturating_sub(earlier.major_faults),
            read_bytes: self.read_bytes.saturating_sub(earlier.read_bytes),
        }
    }
}

/// The process's major page faults, from the kernel's status line of the process.
#[cfg(target_os = "linux")]
fn major_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name is in parentheses and may hold spaces: the fields start after it.
    let after = &stat[stat.rfind(')')? + 1..];
    // The fields after the name: state, ppid, pgrp, session, tty, tpgid, flags, minflt, cminflt,
    // majflt. The major faults are the tenth.
    after.split_whitespace().nth(9)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
fn major_faults() -> Option<u64> {
    None
}

/// Bytes the process, every thread of it, had read from storage.
#[cfg(target_os = "linux")]
fn read_bytes() -> Option<u64> {
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    io.lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
}

#[cfg(not(target_os = "linux"))]
fn read_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card that takes projections and declines everything else.
    struct ProjectionsOnly;

    impl Offload for ProjectionsOnly {
        fn projection(&self, _p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>> {
            Some(Ok(vec![0.0; xs.len() / 2]))
        }
        fn sparse_attention(
            &self,
            _q: &[f32],
            _kv: &[f32],
            _sink: &[f32],
            _idxs: &[i32],
            _dims: (usize, usize, usize, usize),
            _scale: f32,
        ) -> Option<Result<Vec<f32>>> {
            None
        }
    }

    fn weight() -> Projection {
        Projection::Dense(crate::tensor::Tensor::zeros((2, 4), crate::tensor::DType::F32).unwrap())
    }

    /// A taken step counts its bytes each way; a declined one counts only that it was declined.
    #[test]
    fn taken_and_declined_steps_are_counted_with_their_bytes() {
        let c = Counting::new(Some(Arc::new(ProjectionsOnly)));
        let w = weight();
        assert!(c.projection(&w, &[1.0; 8]).is_some());
        assert!(c.projection(&w, &[1.0; 4]).is_some());
        assert!(c
            .sparse_attention(&[0.0; 6], &[0.0; 2], &[0.0], &[0], (1, 1, 1, 1), 1.0)
            .is_none());
        let snap: BTreeMap<&str, StepCount> = c.crossings.snapshot().into_iter().collect();
        assert_eq!(
            snap["projection"],
            StepCount {
                taken: 2,
                declined: 0,
                bytes_out: 48,
                bytes_back: 24,
            }
        );
        assert_eq!(
            snap["sparse attention"],
            StepCount {
                taken: 0,
                declined: 1,
                bytes_out: 0,
                bytes_back: 0,
            }
        );
        c.crossings.reset();
        assert!(c.crossings.snapshot().is_empty());
    }

    /// With no card behind it, every step is declined and nothing crosses.
    #[test]
    fn without_a_card_every_step_is_declined() {
        let c = Counting::new(None);
        let w = weight();
        assert!(c.projection(&w, &[1.0; 8]).is_none());
        assert!(c.attn_out(&w, &w, &[0.0; 4], 1, 4, 1, 2).is_none());
        for (_, s) in c.crossings.snapshot() {
            assert_eq!(
                (s.taken, s.declined, s.bytes_out, s.bytes_back),
                (0, 1, 0, 0)
            );
        }
    }

    /// The process counters read and never go backwards.
    #[test]
    fn process_counters_are_read_and_monotonic() {
        let a = ProcessCounters::now();
        let b = ProcessCounters::now();
        let d = b.since(&a);
        assert!(d.major_faults <= b.major_faults);
        assert!(d.read_bytes <= b.read_bytes);
        assert_eq!(a.since(&b).major_faults, 0);
    }
}
