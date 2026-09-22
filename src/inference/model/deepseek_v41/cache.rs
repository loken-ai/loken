//! Per-layer attention state carried across decode steps (unit 2).
//!
//! Decode is prefill at one token: the hyper-connections, norms and MoE are per-token and stateless,
//! so only attention keeps state between steps. A ratio-0 layer keeps a sliding window of the recent
//! KV; a band layer keeps that window plus the compressed KV and index keys it (or its kv-source)
//! builds one group at a time. Everything cached is already rope'd and quantised at its own position,
//! so a cached row is bit-for-bit the value prefill would have recomputed there - which is what the
//! self-consistency gate checks.

use super::engram::EngramState;
use std::collections::VecDeque;

/// One layer's decode state.
#[derive(Default, Clone)]
pub struct AttnCache {
    /// Sliding window of rope'd, fp8-quantised KV, one `head_dim` row per recent token, oldest
    /// first, capped at `window_size`. Both ratio-0 and band layers keep this.
    pub win: VecDeque<Vec<f32>>,
    /// Band kv-source only: rope'd, fp4 compressed KV, `[n_groups * head_dim]` flattened.
    pub comp_kv: Vec<f32>,
    /// Band index-key owner only: rope'd, fp4 (pow2 scale) index keys, `[n_groups * index_head_dim]`.
    pub index_k: Vec<f32>,
    /// Band kv-source only: raw `x` rows buffered since the last group boundary, to pool the next.
    pub group_x: Vec<Vec<f32>>,
    /// Completed compressed groups so far.
    pub n_groups: usize,
}

/// The whole model's decode state: one cache per layer, the recent ids the engram hash reads,
/// and the running position.
#[derive(Clone)]
pub struct DecodeState {
    pub layers: Vec<AttnCache>,
    pub engram: EngramState,
    pub pos: usize,
    /// The block indices whose attention input the DSpark draft reads. Empty leaves the capture off.
    pub capture_layers: Vec<usize>,
    /// The last decode's mean-pooled hidden at each capture layer, in `capture_layers` order. Read
    /// by the draft after a decode; overwritten each decode.
    pub captured: Vec<Vec<f32>>,
}

impl DecodeState {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| AttnCache::default()).collect(),
            engram: EngramState::default(),
            pos: 0,
            capture_layers: Vec::new(),
            captured: Vec::new(),
        }
    }

    /// A copy of the state to return to after a speculative batch is rejected. The window is
    /// capped at `window_size` and the compressed groups grow slowly, so a copy is cheap
    /// beside a forward, and an exact copy rolls back with no boundary reasoning: whatever a
    /// rejected token appended - a window row, a buffered group row, a pooled group, an
    /// engram id - is simply the copy again.
    pub fn checkpoint(&self) -> DecodeState {
        self.clone()
    }

    /// Return to `mark`, discarding everything appended since it was taken.
    pub fn rewind(&mut self, mark: &DecodeState) {
        self.clone_from(mark);
    }
}
