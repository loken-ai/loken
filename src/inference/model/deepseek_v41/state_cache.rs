//! The decode-state cache as the serving paths use it: restore the longest stored prefix of a
//! prompt and prefill the rest, keep the states a prefill passes at block boundaries, and keep the
//! state a finished turn leaves.
//!
//! One `StateCache` is a store, the identity every state in it is written and read under, and the
//! model's compression ratios (which say how a state divides into blocks). The serial backend and
//! the batch worker share it, so a prefix stored by either is found by both.
//!
//! The pipeline head does not use it: a pipelined sequence's state is in two places, the head's
//! layers here and the tail's on another node under its own sequence id, and a restored head
//! against a tail standing at position zero would continue wrongly. Restoring both would need the
//! tail to snapshot and restore under the same identity over the wire; until it does, a pipelined
//! model is prefilled cold.

use super::cache::DecodeState;
use super::model::DeepseekV41Model;
use super::snapshot::{
    block_shape, decode_split, encode_block, encode_grid_manifest, encode_manifest, BlockLayout,
    GridPoint, Identity,
};
use crate::inference::cache::state_disk::StateStore;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// What one more token costs when it is run on a restored state against one prefilled in a batch,
/// measured on this model: a restored state continues token by token through attention while a
/// cold prompt runs each layer once over all its tokens, and 996 continued tokens took 501 s where
/// 2000 cold ones took 122 s. A stored state is used only while its remainder costs less than
/// prefilling the whole prompt: `remainder * ratio <= prompt`.
pub const CONTINUE_COST_RATIO: usize = 8;

#[derive(Clone)]
pub struct StateCache {
    store: Arc<StateStore>,
    identity: Identity,
    ratios: Vec<usize>,
    continue_cost_ratio: usize,
}

/// A prefix restored from the cache.
struct Restored {
    covered: usize,
    state: DecodeState,
    /// The logits after the last of `covered` tokens, when the state was stored with them.
    logits: Option<Vec<f32>>,
}

impl StateCache {
    /// The cache over `store` for `model`, `None` when a block of the store's chain does not hold
    /// whole groups of every layer.
    pub fn new(
        model: &DeepseekV41Model,
        store: Arc<StateStore>,
        identity: Identity,
    ) -> Option<Self> {
        let block = store.block_tokens();
        if let Some(r) = model
            .compress_ratios()
            .iter()
            .find(|&&r| r != 0 && block % r != 0)
        {
            tracing::warn!(
                "state cache off: a block of {block} tokens is not a whole number of groups at ratio {r}"
            );
            return None;
        }
        Some(Self {
            store,
            identity,
            // The ratios of the layers a state has: the model lists the draft's after them.
            ratios: model.compress_ratios()[..model.n_layers().min(model.compress_ratios().len())]
                .to_vec(),
            continue_cost_ratio: CONTINUE_COST_RATIO,
        })
    }

    /// The cost of a continued token against a batched one that decides whether a stored state
    /// pays (see `CONTINUE_COST_RATIO`); 1 uses any stored prefix.
    pub fn with_continue_cost_ratio(mut self, ratio: usize) -> Self {
        self.continue_cost_ratio = ratio.max(1);
        self
    }

    fn layout(&self) -> BlockLayout<'_> {
        BlockLayout {
            ratios: &self.ratios,
            block_tokens: self.store.block_tokens(),
        }
    }

    /// The longest stored state that is a prefix of `ids`. A state that is the whole of `ids` but
    /// holds no logits cannot answer and is passed over; a file that cannot be read or decoded is
    /// logged and dropped.
    fn restore(&self, model: &DeepseekV41Model, ids: &[u32]) -> Option<Restored> {
        let keys = self.store.keys(&self.identity, ids);
        for entry in self.store.candidates(&keys, ids.len()) {
            // Longer remainders cost more than the prompt prefilled whole.
            if (ids.len() - entry.len) * self.continue_cost_ratio > ids.len() {
                tracing::info!(
                    "state cache: the {}-token prefix leaves {} of {} tokens to continue, more than a cold prompt costs",
                    entry.len,
                    ids.len() - entry.len,
                    ids.len()
                );
                continue;
            }
            let manifest = match self.store.read(&entry) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("state cache: {e}; ignored");
                    continue;
                }
            };
            let blocks: std::result::Result<Vec<Vec<u8>>, _> = entry
                .blocks
                .iter()
                .map(|h| self.store.read_block(h))
                .collect();
            let blocks = match blocks {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("state cache: {e}; dropped");
                    self.store.forget(&entry);
                    continue;
                }
            };
            match decode_split(&manifest, &blocks, &self.identity, model.n_layers()) {
                Ok(snap) if snap.state.pos == entry.len => {
                    if entry.len == ids.len() && snap.logits.is_none() {
                        continue;
                    }
                    return Some(Restored {
                        covered: entry.len,
                        state: snap.state,
                        logits: snap.logits,
                    });
                }
                Ok(_) => {
                    tracing::warn!("state cache: a snapshot disagrees with its name; dropped");
                    self.store.forget(&entry);
                }
                Err(e) => {
                    tracing::warn!("state cache: a snapshot is unusable ({e}); dropped");
                    self.store.forget(&entry);
                }
            }
        }
        None
    }

    /// The token counts inside `(from, to)` at which a state is kept: every multiple of the
    /// store's block size, which are the block boundaries.
    fn grid_points(&self, from: usize, to: usize) -> Vec<usize> {
        let b = self.store.block_tokens();
        (from / b + 1..)
            .map(|k| k * b)
            .take_while(|&p| p < to)
            .collect()
    }

    /// Prefill `ids` into `state`, a fresh one: the longest cached prefix is restored and only the
    /// rest computed, a prompt seen whole is answered from its stored logits, and the states the
    /// pass goes through at block boundaries and the one it ends in are written back. Returns the
    /// logits after the last token and the tokens taken from the cache. The caller holds whatever
    /// offload the pass runs under.
    pub fn prefill(
        &self,
        model: &DeepseekV41Model,
        ids: &[u32],
        state: &mut DecodeState,
    ) -> Result<(Tensor, usize)> {
        let capture = state.capture_layers.clone();
        let mut from = 0;
        if let Some(r) = self.restore(model, ids) {
            tracing::info!(
                "state cache: restored {} of {} prompt tokens",
                r.covered,
                ids.len()
            );
            from = r.covered;
            *state = r.state;
            state.capture_layers = capture;
            if let (true, Some(l)) = (r.covered == ids.len(), r.logits) {
                let n = l.len();
                return Ok((Tensor::from_vec(l, (1, n), &Device::Cpu)?, from));
            }
        }
        let grid = self.grid_points(from, ids.len());
        let (out, windows) = if from == 0 {
            model.prefill_into_grid(ids, state, &grid)?
        } else {
            model.prefill_continue_grid(&ids[from..], state, &grid)?
        };
        let logits = out.flatten_all()?.to_vec1::<f32>()?;
        self.save(
            model,
            state,
            ids,
            Some(logits),
            grid.into_iter().zip(windows).collect(),
        );
        Ok((out, from))
    }

    /// The state a finished turn left, which stands after exactly `tokens`.
    pub fn save_turn(&self, model: &DeepseekV41Model, state: &DecodeState, tokens: &[u32]) {
        self.save(model, state, tokens, None, Vec::new());
    }

    /// Write `state`, which stands after `tokens`, and the grid states a prefill passed on the
    /// way. Encoding and writing run on their own thread over a copy, so the request is not held
    /// for them. A state shorter than one block is not worth a file.
    fn save(
        &self,
        model: &DeepseekV41Model,
        state: &DecodeState,
        tokens: &[u32],
        logits: Option<Vec<f32>>,
        grid: Vec<(usize, Vec<Vec<u8>>)>,
    ) {
        if state.pos != tokens.len() || tokens.len() < self.store.block_tokens() {
            return;
        }
        let this = self.clone();
        let (state, tokens) = (state.clone(), tokens.to_vec());
        let grid: Vec<_> = grid
            .into_iter()
            .map(|(p, w)| (p, w, model.engram_after(&tokens[..p])))
            .collect();
        std::thread::spawn(move || {
            let layout = this.layout();
            if let Err(why) = block_shape(&state, &layout) {
                tracing::info!("state cache: states are kept whole, not in blocks: {why}");
            }
            let keys = this.store.keys(&this.identity, &tokens);
            // The states the prefill passed, at block boundaries, so any prompt that shares a
            // prefix with this one restores at the last of them it shares.
            let mut kept = 0;
            for (p, windows, engram) in grid {
                let point = GridPoint {
                    pos: p,
                    windows: &windows,
                    engram,
                };
                let Some((manifest, nb)) =
                    encode_grid_manifest(&state, &point, &this.identity, &layout)
                else {
                    tracing::info!(
                        "state cache: no state kept at token {p}: the final state cannot be cut at that block boundary"
                    );
                    continue;
                };
                match this.store.put(&keys, p, &manifest, nb, |b| {
                    encode_block(&state, b, &layout)
                }) {
                    Ok(()) => kept += 1,
                    Err(e) => tracing::warn!("state cache: not stored at {p}: {e}"),
                }
            }
            if kept > 0 {
                tracing::info!("state cache: kept {kept} states on the block grid");
            }
            let (manifest, nblocks) =
                encode_manifest(&state, logits.as_deref(), &this.identity, Some(&layout));
            let put = this
                .store
                .put(&keys, tokens.len(), &manifest, nblocks, |b| {
                    encode_block(&state, b, &layout)
                });
            match put {
                Ok(()) => tracing::info!(
                    "state cache: stored a {}-token state ({} KiB manifest, {nblocks} blocks)",
                    tokens.len(),
                    manifest.len() >> 10
                ),
                Err(e) => tracing::warn!("state cache: not stored: {e}"),
            }
        });
    }
}
