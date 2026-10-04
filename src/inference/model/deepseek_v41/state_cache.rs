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

#[derive(Clone)]
pub struct StateCache {
    store: Arc<StateStore>,
    identity: Identity,
    ratios: Vec<usize>,
    /// Tokens per block of this model's chain, settled in the store.
    block: usize,
    last_stride: Arc<std::sync::atomic::AtomicUsize>,
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
        token_bytes: usize,
    ) -> Option<Self> {
        // A block holds whole groups of every layer: a multiple of every ratio.
        let granule = model
            .compress_ratios()
            .iter()
            .filter(|&&r| r != 0)
            .fold(1usize, |a, &r| a / gcd(a, r) * r);
        let Some(block) = crate::inference::cache::state_tune::settle_block_tokens(
            &store,
            &identity,
            granule,
            token_bytes,
        ) else {
            tracing::warn!("state cache off: the directory cannot be written to measure it");
            return None;
        };
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
            block,
            last_stride: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            // The ratios of the layers a state has: the model lists the draft's after them.
            ratios: model.compress_ratios()[..model.n_layers().min(model.compress_ratios().len())]
                .to_vec(),
        })
    }

    fn layout(&self) -> BlockLayout<'_> {
        BlockLayout {
            ratios: &self.ratios,
            block_tokens: self.block,
        }
    }

    /// The longest stored state that is a prefix of `ids`. A state that is the whole of `ids` but
    /// holds no logits cannot answer and is passed over; a file that cannot be read or decoded is
    /// logged and dropped.
    fn restore(&self, model: &DeepseekV41Model, ids: &[u32]) -> Option<Restored> {
        let keys = self.store.keys(&self.identity, ids);
        for entry in self.store.candidates(&keys, ids.len()) {
            let bytes = self.store.entry_bytes(&entry);
            let m = self.store.tuning().get();
            if !crate::inference::cache::state_tune::restore_pays(
                entry.len,
                bytes,
                m.restore.secs_per(),
                m.prefill.secs_per(),
            ) {
                tracing::info!(
                    "state cache: a {}-token state is not restored: reading {} MiB is expected to cost more than prefilling the tokens",
                    entry.len,
                    bytes >> 20
                );
                continue;
            }
            let started = std::time::Instant::now();
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
                    self.store
                        .tuning()
                        .restored(bytes, started.elapsed().as_secs_f64());
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

    /// The token counts inside `(from, to)` at which a state is kept: multiples of the stride the
    /// store's measurements give, which are block boundaries.
    fn grid_points(&self, from: usize, to: usize) -> Vec<usize> {
        let m = self.store.tuning().get();
        let s = crate::inference::cache::state_tune::grid_stride(
            self.block,
            self.block,
            m.manifest_bytes,
            m.write.per_sec(),
            m.prefill.secs_per(),
        );
        if self
            .last_stride
            .swap(s, std::sync::atomic::Ordering::Relaxed)
            != s
        {
            tracing::info!(
                "state cache: a state every {s} tokens (block {}, manifest {} KiB, write {} MB/s, prefill {} tokens/s)",
                self.block,
                m.manifest_bytes >> 10,
                m.write.per_sec().map_or(0.0, |w| w / 1e6) as u64,
                m.prefill.per_sec().unwrap_or(0.0) as u64
            );
        }
        (from / s + 1..)
            .map(|k| k * s)
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
        let started = std::time::Instant::now();
        let (out, windows) = if from == 0 {
            model.prefill_into_grid(ids, state, &grid)?
        } else {
            model.prefill_continue_grid(&ids[from..], state, &grid)?
        };
        let secs = started.elapsed().as_secs_f64();
        self.store.tuning().prefilled(ids.len() - from, secs);
        let logits = out.flatten_all()?.to_vec1::<f32>()?;
        self.save(
            model,
            state,
            ids,
            Some(logits),
            grid.into_iter().zip(windows).collect(),
            Some(secs),
        );
        Ok((out, from))
    }

    /// The state a finished turn left, which stands after exactly `tokens`.
    pub fn save_turn(&self, model: &DeepseekV41Model, state: &DecodeState, tokens: &[u32]) {
        self.save(model, state, tokens, None, Vec::new(), None);
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
        prefill_secs: Option<f64>,
    ) {
        if state.pos != tokens.len() || tokens.len() < self.block {
            return;
        }
        let m = self.store.tuning().get();
        let estimate = m.largest_state.max(m.manifest_bytes) * (grid.len() as u64 + 1);
        let Some(ticket) = self.store.admit(estimate, prefill_secs) else {
            tracing::info!("state cache: a write is skipped, the disk is still behind");
            return;
        };
        let this = self.clone();
        let (state, tokens) = (state.clone(), tokens.to_vec());
        let grid: Vec<_> = grid
            .into_iter()
            .map(|(p, w)| (p, w, model.engram_after(&tokens[..p])))
            .collect();
        std::thread::spawn(move || {
            ticket.run(|| {
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
        })
        });
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
