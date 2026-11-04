//! Cutting a recurrent drafter back to the tokens its target kept.
//!
//! A drafter's KV cache is cut by length, but a recurrent state has advanced past every token the
//! drafter was fed and cannot be cut. A draft therefore keeps the state before its first token
//! and the tokens fed. When the target keeps a prefix of them, the state returns to the mark and
//! takes in only that prefix, so the drafter proposes from the sequence the target holds.

use crate::inference::engine::model_backend::ModelBackend;
use crate::tensor::{Device, Tensor};

/// What a recurrent drafter held before a draft, and what the draft fed it.
pub(crate) struct DraftMark {
    checkpoint: Vec<Option<crate::inference::generic_transformer::mixer::MixerState>>,
    /// Position of the first token fed.
    base: usize,
    /// The tokens fed, the one at `base` first.
    fed: Vec<u32>,
}

impl DraftMark {
    /// The mark of a drafter about to be fed from `base`, or nothing when it has no recurrent
    /// state or its state cannot be copied.
    pub(crate) fn take(model: &dyn ModelBackend, base: usize) -> Option<Self> {
        if !model.is_recurrent() {
            return None;
        }
        let checkpoint = model.recurrent_checkpoint().ok()?;
        Some(Self {
            checkpoint,
            base,
            fed: Vec::new(),
        })
    }

    /// The drafter was fed `token` at the next position.
    pub(crate) fn fed(&mut self, token: u32) {
        self.fed.push(token);
    }

    /// Cut the drafter back to `len` tokens: its state returns to the mark and takes in the fed
    /// tokens that lie before `len`. A `len` past the fed tokens leaves the state as it is.
    pub(crate) fn rewind(
        self,
        model: &mut dyn ModelBackend,
        device: &Device,
        len: usize,
    ) -> crate::tensor::Result<()> {
        if len < self.base || len >= self.base + self.fed.len() {
            model.trim_kv(len);
            return Ok(());
        }
        model.recurrent_rewind(&self.checkpoint)?;
        model.trim_kv(self.base);
        for (i, token) in self.fed[..len - self.base].iter().enumerate() {
            let x = Tensor::new(&[*token], device)?.unsqueeze(0)?;
            model.forward(&x, self.base + i)?;
        }
        Ok(())
    }
}
