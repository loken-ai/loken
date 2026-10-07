//! The routed-expert cache on a card needs a card: a build without one has the type the layers
//! hold, never attached, and the figures the plan reads.

use super::moe_weights::MoeWeights;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

pub use super::moe_weights::prefill_bytes_per_token;

/// A layer's cache of its experts on a card: none is ever built here.
pub struct LayerTier;

impl LayerTier {
    pub(super) fn decode(
        self: &Arc<Self>,
        _: &MoeWeights,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub(super) fn decode_rows(
        self: &Arc<Self>,
        _: &MoeWeights,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub fn stream_pays(&self, _rows: usize, _used: usize) -> bool {
        false
    }

    pub fn is_resident(&self) -> bool {
        false
    }

    pub(super) fn card_stacks(&self) -> &[crate::tensor::quantized::QTensor; 3] {
        unreachable!("an expert cache lives on a card")
    }

    pub fn prefetch(&self) -> Result<()> {
        Ok(())
    }

    pub(super) fn stacks(&self) -> Result<&[crate::tensor::quantized::QTensor; 3]> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub(super) fn release_stacks(&self) -> Result<()> {
        Ok(())
    }
}

/// Whether a file's routed experts can have a cache on a card: never without one.
pub fn header_serves(_: &crate::tensor::quantized::gguf_file::Content) -> bool {
    false
}

impl super::GenericHeteroTransformer {
    /// No card holds a cache of the experts: nothing to attach.
    pub(super) fn attach_expert_tiers(
        &mut self,
        _: &std::collections::HashMap<usize, Device>,
        _: &crate::inference::place::layer_executor::HeteroPlan,
        _: &std::collections::HashMap<usize, u64>,
    ) -> Result<()> {
        Ok(())
    }
}
