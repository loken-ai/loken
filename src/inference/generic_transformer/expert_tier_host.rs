//! The routed-expert cache on a card needs a card: a build without one has the type the layers
//! hold, never attached, and the figures the plan reads.

use super::moe_weights::MoeWeights;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

pub use super::moe_weights::prefill_bytes_per_token;

/// A layer's cache of its experts on a card: none is ever built here.
pub struct LayerTier;

/// A shared expert's projections queued in a recorded decode: none without a card.
pub(crate) struct GraphPending;

/// The thread serving a card's recorded decode: none without a card.
pub struct TierWorker;

impl TierWorker {
    pub fn check(&self) -> Result<()> {
        Ok(())
    }

    pub fn failure(&self) -> Option<String> {
        None
    }
}

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

    pub(super) fn keep_hottest_of_prompt(&self, _: &Tensor, _: &()) -> Result<()> {
        Ok(())
    }

    pub(crate) fn rows_through_cache(&self, _rows: usize, _used: usize) -> bool {
        false
    }

    pub(crate) fn graph_begin(
        &self,
        _: &MoeWeights,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<GraphPending> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub(crate) fn graph_end(&self, _: GraphPending) -> Result<Tensor> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub fn mailbox_state(&self) -> Option<String> {
        None
    }

    pub(super) fn graph_ready(&self) -> bool {
        false
    }

    pub(crate) fn records(&self, _: &Device) -> bool {
        false
    }

    pub(crate) fn graph_rows(&self) -> usize {
        0
    }

    pub(crate) fn decode_rows_graph(
        self: &Arc<Self>,
        _: &MoeWeights,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor> {
        crate::tensor::bail!("an expert cache lives on a card")
    }

    pub fn counters(&self) -> (u64, u64, u64) {
        (0, 0, 0)
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

    pub fn hold_stacks(&self, _on: bool) -> Result<()> {
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
        _: bool,
    ) -> Result<()> {
        Ok(())
    }

    /// No card holds a cache of the experts: no slots to lend a prefill.
    pub fn lend_expert_cache(&mut self, _tokens: usize) -> usize {
        0
    }

    /// No card holds a cache of the experts: nothing lent to return.
    pub fn return_expert_cache(&mut self) -> Result<usize> {
        Ok(0)
    }
}
