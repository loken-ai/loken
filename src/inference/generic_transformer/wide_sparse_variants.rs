//! The file each routed expert of a Sinkhorn wide-residual model is read from, when its
//! checkpoint stores the routed experts in several files, each in its own format.
//!
//! A further file is checked to be of the model's checkpoint before any expert is read from it.
//! `place::expert_variants::plan` chooses each expert's file under the policy the fidelity sets.
//! It fills the tiers that keep experts in the order of the routing heat.
//! An expert no tier keeps is read from its file at each use.

use super::wide_sparse_names::ENGRAM_TABLE;
use super::GenericHeteroTransformer;
use crate::inference::load::variant_files::check_variant;
use crate::inference::place::expert_variants::{plan, Tier, VariantChoice, VariantPlan};
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::{Error, Result};
use std::path::{Path, PathBuf};

/// Whether the tensor `name` is a table the layers read by rows: an engram table, per layer or
/// in the 1 table llama.cpp's layout holds every engram layer's rows in.
fn read_by_rows(name: &str) -> bool {
    name.contains("engram_embd") || name == ENGRAM_TABLE
}

/// The further files `blobs` of the checkpoint `model` is read from, each opened and checked
/// to be of it before any expert is read from it.
///
/// A file that is not of the checkpoint refuses them all, naming the file and `model_id`.
pub fn open_checked(model: &Path, blobs: &[PathBuf], model_id: &str) -> Result<Vec<SplitGguf>> {
    let primary = SplitGguf::open(model)
        .map_err(|e| Error::msg(format!("{}: open parts: {e}", model.display())))?;
    blobs
        .iter()
        .map(|blob| {
            let started = std::time::Instant::now();
            let variant = SplitGguf::open(blob)
                .map_err(|e| Error::msg(format!("{}: {e}", blob.display())))?;
            check_variant(&primary, &variant, &read_by_rows).map_err(|e| {
                Error::msg(format!("{} cannot serve {model_id}: {e}", blob.display()))
            })?;
            tracing::info!(
                "expert variant {} checked against the model's file in {:.1}s",
                blob.display(),
                started.elapsed().as_secs_f32()
            );
            Ok(variant)
        })
        .collect()
}

impl GenericHeteroTransformer {
    /// The most files any held layer's routed experts are stored in, 1 without variants.
    pub fn expert_variants(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| b.ffn.variants.stored.len())
            .max()
            .unwrap_or(0)
            .max(1)
    }

    /// Per held layer, the variant each routed expert is read from.
    pub fn expert_variant_choice(&self) -> Vec<Vec<usize>> {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| match b.ffn.variants.chosen.is_empty() {
                true => vec![0; b.ffn.experts.len()],
                false => b.ffn.variants.chosen.clone(),
            })
            .collect()
    }

    /// The tiers that keep the routed experts of a wide layer, in the order a plan fills them.
    ///
    /// 1 tier per card with a room for experts, cards in ordinal order.
    /// A card computes an expert at the same speed whatever its variant: it takes the precise one.
    /// The host's cores run the other experts from their mapping, with no tier of their own.
    pub(super) fn wide_expert_tiers(&self) -> Vec<Tier> {
        self.wide_sparse
            .as_ref()
            .map_or(&[][..], |ws| ws.card_rooms.as_slice())
            .iter()
            .map(|&(_, bytes)| Tier {
                bytes,
                precise: true,
            })
            .collect()
    }

    /// The plan of the routed experts over the tiers that keep them, by `choice` and the heat.
    ///
    /// A variant a layer lacks counts as too large for any tier.
    /// `None` for a model without a Sinkhorn wide layer.
    pub(super) fn plan_wide_experts(&self, choice: VariantChoice) -> Option<VariantPlan> {
        let variants = self.expert_variants();
        let held: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.layers[i].sinkhorn.is_some())
            .collect();
        if held.is_empty() {
            return None;
        }
        let ffn = |i: usize| self.layers[i].sinkhorn.as_ref().map(|b| &b.ffn);
        let experts = held
            .iter()
            .filter_map(|&i| ffn(i))
            .map(|f| f.experts.len())
            .max()
            .unwrap_or(0);
        let sizes: Vec<Vec<Vec<u64>>> = held
            .iter()
            .map(|&i| {
                (0..experts)
                    .map(|e| {
                        (0..variants)
                            .map(|v| {
                                ffn(i)
                                    .and_then(|f| f.variant_bytes(e, v))
                                    .unwrap_or(u64::MAX)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let ranks = self.held_heat();
        let tiers = self.wide_expert_tiers();
        Some(plan(
            held.len(),
            experts,
            variants,
            &|l, e, v| sizes[l][e][v],
            &ranks,
            &tiers,
            choice,
        ))
    }

    /// Per held wide layer, its experts ranked by the heat.
    pub(super) fn held_heat(&self) -> Vec<Vec<usize>> {
        let heat = self.routing_heat();
        (0..self.layers.len())
            .filter(|&i| self.layers[i].sinkhorn.is_some())
            .map(|i| heat.get(self.first_layer + i).cloned().unwrap_or_default())
            .collect()
    }

    /// Chooses the variant each routed expert is read from, by `choice` over the heat and the
    /// tiers that keep the experts.
    ///
    /// A variant a layer lacks counts as too large for any tier.
    /// `None` when the experts are stored in 1 file.
    pub fn place_expert_variants(&mut self, choice: VariantChoice) -> Option<VariantPlan> {
        let variants = self.expert_variants();
        if variants < 2 {
            return None;
        }
        let p = self.plan_wide_experts(choice)?;
        let held: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.layers[i].sinkhorn.is_some())
            .collect();
        for (l, &i) in held.iter().enumerate() {
            if let Some(block) = self.layers[i].sinkhorn.as_mut() {
                for (e, &v) in p.chosen[l].iter().enumerate() {
                    block.ffn.choose_variant(e, v);
                }
            }
        }
        tracing::info!(
            "expert variants ({choice:?}): {:?} experts read from each of {variants} files",
            p.per_variant(variants)
        );
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engram_table_is_compared_by_rows() {
        assert!(read_by_rows("blk.1.engram_embd.weight"));
        assert!(read_by_rows(ENGRAM_TABLE));
        assert!(!read_by_rows("token_embd.weight"));
        assert!(!read_by_rows("blk.0.attn_q_b.weight"));
    }
}
