//! What the cluster's planner reads of a Sinkhorn wide-residual model.
//!
//! A node prices a routed expert by timing 1 expert of the model's shape and formats on its cores.
//! The planner places layers across nodes from that time, each node's time for a layer and the
//! bytes the model gives here.
//! The bytes are each held layer's token mixer with its output projection as stored, 1 routed
//! expert as stored, the experts the prior names per layer, and the experts a token routes to.
//! The model also gives the bytes its token mixers, routed experts and always-read weights hold
//! on each card.

use super::mixer::Mixer;
use super::wide_sparse_layer::SinkhornBlock;
use super::GenericHeteroTransformer;
use crate::tensor::quantized::GgmlDType;

impl GenericHeteroTransformer {
    /// The Sinkhorn blocks of the held layers, in layer order.
    ///
    /// Empty for a model of another kind.
    fn wide_blocks(&self) -> impl Iterator<Item = (&Mixer, &SinkhornBlock)> {
        let wide = self.wide_sparse.is_some();
        self.layers
            .iter()
            .filter(move |_| wide)
            .filter_map(|l| l.sinkhorn.as_ref().map(|b| (&l.mixer, b)))
    }

    /// The shape a node times 1 missed routed expert against: `(dim, inter, gate_up_dtype,
    /// down_dtype)`.
    ///
    /// Read from expert 0 of the first held layer.
    /// `None` for a model of another kind, or for experts not stored in a GGUF format.
    pub fn wide_expert_cost_shape(&self) -> Option<(usize, usize, GgmlDType, GgmlDType)> {
        let (_, block) = self.wide_blocks().next()?;
        let expert = block.ffn.experts.first()?;
        let gate_dt = expert.gate.quant_dtype()?;
        let down_dt = expert.down.quant_dtype()?;
        // The gate is `[inter, dim]`.
        let dims = expert.gate.dims();
        Some((*dims.get(1)?, *dims.first()?, gate_dt, down_dt))
    }

    /// The inputs the cluster's cut planner places this model with: `(mixer_bytes_per_layer,
    /// expert_bytes, working_set, activated)`.
    ///
    /// `mixer_bytes_per_layer` is, per held layer, its token mixer and output projection as
    /// stored.
    /// `expert_bytes` is expert 0 of the first held layer as stored.
    /// `working_set` is the mean length of the prior's non-empty lists.
    /// `activated` is the experts a token routes to per layer.
    /// `None` for a model of another kind, a mixer without a stored size, experts of no bytes, or
    /// a model without a prior.
    pub fn wide_placement_inputs(&self) -> Option<(Vec<u64>, u64, u64, u64)> {
        let mixer_bytes = self
            .wide_blocks()
            .map(|(mixer, block)| {
                let mixer = match mixer {
                    Mixer::LatentMqa(m) => m.stored_bytes(),
                    Mixer::CompressedBand(b) => b.stored_bytes(),
                    _ => return None,
                };
                Some((mixer + block.output.bytes()) as u64)
            })
            .collect::<Option<Vec<u64>>>()?;
        if mixer_bytes.is_empty() {
            return None;
        }
        let (_, block) = self.wide_blocks().next()?;
        let expert = block.ffn.experts.first()?;
        let expert_bytes = (expert.gate.bytes() + expert.up.bytes() + expert.down.bytes()) as u64;
        if expert_bytes == 0 {
            return None;
        }
        let lens: Vec<usize> = self
            .routing_prior()
            .iter()
            .map(Vec::len)
            .filter(|&l| l > 0)
            .collect();
        if lens.is_empty() {
            return None;
        }
        let working_set = (lens.iter().sum::<usize>() / lens.len()) as u64;
        Some((
            mixer_bytes,
            expert_bytes,
            working_set,
            self.routed_per_token() as u64,
        ))
    }

    /// Per card, the bytes the token mixers, the routed experts and the weights every token
    /// reads held there take, by ordinal.
    ///
    /// `None` for a model of another kind, or one that holds nothing on a card.
    pub fn wide_card_residency(&self) -> Option<Vec<(usize, u64)>> {
        let mut cards: Vec<(usize, u64)> = Vec::new();
        let mut add =
            |ordinal: usize, bytes: u64| match cards.iter_mut().find(|(o, _)| *o == ordinal) {
                Some((_, held)) => *held += bytes,
                None => cards.push((ordinal, bytes)),
            };
        for (mixer, block) in self.wide_blocks() {
            if let Some((ordinal, bytes)) = held_on_card(mixer) {
                add(ordinal, bytes);
            }
            for (ordinal, bytes) in block.ffn.card.bytes_by_card() {
                add(ordinal, bytes);
            }
            if let Some((ordinal, bytes)) =
                block.engram.as_ref().and_then(|e| e.block.wkv_on_card())
            {
                add(ordinal, bytes as u64);
            }
            if let Some(ordinal) = block.card_mix.ordinal() {
                add(ordinal, block.card_mix.bytes() as u64);
            }
        }
        if let Some(ws) = self.wide_sparse.as_ref() {
            if let Some(ordinal) = ws.card_head.ordinal() {
                add(ordinal, ws.card_head.bytes() as u64);
            }
        }
        cards.sort_unstable();
        (!cards.is_empty()).then_some(cards)
    }
}

/// The card a token mixer runs on and the bytes of weights it holds there.
///
/// `None` for a mixer on the host.
#[cfg(feature = "gpu")]
fn held_on_card(mixer: &Mixer) -> Option<(usize, u64)> {
    match mixer {
        Mixer::LatentMqa(m) => Some((m.card_ordinal()?, m.card_bytes() as u64)),
        Mixer::CompressedBand(b) => Some((b.card_ordinal()?, b.card_bytes() as u64)),
        _ => None,
    }
}

/// A build without cards runs every token mixer on the host.
#[cfg(not(feature = "gpu"))]
fn held_on_card(_: &Mixer) -> Option<(usize, u64)> {
    None
}
