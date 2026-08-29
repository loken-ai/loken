//! The block an encoder-decoder stacks, once.
//!
//! Whisper and Parler are different models - one transcribes speech, the other speaks - but
//! their decoders are the same layer repeated: attend to what has been said so far, attend to
//! what the encoder produced, widen and narrow through two projections, adding each result back
//! to a normalised copy of the input. Both were converted to the same checkpoint convention, so
//! the names are the same too, and the block can read its own weights.
//!
//! What the two do differ in is stated as arguments: whether there is an encoder to attend to,
//! whether the feed-forward projections carry a bias, and which activation sits between them.
//! The attentions themselves come in ready-made, because how a family names and shapes its four
//! projections is the one thing that is genuinely its own.

use crate::inference::model::attention::{Mask, MultiHeadAttention};
use crate::tensor::layer::{layer_norm, linear, linear_no_bias, LayerNorm, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::{Result, Tensor, VarBuilder};

/// Self attention, optional cross attention, and a feed-forward - each added back.
#[derive(Debug, Clone)]
pub struct CrossAttentionBlock {
    self_attn: MultiHeadAttention,
    self_norm: LayerNorm,
    cross: Option<(MultiHeadAttention, LayerNorm)>,
    fc1: Linear,
    fc2: Linear,
    ffn_norm: LayerNorm,
    activation: Activation,
}

impl CrossAttentionBlock {
    /// The three normalisations and the feed-forward, read from `vb`; the attentions are given.
    ///
    /// `cross_attn` is `None` in an encoder, where there is nothing else to attend to - and its
    /// normalisation is then not read either, because the checkpoint does not carry one.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        self_attn: MultiHeadAttention,
        cross_attn: Option<MultiHeadAttention>,
        hidden: usize,
        ffn: usize,
        eps: f32,
        bias: bool,
        activation: Activation,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let project = |from, to, vb: &VarBuilder| match bias {
            true => linear(from, to, vb),
            false => linear_no_bias(from, to, vb),
        };
        let cross = match cross_attn {
            Some(attn) => Some((
                attn,
                layer_norm(hidden, eps, &vb.pp("encoder_attn_layer_norm"))?,
            )),
            None => None,
        };
        Ok(Self {
            self_attn,
            self_norm: layer_norm(hidden, eps, &vb.pp("self_attn_layer_norm"))?,
            cross,
            fc1: project(hidden, ffn, &vb.pp("fc1"))?,
            fc2: project(ffn, hidden, &vb.pp("fc2"))?,
            ffn_norm: layer_norm(hidden, eps, &vb.pp("final_layer_norm"))?,
            activation,
        })
    }

    /// `mask` governs the self attention only: what a query may see of its own past is a
    /// property of the model, while an encoder's output is entirely visible by construction.
    ///
    /// `encoder_xs` is ignored when the block has no cross attention, and a block that has one
    /// and is handed nothing attends to whatever it projected on an earlier call - which is the
    /// point of holding it.
    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        mask: Mask,
    ) -> Result<Tensor> {
        let attended = self.self_attn.forward(&self.self_norm.forward(xs)?, None, mask)?;
        let mut xs = xs.add(&attended)?;
        if let Some((attn, norm)) = &mut self.cross {
            let crossed = attn.forward(&norm.forward(&xs)?, encoder_xs, Mask::All)?;
            xs = xs.add(&crossed)?;
        }
        let widened = self.ffn_norm.forward(&xs)?.apply(&self.fc1)?;
        let narrowed = self.activation.apply(&widened)?.apply(&self.fc2)?;
        xs.add(&narrowed)
    }

    /// Forget both caches - the next call starts a new sequence.
    pub fn clear(&mut self) {
        self.self_attn.clear();
        if let Some((attn, _)) = &mut self.cross {
            attn.clear();
        }
    }
}
