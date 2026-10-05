//! The projector between a speech encoder and the language model.
//!
//! Consecutive encoder frames are stacked along the feature axis (the last group padded
//! with zeros), which divides the sequence length by the stack factor, and a small network
//! maps each stacked row into the language model's embedding space.

use super::encoder::linear;
use super::source::TensorReader;
use crate::inference::model::mmproj::MmprojAudio;
use crate::inference::model_facts::AudioProjector;
use crate::tensor::{Error, Result, Tensor};

/// RMS normalisation epsilon of the gated projector's two norms.
const GATED_NORM_EPS: f32 = 1e-6;

enum Net {
    Mlp {
        up: Tensor,
        down: Tensor,
    },
    Gated {
        pre: Tensor,
        up: Tensor,
        mid: Tensor,
        down: Tensor,
    },
}

pub struct Projector {
    net: Net,
    stack: usize,
}

impl Projector {
    pub(crate) fn load(
        r: &TensorReader<'_>,
        geo: &MmprojAudio,
        kind: AudioProjector,
    ) -> Result<Self> {
        let stacked = geo.width * geo.stack;
        let net = match kind {
            AudioProjector::GeluMlp => Net::Mlp {
                up: r.matrix("mm.a.mlp.1.weight", geo.projection, stacked)?,
                down: r.matrix("mm.a.mlp.2.weight", geo.projection, geo.projection)?,
            },
            AudioProjector::GatedMlp => {
                let inner = r
                    .shape_of("mm.a.mlp.1.weight")
                    .and_then(|d| d.first().copied())
                    .ok_or_else(|| Error("mm.a.mlp.1.weight: missing".to_string()))?;
                if inner % 2 != 0 {
                    return Err(Error(format!("the gated projector's width {inner} is odd")));
                }
                Net::Gated {
                    pre: r.vector("mm.a.norm_pre.weight")?,
                    up: r.matrix("mm.a.mlp.1.weight", inner, stacked)?,
                    mid: r.vector("mm.a.norm_mid.weight")?,
                    down: r.matrix("mm.a.mlp.2.weight", geo.projection, inner / 2)?,
                }
            }
        };
        Ok(Self {
            net,
            stack: geo.stack,
        })
    }

    /// Rows the projector yields for `frames` encoder frames.
    pub fn rows_for(&self, frames: usize) -> usize {
        frames.div_ceil(self.stack)
    }

    /// `frames [F, width]` to embeddings `[ceil(F / stack), projection]`.
    pub fn forward(&self, frames: &Tensor) -> Result<Tensor> {
        let (f, width) = frames.dims2()?;
        let padded = f.div_ceil(self.stack) * self.stack;
        let frames = if padded == f {
            frames.clone()
        } else {
            frames.pad_with_zeros(0, 0, padded - f)?
        };
        let x = frames.reshape((padded / self.stack, width * self.stack))?;
        match &self.net {
            Net::Mlp { up, down } => linear(&linear(&x, up, None)?.gelu_erf()?, down, None),
            Net::Gated { pre, up, mid, down } => {
                let x = x.rms_norm(pre, GATED_NORM_EPS)?;
                let x = linear(&x, up, None)?;
                let half = x.dim(1)? / 2;
                let value = x.narrow(1, 0, half)?;
                let gate = x.narrow(1, half, half)?;
                let x = gate.silu()?.mul(&value)?;
                linear(&x.rms_norm(mid, GATED_NORM_EPS)?, down, None)
            }
        }
    }
}
