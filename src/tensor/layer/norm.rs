//! Normalisation: RMS, layer and group - plus the query/key pair an attention block wants
//! as one thing rather than two fields.
//!
//! Each is a weight, an epsilon and a call into the op that does the work; the arithmetic
//! lives in [`ops`](super::ops), not here. What these add is where the weight came from and
//! at what width the epsilon is spoken: checkpoints publish it as f64 and the kernels take
//! f32, so the conversion happens once, at the boundary, rather than at every call site.

use super::*;

/// The per-head query/key normalisation a DiT applies before attention.
///
/// Two `RmsNorm`s and nothing else - but it existed four times over, once per weight loader,
/// because each port wrote its own constructor and carried a struct along with it. This one
/// takes the norms already built, so it belongs to no loader.
#[derive(Clone, Debug)]
pub struct QkNorm {
    pub query: RmsNorm,
    pub key: RmsNorm,
}

impl QkNorm {
    pub fn new(query: RmsNorm, key: RmsNorm) -> Self {
        Self { query, key }
    }

    /// `q` and `k` are `(.., heads, head_dim)`; each is normalised over its last axis.
    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.query.forward(q)?, self.key.forward(k)?))
    }
}

/// RMS norm layer (weight over the last dim).
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    /// The weight, for callers that read it back - a config layer that reports what a norm
    /// holds, or a placement walk that charges it.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// The epsilon, as `f64`: it is read back to be reported and compared, and the checkpoints
    /// that publish it do so at that width.
    pub fn eps(&self) -> f64 {
        self.eps as f64
    }

    /// Built from a plain tensor, with the epsilon at the width checkpoints publish it in.
    pub fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self::new(weight, eps as f32)
    }

    /// Built from a quantised weight, dequantised on the device that holds it.
    ///
    /// A norm is a vector per layer against a matrix per projection, so keeping it in blocks
    /// saves nothing and costs precision on every element that passes through.
    pub fn from_qtensor(weight: crate::tensor::quantized::QTensor, eps: f64) -> Result<Self> {
        let dev = weight.device();
        Ok(Self::new(weight.dequantize(dev)?, eps as f32))
    }
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.rms_norm(&self.weight, self.eps)
    }

    /// Move the scale weight onto `dev` (CPU↔CUDA).
    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(dev)?,
            eps: self.eps,
        })
    }

    /// Copy on `dev` converted to `dtype`.
    pub fn to_dtype_on(&self, dev: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(dev)?.to_dtype(dtype)?,
            eps: self.eps,
        })
    }
}

pub fn rms_norm(size: usize, eps: f32, vb: &VarBuilder) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vb.get(size, "weight")?, eps))
}

/// LayerNorm layer (affine, mean-subtracting).
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f32,
}

impl LayerNorm {
    /// A norm the checkpoint publishes without a bias.
    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self::new(weight, None, eps as f32)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// The epsilon, at the width checkpoints publish it in.
    pub fn eps(&self) -> f64 {
        self.eps as f64
    }

    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.layer_norm(&self.weight, self.bias.as_ref(), self.eps)
    }
}

pub fn layer_norm(size: usize, eps: f32, vb: &VarBuilder) -> Result<LayerNorm> {
    let w = vb.get(size, "weight")?;
    let b = vb.get(size, "bias")?;
    Ok(LayerNorm::new(w, Some(b), eps))
}

/// The norm a checkpoint publishes no parameters for: unit weight, no bias - the
/// normalisation alone, with the scale and shift left to whatever follows it.
///
/// There is nothing to load, so the weight is synthesised, and `device` says where it
/// is made: on the device the forward runs it rides along, on the host it makes the
/// input come and meet it.
pub fn layer_norm_no_affine(size: usize, eps: f32, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        Tensor::ones(size, DType::F32, device)?,
        None,
        eps,
    ))
}

/// GroupNorm layer (per-(batch,group) normalize + per-channel affine).
#[derive(Clone, Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn new(weight: Tensor, bias: Tensor, num_groups: usize, eps: f32) -> Self {
        Self {
            weight,
            bias,
            num_groups,
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.group_norm(self.num_groups, &self.weight, &self.bias, self.eps)
    }
}

pub fn group_norm(
    num_groups: usize,
    channels: usize,
    eps: f32,
    vb: &VarBuilder,
) -> Result<GroupNorm> {
    let w = vb.get(channels, "weight")?;
    let b = vb.get(channels, "bias")?;
    Ok(GroupNorm::new(w, b, num_groups, eps))
}

// `Module` is how a caller holds a layer without knowing which one it is. Each impl forwards
// to the type's own `forward`, which is the same four lines whichever type it is.
macro_rules! callable {
    ($($ty:ty),+ $(,)?) => {$(
        impl crate::tensor::Module for $ty {
            fn forward(&self, xs: &Tensor) -> Result<Tensor> {
                self.forward(xs)
            }
        }
    )+};
}

callable!(RmsNorm, LayerNorm, GroupNorm);
