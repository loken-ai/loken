//! Projections: the one `Linear` every model uses, in either weight orientation, and the
//! LoRA deltas fused into it.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Which orientation the weight is held in.
///
/// Both reach the same GEMM: `[in, out]` is multiplied directly, and `[out, in]` goes through
/// cuBLAS's N-T flag, which the kernel honours by reading the operand differently rather than
/// by transposing it. What decides the choice is the loader. A path that can stage the
/// transpose host-side hands over the first. A path holding a device-resident half-precision
/// weight hands over the second, because turning it on the device materialises an F32 cast and
/// a transposed copy per tensor, and those transients stay in the mempool.
#[derive(Clone, Debug)]
enum Held {
    Transposed(Tensor),
    Published(Tensor),
}

impl Held {
    fn tensor(&self) -> &Tensor {
        match self {
            Held::Transposed(t) | Held::Published(t) => t,
        }
    }

    fn map(&self, f: impl FnOnce(&Tensor) -> Result<Tensor>) -> Result<Self> {
        Ok(match self {
            Held::Transposed(t) => Held::Transposed(f(t)?),
            Held::Published(t) => Held::Published(f(t)?),
        })
    }
}

/// Affine projection, in either weight orientation (see `Held`).
#[derive(Clone, Debug)]
pub struct Linear {
    held: Held,
    bias: Option<Tensor>,
    in_dim: usize,
    out_dim: usize,
    /// The published `[out, in]` orientation, built on first request when the weight is not
    /// already held that way.
    ///
    /// The forward wants `[in, out]` and that is what is stored, because transposing is a copy
    /// on this substrate rather than a view - doing it per call is what made the compat Linear
    /// copy its whole matrix every pass. But a few callers hand the weight to a kernel that
    /// reads the published orientation and DECLINES when it does not match, falling back to
    /// the generic path without saying so. Those get the original, once, and only they pay for
    /// it: a MoE gate is `[experts, hidden]`, small beside the projections around it.
    weight_orig: std::sync::Arc<std::sync::OnceLock<Tensor>>,
    /// Low-rank adapters applied ON TOP of `wt`, never merged into it.
    ///
    /// A LoRA is a rank-r correction `B*A` to a projection. Merging it would mean
    /// rewriting the base weight - which is impossible without loss once that weight is
    /// quantised, since it would have to be dequantised, added to, and requantised. Kept
    /// as a separate term, the base stays untouched, several adapters compose by simple
    /// addition, and unloading one is dropping it from this list rather than restoring a
    /// checkpoint.
    ///
    /// Cost is negligible: r is 8-128 against dimensions in the thousands, so the two
    /// extra matmuls are a low-percent tax on the projection.
    lora: Vec<LoraDelta>,
    /// The attached adapters fused into one `([1, in, R], [1, R, out])` pair.
    ///
    /// N adapters of ranks `r_i` are mathematically ONE adapter of rank `sum(r_i)`:
    /// stacking the `down` blocks along the rank axis and the `up` blocks along the same
    /// axis makes `(x @ down_cat) @ up_cat` produce exactly the sum of the individual
    /// corrections. Fusing at attach time keeps the forward at two matmuls however many
    /// adapters are stacked, folds each scale into its `up` block so the hot path has no
    /// scaling pass, and pre-shapes to 3-D so it has no reshape either - the correction
    /// runs per denoising step, on every projection, so those passes are not free.
    fused: Option<(Tensor, Tensor)>,
}

impl Linear {
    /// The weight as the checkpoint published it, `[out, in]`.
    ///
    /// Built from the stored transpose on first call and kept, so a caller in a per-token path
    /// does not pay for it twice.
    pub fn weight(&self) -> Result<&Tensor> {
        if let Held::Published(w) = &self.held {
            return Ok(w);
        }
        if let Some(w) = self.weight_orig.get() {
            return Ok(w);
        }
        let _ = self.weight_orig.set(self.held.tensor().t()?);
        Ok(self.weight_orig.get().expect("just set"))
    }

    /// `weight` in the file layout `[out, in]`.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (out_dim, in_dim) = weight.shape().dims2()?;
        let wt = weight.transpose(0, 1)?;
        Ok(Self {
            held: Held::Transposed(wt),
            bias,
            weight_orig: std::sync::Arc::new(std::sync::OnceLock::new()),
            in_dim,
            out_dim,
            lora: Vec::new(),
            fused: None,
        })
    }

    /// Weight ALREADY transposed (`[in, out]`). The big-checkpoint load path
    /// stages the transpose host-side and uploads the final blob: `new` on a
    /// device-resident half-precision weight materializes an F32 cast + a
    /// transposed copy per tensor (transpose routes through the f32 kernels),
    /// and those freed transients stay in the CUDA mempool - for a 12 GB
    /// model that retention starved the device (z-image 1024² OOM).
    pub fn from_transposed(wt: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (in_dim, out_dim) = wt.shape().dims2()?;
        Ok(Self {
            held: Held::Transposed(wt),
            bias,
            weight_orig: std::sync::Arc::new(std::sync::OnceLock::new()),
            in_dim,
            out_dim,
            lora: Vec::new(),
            fused: None,
        })
    }

    /// Move both weight and (optional) bias onto `dev` (CPU↔CUDA), preserving the
    /// already-transposed layout. Mirrors the Wan-VAE per-component `to_device` so a
    /// loaded encoder can be relocated to the planner's chosen card.
    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        // The adapters travel with the layer. Cloning the list alone would leave the
        // deltas on the old device, and the first forward after a relocation would fail
        // on mixed devices - or worse, silently take a fallback path.
        let mut out = Self {
            held: self.held.map(|t| t.to_device(dev))?,
            bias: self.bias.as_ref().map(|b| b.to_device(dev)).transpose()?,
            // Not carried across: the cached original belongs to the device it was built
            // on, and a relocated weight must rebuild it there rather than hand a caller
            // a tensor on the wrong card.
            weight_orig: std::sync::Arc::new(std::sync::OnceLock::new()),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            lora: self
                .lora
                .iter()
                .map(|d| -> Result<LoraDelta> {
                    Ok(LoraDelta {
                        down: d.down.to_device(dev)?,
                        up: d.up.to_device(dev)?,
                        scale: d.scale,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            fused: None,
        };
        out.refuse();
        Ok(out)
    }

    /// Copy on `dev` converted to `dtype` (weight + optional bias).
    pub fn to_dtype_on(&self, dev: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        // Adapters follow the weight in both device AND dtype: the correction is added
        // to the base output, so a mismatch here would surface as a dtype error deep in
        // a matmul rather than at the relocation that caused it.
        let mut out = Self {
            held: self.held.map(|t| t.to_device(dev)?.to_dtype(dtype))?,
            bias: self
                .bias
                .as_ref()
                .map(|b| -> Result<Tensor> { b.to_device(dev)?.to_dtype(dtype) })
                .transpose()?,
            // Rebuilt on the new device rather than carried, for the reason above.
            weight_orig: std::sync::Arc::new(std::sync::OnceLock::new()),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            lora: self
                .lora
                .iter()
                .map(|d| -> Result<LoraDelta> {
                    Ok(LoraDelta {
                        down: d.down.to_device(dev)?.to_dtype(dtype)?,
                        up: d.up.to_device(dev)?.to_dtype(dtype)?,
                        scale: d.scale,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            fused: None,
        };
        out.refuse();
        Ok(out)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error("linear: rank-0 input".into()))?;
        if k != self.in_dim {
            return Err(Error(format!(
                "linear: in {k} != weight in {}",
                self.in_dim
            )));
        }
        let rows = x.elem_count() / k;

        // Multiply in the WEIGHT's width when the two differ, rather than widening the
        // activation to meet it. cuBLAS accumulates half-precision products into F32, so the
        // sum keeps its width either way, while widening the stream costs a full copy of the
        // largest buffer in the block - on a video-sized sequence, the largest allocation
        // there is. The result is handed back in the dtype the caller passed in, so a stream
        // running F32 stays F32 across a half-precision weight.
        let wdt = self.held.tensor().dtype();
        let follow = wdt != x.dtype() && matches!(wdt, DType::BF16 | DType::F16);
        let xm = if follow { x.to_dtype(wdt)? } else { x.clone() };

        let y = match &self.held {
            Held::Transposed(wt) => {
                let wt = if follow || wt.dtype() == xm.dtype() {
                    wt.clone()
                } else {
                    wt.to_dtype(xm.dtype())?
                };
                xm.reshape(vec![1, rows, k])?.matmul(&wt.reshape(vec![
                    1,
                    self.in_dim,
                    self.out_dim,
                ])?)?
            }
            // The N-T flag, not a materialised transpose: the weight is read in the
            // orientation the checkpoint published it in and never copied.
            Held::Published(w) => {
                let w = if follow || w.dtype() == xm.dtype() {
                    w.clone()
                } else {
                    w.to_dtype(xm.dtype())?
                };
                xm.matmul_t(&w)?
            }
        };
        let mut odims = dims;
        *odims.last_mut().unwrap() = self.out_dim;
        let mut y = y.reshape(odims)?;
        if y.dtype() != x.dtype() {
            y = y.to_dtype(x.dtype())?;
        }
        // The fused correction is ((x @ down_cat) @ up_cat). Applied AFTER the base
        // projection and BEFORE the bias, which is where a merged weight would have put
        // it: the bias is not part of the low-rank correction.
        if let Some((down, up)) = &self.fused {
            let delta = x.reshape(vec![1, rows, k])?.matmul(down)?.matmul(up)?;
            let mut ddims = x.dims().to_vec();
            *ddims.last_mut().unwrap() = self.out_dim;
            y = y.add(&delta.reshape(ddims)?)?;
        }
        match &self.bias {
            None => Ok(y),
            Some(b) if b.dtype() == y.dtype() => y.broadcast_add(b),
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
        }
    }

    /// A weight in the orientation the checkpoint published it, `[out, in]`, kept that way.
    ///
    /// For a loader that cannot stage a host-side transpose - a device-resident
    /// half-precision weight, where turning it would materialise an F32 cast and a
    /// transposed copy per tensor.
    pub fn from_published(w: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (out_dim, in_dim) = w.shape().dims2()?;
        Ok(Self {
            held: Held::Published(w),
            bias,
            weight_orig: std::sync::Arc::new(std::sync::OnceLock::new()),
            in_dim,
            out_dim,
            lora: Vec::new(),
            fused: None,
        })
    }

    /// Attach a low-rank correction. Several compose by addition, in the order added.
    pub fn add_lora(&mut self, delta: LoraDelta) -> Result<()> {
        let (din, r) = delta.down.shape().dims2()?;
        let (r2, dout) = delta.up.shape().dims2()?;
        if din != self.in_dim || dout != self.out_dim || r != r2 {
            return Err(Error(format!(
                "lora: down {din}x{r} up {r2}x{dout} does not fit a {}x{} projection",
                self.in_dim, self.out_dim
            )));
        }
        self.lora.push(delta);
        self.refuse();
        Ok(())
    }

    /// Rebuild the fused pair from the attached adapters.
    ///
    /// Adapters at scale zero are left out of the fusion but kept in the list: a caller
    /// who asked for an adapter at strength 0 wants it disabled, not reported as
    /// unmatched, and a zero block would otherwise cost the same as a live one.
    fn refuse(&mut self) {
        self.fused = fuse_loras(&self.lora, self.in_dim, self.out_dim);
    }

    /// Drop every attached adapter, returning the layer to the base checkpoint.
    pub fn clear_lora(&mut self) {
        self.lora.clear();
        self.fused = None;
    }

    pub fn lora_count(&self) -> usize {
        self.lora.len()
    }
}

impl LoraDelta {
    pub fn rank(&self) -> usize {
        self.down.dims()[1]
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = vb.get(out_dim, "bias")?;
    Linear::new(w, Some(b))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    Linear::new(w, None)
}

//
// The `Module` impl lives with the type it makes callable; it used to sit in another
// area of the crate entirely.
impl crate::tensor::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

/// Widen, activate, narrow - the feed-forward every transformer block puts after its attention.
///
/// Three families wrote this out: two projections, one activation between them, and a forward
/// that chains the three. What differs is the checkpoint's names for the two weights and which
/// activation it was trained with, and both of those belong to the caller - so the projections
/// arrive built and the activation is a field.
#[derive(Clone, Debug)]
pub struct Mlp {
    fc1: Linear,
    act: crate::tensor::ops::Activation,
    fc2: Linear,
}

impl Mlp {
    pub fn new(fc1: Linear, act: crate::tensor::ops::Activation, fc2: Linear) -> Self {
        Self { fc1, act, fc2 }
    }
}

impl crate::tensor::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.apply(&self.fc1.forward(xs)?)?)
    }
}

/// The gated feed-forward: `down(silu(gate(x)) * up(x))`.
///
/// Two widening projections rather than one, and the activation applies to only one of
/// them - the other passes through and multiplies it, so the block chooses per channel
/// how much of itself to let through. Checkpoints spell the three weights `gate/up/down`
/// or `w1/w3/w2`; which it is belongs to the loader, so the projections arrive built.
#[derive(Clone, Debug)]
pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.silu()?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&lhs.mul(&rhs)?)
    }
}

impl crate::tensor::Module for SwiGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
