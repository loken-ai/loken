//! Convolutions, 1-D and 2-D, with their configuration.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// 1-D convolution config.
#[derive(Clone, Copy, Debug)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1dConfig {
    /// The kernel laid straight over the input: it starts on the first position, moves one at a
    /// time, keeps its taps side by side, and mixes every channel in a single group.
    ///
    /// Only the padding is nothing. The other three are counts, and a convolution that stepped,
    /// spread or grouped by zero would not be a convolution at all.
    const PLAIN: Self = Self {
        padding: 0,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self::PLAIN
    }
}

/// The one-dimensional convolution that gives back the length it was handed.
///
/// A kernel of `kernel` taps spaced `dilation` apart reaches `(kernel - 1) * dilation`
/// positions past its centre, and half of that on each side is exactly what must be padded for
/// the output to be as long as the input. Nearly every convolution in an audio front-end or a
/// residual stack wants that; each was carrying the arithmetic already done, as a number whose
/// relation to the kernel a reader had to reconstruct.
///
/// A convolution that also strides takes this and says so: `Conv1dConfig { stride: 2,
/// ..same_length_1d(3, 1) }` is the halving front-end every whisper-shaped encoder opens with.
pub fn same_length_1d(kernel: usize, dilation: usize) -> Conv1dConfig {
    Conv1dConfig {
        dilation,
        padding: (kernel - 1) * dilation / 2,
        ..Default::default()
    }
}

/// 1-D convolution layer: weight `[c_out, c_in/groups, k]` + per-channel bias.
#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    cfg: Conv1dConfig,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, cfg: Conv1dConfig) -> Self {
        Self { weight, bias, cfg }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(
            &self.weight,
            self.cfg.padding,
            self.cfg.stride,
            self.cfg.dilation,
            self.cfg.groups,
        )?;
        match self.bias.as_ref() {
            Some(b) => y.add_channel_bias(b),
            None => Ok(y),
        }
    }
}

pub fn conv1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: &VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c / cfg.groups, k), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// 2-D convolution config (padding/stride/dilation/groups quadruple).
#[derive(Clone, Copy, Debug)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv2dConfig {
    /// Both axes take the one-dimensional quadruple. There is one set of these numbers in the
    /// module, not two tables that have to be kept in agreement.
    fn default() -> Self {
        let Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
        } = Conv1dConfig::PLAIN;
        Self {
            padding,
            stride,
            dilation,
            groups,
        }
    }
}

/// 2-D convolution layer: weight `[c_out, c_in/groups, kh, kw]` + per-channel
/// bias.
#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    cfg: Conv2dConfig,
    /// Low-rank corrections, applied ALONGSIDE the kernel rather than merged into it.
    ///
    /// Merging would need a copy of every adapted kernel to restore on detach, and the
    /// widest levels of a UNet are tens of megabytes each - hundreds of megabytes of
    /// VRAM held for as long as an adapter is attached, on a machine where VRAM is the
    /// binding constraint. Two small convolutions per step cost a few percent of the
    /// kernel they correct and nothing at rest, which is the same trade `Linear` makes.
    lora: Vec<LoraDelta>,
    /// This convolution's path in the checkpoint, so an adapter can be looked up.
    path: String,
}

impl Conv2d {
    /// The kernel, for callers that need to match an input to it.
    ///
    /// A vision tower reads the dtype off this to cast its patches before the convolution: the
    /// weight's precision is decided at load by what the device can hold, and an input that
    /// arrives at another one fails inside the kernel rather than at the boundary.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn new(weight: Tensor, bias: Option<Tensor>, cfg: Conv2dConfig) -> Self {
        Self {
            weight,
            bias,
            cfg,
            lora: Vec::new(),
            path: String::new(),
        }
    }

    /// Record where this convolution came from (adapters are keyed on it).
    pub fn with_path(mut self, path: &str) -> Self {
        self.path = path.to_string();
        self
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Attach a convolution adapter: `down` `[r, in, kh, kw]`, `up` `[out, r, 1, 1]`.
    ///
    /// `down` carries the kernel and therefore the geometry - padding, stride and
    /// dilation must match the layer it corrects, or the correction lands on different
    /// pixels than the output it is added to. `up` is a 1x1 mixing of the rank.
    pub fn add_lora(&mut self, delta: LoraDelta) -> Result<()> {
        let (r, din, kh, kw) = delta.down.shape().dims4()?;
        let (dout, r2, uh, uw) = delta.up.shape().dims4()?;
        let (out, wi, wkh, wkw) = self.weight.shape().dims4()?;
        if r != r2 || din != wi || kh != wkh || kw != wkw || dout != out || uh != 1 || uw != 1 {
            return Err(Error(format!(
                "lora: down {r}x{din}x{kh}x{kw} up {dout}x{r2}x{uh}x{uw} does not fit a \
                 {out}x{wi}x{wkh}x{wkw} convolution"
            )));
        }
        self.lora.push(delta);
        Ok(())
    }

    /// Drop every attached adapter, returning the layer to the base checkpoint.
    ///
    /// Exact by construction: the kernel was never modified, so there is nothing to
    /// undo and no arithmetic to drift.
    pub fn clear_lora(&mut self) {
        self.lora.clear();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(
            &self.weight,
            self.cfg.padding,
            self.cfg.stride,
            self.cfg.dilation,
            self.cfg.groups,
        )?;
        let mut y = match self.bias.as_ref() {
            Some(b) => y.add_channel_bias(b)?,
            None => y,
        };
        for d in &self.lora {
            if d.scale == 0.0 {
                continue;
            }
            let h = x.conv2d(
                &d.down,
                self.cfg.padding,
                self.cfg.stride,
                self.cfg.dilation,
                1,
            )?;
            let c = h.conv2d(&d.up, 0, 1, 1, 1)?;
            y = y.add(&c.scale(d.scale)?)?;
        }
        Ok(y)
    }
}

pub fn conv2d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv2dConfig,
    vb: &VarBuilder,
) -> Result<Conv2d> {
    let weight = vb.get((out_c, in_c / cfg.groups, k, k), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv2d::new(weight, Some(bias), cfg).with_path(vb.prefix()))
}

/// A convolution with NO bias term. Networks that follow every convolution with a
/// batch norm (the ResNet family and its descendants) store no conv bias at all -
/// the norm provides it - so asking for one fails the load.
pub fn conv2d_no_bias(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv2dConfig,
    vb: &VarBuilder,
) -> Result<Conv2d> {
    let weight = vb.get((out_c, in_c / cfg.groups, k, k), "weight")?;
    Ok(Conv2d::new(weight, None, cfg).with_path(vb.prefix()))
}

// The `Module` impl lives with the type it makes callable, not with whoever loads it.
impl crate::tensor::Module for Conv2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
