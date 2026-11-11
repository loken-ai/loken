//! The convolutional stages of the codec decoder: the upsample stages after the transformer,
//! and the snake stack that turns them into the waveform.
//!
//! Every width, kernel and count is read from the stored tensors.
//! Every stride is read from the codec's configuration.

use super::config::Reach;
use crate::inference::model::codec_blocks::{causal_conv1d, causal_convtr1d};
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// The epsilon the reference adds to `exp(beta)` before it divides by it.
pub const SNAKE_EPS: f32 = 1e-9;
/// The epsilon of the layer norm of a ConvNeXt block.
pub const CONVNEXT_LN_EPS: f32 = 1e-6;
/// The residual unit `i` of a block dilates its conv by `RESIDUAL_DILATION_BASE^i`.
pub const RESIDUAL_DILATION_BASE: usize = 3;

/// A conv `[out, in / groups, k]` or a transposed conv `[in, out, k]`, with its bias.
pub(super) struct Conv {
    pub(super) w: Tensor,
    pub(super) b: Tensor,
}

impl Conv {
    /// `{name}.weight` and `{name}.bias`, the bias over the output channels at `out_axis`.
    fn load(vb: &VarBuilder, name: &str, out_axis: usize) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        Ok(Self {
            b: vb.get(shape[out_axis], &format!("{name}.bias"))?,
            w: vb.get(shape, &format!("{name}.weight"))?,
        })
    }

    pub(super) fn load_conv(vb: &VarBuilder, name: &str) -> Result<Self> {
        Self::load(vb, name, 0)
    }

    pub(super) fn load_tr(vb: &VarBuilder, name: &str) -> Result<Self> {
        Self::load(vb, name, 1)
    }

    /// Taps of the kernel.
    pub(super) fn kernel(&self) -> usize {
        self.w.dims()[2]
    }

    /// Input channels of a conv of `groups` groups.
    pub(super) fn conv_in(&self, groups: usize) -> usize {
        self.w.dims()[1] * groups
    }

    /// The causal conv of `x [1, in, L]` with `dilation`.
    pub(super) fn forward(&self, x: &Tensor, dilation: usize, groups: usize) -> Result<Tensor> {
        causal_conv1d(x, &self.w, Some(&self.b), 1, dilation, groups)
    }

    /// The causal transposed conv of `x [1, in, L]` to `[1, out, L * stride]`.
    pub(super) fn forward_tr(&self, x: &Tensor, stride: usize) -> Result<Tensor> {
        causal_convtr1d(x, &self.w, Some(&self.b), stride, 1)
    }
}

/// SnakeBeta `x + 1 / (exp(beta) + eps) * sin(x * exp(alpha))^2`, per channel.
///
/// `exp(alpha)` and `1 / (exp(beta) + eps)` are computed at load, the 2 inputs `snake1d`
/// takes.
pub(super) struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    /// `{name}.alpha` and `{name}.beta`, of one value per channel.
    pub(super) fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        let alpha = format!("{name}.alpha");
        let channels = vb.shape_of(&alpha)?[0];
        let dev = vb.device();
        let host = |t: Tensor| t.to_device(&Device::Cpu);
        let a = host(vb.get(channels, &alpha)?)?.exp()?;
        let b = host(vb.get(channels, &format!("{name}.beta"))?)?.exp()?;
        Ok(Self {
            alpha: a.to_device(dev)?,
            inv_beta: b.affine(1.0, SNAKE_EPS)?.recip()?.to_device(dev)?,
        })
    }

    /// `x [1, C, L]` through the activation.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.snake1d(&self.alpha, &self.inv_beta)
    }
}

/// A ConvNeXt block: a depthwise causal conv, a layer norm over the channels, 2 linear
/// layers with an exact GELU between them, a per channel scale, and the residual.
struct ConvNeXt {
    dwconv: Conv,
    norm_w: Tensor,
    norm_b: Tensor,
    pw1_w: Tensor,
    pw1_b: Tensor,
    pw2_w: Tensor,
    pw2_b: Tensor,
    gamma: Tensor,
}

impl ConvNeXt {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let dwconv = Conv::load_conv(vb, "dwconv.conv")?;
        let dim = dwconv.b.dims()[0];
        let ffn = vb.shape_of("pwconv1.weight")?[0];
        Ok(Self {
            dwconv,
            norm_w: vb.get(dim, "norm.weight")?,
            norm_b: vb.get(dim, "norm.bias")?,
            pw1_w: vb.get((ffn, dim), "pwconv1.weight")?,
            pw1_b: vb.get(ffn, "pwconv1.bias")?,
            pw2_w: vb.get((dim, ffn), "pwconv2.weight")?,
            pw2_b: vb.get(dim, "pwconv2.bias")?,
            gamma: vb.get(dim, "gamma")?,
        })
    }

    /// `x [1, C, L]` through the block.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, l) = x.shape().dims3()?;
        let h = self.dwconv.forward(x, 1, c)?;
        let h = h.reshape((c, l))?.transpose(0, 1)?.contiguous()?;
        let h = h.layer_norm(&self.norm_w, Some(&self.norm_b), CONVNEXT_LN_EPS)?;
        let h = h
            .matmul_t(&self.pw1_w)?
            .broadcast_add(&self.pw1_b.reshape((1, self.pw1_b.dims()[0]))?)?
            .gelu_erf()?;
        let h = h
            .matmul_t(&self.pw2_w)?
            .broadcast_add(&self.pw2_b.reshape((1, c))?)?
            .broadcast_mul(&self.gamma.reshape((1, c))?)?;
        let h = h.transpose(0, 1)?.contiguous()?.reshape((1, c, l))?;
        x.add(&h)
    }
}

/// An upsample stage after the transformer: a causal transposed conv, then a ConvNeXt block.
pub struct Upsample {
    convtr: Conv,
    block: ConvNeXt,
    stride: usize,
}

impl Upsample {
    /// The stage under `vb` that multiplies the rate by `stride`.
    pub fn load(vb: &VarBuilder, stride: usize) -> Result<Self> {
        Ok(Self {
            convtr: Conv::load_tr(vb, "0.conv")?,
            block: ConvNeXt::load(&vb.pp("1"))?,
            stride,
        })
    }

    /// `x [1, C, L]` to `[1, C, L * stride]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.block.forward(&self.convtr.forward_tr(x, self.stride)?)
    }

    /// The stages an output of this one reads through, in order.
    pub fn reach(&self) -> [Reach; 2] {
        [
            Reach::ConvTr {
                kernel: self.convtr.kernel(),
                stride: self.stride,
            },
            Reach::Conv {
                kernel: self.block.dwconv.kernel(),
                dilation: 1,
            },
        ]
    }

    /// The factor the stage multiplies the rate by.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Channels in and out, and the width of the feed-forward.
    pub(super) fn widths(&self) -> (usize, usize) {
        (self.block.dwconv.b.dims()[0], self.block.pw1_b.dims()[0])
    }
}

/// A residual unit: snake, a dilated causal conv, snake, a causal conv, and the residual.
pub(super) struct ResidualUnit {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
    dilation: usize,
}

impl ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self
            .conv1
            .forward(&self.act1.forward(x)?, self.dilation, 1)?;
        let h = self.conv2.forward(&self.act2.forward(&h)?, 1, 1)?;
        h.add(x)
    }
}

/// A block of the snake stack: snake, a causal transposed conv, then the residual units.
pub struct SnakeBlock {
    act: Snake,
    pub(super) convtr: Conv,
    pub(super) units: Vec<ResidualUnit>,
    stride: usize,
}

impl SnakeBlock {
    /// The block under `vb` that multiplies the rate by `stride`.
    ///
    /// `block.0` is the snake, `block.1` the transposed conv, the entries after it the units.
    pub fn load(vb: &VarBuilder, stride: usize) -> Result<Self> {
        let b = vb.pp("block");
        let units = (2..vb.count_children("block"))
            .map(|i| {
                let u = b.pp(i.to_string());
                Ok(ResidualUnit {
                    act1: Snake::load(&u, "act1")?,
                    conv1: Conv::load_conv(&u, "conv1.conv")?,
                    act2: Snake::load(&u, "act2")?,
                    conv2: Conv::load_conv(&u, "conv2.conv")?,
                    dilation: RESIDUAL_DILATION_BASE.pow((i - 2) as u32),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            act: Snake::load(&b, "0")?,
            convtr: Conv::load_tr(&b, "1.conv")?,
            units,
            stride,
        })
    }

    /// `x [1, in, L]` to `[1, out, L * stride]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.convtr.forward_tr(&self.act.forward(x)?, self.stride)?;
        for unit in &self.units {
            h = unit.forward(&h)?;
        }
        Ok(h)
    }

    /// The factor the block multiplies the rate by.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The stages an output of this one reads through, in order.
    pub fn reach(&self) -> Vec<Reach> {
        let mut r = vec![Reach::ConvTr {
            kernel: self.convtr.kernel(),
            stride: self.stride,
        }];
        for u in &self.units {
            r.push(Reach::Conv {
                kernel: u.conv1.kernel(),
                dilation: u.dilation,
            });
            r.push(Reach::Conv {
                kernel: u.conv2.kernel(),
                dilation: 1,
            });
        }
        r
    }
}

/// The bytes of F32 values the largest step of a stage holds at once, per position of its
/// input.
pub(super) trait StageBytes {
    /// `rate` is the positions of its input per frame.
    fn bytes_per_frame(&self, rate: usize) -> u64;
}

/// A causal conv holds its input, the padded copy, the unfolded columns and its output.
fn conv_bytes(c_in: usize, kernel: usize, c_out: usize, positions: usize) -> u64 {
    (positions * (c_in * (kernel + 2) + c_out) * std::mem::size_of::<f32>()) as u64
}

/// A transposed conv holds its input, the columns of every tap and its output.
fn convtr_bytes(c_in: usize, kernel: usize, c_out: usize, stride: usize, positions: usize) -> u64 {
    (positions * (c_in + c_out * kernel + c_out * stride) * std::mem::size_of::<f32>()) as u64
}

impl StageBytes for Upsample {
    fn bytes_per_frame(&self, rate: usize) -> u64 {
        let (c, ffn) = self.widths();
        let k = self.convtr.kernel();
        let out = rate * self.stride;
        let tr = convtr_bytes(c, k, c, self.stride, rate);
        let dw = conv_bytes(c, self.block.dwconv.kernel(), c, out);
        let mlp = (out * (3 * c + ffn) * std::mem::size_of::<f32>()) as u64;
        tr.max(dw).max(mlp)
    }
}

impl StageBytes for SnakeBlock {
    fn bytes_per_frame(&self, rate: usize) -> u64 {
        let c_in = self.convtr.w.dims()[0];
        let c_out = self.convtr.w.dims()[1];
        let out = rate * self.stride;
        let tr = convtr_bytes(c_in, self.convtr.kernel(), c_out, self.stride, rate);
        let units = self.units.iter().map(|u| {
            let held = (out * c_out * std::mem::size_of::<f32>()) as u64;
            held + conv_bytes(u.conv1.conv_in(1), u.conv1.kernel(), c_out, out)
        });
        units.fold(tr, u64::max)
    }
}

impl StageBytes for Conv {
    fn bytes_per_frame(&self, rate: usize) -> u64 {
        conv_bytes(self.conv_in(1), self.kernel(), self.w.dims()[0], rate)
    }
}
