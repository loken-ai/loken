//! The blocks a 2-D convolutional autoencoder is built from.
//!
//! Every VAE in this tree resamples by a factor of two and passes through the same residual
//! block on the way, so the block, the downsampler and the upsampler are stated here and the
//! models keep only their channel schedules. What differs between checkpoints is what they
//! call the shortcut and how many groups the normalisation takes - both parameters.

use crate::tensor::layer::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm};
use crate::tensor::VarBuilder;
use crate::tensor::{Result, Tensor};

/// A 3x3 convolution that hands back the grid it was given.
///
/// A three-tap kernel reaches one position past its centre, so one row and one column of padding
/// on every side is exactly what the edges need for the output to keep the input's size. Every
/// convolution in this file that is not a channel mixing wants that.
fn conv3x3(from: usize, to: usize, vb: &VarBuilder, name: &str) -> Result<Conv2d> {
    let keep_size = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    conv2d(from, to, 3, keep_size, &vb.pp(name))
}

/// A 1x1 convolution: a mixing of the channels at each position, with no geometry of its own.
fn conv1x1(from: usize, to: usize, vb: &VarBuilder, name: &str) -> Result<Conv2d> {
    conv2d(from, to, 1, Default::default(), &vb.pp(name))
}

/// A residual block: normalise, convolve, normalise, convolve, add.
///
/// The shortcut is a 1x1 convolution and exists only when the block changes the channel count.
/// Its tensor is named `nin_shortcut` in one lineage of checkpoints and `conv_shortcut` in the
/// other, which is the only reason the caller has to say.
#[derive(Debug, Clone)]
pub struct ResnetBlock2d {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_groups: usize,
        shortcut_name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        // A block that keeps its width has nothing to reconcile, so the checkpoint holds no
        // shortcut for it and the input is added to the residual as it stands.
        let shortcut = if in_channels == out_channels {
            None
        } else {
            Some(conv1x1(in_channels, out_channels, &vb, shortcut_name)?)
        };
        Ok(Self {
            norm1: group_norm(num_groups, in_channels, 1e-6, &vb.pp("norm1"))?,
            conv1: conv3x3(in_channels, out_channels, &vb, "conv1")?,
            norm2: group_norm(num_groups, out_channels, 1e-6, &vb.pp("norm2"))?,
            conv2: conv3x3(out_channels, out_channels, &vb, "conv2")?,
            shortcut,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(xs)?.silu()?;
        let h = self.norm2.forward(&self.conv1.forward(&h)?)?.silu()?;
        let h = self.conv2.forward(&h)?;
        match self.shortcut.as_ref() {
            None => xs.add(&h),
            Some(conv) => conv.forward(xs)?.add(&h),
        }
    }
}

/// Halve both spatial dimensions with a stride-2 convolution.
#[derive(Debug, Clone)]
pub struct Downsample2d {
    conv: Conv2d,
}

impl Downsample2d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        // Stride two, and no padding of the convolution's own: the forward pads by hand,
        // because where the missing row and column go is the whole point.
        let halve = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(channels, channels, 3, halve, &vb.pp("conv"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The padding is asymmetric - one row at the bottom and one column at the right - so
        // that an odd size loses its edge rather than its centre.
        let rank = xs.rank();
        let xs = xs.pad_with_zeros(rank - 1, 0, 1)?;
        let xs = xs.pad_with_zeros(rank - 2, 0, 1)?;
        self.conv.forward(&xs)
    }
}

/// Double both spatial dimensions: nearest-neighbour, then a convolution to smooth it.
#[derive(Debug, Clone)]
pub struct Upsample2d {
    conv: Conv2d,
}

impl Upsample2d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv3x3(channels, channels, &vb, "conv")?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.shape().dims4()?;
        self.conv.forward(&xs.upsample_nearest2d(h * 2, w * 2)?)
    }
}

// -- the autoencoder those blocks make -------------------------------------------------------

use crate::tensor::layer::{linear, Linear};

/// How a checkpoint stores the middle block's four projections.
///
/// A 1x1 convolution over `[b, c, h, w]` and a linear over `[b, hw, c]` are the same map with
/// the weight written two ways; the checkpoint decides which, so each stays on its own side and
/// the arithmetic of neither moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    Conv1x1,
    Linear,
}

#[derive(Debug, Clone)]
enum Projection {
    Conv(Conv2d),
    Linear(Linear),
}

impl Projection {
    fn load(kind: ProjectionKind, channels: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        Ok(match kind {
            ProjectionKind::Conv1x1 => Projection::Conv(conv1x1(channels, channels, vb, name)?),
            ProjectionKind::Linear => Projection::Linear(linear(channels, channels, &vb.pp(name))?),
        })
    }

    /// `[b, c, h, w]` in, `[b, hw, c]` out - the shape the attention wants.
    fn to_sequence(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            // The convolution takes the grid as it is and the flattening follows.
            Self::Conv(c) => c.forward(xs)?.flatten_from(2)?.transpose(1, 2),
            // The linear needs the grid flattened first.
            Self::Linear(l) => l.forward(&xs.flatten_from(2)?.transpose(1, 2)?),
        }
    }

    /// `[b, hw, c]` in, `[b, c, h, w]` out.
    fn to_grid(&self, xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, _, c) = xs.dims3()?;
        match self {
            Self::Conv(conv) => conv.forward(&xs.transpose(1, 2)?.reshape((b, c, h, w))?),
            Self::Linear(l) => l.forward(xs)?.transpose(1, 2)?.reshape((b, c, h, w)),
        }
    }
}

/// What the checkpoint calls the attention's parts.
#[derive(Debug, Clone, Copy)]
pub struct AttnNaming {
    pub kind: ProjectionKind,
    pub norm: &'static str,
    pub q: &'static str,
    pub k: &'static str,
    pub v: &'static str,
    pub out: &'static str,
}

/// Every position attending to every other, over the whole grid.
///
/// The middle of an autoencoder is where the picture as a whole is decided - it is what keeps
/// colour and brightness consistent across an image - which is why a tiled decode has to accept
/// that each tile attends only to itself.
#[derive(Debug, Clone)]
pub struct SelfAttention2d {
    norm: GroupNorm,
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
}

impl SelfAttention2d {
    pub fn new(
        channels: usize,
        num_groups: usize,
        names: &AttnNaming,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm: group_norm(num_groups, channels, 1e-6, &vb.pp(names.norm))?,
            q: Projection::load(names.kind, channels, &vb, names.q)?,
            k: Projection::load(names.kind, channels, &vb, names.k)?,
            v: Projection::load(names.kind, channels, &vb, names.v)?,
            out: Projection::load(names.kind, channels, &vb, names.out)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let normed = self.norm.forward(xs)?;
        let (q, k, v) = (
            self.q.to_sequence(&normed)?,
            self.k.to_sequence(&normed)?,
            self.v.to_sequence(&normed)?,
        );
        let scale = 1.0 / (*q.dims().last().unwrap_or(&1) as f32).sqrt();
        let attended = crate::inference::model::sdpa::sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        self.out.to_grid(&attended, h, w)?.add(xs)
    }
}

/// The channel schedule an autoencoder is shaped by.
#[derive(Debug, Clone)]
pub struct Shape {
    /// What each stage widens to, finest first. The decoder walks it backwards.
    pub stages: Vec<usize>,
    /// What enters the first encoder stage, straight out of the input convolution.
    pub stem: usize,
    /// Residual blocks per encoder stage. A decoder stage carries one more - that is what the
    /// checkpoints hold, and it is the extra capacity the harder direction is given.
    pub blocks_per_stage: usize,
    pub groups: usize,
    /// Image channels in and out, and the width of the latent.
    pub image_channels: usize,
    pub latent_channels: usize,
}

/// Where a checkpoint keeps the parts, and what it calls them.
///
/// Two lineages of this autoencoder are in the tree and they agree on every number; they
/// disagree only here. One nests a stage's residual blocks under `block` and its resampler
/// beside them, the other under `resnets` with the resampler in a list of its own; one numbers
/// the decoder's stages the way the encoder numbered them and walks backwards, the other
/// renumbers from the deepest.
#[derive(Debug, Clone, Copy)]
pub struct Naming {
    pub down: &'static str,
    pub up: &'static str,
    pub resnets: &'static str,
    pub downsample: &'static str,
    pub upsample: &'static str,
    pub shortcut: &'static str,
    pub mid_first: &'static str,
    pub mid_attn: &'static str,
    pub mid_second: &'static str,
    pub norm_out: &'static str,
    pub attn: AttnNaming,
    /// True when the decoder's stages keep the encoder's numbering, so the deepest holds the
    /// highest index and the walk runs backwards.
    pub up_keeps_encoder_numbering: bool,
}

/// One stage: some residual blocks, then optionally a change of resolution.
#[derive(Debug, Clone)]
pub struct Stage {
    blocks: Vec<ResnetBlock2d>,
    down: Option<Downsample2d>,
    up: Option<Upsample2d>,
}

impl Stage {
    fn load(
        in_channels: usize,
        out_channels: usize,
        count: usize,
        shape: &Shape,
        names: &Naming,
        vb: &VarBuilder,
    ) -> Result<Vec<ResnetBlock2d>> {
        let vb = vb.pp(names.resnets);
        (0..count)
            .map(|i| {
                // Only the first block of a stage changes the width; the rest keep it.
                let from = if i == 0 { in_channels } else { out_channels };
                ResnetBlock2d::new(from, out_channels, shape.groups, names.shortcut, vb.pp(i))
            })
            .collect()
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The residual blocks feed one another in order; the resampler, if the stage has one,
        // sees only what the last of them produced.
        let h = self
            .blocks
            .iter()
            .try_fold(xs.clone(), |h, block| block.forward(&h))?;
        match (&self.down, &self.up) {
            (Some(d), _) => d.forward(&h),
            (_, Some(u)) => u.forward(&h),
            _ => Ok(h),
        }
    }
}

/// The middle: a residual block, attention over the whole grid, another residual block.
#[derive(Debug, Clone)]
pub struct Mid {
    first: ResnetBlock2d,
    attention: SelfAttention2d,
    second: ResnetBlock2d,
}

impl Mid {
    fn load(channels: usize, shape: &Shape, names: &Naming, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            first: ResnetBlock2d::new(
                channels,
                channels,
                shape.groups,
                names.shortcut,
                vb.pp(names.mid_first),
            )?,
            attention: SelfAttention2d::new(
                channels,
                shape.groups,
                &names.attn,
                vb.pp(names.mid_attn),
            )?,
            second: ResnetBlock2d::new(
                channels,
                channels,
                shape.groups,
                names.shortcut,
                vb.pp(names.mid_second),
            )?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.first.forward(xs)?;
        self.second.forward(&self.attention.forward(&h)?)
    }
}

/// Image to latent: halve the resolution stage by stage, then say the mean and the log variance.
#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    stages: Vec<Stage>,
    mid: Mid,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(shape: &Shape, names: &Naming, vb: VarBuilder) -> Result<Self> {
        let deepest = *shape.stages.last().unwrap_or(&shape.stem);
        let vb_down = vb.pp(names.down);
        let stages = shape
            .stages
            .iter()
            .enumerate()
            .map(|(i, &out)| {
                let from = if i == 0 {
                    shape.stem
                } else {
                    shape.stages[i - 1]
                };
                let vb = vb_down.pp(i);
                let blocks = Stage::load(from, out, shape.blocks_per_stage, shape, names, &vb)?;
                // Every stage but the last halves the grid on its way out.
                let down = if i + 1 < shape.stages.len() {
                    Some(Downsample2d::new(out, vb.pp(names.downsample))?)
                } else {
                    None
                };
                Ok(Stage {
                    blocks,
                    down,
                    up: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: conv3x3(shape.image_channels, shape.stem, &vb, "conv_in")?,
            stages,
            mid: Mid::load(deepest, shape, names, &vb)?,
            norm_out: group_norm(shape.groups, deepest, 1e-6, &vb.pp(names.norm_out))?,
            // Mean and log variance, side by side.
            conv_out: conv3x3(deepest, 2 * shape.latent_channels, &vb, "conv_out")?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(xs)?;
        for stage in &self.stages {
            h = stage.forward(&h)?;
        }
        let h = self.mid.forward(&h)?;
        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)
    }
}

/// Latent to image: the encoder's stages walked backwards, doubling as they go.
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid: Mid,
    stages: Vec<Stage>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(shape: &Shape, names: &Naming, vb: VarBuilder) -> Result<Self> {
        let n = shape.stages.len();
        let deepest = *shape.stages.last().unwrap_or(&shape.stem);
        let finest = *shape.stages.first().unwrap_or(&shape.stem);
        let vb_up = vb.pp(names.up);
        let stages = (0..n)
            .map(|k| {
                let out = shape.stages[n - 1 - k];
                let from = if k == 0 { deepest } else { shape.stages[n - k] };
                let index = if names.up_keeps_encoder_numbering {
                    n - 1 - k
                } else {
                    k
                };
                let vb = vb_up.pp(index);
                let blocks = Stage::load(from, out, shape.blocks_per_stage + 1, shape, names, &vb)?;
                // Every stage but the last doubles the grid on its way out.
                let up = if k + 1 < n {
                    Some(Upsample2d::new(out, vb.pp(names.upsample))?)
                } else {
                    None
                };
                Ok(Stage {
                    blocks,
                    down: None,
                    up,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: conv3x3(shape.latent_channels, deepest, &vb, "conv_in")?,
            mid: Mid::load(deepest, shape, names, &vb)?,
            stages,
            norm_out: group_norm(shape.groups, finest, 1e-6, &vb.pp(names.norm_out))?,
            conv_out: conv3x3(finest, shape.image_channels, &vb, "conv_out")?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.mid.forward(&self.conv_in.forward(xs)?)?;
        for stage in &self.stages {
            h = stage.forward(&h)?;
        }
        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)
    }

    /// How many image pixels one latent pixel becomes along each axis: two per upsampling
    /// stage.
    pub fn upscale(&self) -> usize {
        1 << self.stages.iter().filter(|s| s.up.is_some()).count()
    }

    /// [`Self::forward`] in overlapping spatial tiles, bounding the peak memory by one tile
    /// (see [`decode_in_tiles`]). A latent no larger than `max_tile` is one tile, which is
    /// the whole decode.
    pub fn forward_tiled(&self, xs: &Tensor, max_tile: usize, overlap: usize) -> Result<Tensor> {
        decode_in_tiles(xs, max_tile, overlap, self.upscale(), |t| self.forward(t))
    }
}

/// Decode a `[b, c, h, w]` latent in tiles of at most `max_tile` latent pixels a side.
///
/// Each tile is decoded with `overlap` latent pixels of context on every side it has a
/// neighbour, then cut back to its own band (scaled by `upscale`, the decoder's pixels per
/// latent pixel), so the convolutions see their context and the seams carry no boundary.
/// The bands partition the grid and are stitched back in order. The decoder's mid-block
/// attention sees one tile at a time, which is the one way tiles can drift apart.
pub fn decode_in_tiles(
    latent: &Tensor,
    max_tile: usize,
    overlap: usize,
    upscale: usize,
    decode: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let d = latent.dims();
    let (hl, wl) = (d[2], d[3]);
    let bounds = |len: usize| -> Vec<usize> {
        let n = len.div_ceil(max_tile.max(1)).max(1);
        (0..=n).map(|i| (i * len) / n).collect()
    };
    let (hb, wb) = (bounds(hl), bounds(wl));
    let mut rows: Vec<Tensor> = Vec::with_capacity(hb.len() - 1);
    for band in hb.windows(2) {
        let (h0, h1) = (band[0], band[1]);
        let (hs, he) = (h0.saturating_sub(overlap), (h1 + overlap).min(hl));
        let mut cols: Vec<Tensor> = Vec::with_capacity(wb.len() - 1);
        for col in wb.windows(2) {
            let (w0, w1) = (col[0], col[1]);
            let (ws, we) = (w0.saturating_sub(overlap), (w1 + overlap).min(wl));
            let tile = latent
                .narrow(2, hs, he - hs)
                .and_then(|t| t.narrow(3, ws, we - ws))?;
            let decoded = decode(&tile)?;
            cols.push(
                decoded
                    .narrow(2, (h0 - hs) * upscale, (h1 - h0) * upscale)
                    .and_then(|t| t.narrow(3, (w0 - ws) * upscale, (w1 - w0) * upscale))?,
            );
        }
        let refs: Vec<&Tensor> = cols.iter().collect();
        rows.push(Tensor::cat(&refs, 3)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&refs, 2)
}

#[cfg(test)]
mod tiling_tests {
    use super::*;

    /// A decoder that scales every latent pixel into a `2 x 2` block: tiles of it must
    /// stitch back into exactly the whole decode, whatever the tile and overlap.
    #[test]
    fn tiles_stitch_back_into_the_whole_decode() {
        let dev = crate::tensor::Device::Cpu;
        let (h, w) = (13usize, 9usize);
        let v: Vec<f32> = (0..2 * h * w).map(|i| (i as f32 * 0.37).sin()).collect();
        let latent = Tensor::from_vec(v, (1, 2, h, w), &dev).unwrap();
        let decode = |t: &Tensor| {
            let (_, _, th, tw) = t.shape().dims4()?;
            t.affine(3.0, 0.5)?.upsample_nearest2d(th * 2, tw * 2)
        };
        let whole = decode(&latent)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (tile, overlap) in [(4usize, 1usize), (5, 2), (13, 0), (64, 8)] {
            let tiled = decode_in_tiles(&latent, tile, overlap, 2, decode)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(tiled, whole, "tile {tile} overlap {overlap}");
        }
    }
}
