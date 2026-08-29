//! ControlNet for SDXL: structural conditioning from a control image.
//!
//! A prompt says WHAT to draw; it cannot say where the limbs are. That is why "too many
//! legs" is not fixable by more steps or more guidance - nothing in the conditioning
//! constrains the geometry. ControlNet does: a second network reads a control image (a
//! pose skeleton, an edge map) and emits RESIDUALS that are added into the UNet's skip
//! connections, so the structure is imposed rather than hoped for.
//!
//! SHAPE OF THE THING. It is a copy of the UNet's DOWN path plus its middle block, with
//! a zero-initialised 1x1 convolution after each - nine for the skips our UNet pushes,
//! one for the middle. The zero-init is what makes it safe to attach: at load the
//! residuals are exactly zero, so the base model is untouched and training moves away
//! from that only where the control actually helps.
//!
//! NAMING. These checkpoints are published in the DIFFUSERS layout
//! (`down_blocks.1.resnets.0.norm1`), while our UNet loads the single-file LDM one
//! (`input_blocks.4.0.in_layers.0`). The two describe the same graph. Rather than bend
//! one into the other, this module loads its own weights under their own names and only
//! the RESIDUALS cross over - a much smaller contract than a name mapping over every
//! tensor, and one that cannot silently mis-pair a weight.

use crate::inference::model::sdxl::unet::{
    linear_wb, sinusoidal, Layer, ResBlock, SpatialTransformer, BASE, EMB_DIM, INPUT_SPEC,
    LABEL_DIM, LATENT_CHANNELS,
};
use crate::tensor::layer::{conv2d, Conv2d, Conv2dConfig, Linear};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// Channel widths of the control-image encoder, read from the checkpoint rather than
/// assumed: 3 -> 16 -> 32 -> 96 -> 256 -> 320, with a stride-2 step at every widening.
///
/// It exists to bring a full-resolution control image down to the latent's own grid, so
/// its total stride must match the VAE's factor of 8. Three stride-2 blocks plus the
/// entry gives exactly that.
const COND_CHANNELS: [(usize, usize, usize); 6] = [
    (16, 16, 1),
    (16, 32, 2),
    (32, 32, 1),
    (32, 96, 2),
    (96, 96, 1),
    (96, 256, 2),
];

/// Encodes the control image into something shaped like the first UNet feature map.
pub struct CondEmbedding {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    conv_out: Conv2d,
}

impl CondEmbedding {
    pub fn new(vb: &VarBuilder) -> Result<Self> {
        let pad = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let stride2 = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv_in = conv2d(3, 16, 3, pad, &vb.pp("conv_in"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(COND_CHANNELS.len());
        for (i, (ic, oc, stride)) in COND_CHANNELS.iter().enumerate() {
            let cfg = if *stride == 2 { stride2 } else { pad };
            blocks.push(conv2d(*ic, *oc, 3, cfg, &vb_b.pp(i.to_string()))?);
        }
        let conv_out = conv2d(256, 320, 3, pad, &vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    /// `xs`: `[b, 3, h, w]` in 0..1. Returns `[b, 320, h/8, w/8]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = silu(&self.conv_in.forward(xs)?)?;
        for b in &self.blocks {
            h = silu(&b.forward(&h)?)?;
        }
        // conv_out is the zero-initialised hand-off; no activation after it, the same
        // way the residual projections below are taken raw.
        self.conv_out.forward(&h)
    }
}

/// SiLU, the activation this network shares with the UNet.
fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.mul(&xs.neg()?.exp()?.affine(1.0, 1.0)?.recip()?)
}

/// The 1x1 projections that turn a captured feature map into a residual.
///
/// One per skip the UNet pushes, plus one for the middle block. Their count is not a
/// free parameter: it must equal the number of skips, or a residual would be added to
/// the wrong stage - which shifts the structure without failing, the worst kind of bug
/// for a conditioning signal.
pub struct ResidualProjections {
    down: Vec<Conv2d>,
    mid: Conv2d,
}

impl ResidualProjections {
    pub fn new(vb: &VarBuilder, widths: &[usize], mid_width: usize) -> Result<Self> {
        let one = Conv2dConfig::default();
        let vb_d = vb.pp("controlnet_down_blocks");
        let mut down = Vec::with_capacity(widths.len());
        for (i, w) in widths.iter().enumerate() {
            down.push(conv2d(*w, *w, 1, one, &vb_d.pp(i.to_string()))?);
        }
        let mid = conv2d(mid_width, mid_width, 1, one, &vb.pp("controlnet_mid_block"))?;
        Ok(Self { down, mid })
    }

    pub fn len(&self) -> usize {
        self.down.len()
    }

    pub fn is_empty(&self) -> bool {
        self.down.is_empty()
    }

    pub fn project_down(&self, i: usize, xs: &Tensor) -> Result<Tensor> {
        self.down[i].forward(xs)
    }

    pub fn project_mid(&self, xs: &Tensor) -> Result<Tensor> {
        self.mid.forward(xs)
    }
}

/// The widths our UNet's nine skips carry, in push order.
///
/// READ FROM THE CHECKPOINT, not guessed: a real openpose ControlNet's nine residual
/// projections are exactly these. My first attempt derived them from the UNet's
/// output-side table and got 320,320,640,320,640,1280,640,1280,1280 - plausible, wrong,
/// and it would have added residuals to the wrong stages, which shifts the structure the
/// control was meant to impose WITHOUT failing on shape. `skip_widths()` now walks the
/// down path instead, and the test holds the two together.
pub const SKIP_WIDTHS: [usize; 9] = [320, 320, 320, 320, 640, 640, 640, 1280, 1280];
pub const MID_WIDTH: usize = 1280;

/// Where the checkpoint lives, relative to the models directory.
pub const OPENPOSE_SDXL: &str = "controlnet/openpose-sdxl.safetensors";

/// Load the whole network.
pub fn load(path: &str, device: &Device, dtype: DType) -> Result<SdxlControlNet> {
    let vb = unsafe { VarBuilder::from_files(&[path], dtype, device) }?;
    SdxlControlNet::new(&vb, device, dtype)
}

/// The attention depth of the middle block, matching the UNet's.
const MID_DEPTH: usize = 10;

/// The residuals one ControlNet pass produces: one per UNet skip, plus the middle.
pub struct ControlResiduals {
    pub down: Vec<Tensor>,
    pub mid: Tensor,
}

/// The full network: a copy of the UNet's down path and middle block, reading a control
/// image, emitting residuals.
pub struct SdxlControlNet {
    time_1: Linear,
    time_2: Linear,
    label_1: Linear,
    label_2: Linear,
    conv_in: Conv2d,
    cond: CondEmbedding,
    /// One entry per `input_blocks.N` of the UNet, in the same order, so residual `i`
    /// pairs with skip `i`.
    blocks: Vec<Vec<Layer>>,
    middle: Vec<Layer>,
    proj: ResidualProjections,
    device: Device,
    dtype: DType,
}

impl SdxlControlNet {
    /// Build the trunk by walking the UNet's OWN `INPUT_SPEC`.
    ///
    /// The two networks must stay structurally identical or the residuals pair with the
    /// wrong skips, so the shape is read from one table rather than transcribed twice.
    /// Only the NAMES differ, and they differ mechanically: the LDM layout numbers the
    /// blocks flat (`input_blocks.4.0`), diffusers groups them by resolution level
    /// (`down_blocks.1.resnets.0`), which is what the walk below tracks.
    pub fn new(vb: &VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let pad = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let stride2 = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let vb_d = vb.pp("down_blocks");

        let mut blocks: Vec<Vec<Layer>> = Vec::with_capacity(INPUT_SPEC.len());
        let mut level = 0usize;
        let mut within = 0usize;
        let mut width = BASE;
        // Where each level's cross-attentions start in the identity adapter's table.
        // ControlNet never receives identity conditioning, so this only has to be
        // consistent, not shared with the UNet's numbering.
        for (in_c, out_c, depth) in INPUT_SPEC.iter() {
            if *in_c == 0 && *out_c == 0 && *depth == 0 {
                let conv = conv2d(
                    width,
                    width,
                    3,
                    stride2,
                    &vb_d
                        .pp(level.to_string())
                        .pp("downsamplers")
                        .pp("0")
                        .pp("conv"),
                )?;
                blocks.push(vec![Layer::Down(conv)]);
                level += 1;
                within = 0;
                continue;
            }
            let vb_l = vb_d.pp(level.to_string());
            let mut layers = vec![Layer::Res(ResBlock::new_diffusers(
                *in_c,
                *out_c,
                &vb_l.pp("resnets").pp(within.to_string()),
            )?)];
            if *depth > 0 {
                layers.push(Layer::Attn(SpatialTransformer::new(
                    *out_c,
                    *depth,
                    &vb_l.pp("attentions").pp(within.to_string()),
                )?));
            }
            blocks.push(layers);
            width = *out_c;
            within += 1;
        }

        let vb_m = vb.pp("mid_block");
        let middle = vec![
            Layer::Res(ResBlock::new_diffusers(
                MID_WIDTH,
                MID_WIDTH,
                &vb_m.pp("resnets").pp("0"),
            )?),
            Layer::Attn(SpatialTransformer::new(
                MID_WIDTH,
                MID_DEPTH,
                &vb_m.pp("attentions").pp("0"),
            )?),
            Layer::Res(ResBlock::new_diffusers(
                MID_WIDTH,
                MID_WIDTH,
                &vb_m.pp("resnets").pp("1"),
            )?),
        ];

        Ok(Self {
            time_1: linear_wb(EMB_DIM, BASE, &vb.pp("time_embedding").pp("linear_1"))?,
            time_2: linear_wb(EMB_DIM, EMB_DIM, &vb.pp("time_embedding").pp("linear_2"))?,
            label_1: linear_wb(EMB_DIM, LABEL_DIM, &vb.pp("add_embedding").pp("linear_1"))?,
            label_2: linear_wb(EMB_DIM, EMB_DIM, &vb.pp("add_embedding").pp("linear_2"))?,
            conv_in: conv2d(LATENT_CHANNELS, BASE, 3, pad, &vb.pp("conv_in"))?,
            cond: CondEmbedding::new(&vb.pp("controlnet_cond_embedding"))?,
            blocks,
            middle,
            proj: ResidualProjections::new(vb, &SKIP_WIDTHS, MID_WIDTH)?,
            device: device.clone(),
            dtype,
        })
    }

    /// Run the trunk and project every captured feature map into a residual.
    ///
    /// `control` is `[b, 3, h, w]` in 0..1 at the IMAGE resolution; the encoder brings it
    /// to the latent grid. `scale` attenuates the whole set - 0 reproduces the base model
    /// exactly, which is what makes the feature safe to leave wired in.
    pub fn forward(
        &self,
        x: &Tensor,
        t: f32,
        ctx: &Tensor,
        label: &Tensor,
        control: &Tensor,
        scale: f32,
    ) -> Result<ControlResiduals> {
        let t_emb = sinusoidal(&[t], BASE)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let emb = self.time_2.forward(&self.time_1.forward(&t_emb)?.silu()?)?;
        let ctx = ctx.to_device(&self.device)?.to_dtype(self.dtype)?;
        let label = label.to_device(&self.device)?.to_dtype(self.dtype)?;
        let y = self
            .label_2
            .forward(&self.label_1.forward(&label)?.silu()?)?;
        let emb = emb.add(&y)?;

        // The control signal enters ONCE, added to the stem - it is not re-injected per
        // block. Everything downstream is the UNet's own computation over a stem that
        // now carries the structure.
        let cond = self
            .cond
            .forward(&control.to_device(&self.device)?.to_dtype(self.dtype)?)?;
        let mut h = self
            .conv_in
            .forward(&x.to_device(&self.device)?.to_dtype(self.dtype)?)?
            .add(&cond)?;

        let mut captured = vec![h.clone()];
        for block in &self.blocks {
            for layer in block {
                h = layer.forward(&h, &emb, &ctx)?;
            }
            captured.push(h.clone());
        }
        for layer in &self.middle {
            h = layer.forward(&h, &emb, &ctx)?;
        }

        if captured.len() != self.proj.len() {
            return Err(crate::tensor::Error(format!(
                "controlnet captured {} feature maps but has {} projections",
                captured.len(),
                self.proj.len()
            )));
        }
        let mut down = Vec::with_capacity(captured.len());
        for (i, c) in captured.iter().enumerate() {
            down.push(self.proj.project_down(i, c)?.affine(scale, 0.0)?);
        }
        let mid = self.proj.project_mid(&h)?.affine(scale, 0.0)?;
        Ok(ControlResiduals { down, mid })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One residual projection per skip, at the width that skip carries.
    ///
    /// A ControlNet residual is added to a specific stage of the UNet. If the counts or
    /// the widths disagree the addition either fails on shape - the good case - or lands
    /// on the wrong stage and silently shifts the structure the control was meant to
    /// impose. Pin both against the UNet's own skip table.
    #[test]
    fn the_residual_projections_match_our_skips() {
        use crate::inference::model::sdxl::unet::skip_widths;
        assert_eq!(
            SKIP_WIDTHS.to_vec(),
            skip_widths(),
            "the ControlNet residual widths no longer match the UNet's skip stack"
        );
    }

    /// The control encoder's total stride must be the VAE's factor of 8.
    ///
    /// It hands its output to a feature map on the latent grid; any other stride and the
    /// control image is misaligned with the image it is supposed to constrain - by a
    /// factor, so every pose lands in the wrong place.
    #[test]
    fn the_control_encoder_downsamples_by_exactly_eight() {
        let stride: usize = COND_CHANNELS.iter().map(|(_, _, s)| *s).product();
        assert_eq!(
            stride, 8,
            "control encoder strides to 1/{stride}, but the latent is 1/8"
        );
        // And the channel chain has to be continuous: each block's input is the previous
        // block's output, or the weights would not load.
        for w in COND_CHANNELS.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "channel chain breaks between {:?} and {:?}",
                w[0], w[1]
            );
        }
    }
}
