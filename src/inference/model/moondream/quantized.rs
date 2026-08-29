//! Moondream: a picture read by a vision tower, answered by a text model.
//!
//! Two models that were trained together and are loaded together. The tower is a plain ViT over
//! square patches; the answer comes from the mixformer decoder, which knows nothing about
//! images. What joins them is the projection at the bottom of this file - two linear layers
//! that carry the tower's width to the decoder's, and that is the whole interface between
//! them.
//!
//! The weights arrive quantised and there is no exception: even the patch embedding, which
//! most vision towers spell as a strided convolution, is a quantised linear here - the picture
//! is cut into 14x14 RGB squares, each flattened to its 588 values, and projected to the
//! tower's width in one matmul.

use crate::inference::model::mixformer::MixFormerSequentialForCausalLM as PhiModel;

/// Moondream configuration and vision configuration, declared here
/// with public fields so the whole model lives in this crate.
#[derive(Debug, Clone)]
pub struct Config {
    pub phi_config: crate::inference::model::mixformer::Config,
    pub vision_config: VisionConfig,
}
impl Config {
    pub fn v2() -> Self {
        Self {
            phi_config: crate::inference::model::mixformer::Config::v1_5(),
            vision_config: VisionConfig::v2(),
        }
    }
}
/// What the tower is, and where its output has to land.
///
/// The tower's width is one field, not two: the projection's input width IS the tower's output
/// width, or the two would not fit together, and a quantity spelled twice is a quantity that
/// can disagree with itself.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// What a patch becomes and what every block carries.
    pub embed_dim: usize,
    /// The width a block's own feed-forward widens to, inside the tower.
    pub hidden_features: usize,
    /// How many patch positions the learnt position table was trained for.
    pub embed_len: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    /// The projection's intermediate width, and the decoder width it has to reach.
    pub hidden_dim: usize,
    pub model_dim: usize,
    pub act: crate::tensor::ops::Activation,
}
impl VisionConfig {
    pub fn v2() -> Self {
        // The tower's width, which is also what it hands the projection.
        const WIDTH: usize = 1152;
        // The decoder's width, which is what the projection has to reach.
        const TEXT_WIDTH: usize = 2048;
        // A 378-pixel square cut into 14-pixel patches is 27 positions on a side.
        const GRID: usize = 27;
        Self {
            embed_dim: WIDTH,
            hidden_features: 4304,
            embed_len: GRID * GRID,
            num_blocks: 27,
            num_heads: 16,
            hidden_dim: TEXT_WIDTH * 4,
            model_dim: TEXT_WIDTH,
            act: crate::tensor::ops::Activation::GeluPytorchTanh,
        }
    }
}
use crate::inference::model::vit::{Qkv, VitBlock};
use crate::tensor::layer::qlinear::{
    q_layer_norm as layer_norm, qlinear_b as linear_b, QLinear as Linear, QMlp,
};
use crate::tensor::quantized::QVarBuilder as VarBuilder;
use crate::tensor::{Module, Result, Tensor};

/// The two-layer MLP that projects the tower's output into the text model's space.
///
/// The blocks' own MLPs come with them from [`vit`]; this one sits after the tower and has no
/// residual around it, so it stays here.
/// The vision tower and the projection that puts its output where the text model can read it.
///
/// An image arrives as pixels and leaves as a sequence: a 14x14 patch becomes one position,
/// its 588 channel-major values become one vector through a single linear layer, and a learnt
/// position embedding is added because nothing else tells a patch where it was.
///
/// The checkpoint nests all of this - `encoder.model.visual` for the tower, `projection.mlp`
/// for the projection - and that nesting was the only thing the four wrapper types this
/// replaces contributed.
#[derive(Debug)]
pub struct VisionEncoder {
    patch_embed: Linear,
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    norm: crate::tensor::layer::LayerNorm,
    projection: QMlp,
}

impl VisionEncoder {
    /// The edge of a patch, in pixels. It is the checkpoint's, not a choice: the patch
    /// projection's input width is `channels * PATCH * PATCH`.
    const PATCH: usize = 14;

    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let tower = vb.pp("encoder").pp("model.visual");
        // The checkpoint fuses the three projections into one `attn.qkv` and names the block's
        // norms and MLP plainly, so a block is read off in one place here.
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                let b = tower.pp(format!("blocks.{i}"));
                let (dim, hidden) = (cfg.embed_dim, cfg.hidden_features);
                VitBlock::new(
                    layer_norm(dim, 1e-5, &b.pp("norm1"))?,
                    Qkv::Fused(linear_b(dim, dim * 3, true, &b.pp("attn").pp("qkv"))?),
                    linear_b(dim, dim, true, &b.pp("attn").pp("proj"))?,
                    layer_norm(dim, 1e-5, &b.pp("norm2"))?,
                    QMlp::new(
                        linear_b(dim, hidden, true, &b.pp("mlp").pp("fc1"))?,
                        cfg.act,
                        linear_b(hidden, dim, true, &b.pp("mlp").pp("fc2"))?,
                    ),
                    dim,
                    cfg.num_heads,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            // One patch, flattened, projected to the tower's width. The input width is the
            // patch itself - three colour planes of PATCH by PATCH - rather than a number
            // repeated here, and the output is the width the rest of the tower carries.
            patch_embed: linear_b(
                3 * Self::PATCH * Self::PATCH,
                cfg.embed_dim,
                true,
                &tower.pp("patch_embed").pp("linear"),
            )?,
            // `get_f32` dequantises on the way out; the position table is read once and used
            // as a float from then on.
            pos_embed: tower.get_f32((1, cfg.embed_len, cfg.embed_dim), "pos_embed")?,
            blocks,
            norm: layer_norm(cfg.embed_dim, 1e-5, &tower.pp("norm"))?,
            projection: {
                // After the tower and outside any residual: it carries the tower's width to
                // the decoder's, which is the whole interface between the two models.
                let p = vb.pp("projection").pp("mlp");
                QMlp::new(
                    linear_b(cfg.embed_dim, cfg.hidden_dim, true, &p.pp("fc1"))?,
                    cfg.act,
                    linear_b(cfg.hidden_dim, cfg.model_dim, true, &p.pp("fc2"))?,
                )
            },
        })
    }
}

impl Module for VisionEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = crate::inference::model::patches::cut(
            xs,
            Self::PATCH,
            crate::inference::model::patches::Order::ByChannel,
        )?
        .apply(&self.patch_embed)?
        .broadcast_add(&self.pos_embed)?;
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?;
        }
        xs.apply(&self.norm)?.apply(&self.projection)
    }
}

pub struct Model {
    pub text_model: PhiModel,
    pub vision_encoder: VisionEncoder,
}

impl Model {
    /// The two halves, each rooted at the name the checkpoint files it under.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            text_model: PhiModel::new_v2(&config.phi_config, vb.pp("text_model"))?,
            vision_encoder: VisionEncoder::new(&config.vision_config, vb.pp("vision_encoder"))?,
        })
    }

    pub fn vision_encoder(&self) -> &VisionEncoder {
        &self.vision_encoder
    }

    pub fn text_model(&mut self) -> &mut PhiModel {
        &mut self.text_model
    }
}
