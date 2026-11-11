//! The transformer of the codec decoder, at the frame rate.
//!
//! Its layers are a transformer of the generic route, placed by `HeteroPlan` and the ladder
//! with their own demand.
//! Each layer windows its attention to the codec's sliding window.
//! The glue projects the frames' rows into the layers' width before them and back after the
//! final norm.

use crate::inference::generic_transformer::class_load::{self, Rows};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::tensor::safetensors::MmapedSafetensors;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Error, Result, Tensor};

/// The class of the codec's transformer, nested in the codec's configuration.
pub const CODEC_TRANSFORMER_CLASS: &str = "Qwen3TTSTokenizerV2DecoderTransformerModel";

/// The prefix of the transformer's tensors in the codec's file.
pub(super) const PREFIX: &str = "decoder.pre_transformer.";

/// A linear layer `[out, in]` with its optional bias `[out]`.
pub(super) struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    /// The stored `{name}.weight`, with `{name}.bias` when the checkpoint has it.
    pub(super) fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let bias = format!("{name}.bias");
        let b = match vb.contains(&bias) {
            true => Some(vb.get(shape[0], &bias)?),
            false => None,
        };
        Ok(Self {
            w: vb.get(shape, &format!("{name}.weight"))?,
            b,
        })
    }

    /// `x [S, in]` to `[S, out]`.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul_t(&self.w)?;
        match &self.b {
            None => Ok(y),
            Some(b) => y.broadcast_add(&b.reshape((1, b.dims()[0]))?),
        }
    }
}

/// The input projection, the layers with their final norm, and the output projection.
///
/// The projections sit on the decoder's device.
/// The layers sit where their placement put them.
pub struct Transformer {
    input_proj: Linear,
    layers: std::sync::Mutex<GenericHeteroTransformer>,
    count: usize,
    output_proj: Linear,
}

impl Transformer {
    /// The projections under `vb`, `pre_transformer` of the decoder, and `layers`.
    pub fn new(vb: &VarBuilder, layers: GenericHeteroTransformer) -> Result<Self> {
        Ok(Self {
            input_proj: Linear::load(vb, "input_proj")?,
            count: layers.layers().len(),
            layers: std::sync::Mutex::new(layers),
            output_proj: Linear::load(vb, "output_proj")?,
        })
    }

    /// The layers of the codec configuration `root` and its file `weights`, placed on `cards`.
    ///
    /// The cache holds `rows`, the positions a pass of the front reaches.
    pub fn layers(
        root: &serde_json::Value,
        weights: &std::path::Path,
        rows: usize,
        cards: &[(usize, u64)],
    ) -> Result<GenericHeteroTransformer> {
        let shards = [weights.to_string_lossy().to_string()];
        let loaded = class_load::load_class_from(
            root,
            &shards,
            CODEC_TRANSFORMER_CLASS,
            Rows::Exactly(rows),
            0,
            cards,
        )?;
        Ok(loaded.model)
    }

    /// `x [S, latent]` of a pass that starts afresh to `[S, latent]`.
    ///
    /// The layers see the rows at positions `0..S`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let h = self.input_proj.forward(x)?;
        let rows = {
            let mut layers = self
                .layers
                .lock()
                .map_err(|_| Error::msg("the codec's transformer is poisoned"))?;
            layers.forward_embeddings_rows(&h)?
        };
        let rows = match rows.dtype() {
            DType::F32 => rows,
            _ => rows.to_dtype(DType::F32)?,
        };
        self.output_proj.forward(&rows.to_device(&device)?)
    }

    /// Where the layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        match self.layers.lock() {
            Ok(layers) => super::super::layer_runs(&layers),
            Err(_) => Vec::new(),
        }
    }

    /// Layers of the transformer.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the transformer has no layer.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// The bytes of the layers and the final norm in the codec's file `st`, which their placement
/// holds apart from the decoder.
pub(super) fn layer_bytes(st: &MmapedSafetensors) -> u64 {
    let layers = format!("{PREFIX}layers.");
    let norm = format!("{PREFIX}norm.");
    st.byte_sizes()
        .iter()
        .filter(|(n, _)| n.starts_with(&layers) || n.starts_with(&norm))
        .map(|(_, b)| *b as u64)
        .sum()
}
