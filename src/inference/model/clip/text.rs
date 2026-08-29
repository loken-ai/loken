//! CLIP's text tower - FLUX's second text encoder, the one that pools.
//!
//! Runs on the host in f32. At ~120M parameters it costs a fraction of a second once per
//! image, so the card is left to the model that needs it, and f32 removes any question of
//! whether the conditioning matches the reference.
//!
//! Two outputs, and which one a caller wants is not a detail: [`Transformer::forward`] pools
//! the final-norm state at the EOS position, which is what FLUX conditions on, while
//! [`Transformer::hidden_state`] stops short of the last layers and skips the final norm,
//! which is what SDXL conditions on.

use crate::tensor::Module;
use crate::tensor::layer::{embedding, layer_norm, linear, Embedding, LayerNorm, Mlp};
use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention};
use crate::tensor::ops::Activation;
use crate::tensor::VarBuilder;
use crate::tensor::{self, Tensor};

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub activation: Activation,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub pad_with: Option<String>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
}

#[derive(Clone, Debug)]
struct Embeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
}

impl Embeddings {
    fn new(vs: &VarBuilder, c: &Config) -> tensor::Result<Self> {
        Ok(Self {
            token_embedding: embedding(c.vocab_size, c.embed_dim, &vs.pp("token_embedding"))?,
            position_embedding: embedding(
                c.max_position_embeddings,
                c.embed_dim,
                &vs.pp("position_embedding"),
            )?,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> tensor::Result<Tensor> {
        let (_b, seq) = input_ids.shape().dims2()?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let pos_ids = Tensor::from_vec_u32((0..seq as u32).collect(), vec![1, seq])?;
        let position_embedding = self.position_embedding.forward(&pos_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

/// One attention of this tower, read off the four names its checkpoint uses.
///
/// It keeps nothing: the tower is handed a whole prompt at a time, so there is no past to
/// carry between calls.
fn attention(vs: &VarBuilder, c: &Config) -> tensor::Result<MultiHeadAttention> {
    let width = c.embed_dim;
    let one = |name: &str| linear(width, width, &vs.pp(name));
    MultiHeadAttention::new(
        one("q_proj")?,
        one("k_proj")?,
        one("v_proj")?,
        one("out_proj")?,
        width,
        c.num_attention_heads,
        c.num_attention_heads,
        Kv::None,
    )
}

/// The block's feed-forward, read off this checkpoint's two names.
fn mlp(vs: &VarBuilder, c: &Config) -> tensor::Result<Mlp> {
    Ok(Mlp::new(
        linear(c.embed_dim, c.intermediate_size, &vs.pp("fc1"))?,
        c.activation,
        linear(c.intermediate_size, c.embed_dim, &vs.pp("fc2"))?,
    ))
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    self_attn: MultiHeadAttention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(vs: &VarBuilder, c: &Config) -> tensor::Result<Self> {
        Ok(Self {
            self_attn: attention(&vs.pp("self_attn"), c)?,
            layer_norm1: layer_norm(c.embed_dim, 1e-5, &vs.pp("layer_norm1"))?,
            mlp: mlp(&vs.pp("mlp"), c)?,
            layer_norm2: layer_norm(c.embed_dim, 1e-5, &vs.pp("layer_norm2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> tensor::Result<Tensor> {
        let h = self.layer_norm1.forward(xs)?;
        // The mask this tower builds is ALREADY the shape of the scores - one row per query,
        // one column per key, behind a batch and a head axis to broadcast over - so it is
        // added as it stands.
        //
        // It is NOT a square table to be cut down. That is what a CACHED decode needs, where
        // the queries are the tail of the sequence and the rows have to be taken from where
        // the keys end. This tower encodes the whole prompt in one pass and keeps no cache,
        // so there is no tail to find: asking for the cut reads `seq` rows off the leading
        // axis, and that axis is one long.
        let mask = match causal_mask {
            Some(shaped_like_the_scores) => Mask::Added(shaped_like_the_scores),
            None => Mask::All,
        };
        let h = self.self_attn.forward_stateless(&h, None, mask)?;
        let xs = xs.add(&h)?;
        let h2 = self.layer_norm2.forward(&xs)?;
        let h2 = self.mlp.forward(&h2)?;
        xs.add(&h2)
    }
}

#[derive(Clone, Debug)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(vs: &VarBuilder, c: &Config) -> tensor::Result<Self> {
        let vs = vs.pp("layers");
        let mut layers = Vec::new();
        for index in 0..c.num_hidden_layers {
            layers.push(EncoderLayer::new(&vs.pp(index.to_string()), c)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> tensor::Result<Tensor> {
        self.forward_upto(xs, causal_mask, self.layers.len())
    }

    /// Run the first `stop` layers. SDXL conditions on the PENULTIMATE hidden state
    /// (`layer_idx = -2`) of both text towers, so the encoder has to be able to stop
    /// one layer early - taking the final output instead shifts the whole
    /// conditioning and yields a plausible-but-wrong image.
    fn forward_upto(
        &self,
        xs: &Tensor,
        causal_mask: Option<&Tensor>,
        stop: usize,
    ) -> tensor::Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter().take(stop.min(self.layers.len())) {
            xs = layer.forward(&xs, causal_mask)?;
        }
        Ok(xs)
    }
}

/// CLIP's text tower: embeddings, `num_hidden_layers` encoder layers, a final norm.
#[derive(Clone, Debug)]
pub struct Transformer {
    embeddings: Embeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
}

impl Transformer {
    /// Rooted at the model file: callers pass `vb.pp("text_model")`.
    pub fn new(vs: VarBuilder, c: &Config) -> tensor::Result<Self> {
        Ok(Self {
            embeddings: Embeddings::new(&vs.pp("embeddings"), c)?,
            encoder: Encoder::new(&vs.pp("encoder"), c)?,
            final_layer_norm: layer_norm(c.embed_dim, 1e-5, &vs.pp("final_layer_norm"))?,
        })
    }

    fn causal_mask(seq: usize, mask_after: usize) -> tensor::Result<Tensor> {
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..seq).map(move |j| {
                    if j > i || j > mask_after {
                        f32::MIN
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec_f32(mask, vec![1, 1, seq, seq])
    }

    fn forward_ids(
        &self,
        ids: &[u32],
        b: usize,
        seq: usize,
        mask_after: usize,
    ) -> tensor::Result<Tensor> {
        let input_ids = Tensor::from_vec_u32(ids.to_vec(), vec![b, seq])?;
        let xs = self.embeddings.forward(&input_ids)?;
        let mask = Self::causal_mask(seq, mask_after)?;
        let xs = self.encoder.forward(&xs, Some(&mask))?;
        self.final_layer_norm.forward(&xs)
    }

    /// The state after `num_hidden_layers - skip` layers, with NO final norm - the
    /// conditioning SDXL feeds its UNet.
    ///
    /// `skip = 1` is the usual "penultimate" choice: the reference stops one layer
    /// short AND skips the final norm for both towers, so both details are folded in
    /// here rather than left to the caller.
    pub fn hidden_state(
        &self,
        ids: &[u32],
        b: usize,
        seq: usize,
        skip: usize,
    ) -> tensor::Result<Tensor> {
        let input_ids = Tensor::from_vec_u32(ids.to_vec(), vec![b, seq])?;
        let xs = self.embeddings.forward(&input_ids)?;
        let mask = Self::causal_mask(seq, usize::MAX)?;
        let stop = self.encoder.layers.len().saturating_sub(skip);
        self.encoder.forward_upto(&xs, Some(&mask), stop)
    }

    /// The final-norm state at each sequence's EOS position - taken as the highest token id
    /// in the row, which is what the tokenizer's EOS is by construction.
    pub fn forward(
        &self,
        input_ids: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let out = self.forward_ids(&ids, b, seq, usize::MAX)?;
        // pooled: row at the per-sequence max token id (the EOS position)
        let dim = out.dims()[2];
        let mut pooled = Vec::with_capacity(b);
        for bi in 0..b {
            let row_ids = &ids[bi * seq..][..seq];
            let eos = row_ids
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let row = out
                .narrow(0, bi, 1)
                .and_then(|t| t.narrow(1, eos, 1))
                .and_then(|t| t.reshape(vec![1, dim]))?;
            pooled.push(row);
        }
        let refs: Vec<&Tensor> = pooled.iter().collect();
        let pooled = Tensor::cat(&refs, 0)?;
        pooled.to_device(&input_ids.device())
    }
}
