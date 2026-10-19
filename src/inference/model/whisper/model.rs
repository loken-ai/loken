//! An audio encoder over mel frames and a text decoder that attends to it.
//!
//! The encoder takes a mel spectrogram, halves its length through two convolutions, and runs a
//! stack of blocks over the result - once per clip. The decoder runs its own stack over the
//! tokens emitted so far, each block reading the encoder's output as well as its own past, and
//! projects the last hidden state back through the embedding table to score the next token. The
//! audio front-end, the config and the tokenizer constants live alongside, in the module above.
//!
//! On a card the projections a decode step runs, the head included, are held in half precision
//! and the stream between them runs F32; the projections over the whole audio window are held
//! in half precision when the card multiplies them faster that way ([`half_matmul_pays`]), and
//! in F32 otherwise. The convolutions, norms and biases stay F32. Both embedding tables
//! stay on HOST: a decode step looks up its token and position there and uploads the one row
//! it built. The decoder's self attention writes into buffers reserved for its whole context.

use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention, Reserved};
use crate::inference::model::block::CrossAttentionBlock;
use crate::inference::model::half::half_matmul_pays;
use crate::inference::model::whisper::Config;
use crate::tensor::layer::{
    conv1d, embedding, layer_norm, same_length_1d, Conv1d, Conv1dConfig, Embedding, LayerNorm,
    Linear,
};
use crate::tensor::ops::Activation;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// The `[to, from]` projection under `vb`, with its bias when `bias` is set: read on the host
/// and held at its resident half precision on a card when `half` is set, and in F32 otherwise.
fn projection(from: usize, to: usize, bias: bool, half: bool, vb: &VarBuilder) -> Result<Linear> {
    if !(half && vb.device().is_cuda()) {
        let w = vb.get((to, from), "weight")?;
        let b = bias.then(|| vb.get(to, "bias")).transpose()?;
        return Linear::new(w, b);
    }
    let host = vb.to(DType::F32, &Device::Cpu);
    let w = host.get((to, from), "weight")?;
    let b = bias.then(|| host.get(to, "bias")).transpose()?;
    crate::inference::model::half::linear(&w, b.as_ref(), vb.device())
}

/// One attention of this model, read off the four names its checkpoint uses.
///
/// The key projection carries no bias and the other three do - the model was trained that way,
/// and a bias of zeros would answer the same but cost a row of weights per layer to say so.
///
/// `kv` is what the attention keeps: the encoder's own attention keeps nothing, the decoder's
/// self attention writes each step into its reserved buffers, and the decoder's cross attention
/// projects the encoder's output once and holds it for the whole generation.
///
/// `half` says, for the query and output projections and then for the key and value ones,
/// whether they are held in half precision.
fn attention(
    n_state: usize,
    n_head: usize,
    kv: Kv,
    [half_q, half_kv]: [bool; 2],
    vb: VarBuilder,
) -> Result<MultiHeadAttention> {
    MultiHeadAttention::new(
        projection(n_state, n_state, true, half_q, &vb.pp("q_proj"))?,
        projection(n_state, n_state, false, half_kv, &vb.pp("k_proj"))?,
        projection(n_state, n_state, true, half_kv, &vb.pp("v_proj"))?,
        projection(n_state, n_state, true, half_q, &vb.pp("out_proj"))?,
        n_state,
        n_head,
        n_head,
        kv,
    )
    .map(MultiHeadAttention::with_half_cache)
}

/// One block of either stack, read off the names whisper's checkpoint uses.
///
/// `context` is how many positions the decoder's self attention reserves; the audio tower has
/// none, attends to all of its positions at once and carries no cross-attention weights. Both
/// feed-forward projections carry a bias here, and the activation between them is the
/// error-function GELU.
///
/// `window_half` says whether the projections that read the whole audio window - all of the
/// tower's, and the keys and values the decoder projects from it - are held in half precision.
/// The decoder's other projections read one step at a time and always are.
fn block(
    n_state: usize,
    n_head: usize,
    context: Option<usize>,
    window_half: bool,
    vb: VarBuilder,
) -> Result<CrossAttentionBlock> {
    let (self_kv, cross_attn, step_half) = match context {
        Some(span) => (
            Kv::Reserved(Reserved::new(span)),
            Some(attention(
                n_state,
                n_head,
                Kv::Fixed(None),
                [true, window_half],
                vb.pp("encoder_attn"),
            )?),
            true,
        ),
        None => (Kv::None, None, window_half),
    };
    let self_attn = attention(
        n_state,
        n_head,
        self_kv,
        [step_half, step_half],
        vb.pp("self_attn"),
    )?;
    CrossAttentionBlock::with_projections(
        self_attn,
        cross_attn,
        n_state,
        n_state * 4,
        1e-5,
        Activation::Gelu,
        &vb,
        |from, to, vb| projection(from, to, true, step_half, vb),
    )
}

/// Whether the projections over the whole audio window are held in half precision on the
/// device `vb` loads to.
fn window_half(vb: &VarBuilder, cfg: &Config) -> Result<bool> {
    half_matmul_pays(vb.device(), cfg.max_source_positions, cfg.d_model)
}

/// `[length, channels]` sin/cos table, built host-side.
fn sinusoids(length: usize, channels: usize) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let half = channels / 2;
    let mut data = vec![0f32; length * channels];
    for t in 0..length {
        for i in 0..half {
            let scaled = t as f32 * (i as f32 * (-log_timescale_increment)).exp();
            data[t * channels + i] = scaled.sin();
            data[t * channels + half + i] = scaled.cos();
        }
    }
    Tensor::from_vec_f32(data, vec![length, channels])
}

/// The audio tower: two convolutions over the mel bins, then a stack that attends to itself.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// `[n_ctx, n_state]`, kept on HOST; the per-chunk slice uploads once.
    positional_embedding: Tensor,
    blocks: Vec<CrossAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    /// Public so Ultravox can load its Whisper-large-v3-turbo audio tower (the
    /// `audio_tower.*` tensors use the same names) without the text decoder.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let (width, heads) = (cfg.d_model, cfg.encoder_attention_heads);
        let window_half = window_half(&vb, cfg)?;
        // The mel front-end: a three-tap convolution that keeps the length, then the same
        // one striding by two, which is where the sequence halves.
        let taps = same_length_1d(3, 1);
        let halving = Conv1dConfig { stride: 2, ..taps };
        Ok(Self {
            conv1: conv1d(cfg.num_mel_bins, width, 3, taps, &vb.pp("conv1"))?,
            conv2: conv1d(width, width, 3, halving, &vb.pp("conv2"))?,
            // Nothing was stored for the audio positions, so the table is computed.
            positional_embedding: sinusoids(cfg.max_source_positions, width)?,
            blocks: (0..cfg.encoder_layers)
                .map(|i| {
                    block(
                        width,
                        heads,
                        None,
                        window_half,
                        vb.pp(format!("layers.{i}")),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            ln_post: layer_norm(width, 1e-5, &vb.pp("layer_norm"))?,
        })
    }

    /// Override the (default sinusoidal) positional embedding - Ultravox loads the
    /// STORED `embed_positions.weight` so the encoder is bit-exact vs the frozen
    /// Whisper the projector was trained against (computed sinusoids differ ~1e-5).
    pub fn set_positional_embedding(&mut self, pos: Tensor) {
        self.positional_embedding = pos;
    }

    /// Mel `[1, n_mels, frames]` -> audio features `[1, frames/2, n_state]`.
    pub fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.shape().dims3()?;
        let positional_embedding = self
            .positional_embedding
            .narrow(0, 0, seq_len)?
            .to_device(&x.device())?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            if flush_kv_cache {
                block.clear();
            }
            x = block.forward(&x, None, Mask::All)?;
        }
        self.ln_post.forward(&x)
    }
}

/// The text tower: token and position embeddings, a stack that also attends to the audio, and
/// a head that is the embedding table read the other way round.
#[derive(Debug, Clone)]
pub struct TextDecoder {
    /// `[vocab, n_state]` on HOST: a step looks up its own tokens there.
    token_embedding: Embedding,
    /// `[n_ctx, n_state]` on HOST, read beside the token rows.
    positional_embedding: Tensor,
    /// The weight-tied head: the token table as a projection onto the vocabulary, on the
    /// compute device.
    head: Linear,
    blocks: Vec<CrossAttentionBlock>,
    ln: LayerNorm,
    /// How many tokens have flowed into the self-attn KV cache so far.
    /// Reset to 0 on flush, incremented each forward - picks the right
    /// slice of `positional_embedding` so incremental decode steps line
    /// up with what the full-prefix forward would produce.
    cache_offset: usize,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let (width, heads) = (cfg.d_model, cfg.decoder_attention_heads);
        let span = cfg.max_target_positions;
        let host = vb.to(DType::F32, &Device::Cpu);
        let token_embedding = embedding(cfg.vocab_size, width, &host.pp("embed_tokens"))?;
        // Text positions were learned rather than computed, so the table is read.
        let positional_embedding = host.get((span, width), "embed_positions.weight")?;
        let head = if vb.device().is_cuda() {
            crate::inference::model::half::linear(token_embedding.table(), None, vb.device())?
        } else {
            Linear::new(token_embedding.table().to_device(vb.device())?, None)?
        };
        let window_half = window_half(&vb, cfg)?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                let vb = vb.pp(format!("layers.{i}"));
                block(width, heads, Some(span), window_half, vb)
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(width, 1e-5, &vb.pp("layer_norm"))?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            head,
            blocks,
            ln,
            cache_offset: 0,
        })
    }

    /// Token ids `[1, n]` (u32) + audio features -> hidden `[1, n, n_state]`.
    ///
    /// The ids are read on the host, where both tables are; the embedded rows are what goes to
    /// the device. Each block's self attention sees the positions up to its own.
    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let last = *x.dims().last().unwrap_or(&0);
        if flush_kv_cache {
            self.reset_kv_cache();
        }
        let token_embedding = self.token_embedding.forward(&x.to_device(&Device::Cpu)?)?;
        // With incremental decode (self-attn KV cache + caller supplying
        // only the new token), `x` is the tail of the prefix that hasn't
        // been seen yet. The positional embedding must therefore start at
        // `cache_offset`, not 0, so position ids line up with the cached
        // K/V layout.
        let positional_embedding = self
            .positional_embedding
            .narrow(0, self.cache_offset, last)?;
        let mut x = token_embedding
            .broadcast_add(&positional_embedding)?
            .to_device(&xa.device())?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, Some(xa), Mask::Causal)?;
        }
        self.cache_offset += last;
        self.ln.forward(&x)
    }

    /// Hidden `[1, n, n_state]` -> logits `[1, n, vocab]` (weight-tied head).
    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(x)
    }

    /// Forget both caches: the next forward starts a new sequence over new audio.
    pub fn reset_kv_cache(&mut self) {
        self.cache_offset = 0;
        for block in self.blocks.iter_mut() {
            block.clear();
        }
    }

    /// Forget the tokens but keep what was projected from the audio: the next forward starts a
    /// new sequence over the same encoder output.
    pub fn restart_sequence(&mut self) {
        self.cache_offset = 0;
        for block in self.blocks.iter_mut() {
            block.clear_self();
        }
    }
}

/// The two towers and the config they were shaped from.
#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    /// Both stacks hang off `model.*` in the checkpoint, one prefix each.
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        Ok(Self {
            encoder: AudioEncoder::load(vb.pp("model.encoder"), &config)?,
            decoder: TextDecoder::load(vb.pp("model.decoder"), &config)?,
            config,
        })
    }

    /// Forget what both towers are holding - the next call starts on a fresh clip.
    pub fn reset_kv_cache(&mut self) {
        for block in &mut self.encoder.blocks {
            block.clear();
        }
        self.decoder.reset_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;
    use std::collections::HashMap;

    /// A block's weights, deterministic and distinct, under the names the checkpoint uses.
    fn block_weights(n_state: usize, cross: bool) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let mat = |seed: usize, rows: usize, cols: usize| -> Tensor {
            let v: Vec<f32> = (0..rows * cols)
                .map(|i| (((i * 31 + seed * 17) % 83) as f32) * 0.017 - 0.7)
                .collect();
            Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
        };
        let vec = |seed: usize, n: usize| -> Tensor {
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 7 + seed * 5) % 29) as f32) * 0.011 - 0.15)
                .collect();
            Tensor::from_vec(v, n, &Device::Cpu).unwrap()
        };

        // Every tensor gets its own seed, so a projection read into the wrong slot shows.
        let attention_at = |map: &mut HashMap<String, Tensor>, p: &str, base: usize| {
            for (i, name) in ["q_proj", "v_proj", "out_proj"].iter().enumerate() {
                map.insert(
                    format!("{p}.{name}.weight"),
                    mat(base + i * 2, n_state, n_state),
                );
                map.insert(format!("{p}.{name}.bias"), vec(base + i * 2 + 1, n_state));
            }
            // The key projection is the one without a bias.
            map.insert(
                format!("{p}.k_proj.weight"),
                mat(base + 7, n_state, n_state),
            );
        };
        attention_at(&mut map, "self_attn", 1);
        if cross {
            attention_at(&mut map, "encoder_attn", 20);
        }
        for (i, p) in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ]
        .iter()
        .enumerate()
        {
            map.insert(format!("{p}.weight"), vec(40 + i * 2, n_state));
            map.insert(format!("{p}.bias"), vec(41 + i * 2, n_state));
        }
        map.insert("fc1.weight".into(), mat(50, n_state * 4, n_state));
        map.insert("fc1.bias".into(), vec(51, n_state * 4));
        map.insert("fc2.weight".into(), mat(52, n_state, n_state * 4));
        map.insert("fc2.bias".into(), vec(53, n_state));
        map
    }

    /// A decoder block fed one token at a time answers what it answers fed the whole prefix.
    ///
    /// This is the wiring rather than the arithmetic: that the self attention is the one that
    /// accumulates, that the cross attention is the one that is projected once and kept, and
    /// that the causal positions count from where the cache ends rather than from zero.
    /// Each of those is a choice made where the block is built, and each of them fails by
    /// returning a plausible tensor of the right shape.
    #[test]
    fn a_decoder_block_stepped_matches_the_same_block_given_the_prefix() {
        let (n_state, n_head, n, src) = (16usize, 4usize, 5usize, 7usize);
        let vb = VarBuilder::from_tensors(block_weights(n_state, true), DType::F32, &Device::Cpu);

        let seq: Vec<f32> = (0..n * n_state)
            .map(|i| (((i * 13 + 3) % 61) as f32) * 0.021 - 0.6)
            .collect();
        let xs = Tensor::from_vec(seq, (1, n, n_state), &Device::Cpu).unwrap();
        let audio: Vec<f32> = (0..src * n_state)
            .map(|i| (((i * 19 + 7) % 53) as f32) * 0.015 - 0.4)
            .collect();
        let xa = Tensor::from_vec(audio, (1, src, n_state), &Device::Cpu).unwrap();

        let mut whole = block(n_state, n_head, Some(n), true, vb.clone()).unwrap();
        whole.clear();
        let at_once = whole.forward(&xs, Some(&xa), Mask::Causal).unwrap();

        let mut stepped = block(n_state, n_head, Some(n), true, vb).unwrap();
        let mut rows = Vec::new();
        for t in 0..n {
            let step = xs.narrow(1, t, 1).unwrap();
            // Only the first call flushes; the rest extend what it left.
            if t == 0 {
                stepped.clear();
            }
            rows.push(stepped.forward(&step, Some(&xa), Mask::Causal).unwrap());
        }
        let step_by_step = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1).unwrap();

        let a = at_once.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = step_by_step
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gap = a
            .iter()
            .zip(&b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            gap < 1e-4,
            "stepped and whole-prefix decoding differ by {gap}"
        );
    }
}

impl AudioEncoder {
    /// How many blocks the encoder stacks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl TextDecoder {
    /// How many blocks the decoder stacks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}
