//! An audio encoder over mel frames and a text decoder that attends to it.
//!
//! The encoder takes a mel spectrogram, halves its length through two convolutions, and runs a
//! stack of blocks over the result - once per clip. The decoder runs its own stack over the
//! tokens emitted so far, each block reading the encoder's output as well as its own past, and
//! projects the last hidden state back through the embedding table to score the next token. The
//! audio front-end, the config and the tokenizer constants live alongside, in the module above.
//!
//! Everything here is F32. The causal mask and both positional tables stay on HOST and only the
//! rows a step actually needs are uploaded - a decode step wants one `[1, n_state]` row, not a
//! narrow across 1.4 MB of device memory. The self-attention caches grow device-side instead.

use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention};
use crate::inference::model::block::CrossAttentionBlock;
use crate::inference::model::whisper::Config;
use crate::tensor::layer::{
    conv1d, embedding, layer_norm, linear, linear_no_bias, same_length_1d, Conv1d, Conv1dConfig,
    Embedding, LayerNorm,
};
use crate::tensor::ops::Activation;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// One attention of this model, read off the four names its checkpoint uses.
///
/// The key projection carries no bias and the other three do - the model was trained that way,
/// and a bias of zeros would answer the same but cost a row of weights per layer to say so.
///
/// `kv` is what the attention keeps: the encoder's own attention and the decoder's self
/// attention grow their cache a step at a time, while the decoder's cross attention projects
/// the encoder's output once and holds it for the whole generation.
fn attention(n_state: usize, n_head: usize, kv: Kv, vb: VarBuilder) -> Result<MultiHeadAttention> {
    MultiHeadAttention::new(
        linear(n_state, n_state, &vb.pp("q_proj"))?,
        linear_no_bias(n_state, n_state, &vb.pp("k_proj"))?,
        linear(n_state, n_state, &vb.pp("v_proj"))?,
        linear(n_state, n_state, &vb.pp("out_proj"))?,
        n_state,
        n_head,
        n_head,
        kv,
    )
}

/// One block of either stack, read off the names whisper's checkpoint uses.
///
/// `ca` says whether there is an encoder to attend to: the audio tower has none, and its
/// checkpoint carries no cross-attention weights to read. Both feed-forward projections carry a
/// bias here, and the activation between them is the error-function GELU.
fn block(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<CrossAttentionBlock> {
    let self_attn = attention(n_state, n_head, Kv::Growing(None), vb.pp("self_attn"))?;
    let cross_attn = match ca {
        true => Some(attention(
            n_state,
            n_head,
            Kv::Fixed(None),
            vb.pp("encoder_attn"),
        )?),
        false => None,
    };
    CrossAttentionBlock::new(
        self_attn,
        cross_attn,
        n_state,
        n_state * 4,
        1e-5,
        true,
        Activation::Gelu,
        &vb,
    )
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
                .map(|i| block(width, heads, false, vb.pp(format!("layers.{i}"))))
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
    token_embedding: Embedding,
    /// `[n_ctx, n_state]`, kept on HOST; per-step slices upload tiny rows.
    positional_embedding: Tensor,
    /// Weight-tied lm-head, pre-transposed to `[n_state, vocab]` on the
    /// compute device at load (transposing 160 MB per token would hurt).
    lm_head_t: Tensor,
    blocks: Vec<CrossAttentionBlock>,
    ln: LayerNorm,
    /// `[n_ctx, n_ctx]` causal mask on HOST (sliced + uploaded per step).
    mask: Tensor,
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
        let token_embedding = embedding(cfg.vocab_size, width, &vb.pp("embed_tokens"))?;
        // Text positions were learned rather than computed, so the table is read; it stays on
        // host because a step only ever wants a row or two of it.
        let positional_embedding = vb
            .to(vb.dtype(), &Device::Cpu)
            .get((span, width), "embed_positions.weight")?;
        let lm_head_t = token_embedding.table().transpose(0, 1)?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| block(width, heads, true, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(width, 1e-5, &vb.pp("layer_norm"))?;
        // Added to the self-attention scores, so it says what a query is allowed to have seen:
        // row `i` leaves the columns up to its own position alone and sends the rest to minus
        // infinity, where the softmax gives them no weight at all.
        let mut future = vec![0f32; span * span];
        for (i, row) in future.chunks_mut(span).enumerate() {
            row[i + 1..].fill(f32::NEG_INFINITY);
        }
        Ok(Self {
            token_embedding,
            positional_embedding,
            lm_head_t,
            blocks,
            ln,
            mask: Tensor::from_vec_f32(future, vec![span, span])?,
            cache_offset: 0,
        })
    }

    /// Token ids `[1, n]` (u32) + audio features -> hidden `[1, n, n_state]`.
    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let last = *x.dims().last().unwrap_or(&0);
        if flush_kv_cache {
            self.cache_offset = 0;
        }
        let token_embedding = self.token_embedding.forward(x)?;
        // With incremental decode (self-attn KV cache + caller supplying
        // only the new token), `x` is the tail of the prefix that hasn't
        // been seen yet. The positional embedding must therefore start at
        // `cache_offset`, not 0, so position ids line up with the cached
        // K/V layout. For the legacy full-prefix call (cache_offset == 0,
        // x = whole prompt) this trivially reduces to narrow(0, 0, last).
        let positional_embedding = self
            .positional_embedding
            .narrow(0, self.cache_offset, last)?
            .to_device(&token_embedding.device())?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            if flush_kv_cache {
                block.clear();
            }
            x = block.forward(&x, Some(xa), Mask::Table(&self.mask))?;
        }
        self.cache_offset += last;
        self.ln.forward(&x)
    }

    /// Hidden `[1, n, n_state]` -> logits `[1, n, vocab]` (weight-tied head).
    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, d) = x.shape().dims3()?;
        let logits = x.reshape((b * n, d))?.matmul(&self.lm_head_t)?;
        let vocab = *logits.dims().last().unwrap_or(&0);
        logits.reshape((b, n, vocab))
    }

    pub fn reset_kv_cache(&mut self) {
        // Defensive: pair the cache reset with the position-id reset
        // so callers that reset_kv_cache() and then call forward() with
        // flush=false don't see stale positional offsets. The hot
        // transcribe path always passes flush=true on iter 0 so this
        // is rarely the deciding write - but explicit is safer.
        self.cache_offset = 0;
        for block in self.blocks.iter_mut() {
            block.clear();
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
        let mut attention_at = |map: &mut HashMap<String, Tensor>, p: &str, base: usize| {
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

    fn causal_table(n: usize) -> Tensor {
        let mut v = vec![0f32; n * n];
        for r in 0..n {
            for c in (r + 1)..n {
                v[r * n + c] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(v, (n, n), &Device::Cpu).unwrap()
    }

    /// A decoder block fed one token at a time answers what it answers fed the whole prefix.
    ///
    /// This is the wiring rather than the arithmetic: that the self attention is the one that
    /// accumulates, that the cross attention is the one that is projected once and kept, and
    /// that the mask table is cut to the rows being asked for rather than read from the top.
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
        let mask = causal_table(n);

        let mut whole = block(n_state, n_head, true, vb.clone()).unwrap();
        whole.clear();
        let at_once = whole.forward(&xs, Some(&xa), Mask::Table(&mask)).unwrap();

        let mut stepped = block(n_state, n_head, true, vb).unwrap();
        let mut rows = Vec::new();
        for t in 0..n {
            let step = xs.narrow(1, t, 1).unwrap();
            // Only the first call flushes; the rest extend what it left.
            if t == 0 {
                stepped.clear();
            }
            rows.push(
                stepped
                    .forward(&step, Some(&xa), Mask::Table(&mask))
                    .unwrap(),
            );
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
