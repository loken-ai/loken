//! Parler-TTS: audio tokens from a text description and a prompt.
//!
//! Three models in a trench coat. A T5 encoder turns the voice description into a sequence
//! the decoder attends to; the decoder is a causal transformer that predicts several
//! codebooks of audio tokens at once, one head per codebook; and the DAC codec turns those
//! tokens back into a waveform. This module holds the middle one and the loop that drives
//! all three.
//!
//! What is unusual about the decoder is the codebooks. A step embeds one token per codebook
//! and SUMS the embeddings into a single hidden state, so the codebooks share a trunk and
//! differ only at the heads; and a codebook does not start until the step reaches it, which
//! is the delay pattern the codec was trained with.

use crate::inference::codec::dac;
use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention};
use crate::inference::model::block::CrossAttentionBlock;
use crate::inference::model::t5::encoder as t5;
use crate::inference::sample::token_sampling::LogitsProcessor;
use crate::tensor::layer::{embedding, layer_norm, linear, linear_no_bias};
use crate::tensor::layer::{Embedding, LayerNorm, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::VarBuilder;
use crate::tensor::{IndexOp, Result, Tensor};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct DecoderConfig {
    /// Audio tokens per codebook, which is what the decoder samples from.
    pub vocab_size: usize,
    /// The learnt position table's length, and so the longest utterance.
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    /// The width the feed-forward widens to between its two projections.
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    /// Key heads, when fewer than query heads. Absent means one each.
    pub num_key_value_heads: Option<usize>,
    /// The same for the attention over the text encoder's output, which is allowed to group
    /// differently from the decoder's attention to its own past.
    pub num_cross_attention_key_value_heads: Option<usize>,
    pub activation_function: Activation,
    pub hidden_size: usize,
    /// Whether the embedding is scaled by the square root of the width on the way in.
    pub scale_embedding: bool,
    /// How many streams the audio is split across. They are decoded together, one step apart.
    pub num_codebooks: usize,
    /// The token a finished codebook holds, and the two that open and close an utterance.
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    /// Whether the output projection reuses the embedding's weights.
    pub tie_word_embeddings: bool,
    /// Rotary positions instead of the learnt table. Refused here - this family ships with the
    /// table, and the rotary path was never exercised.
    pub rope_embeddings: bool,
    pub rope_theta: f64,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    /// What every codebook holds before it has said anything.
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub decoder: DecoderConfig,
    /// The voice description is read by a T5 encoder, and the audio tokens are turned back into
    /// sound by a DAC. Both are whole models of their own, configured here.
    pub text_encoder: t5::Config,
    /// The PROMPT's vocabulary - text, not audio. The decoder's own is in `decoder`.
    pub vocab_size: usize,
    pub audio_encoder: dac::Config,
}

/// One attention of this decoder, read off the four names its checkpoint uses.
///
/// None of the four carries a bias. `kv_heads` may be fewer than the query heads - the model
/// shares key heads across groups - and the two attentions of a layer are allowed to group
/// differently, which is why it is a parameter rather than read from the config here.
fn attention(
    cfg: &DecoderConfig,
    kv_heads: usize,
    kv: Kv,
    vb: VarBuilder,
) -> Result<MultiHeadAttention> {
    if cfg.rope_embeddings {
        crate::tensor::bail!("rope embeddings are not supported");
    }
    let width = cfg.hidden_size;
    let kv_width = kv_heads * (width / cfg.num_attention_heads);
    MultiHeadAttention::new(
        linear_no_bias(width, width, &vb.pp("q_proj"))?,
        linear_no_bias(width, kv_width, &vb.pp("k_proj"))?,
        linear_no_bias(width, kv_width, &vb.pp("v_proj"))?,
        linear_no_bias(width, width, &vb.pp("out_proj"))?,
        width,
        cfg.num_attention_heads,
        kv_heads,
        kv,
    )
}

/// One decoder layer, read off the names parler's checkpoint uses.
///
/// The decoder attends to its own past a step at a time, and to the text encoder's output  -
/// which does not change across a generation - through a projection taken once. Neither
/// feed-forward projection carries a bias, and the activation between them is the config's.
fn layer(cfg: &DecoderConfig, vb: VarBuilder) -> Result<CrossAttentionBlock> {
    let kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_heads_cross = cfg.num_cross_attention_key_value_heads.unwrap_or(kv_heads);
    CrossAttentionBlock::new(
        attention(cfg, kv_heads, Kv::Growing(None), vb.pp("self_attn"))?,
        Some(attention(
            cfg,
            kv_heads_cross,
            Kv::Fixed(None),
            vb.pp("encoder_attn"),
        )?),
        cfg.hidden_size,
        cfg.ffn_dim,
        1e-5,
        false,
        cfg.activation_function,
        &vb,
    )
}

/// The `n` numbered children of one subtree, in order.
///
/// The codebooks and the layers are stored the same way - a parent name, then one subtree per
/// index - so the walk down to `parent.0`, `parent.1`, ... is written once here, and a caller
/// says only what to read out of one of them.
fn numbered<T>(
    vb: &VarBuilder,
    parent: &str,
    n: usize,
    read: impl Fn(VarBuilder) -> Result<T>,
) -> Result<Vec<T>> {
    let parent = vb.pp(parent);
    (0..n).map(|i| read(parent.pp(i))).collect()
}

#[derive(Debug, Clone)]
pub struct Decoder {
    embed_tokens: Vec<Embedding>,
    embed_positions: Tensor,
    layers: Vec<CrossAttentionBlock>,
    layer_norm: LayerNorm,
    lm_heads: Vec<Linear>,
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        // Everything the trunk needs hangs off one prefix; the per-codebook output heads hang
        // beside it rather than inside it.
        let trunk = vb.pp("model.decoder");
        let books = cfg.num_codebooks;
        // Positions are a learnt table that a step slices rather than looks up, so it is read
        // whole and stays a tensor.
        let table = (cfg.max_position_embeddings, cfg.hidden_size);
        Ok(Self {
            // A codebook is its own vocabulary at both ends of the shared trunk, and the
            // embedding is one entry wider than the head: the token a codebook holds before its
            // turn comes is fed back in, but is never something to predict.
            embed_tokens: numbered(&trunk, "embed_tokens", books, |vb| {
                embedding(cfg.vocab_size + 1, cfg.hidden_size, &vb)
            })?,
            lm_heads: numbered(&vb, "lm_heads", books, |vb| {
                linear_no_bias(cfg.hidden_size, cfg.vocab_size, &vb)
            })?,
            embed_positions: trunk.get(table, "embed_positions.weights")?,
            layers: numbered(&trunk, "layers", cfg.num_hidden_layers, |vb| layer(cfg, vb))?,
            layer_norm: layer_norm(cfg.hidden_size, 1e-5, &trunk.pp("layer_norm"))?,
        })
    }

    /// How many streams the audio is split across - one embedding and one head each.
    pub fn codebooks(&self) -> usize {
        self.embed_tokens.len()
    }

    /// The codebooks summed into the one hidden state the trunk carries.
    ///
    /// `input_ids` holds a token per codebook per step. Each is looked up in its own table and
    /// the results are added together, which is what leaves the streams sharing everything
    /// below and differing only at the heads.
    fn sum_over_codebooks(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut summed: Option<Tensor> = None;
        for (book, table) in self.embed_tokens.iter().enumerate() {
            let one = input_ids.i((.., book))?.apply(table)?;
            summed = Some(match summed {
                None => one,
                Some(so_far) => so_far.add(&one)?,
            });
        }
        match summed {
            Some(stream) => Ok(stream),
            None => crate::tensor::bail!("a decoder without codebooks has nothing to embed"),
        }
    }

    /// One step of the residual stream, or the prompt and the first step together.
    ///
    /// `prompt_hidden_states` is the text prefix and is handed over only on the call that opens
    /// a generation; afterwards it lives in the caches, and `seqlen_offset` is what places the
    /// step's own position past it in the table.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        prompt_hidden_states: Option<&Tensor>,
        encoder_xs: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Vec<Tensor>> {
        if input_ids.dim(1)? != self.codebooks() {
            crate::tensor::bail!("unexpected num codebooks in input {:?}", input_ids.shape())
        }
        let mut xs = self.sum_over_codebooks(input_ids)?;
        if let Some(prefix) = prompt_hidden_states {
            xs = Tensor::cat(&[prefix, &xs], 1)?;
        }
        let span = seqlen_offset..seqlen_offset + xs.dim(1)?;
        xs = xs.add(&self.embed_positions.i(span)?.unsqueeze(0)?)?;
        for block in self.layers.iter_mut() {
            xs = block.forward(&xs, Some(encoder_xs), Mask::Causal)?;
        }
        let xs = xs.apply(&self.layer_norm)?;
        self.lm_heads.iter().map(|head| xs.apply(head)).collect()
    }

    pub fn clear_kv_cache(&mut self) {
        for block in self.layers.iter_mut() {
            block.clear()
        }
    }
}

/// Which codebooks are sampled at this step.
///
/// The codebooks are offset from one another: the first speaks from the start, the second from
/// the step after, and so on, so that each is conditioned on what the ones before it have
/// already said. A codebook that has not started yet has nothing to sample, and one that has
/// emitted the padding token has finished - its slot stays padded and is never sampled again,
/// which is what makes "all of them padded" the end of the utterance.
fn speaking_at(step: usize, tokens: &[u32], pad: u32) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .take(step + 1)
        .filter(|(_, &t)| t != pad)
        .map(|(book, _)| book)
        .collect()
}

#[derive(Debug, Clone)]
pub struct Model {
    pub embed_prompts: Embedding,
    pub enc_to_dec_proj: Option<Linear>,
    pub decoder: Decoder,
    pub text_encoder: t5::T5EncoderModel,
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub audio_encoder: dac::Model,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, t5_vb: crate::tensor::VarBuilder) -> Result<Self> {
        let text_encoder = t5::T5EncoderModel::load(t5_vb.pp("text_encoder"), &cfg.text_encoder)
            .map_err(|e| crate::tensor::Error::msg(format!("parler t5: {}", e.0)))?;
        let decoder = Decoder::new(&cfg.decoder, vb.pp("decoder"))?;
        // The two halves need not be the same width. When they are not, a projection stands
        // between them - and it is the one projection in this model that carries a bias.
        let (text_width, audio_width) = (cfg.text_encoder.d_model, cfg.decoder.hidden_size);
        let enc_to_dec_proj = match text_width == audio_width {
            true => None,
            false => Some(linear(text_width, audio_width, &vb.pp("enc_to_dec_proj"))?),
        };
        Ok(Self {
            decoder,
            // What reads the description, and what fits its output to the decoder's width.
            text_encoder,
            enc_to_dec_proj,
            embed_prompts: embedding(cfg.vocab_size, audio_width, &vb.pp("embed_prompts"))?,
            // DAC runs on the native substrate; it reads the same weight files
            // through the native VarBuilder that already serves the T5 subtree.
            audio_encoder: dac::Model::new(&cfg.audio_encoder, t5_vb.pp("audio_encoder.model"))
                .map_err(|e| crate::tensor::Error::msg(format!("parler dac: {}", e.0)))?,
            decoder_start_token_id: cfg.decoder_start_token_id,
            pad_token_id: cfg.pad_token_id,
        })
    }

    /// Note that the returned tensor uses the CPU device.
    /// Encode the description through the T5 text encoder + the
    /// projection to decoder dim. Callers that synthesise the same voice
    /// repeatedly can cache this result instead of re-running the
    /// encoder on every `generate()` call.
    pub fn encode_description(&mut self, description_tokens: &Tensor) -> Result<Tensor> {
        self.text_encoder.clear_kv_cache();
        let encoded = self.text_encoder.forward(description_tokens)?;
        match self.enc_to_dec_proj.as_ref() {
            None => Ok(encoded),
            Some(proj) => encoded.apply(proj),
        }
    }

    /// Same as `generate` but takes a pre-encoded description (from
    /// `encode_description`). Saves the per-call T5 encoder pass when
    /// the same voice description is reused.
    pub fn generate_with_encoded(
        &mut self,
        prompt_tokens: &Tensor,
        encoded: &Tensor,
        sampler: LogitsProcessor,
        max_steps: usize,
    ) -> Result<Tensor> {
        self.generate_inner(prompt_tokens, encoded, sampler, max_steps)
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &Tensor,
        description_tokens: &Tensor,
        sampler: LogitsProcessor,
        max_steps: usize,
    ) -> Result<Tensor> {
        let encoded = self.encode_description(description_tokens)?;
        self.generate_inner(prompt_tokens, &encoded, sampler, max_steps)
    }

    fn generate_inner(
        &mut self,
        prompt_tokens: &Tensor,
        encoded: &Tensor,
        mut sampler: LogitsProcessor,
        max_steps: usize,
    ) -> Result<Tensor> {
        let books = self.decoder.codebooks();
        self.decoder.clear_kv_cache();
        // The description is already encoded. The prompt is text, and it enters as hidden
        // states, once, in front of the audio the decoder is about to invent.
        let prompt = prompt_tokens.apply(&self.embed_prompts)?;
        let prompt_len = prompt.dim(1)?;
        // One token per codebook is the whole state carried from step to step; `spoken` keeps
        // the ones that were audio, which is everything that is not a marker.
        let mut held = vec![self.decoder_start_token_id; books];
        let mut spoken: Vec<Vec<u32>> = vec![Vec::new(); books];

        for step in 0..max_steps {
            // Autoregressive decode runs in spawn_blocking, so a dropped request can
            // only be observed here. See `cancel::scoped` for why this loop reads a
            // thread-published token instead of taking one.
            crate::inference::serve::cancel::scoped::bail()?;
            let fed = Tensor::from_slice(held.as_slice(), (1, books, 1), &prompt_tokens.device())?;
            // The prompt goes in on the opening step alone; every step after it lands past the
            // prompt in the position table, because the prompt is already in the caches.
            let (prefix, pos) = match step {
                0 => (Some(&prompt), 0),
                _ => (None, step + prompt_len),
            };
            let logits = self.decoder.forward(&fed, prefix, encoded, pos)?;
            for book in speaking_at(step, &held, self.pad_token_id) {
                let last = logits[book].dim(1)? - 1;
                held[book] = sampler.sample(&logits[book].i((0, last))?)?;
            }
            if held.iter().all(|&token| token == self.pad_token_id) {
                break;
            }
            for (book, &token) in held.iter().enumerate() {
                let marker = token == self.decoder_start_token_id || token == self.pad_token_id;
                if !marker {
                    spoken[book].push(token);
                }
            }
        }

        // The codebooks are offset from one another, so they do not finish together; the codec
        // wants a rectangle, and the shortest stream is as far as all of them have spoken.
        let shortest = spoken.iter().map(|stream| stream.len()).min().unwrap_or(0);
        for stream in spoken.iter_mut() {
            stream.truncate(shortest);
        }
        Tensor::new(spoken, &crate::tensor::Device::Cpu)
    }
}

#[cfg(test)]
mod delay_pattern_tests {
    use super::speaking_at;

    const PAD: u32 = 1024;
    const START: u32 = 1025;

    /// A codebook speaks from its own step onwards, and stops for good once it pads.
    ///
    /// Both halves matter and neither shows in the shape of the output: a codebook sampled
    /// before its turn is conditioned on nothing, and one sampled after it has padded overwrites
    /// the end of its own stream - so the utterance would simply be wrong, at the right length.
    #[test]
    fn a_codebook_speaks_from_its_own_step_and_stops_when_it_pads() {
        let books = 4;
        let mut tokens = vec![START; books];

        // Nothing has padded: the number speaking grows by one a step, then stays at all of them.
        for step in 0..8 {
            assert_eq!(
                speaking_at(step, &tokens, PAD),
                (0..books.min(step + 1)).collect::<Vec<_>>(),
                "step {step}"
            );
        }

        // The second one finishes; the rest carry on without it.
        tokens[1] = PAD;
        assert_eq!(speaking_at(5, &tokens, PAD), vec![0, 2, 3]);
        // And it does not come back, however long the utterance runs.
        assert_eq!(speaking_at(50, &tokens, PAD), vec![0, 2, 3]);

        // When every one of them has padded, nobody speaks - which is what ends the loop.
        tokens.iter_mut().for_each(|t| *t = PAD);
        assert!(speaking_at(9, &tokens, PAD).is_empty());
    }
}

#[cfg(test)]
mod causal_mask_tests {
    use crate::inference::model::acestep::ops::sdpa;
    use crate::tensor::{Device, Result, Tensor};

    /// The decoder used to build its own additive mask and add it to the scores; it now asks
    /// `sdpa` for a causal one. The two must be the same mask, at the prefill step where the
    /// query covers the whole prompt AND at every step after it, where one query attends to
    /// everything before it and the mask is the identity.
    ///
    /// The old rule, kept here as the reference: key `j` is visible to query `i` when
    /// `i + kv_len >= j + q_len`.
    #[test]
    fn asking_for_a_causal_mask_is_the_mask_the_decoder_used_to_build() -> Result<()> {
        let dev = Device::Cpu;
        let (heads, head_dim) = (2usize, 4usize);
        let value = |i: usize| ((i % 13) as f32 - 6.0) * 0.17;

        for (q_len, kv_len) in [(5usize, 5usize), (1, 5), (1, 1), (3, 7)] {
            let q = Tensor::from_vec_f32(
                (0..heads * q_len * head_dim).map(value).collect(),
                (1, heads, q_len, head_dim),
            )?
            .to_device(&dev)?;
            let k = Tensor::from_vec_f32(
                (0..heads * kv_len * head_dim)
                    .map(|i| value(i + 3))
                    .collect(),
                (1, heads, kv_len, head_dim),
            )?
            .to_device(&dev)?;
            let v = Tensor::from_vec_f32(
                (0..heads * kv_len * head_dim)
                    .map(|i| value(i + 7))
                    .collect(),
                (1, heads, kv_len, head_dim),
            )?
            .to_device(&dev)?;

            let scale = (head_dim as f64).powf(-0.5) as f32;
            let asked = sdpa(&q, &k, &v, None, true, scale, 1.0)?;

            // The rule itself, stated once, then laid out row-major over the qxkv rectangle.
            let visible = |query: usize, key: usize| key + q_len <= query + kv_len;
            let mask: Vec<f32> = (0..q_len * kv_len)
                .map(|at| match visible(at / kv_len, at % kv_len) {
                    true => 0.0,
                    false => f32::NEG_INFINITY,
                })
                .collect();
            let mask = Tensor::from_vec_f32(mask, (q_len, kv_len))?.to_device(&dev)?;
            let built = sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?;

            let a = asked.flatten_all()?.to_vec1::<f32>()?;
            let b = built.flatten_all()?.to_vec1::<f32>()?;
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-6,
                "q={q_len} kv={kv_len}: the two masks differ by {worst:e}"
            );

            // The negative control: without any mask the two are not the same thing, except
            // where the triangle covers everything anyway.
            if q_len > 1 {
                let unmasked = sdpa(&q, &k, &v, None, false, scale, 1.0)?;
                let c = unmasked.flatten_all()?.to_vec1::<f32>()?;
                let moved = a
                    .iter()
                    .zip(&c)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(
                    moved > 1e-4,
                    "q={q_len} kv={kv_len}: masking changed nothing"
                );
            }
        }
        Ok(())
    }
}
