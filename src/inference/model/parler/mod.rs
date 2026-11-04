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
use crate::inference::model::half::projection;
use crate::inference::model::t5::encoder as t5;
use crate::inference::sample::token_sampling::{LogitsProcessor, Sampling};
use crate::tensor::layer::{embedding, layer_norm, linear};
use crate::tensor::layer::{Embedding, LayerNorm, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::VarBuilder;
use crate::tensor::{IndexOp, Result, Tensor};

#[cfg(any(test, feature = "microbench"))]
pub mod synthetic;

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
///
/// With `half` set the four projections are held in half precision on a card.
fn attention(
    cfg: &DecoderConfig,
    kv_heads: usize,
    kv: Kv,
    half: bool,
    vb: VarBuilder,
) -> Result<MultiHeadAttention> {
    if cfg.rope_embeddings {
        crate::tensor::bail!("rope embeddings are not supported");
    }
    let width = cfg.hidden_size;
    let kv_width = kv_heads * (width / cfg.num_attention_heads);
    MultiHeadAttention::new(
        projection(width, width, false, half, &vb.pp("q_proj"))?,
        projection(width, kv_width, false, half, &vb.pp("k_proj"))?,
        projection(width, kv_width, false, half, &vb.pp("v_proj"))?,
        projection(width, width, false, half, &vb.pp("out_proj"))?,
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
///
/// With `half` set every projection of the layer is held in half precision on a card and the
/// norms stay F32.
fn layer(cfg: &DecoderConfig, half: bool, vb: VarBuilder) -> Result<CrossAttentionBlock> {
    let kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_heads_cross = cfg.num_cross_attention_key_value_heads.unwrap_or(kv_heads);
    CrossAttentionBlock::with_projections(
        attention(cfg, kv_heads, Kv::Growing(None), half, vb.pp("self_attn"))?,
        Some(attention(
            cfg,
            kv_heads_cross,
            Kv::Fixed(None),
            half,
            vb.pp("encoder_attn"),
        )?),
        cfg.hidden_size,
        cfg.ffn_dim,
        1e-5,
        cfg.activation_function,
        &vb,
        |from, to, vb| projection(from, to, false, half, vb),
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
    /// With `half` set the layers and the heads are held in half precision on a card and the
    /// stream between them runs F32. The embeddings and the norms stay F32.
    pub fn new(cfg: &DecoderConfig, half: bool, vb: VarBuilder) -> Result<Self> {
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
                projection(cfg.hidden_size, cfg.vocab_size, false, half, &vb)
            })?,
            embed_positions: trunk.get(table, "embed_positions.weights")?,
            layers: numbered(&trunk, "layers", cfg.num_hidden_layers, |vb| {
                layer(cfg, half, vb)
            })?,
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

/// What the checkpoint's `generation_config.json` says about decoding.
///
/// A field the file leaves out is `None`, and the accessors then give what the library that
/// wrote the format gives a field it is not told: every candidate kept up to `top_k`, no
/// nucleus cut, temperature 1, no minimum length.
#[derive(serde::Deserialize, Debug, Clone, Default, PartialEq)]
pub struct Generation {
    pub do_sample: Option<bool>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub temperature: Option<f64>,
    /// Steps during which no codebook may end the utterance.
    pub min_new_tokens: Option<usize>,
    /// The start token plus the steps the model may take.
    pub max_length: Option<usize>,
}

impl Generation {
    /// The library's own defaults for a field the checkpoint does not set.
    const TOP_K: usize = 50;
    const TOP_P: f64 = 1.0;
    const TEMPERATURE: f64 = 1.0;

    /// The file's content; a file that does not parse says nothing.
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_default()
    }

    /// How a token is drawn: the request's value where it gave one, else the file's, else the
    /// library's. A checkpoint that says `do_sample` is false decodes greedily unless the
    /// request names a temperature.
    pub fn sampling(
        &self,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> Sampling {
        if self.do_sample == Some(false) && temperature.is_none() {
            return Sampling::ArgMax;
        }
        let temperature = temperature
            .or(self.temperature)
            .unwrap_or(Self::TEMPERATURE);
        if temperature < 1e-7 {
            return Sampling::ArgMax;
        }
        let k = top_k.or(self.top_k).unwrap_or(Self::TOP_K);
        let p = top_p.or(self.top_p).unwrap_or(Self::TOP_P);
        // Zero candidates and a nucleus of everything are both "no restriction".
        match (k > 0, p < 1.0) {
            (true, true) => Sampling::TopKThenTopP { k, p, temperature },
            (true, false) => Sampling::TopK { k, temperature },
            (false, true) => Sampling::TopP { p, temperature },
            (false, false) => Sampling::All { temperature },
        }
    }

    pub fn min_new_tokens(&self) -> usize {
        self.min_new_tokens.unwrap_or(0)
    }

    /// The most steps an utterance may take: `max_length` less the start token, and never
    /// past the `positions` of the table, which the prompt shares with the audio.
    fn step_limit(&self, positions: usize, prompt_len: usize) -> usize {
        let table = positions.saturating_sub(prompt_len);
        match self.max_length {
            Some(length) => length.saturating_sub(1).min(table),
            None => table,
        }
    }
}

/// When a codebook may emit the end token.
///
/// No codebook may end before `min_new_tokens` steps have passed. After that, only the first
/// codebook that has not ended may: a later one ending first would cut the others short.
#[derive(Debug, Clone)]
struct Ending {
    first_unfinished: usize,
    min_new_tokens: usize,
}

impl Ending {
    fn new(min_new_tokens: usize) -> Self {
        Self {
            first_unfinished: 0,
            min_new_tokens,
        }
    }

    fn may_end(&self, book: usize, step: usize) -> bool {
        step >= self.min_new_tokens && book <= self.first_unfinished
    }

    /// Called after a step's draws: the next codebook becomes the one that may end once the
    /// current one has.
    fn observe(&mut self, held: &[u32], pad: u32) {
        let last = held.len().saturating_sub(1);
        if held[self.first_unfinished] == pad && self.first_unfinished < last {
            self.first_unfinished += 1;
        }
    }
}

/// One step's draws: each codebook that speaks takes a token from its row, never the end token
/// where `ending` forbids it.
fn draw_step(
    step: usize,
    held: &mut [u32],
    pad: u32,
    ending: &mut Ending,
    sampler: &mut LogitsProcessor,
    row: impl Fn(usize) -> Result<Tensor>,
) -> Result<()> {
    for book in speaking_at(step, held, pad) {
        let row = row(book)?;
        held[book] = match ending.may_end(book, step) {
            true => sampler.sample(&row)?,
            false => sampler.sample_avoiding(&row, &[pad])?,
        };
    }
    ending.observe(held, pad);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Model {
    /// The checkpoint's decoding recipe.
    pub generation: Generation,
    /// The learnt position table's length.
    max_positions: usize,
    pub embed_prompts: Embedding,
    pub enc_to_dec_proj: Option<Linear>,
    pub decoder: Decoder,
    pub text_encoder: t5::T5EncoderModel,
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub audio_encoder: dac::Model,
}

impl Model {
    /// `half` holds the decoder's projections in half precision on a card.
    /// The codec reads `t5_vb`.
    /// The text encoder reads `text_vb`, which may sit on another device than the rest: it runs
    /// once per description, and its output follows the device of the ids it is given.
    /// The text encoder and the codec stay F32.
    pub fn new(
        cfg: &Config,
        half: bool,
        vb: VarBuilder,
        t5_vb: crate::tensor::VarBuilder,
        text_vb: crate::tensor::VarBuilder,
    ) -> Result<Self> {
        let text_encoder = t5::T5EncoderModel::load(text_vb.pp("text_encoder"), &cfg.text_encoder)
            .map_err(|e| crate::tensor::Error::msg(format!("parler t5: {}", e.0)))?;
        let decoder = Decoder::new(&cfg.decoder, half, vb.pp("decoder"))?;
        // The two halves need not be the same width. When they are not, a projection stands
        // between them - and it is the one projection in this model that carries a bias.
        let (text_width, audio_width) = (cfg.text_encoder.d_model, cfg.decoder.hidden_size);
        let enc_to_dec_proj = match text_width == audio_width {
            true => None,
            false => Some(linear(text_width, audio_width, &vb.pp("enc_to_dec_proj"))?),
        };
        Ok(Self {
            generation: Generation::default(),
            max_positions: cfg.decoder.max_position_embeddings,
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

    /// The most steps one utterance of a `prompt_len`-token prompt may take.
    ///
    /// The checkpoint's `max_length` less the start token, and never past the position table,
    /// which the prompt shares with the audio.
    pub fn step_limit(&self, prompt_len: usize) -> usize {
        self.generation.step_limit(self.max_positions, prompt_len)
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
        let mut ending = Ending::new(self.generation.min_new_tokens());

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
            let pad = self.pad_token_id;
            draw_step(step, &mut held, pad, &mut ending, &mut sampler, |book| {
                let last = logits[book].dim(1)? - 1;
                logits[book].i((0, last))
            })?;
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
    use crate::inference::model::sdpa::sdpa;
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

impl Decoder {
    /// How many blocks the decoder stacks.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod decoding_tests {
    use super::*;
    use crate::tensor::Device;

    const PAD: u32 = 7;
    const START: u32 = 8;
    const VOCAB: usize = 8;

    /// Logits whose argmax is `leader`, with `PAD` the runner-up.
    fn row_led_by(leader: usize) -> Result<Tensor> {
        let mut v = vec![0.0f32; VOCAB];
        v[PAD as usize] = 5.0;
        v[leader] = 9.0;
        Tensor::from_vec(v, (VOCAB,), &Device::Cpu)
    }

    /// Logits where the end token leads every codebook at every step, which is the worst case
    /// for an utterance that must not stop early.
    fn ending_leads(_book: usize) -> Result<Tensor> {
        let mut v = vec![0.0f32; VOCAB];
        v[PAD as usize] = 9.0;
        v[3] = 5.0;
        Tensor::from_vec(v, (VOCAB,), &Device::Cpu)
    }

    /// The rule of the reference processor: no end before `min_new_tokens` steps, then only the
    /// first codebook that has not ended may end, and the next one is released the step after.
    #[test]
    fn only_the_first_unfinished_codebook_may_end_and_not_before_the_minimum() -> Result<()> {
        let books = 3;
        let mut held = vec![START; books];
        let mut ending = Ending::new(2);
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut trace = Vec::new();
        for step in 0..6 {
            draw_step(
                step,
                &mut held,
                PAD,
                &mut ending,
                &mut sampler,
                ending_leads,
            )?;
            trace.push(held.clone());
        }
        // Steps 0 and 1: the minimum holds every end back, so a speaking codebook takes the
        // runner-up. Codebooks that have not started hold the start token.
        assert_eq!(trace[0], vec![3, START, START]);
        assert_eq!(trace[1], vec![3, 3, START]);
        // Step 2: only codebook 0 may end.
        assert_eq!(trace[2], vec![PAD, 3, 3]);
        // Step 3: codebook 1 is released, codebook 2 still may not.
        assert_eq!(trace[3], vec![PAD, PAD, 3]);
        // Step 4: the last one ends, and with it the utterance.
        assert_eq!(trace[4], vec![PAD, PAD, PAD]);
        assert_eq!(ending.first_unfinished, books - 1);
        Ok(())
    }

    /// A codebook that may end still takes any other token its row leads with.
    #[test]
    fn a_codebook_that_may_end_still_draws_its_row() -> Result<()> {
        let mut held = vec![START; 2];
        let mut ending = Ending::new(0);
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        draw_step(0, &mut held, PAD, &mut ending, &mut sampler, |_| {
            row_led_by(2)
        })?;
        assert_eq!(held, vec![2, START]);
        Ok(())
    }

    #[test]
    fn sampling_takes_the_request_then_the_file_then_the_library() {
        let sampled = Generation::from_json(r#"{"do_sample": true, "min_new_tokens": 10}"#);
        // The file is silent: top_k 50, no nucleus, temperature 1.
        assert_eq!(
            sampled.sampling(None, None, None),
            Sampling::TopK {
                k: 50,
                temperature: 1.0
            }
        );
        // The request wins over the library.
        assert_eq!(
            sampled.sampling(Some(0.5), Some(10), Some(0.9)),
            Sampling::TopKThenTopP {
                k: 10,
                p: 0.9,
                temperature: 0.5
            }
        );
        // The file wins over the library.
        let tuned = Generation::from_json(r#"{"top_k": 20, "top_p": 0.8, "temperature": 0.7}"#);
        assert_eq!(
            tuned.sampling(None, None, None),
            Sampling::TopKThenTopP {
                k: 20,
                p: 0.8,
                temperature: 0.7
            }
        );
        // A zero temperature is greedy, and so is a checkpoint that does not sample.
        assert_eq!(sampled.sampling(Some(0.0), None, None), Sampling::ArgMax);
        let greedy = Generation::from_json(r#"{"do_sample": false}"#);
        assert_eq!(greedy.sampling(None, None, None), Sampling::ArgMax);
        assert_eq!(sampled.min_new_tokens(), 10);
    }

    #[test]
    fn the_step_limit_comes_from_max_length_and_the_position_table() {
        let g = Generation::from_json(r#"{"max_length": 2580}"#);
        assert_eq!(g.step_limit(4096, 40), 2579);
        // A prompt that leaves less room than max_length allows shortens the limit.
        assert_eq!(g.step_limit(2600, 100), 2500);
        // A file without max_length leaves the table as the only bound.
        assert_eq!(Generation::default().step_limit(4096, 40), 4056);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod half_tests {
    use super::synthetic::{self, argmax, Conditioning, Shape};
    use super::*;
    use crate::tensor::{DType, Device};

    /// How far a decoder held in half precision drifts from the same decoder in F32.
    #[derive(Debug)]
    struct Drift {
        /// The largest difference between two logits of the same step and codebook.
        max_logit: f32,
        /// The largest logit magnitude of the F32 run, which the difference is read against.
        scale: f32,
        /// Steps times codebooks compared.
        compared: usize,
        /// Of those, the ones whose argmax differs when both runs are fed the F32 tokens.
        forced_mismatch: usize,
        /// Steps whose codebook 0 token differs when each run feeds back its own tokens.
        free_mismatch: usize,
    }

    /// Both decoders run `steps` steps. The first pass feeds both the tokens the F32 run draws
    /// greedily. The second lets each feed back its own.
    fn drift(
        full: &mut Decoder,
        half: &mut Decoder,
        cond: &Conditioning,
        books: usize,
        steps: usize,
        device: &Device,
    ) -> Result<Drift> {
        let mut d = Drift {
            max_logit: 0.0,
            scale: 0.0,
            compared: 0,
            forced_mismatch: 0,
            free_mismatch: 0,
        };
        full.clear_kv_cache();
        half.clear_kv_cache();
        let mut tokens = vec![0u32; books];
        for n in 0..steps {
            let a = synthetic::step(full, cond, &tokens, n, device)?;
            let b = synthetic::step(half, cond, &tokens, n, device)?;
            for (x, y) in a.iter().zip(&b) {
                for (p, q) in x.iter().zip(y) {
                    d.max_logit = d.max_logit.max((p - q).abs());
                    d.scale = d.scale.max(p.abs());
                }
                d.compared += 1;
                d.forced_mismatch += (argmax(x) != argmax(y)) as usize;
            }
            tokens = a.iter().map(|row| argmax(row)).collect();
        }
        full.clear_kv_cache();
        half.clear_kv_cache();
        let (mut own_full, mut own_half) = (vec![0u32; books], vec![0u32; books]);
        for n in 0..steps {
            let a = synthetic::step(full, cond, &own_full, n, device)?;
            let b = synthetic::step(half, cond, &own_half, n, device)?;
            own_full = a.iter().map(|row| argmax(row)).collect();
            own_half = b.iter().map(|row| argmax(row)).collect();
            d.free_mismatch += (own_full[0] != own_half[0]) as usize;
        }
        Ok(d)
    }

    /// The logits of a synthetic decoder held in half precision stay within a tolerance of its
    /// F32 logits, and its greedy tokens agree.
    #[test]
    fn a_half_decoder_gives_the_logits_of_the_f32_one_within_tolerance() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let shape = Shape {
            hidden: 128,
            ffn: 512,
            layers: 4,
            heads: 4,
            codebooks: 4,
            vocab: 64,
            positions: 64,
        };
        let cond = Conditioning::new(&shape, 8, 6, &device)?;
        let mut full = synthetic::decoder(&shape, false, &device)?;
        let mut half = synthetic::decoder(&shape, true, &device)?;
        let d = drift(&mut full, &mut half, &cond, shape.codebooks, 20, &device)?;
        eprintln!("synthetic decoder: {d:?}");
        assert!(
            d.max_logit > 0.0,
            "the half decoder did not run in half precision"
        );
        assert!(
            d.max_logit <= 2e-2 * d.scale,
            "logits drift by {} against a scale of {}",
            d.max_logit,
            d.scale
        );
        Ok(())
    }

    /// The same comparison on the real mini checkpoint, which the local cache may hold.
    #[test]
    #[ignore = "needs the mini checkpoint in the hub cache and a card"]
    fn the_mini_checkpoint_in_half_precision_stays_within_tolerance() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let repo = hf_hub::Cache::default().model("parler-tts/parler-tts-mini-v1".to_string());
        let (Some(weights), Some(config)) =
            (repo.get("model.safetensors"), repo.get("config.json"))
        else {
            return Ok(());
        };
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config).unwrap())
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let cfg = &config.decoder;
        let build = |half: bool| -> Result<Decoder> {
            let vb = unsafe { VarBuilder::from_files(&[&weights], DType::F32, &device)? };
            Decoder::new(cfg, half, vb.pp("decoder"))
        };
        let shape = Shape {
            hidden: cfg.hidden_size,
            ffn: cfg.ffn_dim,
            layers: cfg.num_hidden_layers,
            heads: cfg.num_attention_heads,
            codebooks: cfg.num_codebooks,
            vocab: cfg.vocab_size,
            positions: cfg.max_position_embeddings,
        };
        // The conditioning is a real one: the prompt embedded and the description encoded
        // by the F32 model on the host, then moved to the card.
        let cond = {
            let cpu = Device::Cpu;
            let open = || unsafe { VarBuilder::from_files(&[&weights], DType::F32, &cpu) };
            let mut model = Model::new(&config, false, open()?, open()?, open()?)?;
            let tokenizer = tokenizers::Tokenizer::from_file(repo.get("tokenizer.json").unwrap())
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
            let ids = |text: &str| -> Result<Tensor> {
                let enc = tokenizer
                    .encode(text, true)
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
                Tensor::new(enc.get_ids(), &cpu)?.unsqueeze(0)
            };
            let description = ids("A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch.")?;
            let prompt = ids("Hello, this is a test of the speech model.")?;
            Conditioning {
                prompt: prompt.apply(&model.embed_prompts)?.to_device(&device)?,
                encoder: model.encode_description(&description)?.to_device(&device)?,
            }
        };
        let (mut full, mut half) = (build(false)?, build(true)?);
        let d = drift(&mut full, &mut half, &cond, shape.codebooks, 20, &device)?;
        eprintln!("mini checkpoint: {d:?}");
        assert!(d.max_logit <= 2e-2 * d.scale);
        Ok(())
    }
}
