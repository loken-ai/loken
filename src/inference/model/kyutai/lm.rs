//! Kyutai `tts-1.6b-en_fr` - LM text-forward (component 5): embeddings + Helium
//! transformer + out_norm + text head.
//!
//! `forward_text(seq[K=33,S])`:
//!   input = Σ_{cb=0..31} emb[cb](seq[cb+1]) + text_emb(seq[0]) + sum_condition
//!   h = out_norm(helium(input, cross_attention_src=ca))
//!   text_logits = text_linear(h)
//! (audio_offset=1; K=33 = 1 text + 32 audio codebooks; out_norm = RMSNorm alpha.)

use crate::inference::model::kyutai::config::{KyutaiConfig, Streams};
use crate::inference::model::kyutai::helium::{Geometry, HeliumCache, HeliumTransformer};
use crate::tensor::{Device, Result, Tensor};

const AUDIO_OFFSET: usize = 1;
const RMS_EPS: f32 = 1e-5;

/// The LM's shape, read from the `config.json` published beside the checkpoint.
struct LmConfig {
    helium: Geometry,
    streams: Streams,
    /// Audio codebooks summed into each frame.
    n_audio: usize,
    /// Rows of each audio embedding: the codebook size and the zero token.
    audio_card: usize,
    /// Rows of the text embedding: the text vocabulary and its padding.
    text_card: usize,
    /// Rows of the text head.
    text_card_out: usize,
}

impl LmConfig {
    fn read(checkpoint: &str) -> Result<Self> {
        let cfg = KyutaiConfig::read(checkpoint)?;
        let text_card = cfg.usize(&["text_card"])?;
        Ok(Self {
            helium: Geometry::from_config(&cfg)?,
            streams: Streams::from_config(&cfg)?,
            n_audio: cfg.usize(&["n_q"])?,
            audio_card: cfg.usize(&["card"])? + 1,
            text_card: text_card + 1,
            text_card_out: cfg.opt_usize(&["text_card_out"]).unwrap_or(text_card),
        })
    }
}

pub struct KyutaiLm {
    audio_emb: Vec<Tensor>, // n_audio x [audio_card, dim]
    text_emb: Tensor,       // [text_card, dim]
    text_out1: Tensor,      // [dim, dim] demux stream-1 projection
    text_out2: Tensor,      // [dim, dim] demux stream-2 projection
    transformer: HeliumTransformer,
    out_norm: Tensor,    // rms alpha [dim]
    text_linear: Tensor, // [text_card_out, dim]
    audio_card: usize,
    text_card: usize,
    streams: Streams,
    device: Device,
}
impl KyutaiLm {
    /// Load the LM through the ladder in `place`: the Helium layers spread over the cards
    /// with their self-attention cache, the embeddings, out_norm and text head on the first
    /// card the plan uses (the input/output side).
    pub fn from_safetensors_hetero(path: &str) -> Result<Self> {
        let cfg = LmConfig::read(path)?;
        let g = cfg.helium;
        let (block_bytes, head_bytes) =
            crate::inference::place::runtime_demand::safetensors_f32_split(&[path], |n| {
                n.starts_with("transformer.layers.")
            })
            .map_err(crate::tensor::Error)?;
        let demand = crate::inference::place::ladder::Demand {
            layers: g.layers,
            block_bytes,
            head_bytes,
            kv_bytes_per_layer: g.kv_bytes_per_layer(),
            runtime_reserve: crate::inference::place::runtime_demand::transformer_work_reserve(
                block_bytes + head_bytes,
                g.context,
                g.dim,
                g.heads,
                g.hidden,
            ),
        };
        crate::inference::place::ladder::load("Kyutai LM", &demand, |plan| {
            Self::build(path, &cfg, plan)
        })
    }

    fn build(
        path: &str,
        cfg: &LmConfig,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
    ) -> Result<Self> {
        let g = cfg.helium;
        let dim = g.dim;
        let layer_devs = crate::inference::place::ladder::layer_devices(plan, g.layers);
        let primary = crate::inference::place::ladder::primary_device(plan);
        let vbs = crate::inference::place::plan::build_vbset(
            path,
            &layer_devs
                .iter()
                .cloned()
                .chain(std::iter::once(primary.clone()))
                .collect::<Vec<_>>(),
        )?;
        let on_gpu = layer_devs
            .iter()
            .filter(|d| crate::inference::place::plan::dev_key(d) != "cpu")
            .count();
        tracing::info!(
            "kyutai helium hetero: {on_gpu}/{} layers on GPU, primary {}",
            g.layers,
            crate::inference::place::plan::dev_key(&primary)
        );
        let vb = crate::inference::place::plan::vb_on(&vbs, &primary);
        let mut audio_emb = Vec::with_capacity(cfg.n_audio);
        for cb in 0..cfg.n_audio {
            audio_emb.push(vb.get((cfg.audio_card, dim), &format!("emb.{cb}.weight"))?);
        }
        Ok(Self {
            audio_emb,
            text_emb: vb.get((cfg.text_card, dim), "text_emb.weight")?,
            text_out1: vb.get((dim, dim), "text_emb.out1.weight")?,
            text_out2: vb.get((dim, dim), "text_emb.out2.weight")?,
            transformer: HeliumTransformer::load_hetero(&vbs, &layer_devs, g)?,
            out_norm: vb.get((1, 1, dim), "out_norm.alpha")?.reshape(dim)?,
            text_linear: vb.get((cfg.text_card_out, dim), "text_linear.weight")?,
            audio_card: cfg.audio_card,
            text_card: cfg.text_card,
            streams: cfg.streams.clone(),
            device: primary,
        })
    }

    /// Text stream is a `demux_second_stream` embedding: a single id packs two
    /// sub-streams as `id = left + (right+1).TEXT_CARD`. The embedding is
    /// `out1.emb[left] + (right>=0 ? out2.emb[right] : 0)` (no bias). For plain
    /// single-stream TTS text (`id < TEXT_CARD`) `right = -1` -> only the out1 term.
    fn text_embed(&self, text_ids: &Tensor) -> Result<Tensor> {
        let s = text_ids.shape().dims1()?;
        let ids = text_ids.to_vec1_u32()?;
        let card = self.text_card as u32;
        let left: Vec<u32> = ids.iter().map(|&t| t % card).collect();
        let left_t = Tensor::from_vec_u32(left, vec![s])?.to_device(&self.device)?;
        let mut y = self
            .text_emb
            .index_select(&left_t, 0)?
            .matmul_t(&self.text_out1)?;
        if ids.iter().any(|&t| t / card >= 1) {
            let right: Vec<u32> = ids.iter().map(|&t| (t / card).saturating_sub(1)).collect();
            let mask: Vec<f32> = ids
                .iter()
                .map(|&t| if t / card >= 1 { 1.0 } else { 0.0 })
                .collect();
            let right_t = Tensor::from_vec_u32(right, vec![s])?.to_device(&self.device)?;
            let mask_t = Tensor::from_vec_f32(mask, (s, 1))?.to_device(&self.device)?;
            let right_e = self
                .text_emb
                .index_select(&right_t, 0)?
                .matmul_t(&self.text_out2)?;
            y = y.add(&right_e.broadcast_mul(&mask_t)?)?;
        }
        Ok(y)
    }

    /// Σ audio-codebook embeddings + demuxed text embedding + `sum_condition`.
    ///
    /// The `zero_token` (-1, arriving as `u32::MAX`) and any out-of-range id map to a ZERO
    /// embedding, matching moshi's `ScaledEmbedding(zero_idx=-1)` - during the audio warm-up
    /// the acoustic codebooks are fed the zero token, and a raw `index_select` on it would
    /// read out of bounds and corrupt the hidden state.
    fn embed(&self, seq: &Tensor, sum_condition: &Tensor) -> Result<Tensor> {
        let s = seq.shape().dims2()?.1;
        let mut input: Option<Tensor> = None;
        for cb in 0..self.audio_emb.len() {
            let ids_host = seq
                .narrow(0, cb + AUDIO_OFFSET, 1)?
                .reshape(s)?
                .to_vec1_u32()?;
            let clamped: Vec<u32> = ids_host
                .iter()
                .map(|&x| if x < self.audio_card as u32 { x } else { 0 })
                .collect();
            let cl = Tensor::from_vec_u32(clamped, vec![s])?.to_device(&self.device)?;
            let mut e = self.audio_emb[cb].index_select(&cl, 0)?; // [S, DIM]
            if ids_host.iter().any(|&x| x >= self.audio_card as u32) {
                let mask: Vec<f32> = ids_host
                    .iter()
                    .map(|&x| if x < self.audio_card as u32 { 1.0 } else { 0.0 })
                    .collect();
                let m = Tensor::from_vec_f32(mask, (s, 1))?.to_device(&self.device)?;
                e = e.broadcast_mul(&m)?;
            }
            input = Some(match input {
                None => e,
                Some(a) => a.add(&e)?,
            });
        }
        let text_ids = seq.narrow(0, 0, 1)?.reshape(s)?;
        // sum_condition is a single shared offset [1, DIM]; broadcast over the S frames.
        input
            .unwrap()
            .add(&self.text_embed(&text_ids)?)?
            .broadcast_add(sum_condition)
    }

    /// `seq: [K=33, S]` (u32), `sum_condition: [1, DIM]`, `ca: [Sc, DIM]`
    /// -> `(transformer_out [S,DIM] after out_norm, text_logits [S, TEXT_CARD_OUT])`.
    pub fn forward_text(
        &self,
        seq: &Tensor,
        sum_condition: &Tensor,
        ca: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let input = self.embed(seq, sum_condition)?;
        // transformer output lands on its tail device (hetero) -> back to the LM device where
        // out_norm + text_linear (and the downstream depformer) live.
        let tr_out = self
            .transformer
            .forward(&input, ca)?
            .to_device(&self.device)?;
        let h = tr_out.rms_norm(&self.out_norm, RMS_EPS)?;
        let logits = h.matmul_t(&self.text_linear)?;
        Ok((h, logits))
    }

    /// Streaming step over the FULL frame history `seq: [K=33, T]` (all frames generated
    /// so far). Runs the causal Helium over the whole sequence (correct absolute-position
    /// rope + attention over all prior steps) and returns the LAST position's
    /// `(transformer_out [1,DIM] after out_norm, text_logits [1, TEXT_CARD_OUT])`.
    ///
    /// A single-token `forward_text` has no memory of prior steps (its rope position is 0
    /// and its self-attention sees only that token) - correct only at step 0. Re-running
    /// the whole history each step is O(T²) but reuses the validated multi-token path.
    pub fn forward_text_seq(
        &self,
        seq: &Tensor,
        sum_condition: &Tensor,
        ca: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let t = seq.shape().dims2()?.1;
        let input = self.embed(seq, sum_condition)?;
        let tr_out = self
            .transformer
            .forward(&input, ca)?
            .to_device(&self.device)?;
        let last = tr_out.narrow(0, t - 1, 1)?; // [1, DIM]
        let h = last.rms_norm(&self.out_norm, RMS_EPS)?;
        let logits = h.matmul_t(&self.text_linear)?;
        Ok((h, logits))
    }

    /// Build a streaming KV cache bound to the conditioning `ca: [Sc, DIM]`.
    pub fn new_cache(&self, ca: &Tensor) -> Result<HeliumCache> {
        self.transformer.new_cache(ca)
    }

    /// Streaming step: `seq: [K=33, 1]` (this step's frame), `sum_condition: [1, DIM]`,
    /// mutable `cache` -> `(transformer_out [1,DIM] post-out_norm, text_logits [1,CARD])`.
    /// The O(1)-in-history equivalent of `forward_text_seq`; cross-attn reuses the cache's
    /// precomputed K/V and self-attn attends the accumulated cache.
    pub fn forward_text_step(
        &self,
        seq: &Tensor,
        sum_condition: &Tensor,
        cache: &mut HeliumCache,
    ) -> Result<(Tensor, Tensor)> {
        let input = self.embed(seq, sum_condition)?; // [1, DIM]
        let tr_out = self
            .transformer
            .forward_step(&input, cache)?
            .to_device(&self.device)?; // [1, DIM]
        let h = tr_out.rms_norm(&self.out_norm, RMS_EPS)?;
        let logits = h.matmul_t(&self.text_linear)?;
        Ok((h, logits))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// How the frames this LM reads and writes are laid out.
    pub fn streams(&self) -> &Streams {
        &self.streams
    }
}

impl KyutaiLm {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        self.transformer.placement()
    }
}
