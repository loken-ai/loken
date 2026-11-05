//! The block draft: a draft head of several stages that guesses a block of tokens in one pass,
//! the draft kind of a Sinkhorn wide-residual model beside the one-layer head of `draft_head`.
//!
//! The head reads the input of the target layers the file names, at each committed position.
//! The row of a position is the mean of the residual's streams at each such layer, the layers
//! side by side.
//! The row goes through the main projection and its norm, then through each stage's key
//! projection, rotated by its position and rounded to the fp8 grid.
//! Each stage keeps those keys in a ring of `window` slots, position `p` in slot `p % window`.
//!
//! A pass starts from the last committed token at its position.
//! The block holds that token's embedding, then the noise token's in every other slot.
//! Each stage is a latent attention over 1 shared key head, then routed experts, each wrapped in
//! the stage's Sinkhorn residual mixers.
//! A block query attends over the ring slots its position covers and over every key of the block.
//! The block's rows collapse, are normed and go through the target's output head.
//! The Markov head adds the projection of the previous guess's embedding to each row's logits.
//! A row's guess is its argmax.
//!
//! The ring holds rows of committed positions only.
//! A forward from position `p`, and a trim to `p`, drop the rows the capture held from `p` on.
//! The capture holds the last `window` rows, which cover every slot of the ring.

use super::compressed_band::attend_slots;
use super::hyper::sinkhorn::SinkhornMixer;
use super::latent_mqa::{frequencies, phases, LatentMqa};
use super::wide_sparse_layer::RoutedFfn;
use super::GenericHeteroTransformer;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::{DType, Device, Error, Result, Tensor};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The ending of the metadata key whose presence declares a block draft.
const DECLARES: &str = ".dspark.block_size";

/// The shape of a block draft, from the checkpoint's configuration or from a file's metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDraftMeta {
    pub stages: usize,
    /// Tokens of a block.
    pub block: usize,
    /// The token in every slot of the block after the first.
    pub noise_token: u32,
    /// The target layers whose input the head reads, in the order of the row.
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub routed: usize,
    pub used: usize,
    /// Slots of each stage's ring.
    pub window: usize,
    pub dim: usize,
    /// Parallel copies of the hidden state in each stage's residual.
    pub streams: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub gate_temp: f32,
    pub route_scale: f32,
    /// The bound of the experts' SwiGLU, `0` for none.
    pub swiglu_limit: f32,
    pub norm_topk: bool,
    pub rms_eps: f32,
    /// The base the stages turn at, unstretched.
    pub rope_theta: f32,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub vocab: usize,
}

impl BlockDraftMeta {
    /// The shape `config.json` under `dir` gives the draft.
    pub fn from_config_json(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        Self::from_config_value(&v)
    }

    /// The shape a checkpoint's configuration gives the draft, under `text_config` when it has
    /// one.
    pub fn from_config_value(v: &serde_json::Value) -> Result<Self> {
        let t = v.get("text_config").unwrap_or(v);
        let u = |k: &str| -> Result<usize> {
            t.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::msg(format!("block draft: the configuration has no {k}")))
        };
        let f = |k: &str, default: f32| {
            t.get(k)
                .and_then(|x| x.as_f64())
                .map_or(default, |x| x as f32)
        };
        let target_layers = t
            .get("dspark_target_layer_ids")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            stages: u("num_nextn_predict_layers")?,
            block: u("dspark_block_size")?,
            noise_token: u("dspark_noise_token_id")? as u32,
            target_layers,
            markov_rank: u("dspark_markov_rank")?,
            routed: u("dspark_n_routed_experts")?,
            used: u("dspark_num_experts_per_tok")?,
            window: u("sliding_window")?,
            dim: u("hidden_size")?,
            streams: u("hc_mult")?,
            heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_dim: u("qk_rope_head_dim")?,
            o_groups: u("o_groups")?,
            o_lora_rank: u("o_lora_rank")?,
            gate_temp: f("gate_temp", 1.0),
            route_scale: f("routed_scaling_factor", 1.0),
            swiglu_limit: f("swiglu_limit", 0.0),
            norm_topk: t
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            rms_eps: f("rms_norm_eps", 1e-6),
            rope_theta: f("rope_theta", 10000.0),
            sinkhorn_iters: u("hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: f("hc_eps", 1e-6),
            vocab: u("vocab_size")?,
        })
    }

    /// The shape a file's metadata declares, `None` when it declares no block draft.
    ///
    /// The keys share the prefix of the key that ends in [`DECLARES`].
    /// A file that declares some of the keys and not the others is refused.
    pub fn from_metadata(md: &HashMap<String, Value>) -> Result<Option<Self>> {
        let Some(prefix) = md.keys().find_map(|k| {
            k.strip_suffix("block_size")
                .filter(|_| k.ends_with(DECLARES))
        }) else {
            return Ok(None);
        };
        let get = |name: &str| {
            md.get(&format!("{prefix}{name}"))
                .ok_or_else(|| Error::msg(format!("block draft: the file has no {prefix}{name}")))
        };
        let u = |name: &str| -> Result<usize> { Ok(get(name)?.to_u32()? as usize) };
        let f = |name: &str| -> Result<f32> { get(name)?.to_f32() };
        let target_layers = get("target_layers")?
            .to_vec()?
            .iter()
            .map(|v| v.to_u32().map(|l| l as usize))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Self {
            stages: u("stage_count")?,
            block: u("block_size")?,
            noise_token: u("noise_token_id")? as u32,
            target_layers,
            markov_rank: u("markov_rank")?,
            routed: u("expert_count")?,
            used: u("expert_used_count")?,
            window: u("sliding_window")?,
            dim: u("embedding_length")?,
            streams: u("hyper_connection.count")?,
            heads: u("attention.head_count")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            o_groups: u("attention.o_groups")?,
            o_lora_rank: u("attention.o_lora_rank")?,
            gate_temp: f("expert_gating_temperature")?,
            route_scale: f("expert_weights_scale")?,
            swiglu_limit: f("expert_swiglu_limit")?,
            norm_topk: get("expert_weights_norm")?.to_bool()?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            rope_theta: f("rope.freq_base")?,
            sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            vocab: u("vocab_size")?,
        }))
    }
}

/// One stage: latent attention over the ring and the block, then routed experts.
pub struct DraftStage {
    /// The attention's projections, norms and sink.
    /// Its own state stays empty: the stage attends over its ring.
    pub(super) attn: LatentMqa,
    /// `[dim, o_groups * o_lora_rank]`: what the attention's output goes through.
    pub(super) wo_b: Projection,
    /// `[dim]` F32.
    pub(super) attn_norm: Tensor,
    /// `[dim]` F32.
    pub(super) ffn_norm: Tensor,
    pub(super) attn_mix: SinkhornMixer,
    pub(super) ffn_mix: SinkhornMixer,
    pub(super) ffn: RoutedFfn,
}

/// The weights of a block draft.
pub struct BlockDraftHead {
    pub(super) meta: BlockDraftMeta,
    pub(super) stages: Vec<DraftStage>,
    /// `[dim, dim * target layers]`.
    pub(super) main_proj: Projection,
    /// `[dim]` F32.
    pub(super) main_norm: Tensor,
    /// `[dim]` F32.
    pub(super) final_norm: Tensor,
    /// `[vocab, markov_rank]` F32.
    pub(super) markov_embed: Tensor,
    /// `[vocab, markov_rank]`.
    pub(super) markov_head: Projection,
    /// Rotation frequency of each lane pair of the rotated tail, at the base theta.
    pub(super) inv_freq: Vec<f32>,
}

/// Each stage's ring of keys, `window * head_dim` values per stage.
pub struct DraftRing {
    keys: Vec<Vec<f32>>,
}

/// The last of the largest values of `v`.
fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

impl BlockDraftHead {
    /// The head over its stages and weights.
    ///
    /// Refused when the stage count is not the one `meta` gives.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        meta: BlockDraftMeta,
        stages: Vec<DraftStage>,
        main_proj: Projection,
        main_norm: Tensor,
        final_norm: Tensor,
        markov_embed: Tensor,
        markov_head: Projection,
    ) -> Result<Self> {
        if stages.is_empty() || stages.len() != meta.stages {
            crate::tensor::bail!(
                "block draft: {} stages, the header says {}",
                stages.len(),
                meta.stages
            );
        }
        // The stages turn at the base theta, unstretched.
        let inv_freq = frequencies(meta.rope_dim, 0, meta.rope_theta, 1.0);
        Ok(Self {
            meta,
            stages,
            main_proj,
            main_norm,
            final_norm,
            markov_embed,
            markov_head,
            inv_freq,
        })
    }

    pub fn meta(&self) -> &BlockDraftMeta {
        &self.meta
    }

    /// Every stage's ring, its slots zero.
    pub fn ring(&self) -> DraftRing {
        let width = self.meta.window.max(1) * self.meta.head_dim;
        DraftRing {
            keys: vec![vec![0f32; width]; self.stages.len()],
        }
    }

    /// The tokens of the block that starts from `token`: `token`, then the noise token.
    pub fn block_ids(&self, token: u32) -> Vec<u32> {
        let mut ids = vec![self.meta.noise_token; self.meta.block];
        if let Some(first) = ids.first_mut() {
            *first = token;
        }
        ids
    }

    /// The captured row `row` through the main projection and its norm, `[1, dim]`.
    fn main_x(&self, row: &[f32]) -> Result<Tensor> {
        let m = &self.meta;
        let width = m.dim * m.target_layers.len();
        let mh = Tensor::from_vec(row.to_vec(), (1, width), &Device::Cpu)?;
        rms_norm(&self.main_proj.apply_here(&mh)?, &self.main_norm, m.rms_eps)
    }

    /// Writes each stage's key of `main_x` `[1, dim]` at position `pos` into its slot of `ring`.
    fn write_keys(&self, ring: &mut DraftRing, main_x: &Tensor, pos: usize) -> Result<()> {
        let (window, hd) = (self.meta.window.max(1), self.meta.head_dim);
        let slot = pos % window;
        let (cos, sin) = phases(&self.inv_freq, pos..pos + 1);
        for (stage, keys) in self.stages.iter().zip(ring.keys.iter_mut()) {
            let kv = stage.attn.key_rows(main_x, &cos, &sin)?;
            keys[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
        }
        Ok(())
    }

    /// Writes the keys of the captured row `row` of position `pos` into `ring`.
    pub fn write_row(&self, ring: &mut DraftRing, row: &[f32], pos: usize) -> Result<()> {
        let main_x = self.main_x(row)?;
        self.write_keys(ring, &main_x, pos)
    }

    /// The block's guesses and each row's logits after the Markov bias.
    ///
    /// `row` is the captured row of `start_pos`, written into the ring first.
    /// `token` is the committed token at `start_pos`.
    /// `emb` is the embedding of [`Self::block_ids`] of `token`, `[block, dim]`.
    /// `head` is the target's output head.
    pub fn draft(
        &self,
        ring: &mut DraftRing,
        row: &[f32],
        token: u32,
        start_pos: usize,
        emb: &Tensor,
        head: &Projection,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let m = &self.meta;
        let (dim, hc, bs, eps) = (m.dim, m.streams, m.block, m.rms_eps);
        let main_x = self.main_x(row)?;
        self.write_keys(ring, &main_x, start_pos)?;
        let coverage = m.window.max(1).min(start_pos + 1);

        // Every row of the block copied to each stream.
        let emb = emb
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((1, bs, dim))?;
        let mut x = emb
            .unsqueeze(2)?
            .broadcast_as((1, bs, hc, dim))?
            .contiguous()?
            .reshape((1, bs, hc * dim))?;
        // The first stage's attention collapses each row onto stream 0.
        let mut pre: Vec<Vec<f32>> = (0..bs)
            .map(|_| {
                let mut w = vec![0f32; hc];
                if let Some(first) = w.first_mut() {
                    *first = 1.0;
                }
                w
            })
            .collect();
        for (s, stage) in self.stages.iter().enumerate() {
            let am = stage.attn_mix.mixes(&x)?;
            let input = rms_norm(&stage.attn_mix.pre(&x, &pre)?, &stage.attn_norm, eps)?;
            let out = self.attend(stage, &ring.keys[s], &input, start_pos, coverage)?;
            x = stage.attn_mix.post(&out, &x, &am.post, &am.comb)?;
            pre = am.pre;
            let fm = stage.ffn_mix.mixes(&x)?;
            let input = rms_norm(&stage.ffn_mix.pre(&x, &pre)?, &stage.ffn_norm, eps)?;
            let out = stage.ffn.forward(&input)?;
            x = stage.ffn_mix.post(&out, &x, &fm.post, &fm.comb)?;
            pre = fm.pre;
        }

        // The rows collapse with the last stage's pre mix, then the norm and the head.
        let last = self
            .stages
            .last()
            .ok_or_else(|| Error::msg("block draft: no stage"))?;
        let h = rms_norm(&last.ffn_mix.pre(&x, &pre)?, &self.final_norm, eps)?;
        let mut logits = head.apply_here(&h.reshape((bs, dim))?)?.to_vec2::<f32>()?;
        // Each row adds the Markov bias of the guess before it, the first the bias of `token`.
        let mut guesses = Vec::with_capacity(bs);
        let mut prev = token;
        for row in logits.iter_mut() {
            let e = self.markov_embed.narrow(0, prev as usize, 1)?;
            let bias = self
                .markov_head
                .apply_here(&e)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (l, b) in row.iter_mut().zip(&bias) {
                *l += *b;
            }
            prev = argmax(row);
            guesses.push(prev);
        }
        Ok((guesses, logits))
    }

    /// A stage's attention for the block rows `input` `[1, block, dim]`, which stand at
    /// `start_pos + 1..`, over the first `coverage` slots of `ring` and the block's own keys.
    /// Returns `[1, block, dim]`.
    fn attend(
        &self,
        stage: &DraftStage,
        ring: &[f32],
        input: &Tensor,
        start_pos: usize,
        coverage: usize,
    ) -> Result<Tensor> {
        let m = &self.meta;
        let (bs, window, heads, hd) = (m.block, m.window.max(1), m.heads, m.head_dim);
        let a = &stage.attn;
        let x2 = input.reshape((bs, m.dim))?;
        let (cos, sin) = phases(&self.inv_freq, start_pos + 1..start_pos + 1 + bs);
        let q = a.queries(&a.query_factor(&x2)?, &cos, &sin)?;
        let own = a.key_rows(&x2, &cos, &sin)?;
        // The ring's slots in slot order, then the block's keys.
        let mut keys = Vec::with_capacity((window + bs) * hd);
        keys.extend_from_slice(ring);
        keys.extend_from_slice(&own);
        // Every block row sees the covered slots and every key of the block.
        let mut seen: Vec<i32> = (0..coverage as i32).collect();
        seen.extend(window as i32..(window + bs) as i32);
        let width = seen.len();
        let slots: Vec<i32> = (0..bs).flat_map(|_| seen.iter().copied()).collect();
        let o = attend_slots(&q, &keys, &a.w.sink, &slots, width, (heads, hd), a.scale());
        let mixed = a.project_out(o, &cos, &sin, bs)?;
        stage.wo_b.apply_here(&mixed)?.reshape((1, bs, m.dim))
    }
}

/// The captured row of a position.
struct Held {
    pos: usize,
    token: u32,
    /// The target layers' stream means side by side.
    row: Vec<f32>,
}

/// What the capture holds between forwards.
struct CaptureLog {
    /// Ascending positions.
    held: VecDeque<Held>,
    /// Rows held at most.
    keep: usize,
    /// Width of each target layer's part of a row.
    dim: usize,
    /// Target layers a row holds.
    layers: usize,
    /// Positions below this are in the ring.
    folded: usize,
    /// Positions the forwards cover, whatever rows are held.
    covered: usize,
    /// A forward or a trim ran since the head last drafted.
    fresh: bool,
}

/// The rows of the target layers' input that the head reads, by position.
pub struct HiddenCapture(Mutex<CaptureLog>);

/// What the head drafts from: its start, and the rows the ring has yet to take.
struct Pending {
    start_pos: usize,
    token: u32,
    row: Vec<f32>,
    /// `(position, row)` of the positions before `start_pos` that the ring has yet to take.
    before: Vec<(usize, Vec<f32>)>,
}

impl HiddenCapture {
    /// A capture of `layers` parts of `dim` per row, holding the last `keep` rows.
    pub fn new(keep: usize, layers: usize, dim: usize) -> Self {
        Self(Mutex::new(CaptureLog {
            held: VecDeque::new(),
            keep: keep.max(1),
            dim,
            layers,
            folded: 0,
            covered: 0,
            fresh: false,
        }))
    }

    fn with<R>(&self, f: impl FnOnce(&mut CaptureLog) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the forward of `tokens` at the positions `first..`: the rows from `first` on are
    /// dropped, then each token takes a row of zeros.
    pub fn open(&self, first: usize, tokens: &[u32]) {
        self.with(|c| {
            c.held.retain(|h| h.pos < first);
            let width = c.dim * c.layers;
            // Only the last `keep` rows of the forward stay.
            let skip = tokens.len().saturating_sub(c.keep);
            for (i, &token) in tokens.iter().enumerate().skip(skip) {
                c.held.push_back(Held {
                    pos: first + i,
                    token,
                    row: vec![0f32; width],
                });
            }
            while c.held.len() > c.keep {
                c.held.pop_front();
            }
            c.folded = c.folded.min(first);
            c.covered = first + tokens.len();
            c.fresh = true;
        })
    }

    /// Keeps the rows of the positions below `len`.
    pub fn trim(&self, len: usize) {
        self.with(|c| {
            c.held.retain(|h| h.pos < len);
            c.folded = c.folded.min(len);
            c.covered = c.covered.min(len);
            c.fresh = len > 0;
        })
    }

    /// Adds, into part `slot` of the rows at positions `first..`, the mean over the streams of
    /// the residual rows `r` `[1, rows, streams * dim]`.
    ///
    /// A row that the capture does not hold is skipped.
    pub fn add(&self, slot: usize, first: usize, r: &Tensor) -> Result<()> {
        let (_, rows, wide) = r.dims3()?;
        let (oldest, dim) = self.with(|c| (c.held.front().map(|h| h.pos), c.dim));
        let Some(oldest) = oldest else {
            return Ok(());
        };
        if dim == 0 || !wide.is_multiple_of(dim) {
            crate::tensor::bail!("a residual of {wide} does not hold streams of {dim}");
        }
        let streams = wide / dim;
        let skip = oldest.saturating_sub(first).min(rows);
        if skip == rows {
            return Ok(());
        }
        let values = r
            .narrow(1, skip, rows - skip)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        self.with(|c| {
            for (j, residual) in values.chunks_exact(wide).enumerate() {
                let pos = first + skip + j;
                let Some(h) = c.held.iter_mut().find(|h| h.pos == pos) else {
                    continue;
                };
                let Some(out) = h.row.get_mut(slot * dim..(slot + 1) * dim) else {
                    continue;
                };
                for copy in residual.chunks_exact(dim).take(streams) {
                    for (o, v) in out.iter_mut().zip(copy) {
                        *o += v / streams as f32;
                    }
                }
            }
        });
        Ok(())
    }

    /// Positions the capture covers: the forwards' end, cut by each trim.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.with(|c| c.covered)
    }

    /// Every row held, with its position.
    #[cfg(test)]
    pub(super) fn rows(&self) -> Vec<(usize, Vec<f32>)> {
        self.with(|c| c.held.iter().map(|h| (h.pos, h.row.clone())).collect())
    }

    /// What the head drafts from, when a forward or a trim ran since it last drafted.
    ///
    /// The start is the last row held.
    /// The rows before it from the first not yet in the ring are taken with it.
    /// They count as in the ring from here.
    fn take_pending(&self) -> Option<Pending> {
        self.with(|c| {
            if !c.fresh {
                return None;
            }
            let last = c.held.back()?;
            let (start_pos, token, row) = (last.pos, last.token, last.row.clone());
            let before = c
                .held
                .iter()
                .filter(|h| h.pos >= c.folded && h.pos < start_pos)
                .map(|h| (h.pos, h.row.clone()))
                .collect();
            c.folded = start_pos + 1;
            c.fresh = false;
            Some(Pending {
                start_pos,
                token,
                row,
                before,
            })
        })
    }
}

/// A guess of a block from the head, for a test that scripts it: the positions the cache holds,
/// the token chosen after them and the most tokens wanted.
#[cfg(test)]
pub(crate) type ScriptedDraft = Box<dyn FnMut(usize, u32, usize) -> Vec<u32> + Send>;

/// What guesses the block.
enum BlockDraftKind {
    Head {
        head: Box<BlockDraftHead>,
        ring: DraftRing,
    },
    #[cfg(test)]
    Scripted(Mutex<ScriptedDraft>),
}

/// The block draft a model holds, and the capture of the rows it reads.
pub struct BlockDraft {
    kind: BlockDraftKind,
    /// Shared with each target layer.
    pub capture: Arc<HiddenCapture>,
}

/// Where a target layer adds its input to the capture.
pub struct LayerCapture {
    pub log: Arc<HiddenCapture>,
    /// The layer's part of a row.
    pub slot: usize,
}

impl GenericHeteroTransformer {
    /// Whether the model drafts blocks of tokens.
    pub fn has_block_draft(&self) -> bool {
        self.wide_sparse
            .as_ref()
            .is_some_and(|ws| ws.draft.is_some())
    }

    /// Takes `head` as the model's block draft, in place of any it had.
    ///
    /// Each target layer adds its input to the capture from the next forward on.
    /// Refused when the model is not a Sinkhorn wide residual, when a target layer is past the
    /// stack, or when the head's width or vocabulary is not the model's.
    pub fn attach_block_draft(&mut self, head: BlockDraftHead) -> Result<()> {
        let m = head.meta.clone();
        let vocab = match self.wide_sparse.as_ref() {
            Some(ws) => ws.head.dims()[0],
            None => crate::tensor::bail!("a block draft needs a Sinkhorn wide residual"),
        };
        if m.vocab != vocab {
            crate::tensor::bail!(
                "block draft: a vocabulary of {}, the model's is {vocab}",
                m.vocab
            );
        }
        for &l in &m.target_layers {
            let dim = self
                .layers
                .get(l)
                .and_then(|layer| layer.sinkhorn.as_ref())
                .map(|b| b.attn_mix.dim());
            if dim != Some(m.dim) {
                crate::tensor::bail!(
                    "block draft: target layer {l} of {} does not hold rows of {}",
                    self.layers.len(),
                    m.dim
                );
            }
        }
        let capture = Arc::new(HiddenCapture::new(m.window, m.target_layers.len(), m.dim));
        self.set_capture(&m.target_layers, &capture);
        let ring = head.ring();
        self.set_block_draft(BlockDraft {
            kind: BlockDraftKind::Head {
                head: Box::new(head),
                ring,
            },
            capture,
        });
        Ok(())
    }

    /// Takes `script` as the model's block draft, for a test that scripts the guesses.
    #[cfg(test)]
    pub(crate) fn set_scripted_block_draft(&mut self, script: ScriptedDraft) -> Result<()> {
        if self.wide_sparse.is_none() {
            crate::tensor::bail!("a block draft needs a Sinkhorn wide residual");
        }
        let capture = Arc::new(HiddenCapture::new(1, 0, 0));
        self.set_capture(&[], &capture);
        self.set_block_draft(BlockDraft {
            kind: BlockDraftKind::Scripted(Mutex::new(script)),
            capture,
        });
        Ok(())
    }

    /// Each layer of `targets` adds its input to part `i` of `capture`, `i` the first index the
    /// layer takes in `targets`.
    /// Every other layer adds nothing.
    fn set_capture(&mut self, targets: &[usize], capture: &Arc<HiddenCapture>) {
        for (il, layer) in self.layers.iter_mut().enumerate() {
            if let Some(block) = layer.sinkhorn.as_mut() {
                block.capture = targets
                    .iter()
                    .position(|&l| l == il)
                    .map(|slot| LayerCapture {
                        log: Arc::clone(capture),
                        slot,
                    });
            }
        }
    }

    fn set_block_draft(&mut self, draft: BlockDraft) {
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.draft = Some(draft);
        }
    }

    /// The capture keeps the rows of the positions below `len`.
    pub(super) fn block_draft_trim(&mut self, len: usize) {
        if let Some(d) = self.wide_sparse.as_ref().and_then(|ws| ws.draft.as_ref()) {
            d.capture.trim(len);
        }
    }

    /// Up to `k` tokens the block draft guesses after `next`, the token chosen after the last
    /// position the cache holds and not yet in it.
    ///
    /// The head starts from the last committed token at its position.
    /// Its first guess stands for `next`: the rest are returned only when it is `next`.
    /// Empty when no forward or trim ran since the head last drafted.
    pub fn block_draft_tokens(&mut self, next: u32, k: usize) -> Result<Vec<u32>> {
        let Some(capture) = self
            .wide_sparse
            .as_ref()
            .and_then(|ws| ws.draft.as_ref())
            .map(|d| Arc::clone(&d.capture))
        else {
            return Ok(Vec::new());
        };
        #[cfg(test)]
        if let Some(BlockDraftKind::Scripted(script)) = self
            .wide_sparse
            .as_ref()
            .and_then(|ws| ws.draft.as_ref())
            .map(|d| &d.kind)
        {
            let mut held = script.lock().unwrap_or_else(|e| e.into_inner());
            let script: &mut ScriptedDraft = &mut held;
            return Ok(script(capture.len(), next, k));
        }
        let Some(p) = capture.take_pending() else {
            return Ok(Vec::new());
        };
        let ids = match self
            .wide_sparse
            .as_ref()
            .and_then(|ws| ws.draft.as_ref())
            .map(|d| &d.kind)
        {
            Some(BlockDraftKind::Head { head, .. }) => head.block_ids(p.token),
            _ => return Ok(Vec::new()),
        };
        // The block's embeddings as the model's embedding stage gives them.
        let edev = self.embeddings.embeddings().device().clone();
        let ids = Tensor::from_vec(ids.clone(), (ids.len(),), &edev)?;
        let mut emb = self.embeddings.forward(&ids)?.to_dtype(DType::F32)?;
        if let Some(scale) = self.config.embed_scale {
            emb = (emb * scale)?;
        }
        let Some(ws) = self.wide_sparse.as_mut() else {
            return Ok(Vec::new());
        };
        let output = &ws.head;
        let Some(BlockDraftKind::Head { head, ring }) = ws.draft.as_mut().map(|d| &mut d.kind)
        else {
            return Ok(Vec::new());
        };
        for (pos, row) in &p.before {
            head.write_row(ring, row, *pos)?;
        }
        let (guesses, _) = head.draft(ring, &p.row, p.token, p.start_pos, &emb, output)?;
        Ok(match guesses.split_first() {
            Some((&first, rest)) if first == next => rest.iter().copied().take(k).collect(),
            _ => Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::deepseek_v41::dspark::{Dspark, DsparkConfig, DsparkState};
    use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
    use crate::inference::model::deepseek_v41::source::WeightSource;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::open_mapped;
    use crate::tensor::quantized::GgmlDType;
    use safetensors::SafeTensors;

    /// The fields both readers give, side by side.
    fn same_shape(got: &BlockDraftMeta, want: &DsparkConfig) {
        assert_eq!(
            (got.stages, got.block, got.noise_token, &got.target_layers),
            (
                want.n_stages,
                want.block_size,
                want.noise_token,
                &want.target_layers
            )
        );
        assert_eq!(
            (
                got.markov_rank,
                got.routed,
                got.used,
                got.window,
                got.dim,
                got.streams
            ),
            (
                want.markov_rank,
                want.n_routed,
                want.n_activated,
                want.window_size,
                want.dim,
                want.hc_mult
            )
        );
        assert_eq!(
            (
                got.heads,
                got.head_dim,
                got.rope_dim,
                got.o_groups,
                got.o_lora_rank
            ),
            (
                want.n_heads,
                want.head_dim,
                want.rope_head_dim,
                want.o_groups,
                want.o_lora_rank
            )
        );
        assert_eq!(
            (
                got.gate_temp,
                got.route_scale,
                got.swiglu_limit,
                got.norm_topk
            ),
            (
                want.gate_temp,
                want.route_scale,
                want.swiglu_limit,
                want.norm_topk
            )
        );
        assert_eq!(
            (
                got.rms_eps,
                got.rope_theta,
                got.sinkhorn_iters,
                got.hc_eps,
                got.vocab
            ),
            (
                want.rms_eps,
                want.rope_theta,
                want.hc_sinkhorn_iters,
                want.hc_eps,
                want.vocab
            )
        );
    }

    fn vectors_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/deepseek_v41_dspark")
    }

    /// Port of the V4.1 test `the_draft_config_survives_the_file`: the configuration and the
    /// keys the V4.1 writer puts in a file read as the V4.1 reader reads them, and a file without
    /// the keys declares no draft.
    #[test]
    fn the_draft_shape_reads_as_the_v41_reader_reads_it() {
        let dir = vectors_dir();
        let want = DsparkConfig::from_config_json(&dir).unwrap();
        let got = BlockDraftMeta::from_config_json(&dir).unwrap();
        same_shape(&got, &want);

        let mut md = vec![(
            "general.architecture".to_string(),
            Value::String(crate::inference::model::deepseek_v41::ARCH.into()),
        )];
        md.extend(want.to_metadata());
        let entries = vec![GgufEntry {
            name: "output_norm".into(),
            dims: vec![4],
            dtype: GgmlDType::F32,
            data: vec![0; 16],
        }];
        let path = std::env::temp_dir().join(format!(
            "loken-block-draft-meta-{}.gguf",
            std::process::id()
        ));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let from_file = BlockDraftMeta::from_metadata(&g.content.metadata)
            .unwrap()
            .unwrap();
        same_shape(&from_file, &want);
        assert_eq!(from_file, got);

        write_gguf_with_metadata(&path, &md[..1], &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            BlockDraftMeta::from_metadata(&g.content.metadata).unwrap(),
            None
        );
    }

    /// Port of the V4.1 test `the_draft_matches_the_reference`, on the generic head.
    ///
    /// The weights, the hidden rows, the tokens and the positions are the reference's, written
    /// at a small size beside its block logits.
    /// The ring is seeded by a prompt longer than the window, then each step drafts.
    /// The generic head's block logits stay within the V4.1 test's bounds of the reference's.
    /// Each step's guesses are those of the V4.1 head on the same inputs.
    #[test]
    fn the_generic_draft_matches_the_reference_and_guesses_as_the_v41_draft() {
        let dir = vectors_dir();
        let vec_bytes = std::fs::read(dir.join("vectors.safetensors")).unwrap();
        let vecs = SafeTensors::deserialize(&vec_bytes).unwrap();
        let f32s = |name: &str| -> Vec<f32> {
            vecs.tensor(name)
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let i32s = |name: &str| -> Vec<i32> {
            vecs.tensor(name)
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let meta = BlockDraftMeta::from_config_json(&dir).unwrap();
        let ck = super::super::dense::BlockScaledCheckpoint::open(&dir).unwrap();
        let head = BlockDraftHead::from_checkpoint(&ck, meta.clone()).unwrap();
        let embed = ck.dense_f32("token_embd").unwrap();
        let output = Projection::Dense(ck.dense_f32("output").unwrap());

        let cfg = DsparkConfig::from_config_json(&dir).unwrap();
        let src = SafeTensorsSource::open(&dir).unwrap();
        let v41 = Dspark::load(&src, cfg.clone()).unwrap();
        let v41_embed = src.dense_f32("embed.weight").unwrap();
        let v41_head = Projection::Dense(src.dense_f32("head.weight").unwrap());

        let width = meta.dim * meta.target_layers.len();
        let prefill = f32s("prefill_main_hidden");
        let mut ring = head.ring();
        let mut v41_state = DsparkState::new(v41.stages.len(), cfg.window_size, cfg.head_dim);
        for (pos, row) in prefill.chunks_exact(width).enumerate() {
            head.write_row(&mut ring, row, pos).unwrap();
            v41.update_window_from_hidden(&mut v41_state, row, pos)
                .unwrap();
        }
        let (tokens, poss) = (i32s("token"), i32s("pos"));
        let (hidden, logits) = (f32s("main_hidden"), f32s("draft_logits"));
        let (bs, vocab) = (meta.block, meta.vocab);
        let (mut worst_cos, mut worst_rel, mut worst_apart) = (1f32, 0f32, 0f32);
        for step in 0..tokens.len() {
            let row = &hidden[step * width..(step + 1) * width];
            let (token, pos) = (tokens[step] as u32, poss[step] as usize);
            let rows: Vec<Tensor> = head
                .block_ids(token)
                .iter()
                .map(|&t| embed.narrow(0, t as usize, 1).unwrap())
                .collect();
            let emb = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
            let (got_ids, got) = head
                .draft(&mut ring, row, token, pos, &emb, &output)
                .unwrap();
            let (want_ids, v41_logits) = v41
                .forward_draft_logits(&mut v41_state, row, token, pos, &v41_embed, &v41_head)
                .unwrap();
            assert_eq!(
                got_ids, want_ids,
                "step {step}: the guesses of the V4.1 draft"
            );
            for i in 0..bs {
                let want = &logits[(step * bs + i) * vocab..(step * bs + i + 1) * vocab];
                let g = &got[i];
                let dot: f32 = g.iter().zip(want).map(|(a, b)| a * b).sum();
                let na = g.iter().map(|a| a * a).sum::<f32>().sqrt();
                let nb = want.iter().map(|a| a * a).sum::<f32>().sqrt();
                let num: f32 = g.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
                worst_cos = worst_cos.min(dot / (na * nb));
                worst_rel = worst_rel.max(num.sqrt() / nb);
                let apart = g
                    .iter()
                    .zip(&v41_logits[i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                worst_apart = worst_apart.max(apart);
            }
        }
        eprintln!(
            "generic draft against reference: worst cosine {worst_cos}, worst relative L2 {worst_rel}; against V4.1: {worst_apart} apart at most"
        );
        assert!(worst_cos > 0.999, "block logits turned: cosine {worst_cos}");
        assert!(
            worst_rel < 0.05,
            "block logits drifted: relative L2 {worst_rel}"
        );
    }

    /// The capture holds the last rows of a forward, adds each target layer's stream mean into
    /// its part, drops what a later forward or a trim replaces, and hands the head each
    /// committed row once.
    #[test]
    fn the_capture_hands_each_committed_row_once() {
        let (dim, streams) = (2usize, 2usize);
        let capture = HiddenCapture::new(3, 2, dim);
        // 4 rows from position 0: the capture keeps the last 3.
        capture.open(0, &[10, 11, 12, 13]);
        let values: Vec<f32> = (0..4 * streams * dim).map(|i| i as f32).collect();
        let r = Tensor::from_vec(values, (1, 4, streams * dim), &Device::Cpu).unwrap();
        capture.add(1, 0, &r).unwrap();
        let p = capture.take_pending().unwrap();
        assert_eq!((p.start_pos, p.token), (3, 13));
        // Row 3 holds 12..16: streams [12, 13] and [14, 15], mean [13, 14] in part 1.
        assert_eq!(p.row, vec![0.0, 0.0, 13.0, 14.0]);
        assert_eq!(
            p.before.iter().map(|(pos, _)| *pos).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(capture.take_pending().is_none(), "nothing ran since");
        // A verify of 3 rows from 4, then a trim that keeps 1 of them.
        capture.open(4, &[14, 15, 16]);
        capture.trim(5);
        let p = capture.take_pending().unwrap();
        assert_eq!((p.start_pos, p.token), (4, 14));
        assert!(p.before.is_empty(), "rows below 4 are in the ring");
        // A forward from an earlier position replaces the rows from there on.
        capture.open(2, &[20]);
        let p = capture.take_pending().unwrap();
        assert_eq!((p.start_pos, p.token), (2, 20));
        assert_eq!(capture.len(), 3);
        capture.trim(0);
        assert!(
            capture.take_pending().is_none(),
            "a trim to 0 leaves nothing"
        );
    }
}
