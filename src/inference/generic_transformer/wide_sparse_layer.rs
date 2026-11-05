//! A layer of a wide-residual mixture model whose residual mixes through Sinkhorn projections
//! around latent attention, on the host.
//!
//! The residual is `streams` copies of the hidden state.
//! A sublayer reads the copies collapsed by a pre mix, normalised.
//! Its output goes back to every copy through the post and comb mixes of its own Sinkhorn mixer.
//!
//! The pre mix a sublayer collapses with is the one the sublayer before it computed:
//! - the token mixer collapses with the pre mix of the previous layer's feed-forward,
//! - the feed-forward collapses with the pre mix its own layer's token mixer computed,
//! - the first layer of a forward collapses onto stream 0,
//! - the final norm collapses with the pre mix of the last layer's feed-forward.
//!
//! The pre mixes pass from layer to layer through a [`PreMixCarry`] the model shares.
//!
//! The rows of a verify attend 1 at a time, each as the decode step of its row.
//! Their feed-forward runs over every row at once.
//!
//! An engram layer adds hashed n-gram rows to its residual before its token mixer.
//! The hashes read the token ids of the sequence from a [`TokenCarry`] the model shares.
//! The rows of a verify take their engram 1 row at a time, a prompt's all at once.

use super::block_scaled_experts::BlockScaledExpert;
use super::compressed_band::BandCell;
use super::hyper::sinkhorn::SinkhornMixer;
use super::moe_weights::{sqrt_softplus_choice, sqrt_softplus_truncated_choice, Truncation};
use super::ngram::engram::{EngramBlock, EngramMeta};
use super::*;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// The pre mix of a row that collapses onto stream 0 of `streams`.
fn onto_first(streams: usize) -> Vec<f32> {
    let mut row = vec![0f32; streams];
    if let Some(w) = row.first_mut() {
        *w = 1.0;
    }
    row
}

/// The pre mix one layer left for the rows `first..first + rows`.
struct Carried {
    first: usize,
    rows: usize,
    pre: Vec<Vec<f32>>,
}

/// Where each layer of a forward finds the pre mix its token mixer collapses with, and leaves
/// the one its feed-forward computed.
#[derive(Default)]
pub struct PreMixCarry(std::sync::Mutex<Option<Carried>>);

impl PreMixCarry {
    fn with<R>(&self, f: impl FnOnce(&mut Option<Carried>) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the forward over the rows `first..first + rows`: every row collapses onto
    /// stream 0 of `streams`.
    pub fn open(&self, first: usize, rows: usize, streams: usize) {
        let pre = (0..rows).map(|_| onto_first(streams)).collect();
        self.with(|c| *c = Some(Carried { first, rows, pre }));
    }

    /// Leaves `pre` for the next layer of the forward over the rows `first..first + rows`.
    fn put(&self, first: usize, rows: usize, pre: Vec<Vec<f32>>) {
        self.with(|c| *c = Some(Carried { first, rows, pre }));
    }

    /// The pre mix left for the forward over the rows `first..first + rows`.
    fn get(&self, first: usize, rows: usize) -> Result<Vec<Vec<f32>>> {
        self.with(|c| match c {
            Some(c) if c.first == first && c.rows == rows => Ok(c.pre.clone()),
            _ => Err(crate::tensor::Error::msg(format!(
                "no pre mix for the {rows} rows from position {first}: the layer before has not run"
            ))),
        })
    }

    /// The pre mix the last layer left, for a forward over `rows` rows.
    fn last(&self, rows: usize) -> Result<Vec<Vec<f32>>> {
        self.with(|c| match c {
            Some(c) if c.rows == rows => Ok(c.pre.clone()),
            _ => Err(crate::tensor::Error::msg(format!(
                "no pre mix for the {rows} rows of the head: the last layer has not run"
            ))),
        })
    }
}

/// The token ids of a sequence by position.
///
/// A forward from position `first` keeps the ids before `first` and writes its own after them.
#[derive(Debug, Default, Clone)]
pub struct TokenLog(Vec<u32>);

impl TokenLog {
    /// Writes `tokens` at the positions `first..`, after the ids of the positions before `first`.
    ///
    /// Refused, with nothing written, when the log holds fewer than `first` ids.
    pub(super) fn open(&mut self, first: usize, tokens: &[u32]) -> Result<()> {
        if self.0.len() < first {
            crate::tensor::bail!(
                "the token log holds {} positions but the forward starts at {first}",
                self.0.len()
            );
        }
        self.0.truncate(first);
        self.0.extend_from_slice(tokens);
        Ok(())
    }

    /// The table rows engram layer `which` reads for the positions `first..first + rows`.
    ///
    /// A position hashes the compressed ids of the last `max_ngram` positions up to it, oldest
    /// first.
    pub(super) fn engram_hashes(
        &self,
        meta: &EngramMeta,
        which: usize,
        first: usize,
        rows: usize,
    ) -> Result<Vec<Vec<u64>>> {
        if self.0.len() < first + rows {
            crate::tensor::bail!(
                "the token log holds {} positions, not the {rows} from position {first}",
                self.0.len()
            );
        }
        Ok((first..first + rows)
            .map(|p| {
                let start = (p + 1).saturating_sub(meta.max_ngram);
                let recent: Vec<u32> = self.0[start..=p]
                    .iter()
                    .map(|&t| meta.compress(t))
                    .collect();
                meta.hash(which, &recent)
            })
            .collect())
    }
}

/// The token log of the sequence the model's layers hold, shared by the layers that read it.
#[derive(Default)]
pub struct TokenCarry(std::sync::Mutex<TokenLog>);

impl TokenCarry {
    fn with<R>(&self, f: impl FnOnce(&mut TokenLog) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the forward of `tokens` at the positions `first..`, as [`TokenLog::open`] does.
    pub fn open(&self, first: usize, tokens: &[u32]) -> Result<()> {
        self.with(|log| log.open(first, tokens))
    }

    /// Holds `tokens` as the whole sequence.
    pub fn reset(&self, tokens: &[u32]) {
        self.with(|log| *log = TokenLog(tokens.to_vec()));
    }

    /// Exchanges the log held with `other`.
    fn exchange(&self, other: &mut TokenLog) {
        self.with(|log| std::mem::swap(log, other));
    }
}

/// The engram a layer adds to its residual before its token mixer.
pub struct LayerEngram {
    pub block: EngramBlock,
    /// The hash layout of every engram layer of the model.
    pub meta: Arc<EngramMeta>,
    /// The layer's index among the engram layers.
    pub which: usize,
    /// Shared by every layer of the model and its head.
    pub tokens: Arc<TokenCarry>,
}

impl LayerEngram {
    /// The residual rows `r` `[1, rows, streams * hidden]` at the positions `first..` with the
    /// engram added, hashed from the ids `log` holds.
    fn add(&self, r: &Tensor, log: &TokenLog, first: usize) -> Result<Tensor> {
        let rows = r.dims3()?.1;
        let hashes = log.engram_hashes(&self.meta, self.which, first, rows)?;
        self.block.forward(r, &hashes)
    }
}

/// The residual rows `r` at the positions `first..` with the layer's engram added, hashed from
/// the model's token log.
///
/// `r` itself for a layer without an engram.
fn with_engram(block: &SinkhornBlock, r: &Tensor, first: usize) -> Result<Tensor> {
    match block.engram.as_ref() {
        Some(e) => e.tokens.with(|log| e.add(r, log, first)),
        None => Ok(r.clone()),
    }
}

/// Routing that gives up exactness for speed, and what it saved.
///
/// The default routes exactly.
#[derive(Default)]
pub struct LossyRouting {
    pub truncation: Truncation,
    /// The experts counted as in memory.
    ///
    /// `None` counts none: the layer reads every routed expert from its mapping on use.
    pub resident: Option<HashSet<usize>>,
    /// Slots of experts not in memory that the truncation removed or replaced, since last
    /// taken.
    pub saved: AtomicU64,
}

/// The feed-forward of a layer: a sqrt-softplus router over routed experts, beside a shared
/// expert.
pub struct RoutedFfn {
    /// `[experts, hidden]`, dense F32.
    pub router: Projection,
    /// Added to the scores for the choice only, `[experts]`.
    pub bias: Vec<f32>,
    pub experts: Vec<BlockScaledExpert>,
    pub shared: Option<BlockScaledExpert>,
    /// Experts a row is routed to.
    pub used: usize,
    pub temperature: f32,
    /// The chosen weights are divided by their sum.
    pub normalize: bool,
    /// Multiplies every chosen weight.
    pub scale: f32,
    /// The bound of the SwiGLU, `0` for none.
    pub limit: f32,
    pub lossy: LossyRouting,
}

impl RoutedFfn {
    /// Each row's experts and their weights before the scale.
    ///
    /// A row takes the truncated choice when the truncation gives up exactness, the plain
    /// choice otherwise.
    fn route(&self, logits: &[Vec<f32>]) -> Vec<(Vec<usize>, Vec<f32>)> {
        let t = self.lossy.truncation;
        if !t.active() {
            return logits
                .par_iter()
                .map(|row| {
                    sqrt_softplus_choice(
                        row,
                        Some(self.bias.as_slice()),
                        self.used,
                        self.temperature,
                        self.normalize,
                    )
                })
                .collect();
        }
        let resident = |e: usize| {
            self.lossy
                .resident
                .as_ref()
                .is_some_and(|set| set.contains(&e))
        };
        let picks: Vec<(Vec<usize>, Vec<f32>, u64)> = logits
            .par_iter()
            .map(|row| {
                sqrt_softplus_truncated_choice(
                    row,
                    Some(self.bias.as_slice()),
                    self.used,
                    self.temperature,
                    self.normalize,
                    t,
                    &resident,
                )
            })
            .collect();
        let saved: u64 = picks.iter().map(|p| p.2).sum();
        self.lossy.saved.fetch_add(saved, Ordering::Relaxed);
        picks.into_iter().map(|(ids, w, _)| (ids, w)).collect()
    }

    /// The block over the rows of `x` `[.., hidden]` F32 on the host, in `x`'s shape.
    ///
    /// Each row starts from its shared expert's output.
    /// The routed experts run by id, each once over the rows routed to it, in row order.
    /// A row adds the outputs of its experts in that order, each times its weight.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let dim = dims.last().copied().unwrap_or(0);
        let n = x.elem_count() / dim.max(1);
        let x2 = x.reshape((n, dim))?;
        let logits = self.router.apply_here(&x2)?.to_vec2::<f32>()?;
        let picks = self.route(&logits);
        // Per expert, the rows routed to it in row order, with their weights.
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
        for (t, (chosen, w)) in picks.into_iter().enumerate() {
            for (&e, &wi) in chosen.iter().zip(&w) {
                match assign.get_mut(e) {
                    Some(rows) => rows.push((t, wi * self.scale)),
                    None => crate::tensor::bail!("routed to expert {e} of {}", self.experts.len()),
                }
            }
        }
        let mut y: Vec<Vec<f32>> = match &self.shared {
            Some(s) => s.forward(&x2, self.limit)?.to_vec2::<f32>()?,
            None => vec![vec![0f32; dim]; n],
        };
        let rows = x2.flatten_all()?.to_vec1::<f32>()?;
        let active: Vec<usize> = (0..assign.len())
            .filter(|&e| !assign[e].is_empty())
            .collect();
        let outs: Vec<Vec<f32>> = active
            .par_iter()
            .map(|&e| -> Result<Vec<f32>> {
                let picked: Vec<f32> = assign[e]
                    .iter()
                    .flat_map(|&(t, _)| rows[t * dim..(t + 1) * dim].iter().copied())
                    .collect();
                let xe = Tensor::from_vec(picked, (assign[e].len(), dim), &Device::Cpu)?;
                self.experts[e]
                    .forward(&xe, self.limit)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            })
            .collect::<Result<Vec<_>>>()?;
        for (out, &e) in outs.iter().zip(&active) {
            for (r, &(t, w)) in assign[e].iter().enumerate() {
                for (a, b) in y[t].iter_mut().zip(&out[r * dim..(r + 1) * dim]) {
                    *a += w * b;
                }
            }
        }
        let flat: Vec<f32> = y.into_iter().flatten().collect();
        Tensor::from_vec(flat, (n, dim), &Device::Cpu)?.reshape(dims)
    }
}

/// What a layer holds around its token mixer: the 2 Sinkhorn mixers, the 2 norms, the token
/// mixer's output projection and the feed-forward.
pub struct SinkhornBlock {
    pub attn_mix: SinkhornMixer,
    pub ffn_mix: SinkhornMixer,
    /// The norm of the token mixer's input, `[hidden]` F32.
    pub attn_norm: Tensor,
    /// The norm of the feed-forward's input, `[hidden]` F32.
    pub ffn_norm: Tensor,
    pub eps: f32,
    /// `[hidden, mixer output]`: what the token mixer's output goes through.
    pub output: Projection,
    pub ffn: RoutedFfn,
    /// Shared by every layer of the model and its head.
    pub carry: Arc<PreMixCarry>,
    /// What an engram layer adds to the residual before the token mixer reads it.
    pub engram: Option<LayerEngram>,
}

/// The token mixer's half of a layer over the residual rows `r` `[1, seq, streams * hidden]` at
/// positions `pos..`, collapsed by `pre`.
///
/// Returns the residual after the half, the pre mix the feed-forward collapses with, and the
/// token mixer's input.
fn attention_half(
    block: &SinkhornBlock,
    mixer: &mut mixer::Mixer,
    r: &Tensor,
    pre: &[Vec<f32>],
    pos: usize,
) -> Result<(Tensor, Vec<Vec<f32>>, Tensor)> {
    let mixes = block.attn_mix.mixes(r)?;
    let input = rms_norm(&block.attn_mix.pre(r, pre)?, &block.attn_norm, block.eps)?;
    let mixed = mixer.forward(&input, pos)?;
    let (b, seq, width) = mixed.dims3()?;
    let out = block
        .output
        .apply_here(&mixed.reshape((b * seq, width))?)?
        .reshape((b, seq, block.attn_mix.dim()))?;
    let next = block.attn_mix.post(&out, r, &mixes.post, &mixes.comb)?;
    Ok((next, mixes.pre, input))
}

impl GenericTransformerLayer {
    /// One layer over the residual rows `x` `[1, seq, streams * hidden]` at positions
    /// `index_pos..`, on the host.
    pub(super) fn forward_sinkhorn(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let r = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let seq = r.dims3()?.1;
        if self.spec_record {
            self.spec_begin(&r, None, index_pos)?;
        }
        // The rows of a verify each attend as the decode step of their row.
        let stepped = seq > 1 && index_pos > 0 && crate::tensor::quantized::DecodeOrder::active();
        let (mid, attn_pre, input) = {
            let block = self
                .sinkhorn
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
            let pre = block.carry.get(index_pos, seq)?;
            if stepped {
                let (mut mids, mut pres, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
                for j in 0..seq {
                    let rj = r.narrow(1, j, 1)?.contiguous()?;
                    let rj = with_engram(block, &rj, index_pos + j)?;
                    let (m, p, i) =
                        attention_half(block, &mut self.mixer, &rj, &pre[j..j + 1], index_pos + j)?;
                    mids.push(m);
                    pres.extend(p);
                    inputs.push(i);
                }
                let mids: Vec<&Tensor> = mids.iter().collect();
                let inputs: Vec<&Tensor> = inputs.iter().collect();
                (Tensor::cat(&mids, 1)?, pres, Tensor::cat(&inputs, 1)?)
            } else {
                let r = with_engram(block, &r, index_pos)?;
                attention_half(block, &mut self.mixer, &r, &pre, index_pos)?
            }
        };
        self.spec_input(&input);
        let block = self
            .sinkhorn
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
        let mixes = block.ffn_mix.mixes(&mid)?;
        let input = rms_norm(
            &block.ffn_mix.pre(&mid, &attn_pre)?,
            &block.ffn_norm,
            block.eps,
        )?;
        let out = block.ffn.forward(&input)?;
        let next = block.ffn_mix.post(&out, &mid, &mixes.post, &mixes.comb)?;
        block.carry.put(index_pos, seq, mixes.pre);
        Ok(next)
    }
}

/// What a model whose layers mix their residual through Sinkhorn projections holds around them:
/// the state the layers pass to each other, the final norm and the head.
pub struct WideSparseHead {
    /// Shared with every layer.
    pub carry: Arc<PreMixCarry>,
    /// Shared with every compressed band layer.
    pub cell: Arc<BandCell>,
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// The final norm, `[hidden]` F32.
    pub norm: Tensor,
    pub eps: f32,
    /// `[vocab, hidden]`.
    pub head: Projection,
    /// Shared with every engram layer, `None` for a model without one.
    pub tokens: Option<Arc<TokenCarry>>,
}

impl WideSparseHead {
    /// The residual the rows of `hidden` `[1, seq, hidden]` at `index_pos..` start from: 1 copy
    /// of each row per stream.
    ///
    /// The carry and the band cell open for the forward over those rows.
    /// The token log takes `input_ids`, the ids of those rows.
    pub fn widen(&self, hidden: &Tensor, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let seq = hidden.dim(1)?;
        if let Some(tokens) = &self.tokens {
            let ids = input_ids
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            tokens.open(index_pos, &ids)?;
        }
        self.carry.open(index_pos, seq, self.streams);
        self.cell.open(index_pos, seq);
        super::wide::widen_embedding(hidden, self.streams)
    }

    /// The logits of the residual rows `hidden` `[1, seq, streams * hidden]`: every row's
    /// `[1, seq, vocab]` with `all_positions`, else the last row's `[1, vocab]`.
    ///
    /// `collapse` is a Sinkhorn mixer of the model.
    /// The rows collapse with the pre mix the last layer left.
    pub fn logits(
        &self,
        collapse: &SinkhornMixer,
        hidden: &Tensor,
        all_positions: bool,
    ) -> Result<Tensor> {
        let r = hidden.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (b, seq, _) = r.dims3()?;
        let pre = self.carry.last(seq)?;
        let (rows, pre) = match all_positions {
            true => (r, pre),
            false => (
                r.narrow(1, seq - 1, 1)?.contiguous()?,
                pre[seq - 1..].to_vec(),
            ),
        };
        let n = rows.dim(1)?;
        let h = rms_norm(&collapse.pre(&rows, &pre)?, &self.norm, self.eps)?;
        let logits = self.head.apply_here(&h.reshape((b * n, collapse.dim()))?)?;
        match all_positions {
            true => {
                let vocab = logits.dim(1)?;
                logits.reshape((b, n, vocab))
            }
            false => Ok(logits),
        }
    }
}

impl GenericHeteroTransformer {
    /// Bytes of the weights every token of a Sinkhorn wide-residual model reads, as held.
    ///
    /// Per layer: the token mixer's resident weights, its output projection, the router with
    /// its bias, the shared expert and the engram projection.
    /// Then the head.
    /// The routed experts are not counted.
    /// `None` for a model of another kind.
    pub fn resident_path_bytes(&self) -> Option<u64> {
        let mut bytes = self.wide_sparse.as_ref()?.head.bytes();
        for layer in &self.layers {
            let block = layer.sinkhorn.as_ref()?;
            bytes += match &layer.mixer {
                mixer::Mixer::LatentMqa(m) => m.resident_bytes(),
                mixer::Mixer::CompressedBand(b) => b.resident_bytes(),
                _ => return None,
            };
            let ffn = &block.ffn;
            let shared = ffn
                .shared
                .as_ref()
                .map_or(0, |s| s.gate.bytes() + s.up.bytes() + s.down.bytes());
            let bias = ffn.bias.len() * std::mem::size_of::<f32>();
            let engram = block.engram.as_ref().map_or(0, |e| e.block.wkv_bytes());
            bytes += block.output.bytes() + ffn.router.bytes() + bias + shared + engram;
        }
        Some(bytes as u64)
    }

    /// The routed feed-forward of every layer that has one, by layer index.
    fn routed_ffns(&mut self) -> impl Iterator<Item = (usize, &mut RoutedFfn)> {
        self.layers
            .iter_mut()
            .enumerate()
            .filter_map(|(il, l)| l.sinkhorn.as_mut().map(|b| (il, &mut b.ffn)))
    }

    /// Routed experts a token passes through beside the shared one, `0` without a routed
    /// feed-forward.
    pub fn routed_per_token(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| l.sinkhorn.as_ref())
            .map_or(0, |b| b.ffn.used)
    }

    /// Sets every routed feed-forward's truncation.
    pub fn set_truncation(&mut self, t: Truncation) {
        for (_, ffn) in self.routed_ffns() {
            ffn.lossy.truncation = t;
        }
    }

    /// The slots the truncation saved since last asked, summed over the layers.
    pub fn take_saved(&self) -> u64 {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| b.ffn.lossy.saved.swap(0, Ordering::Relaxed))
            .sum()
    }

    /// Counts, per layer index, the given experts as the ones in memory.
    ///
    /// `None`, or a layer past the end of `per_layer`, counts none.
    pub fn set_resident_override(&mut self, per_layer: Option<&[HashSet<usize>]>) {
        for (il, ffn) in self.routed_ffns() {
            ffn.lossy.resident = per_layer.and_then(|p| p.get(il).cloned());
        }
    }

    /// The band cell the layers of a Sinkhorn wide residual share.
    fn shared_cell(&self) -> Result<Arc<BandCell>> {
        match self.wide_sparse.as_ref() {
            Some(ws) => Ok(Arc::clone(&ws.cell)),
            None => crate::tensor::bail!("a batch of sequences needs a Sinkhorn wide residual"),
        }
    }

    /// What a sequence keeps before its first token: no state in any layer.
    ///
    /// Refused for a model that is not a Sinkhorn wide residual over latent and band layers.
    pub fn sequence_mixers(&self) -> Result<SequenceMixers> {
        self.shared_cell()?;
        let states = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                mixer::Mixer::LatentMqa(_) => Ok(mixer::MixerState::LatentMqa(None)),
                mixer::Mixer::CompressedBand(_) => Ok(mixer::MixerState::CompressedBand(None)),
                _ => Err(crate::tensor::Error::msg(
                    "a batch of sequences keeps latent and band states only",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SequenceMixers {
            states,
            cell: BandCell::default(),
            tokens: TokenLog::default(),
        })
    }

    /// The token log the engram layers share, `None` for a model without one.
    fn shared_tokens(&self) -> Option<Arc<TokenCarry>> {
        self.wide_sparse.as_ref().and_then(|ws| ws.tokens.clone())
    }

    /// Exchanges every layer's mixer state with the one `seq` holds for that layer.
    fn exchange_states(&mut self, seq: &mut SequenceMixers) -> Result<()> {
        if seq.states.len() != self.layers.len() {
            crate::tensor::bail!(
                "a sequence keeps {} layer states for {} layers",
                seq.states.len(),
                self.layers.len()
            );
        }
        for (layer, held) in self.layers.iter_mut().zip(seq.states.iter_mut()) {
            exchange_state(&mut layer.mixer, held)?;
        }
        Ok(())
    }

    /// The prompt `tokens` of the sequence `seq` from position 0: the logits of its last row,
    /// `[1, vocab]`.
    ///
    /// The model runs its serial forward over the sequence's own states.
    /// It holds its own states again once the call returns.
    pub fn prefill_sequence(&mut self, tokens: &[u32], seq: &mut SequenceMixers) -> Result<Tensor> {
        let cell = self.shared_cell()?;
        let log = self.shared_tokens();
        self.exchange_states(seq)?;
        cell.exchange(&seq.cell);
        if let Some(log) = &log {
            log.exchange(&mut seq.tokens);
        }
        let logits = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu)
            .and_then(|ids| self.forward(&ids, 0));
        if let Some(log) = &log {
            log.exchange(&mut seq.tokens);
        }
        cell.exchange(&seq.cell);
        self.exchange_states(seq)?;
        logits
    }

    /// 1 decode step of each sequence of `seqs`: token `tokens[j]` at `positions[j]` of
    /// sequence `j`, the logits of every row, `[seqs, vocab]`.
    ///
    /// The layers run in order, each over every row before the next.
    /// The token mixer runs per sequence, over the sequence's own mixer state and band cell.
    /// Each row keeps its own residual and the pre mix its next sublayer collapses with.
    /// Every step of a row runs the operations of the serial decode of its token, on the same
    /// values, so each row gives the bits of its sequence decoded alone.
    pub fn decode_sequences(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        seqs: &mut [&mut SequenceMixers],
    ) -> Result<Tensor> {
        let rows_in = tokens.len();
        if positions.len() != rows_in || seqs.len() != rows_in {
            crate::tensor::bail!(
                "{rows_in} tokens for {} positions and {} sequences",
                positions.len(),
                seqs.len()
            );
        }
        let (streams, carry) = match self.wide_sparse.as_ref() {
            Some(ws) => (ws.streams, Arc::clone(&ws.carry)),
            None => crate::tensor::bail!("a batch of sequences needs a Sinkhorn wide residual"),
        };
        let cell = self.shared_cell()?;
        let hashed = self.shared_tokens().is_some();
        if seqs.iter().any(|s| s.states.len() != self.layers.len()) {
            crate::tensor::bail!("a sequence keeps states for another model");
        }
        crate::tensor::quant_cpu::set_small_model_prefill(false);

        // Each row's residual as the embedding stage of its decode starts it.
        // A sequence's own token log takes the row's token where an engram layer reads it.
        let emb_device = self.embeddings.embeddings().device();
        let mut rows = Vec::with_capacity(rows_in);
        for (seq, (&t, &pos)) in seqs.iter_mut().zip(tokens.iter().zip(positions)) {
            if hashed {
                seq.tokens.open(pos, &[t])?;
            }
            let ids = Tensor::from_vec(vec![t], (1, 1), &emb_device)?;
            let hidden = self.add_positions(self.embeddings.forward(&ids)?, pos)?;
            let hidden = match self.config.embed_scale {
                Some(scale) => (hidden * scale)?,
                None => hidden,
            };
            rows.push(super::wide::widen_embedding(&hidden, streams)?);
            seq.cell.open(pos, 1);
        }
        let mut pres: Vec<Vec<Vec<f32>>> =
            (0..rows_in).map(|_| vec![onto_first(streams)]).collect();

        for il in 0..self.layers.len() {
            let layer = &mut self.layers[il];
            let mut mids = Vec::with_capacity(rows_in);
            let mut attn_pres = Vec::with_capacity(rows_in);
            for (j, seq) in seqs.iter_mut().enumerate() {
                let r = rows[j]
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let block = layer.sinkhorn.as_ref().ok_or_else(|| {
                    crate::tensor::Error::msg("a Sinkhorn layer without its block")
                })?;
                let r = match block.engram.as_ref() {
                    Some(e) => e.add(&r, &seq.tokens, positions[j])?,
                    None => r,
                };
                exchange_state(&mut layer.mixer, &mut seq.states[il])?;
                cell.exchange(&seq.cell);
                let half = attention_half(block, &mut layer.mixer, &r, &pres[j], positions[j]);
                cell.exchange(&seq.cell);
                exchange_state(&mut layer.mixer, &mut seq.states[il])?;
                let (mid, pre, _) = half?;
                mids.push(mid);
                attn_pres.push(pre);
            }
            let block = layer
                .sinkhorn
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
            let each = rows
                .iter_mut()
                .zip(pres.iter_mut())
                .zip(mids.iter().zip(&attn_pres));
            for ((row, pre), (mid, attn_pre)) in each {
                let mixes = block.ffn_mix.mixes(mid)?;
                let input = rms_norm(
                    &block.ffn_mix.pre(mid, attn_pre)?,
                    &block.ffn_norm,
                    block.eps,
                )?;
                let out = block.ffn.forward(&input)?;
                *row = block.ffn_mix.post(&out, mid, &mixes.post, &mixes.comb)?;
                *pre = mixes.pre;
            }
        }

        // Each row's head collapses with the pre mix its last layer left.
        let mut logits = Vec::with_capacity(rows_in);
        for (j, row) in rows.into_iter().enumerate() {
            carry.put(positions[j], 1, std::mem::take(&mut pres[j]));
            logits.push(self.output_stage(row, false, 1, false)?);
        }
        Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0)
    }
}

/// What one sequence of a batch keeps between its steps on a Sinkhorn wide-residual model.
///
/// The model's layers hold 1 sequence's state at a time.
/// A batch brings each sequence's state into its layer for that sequence's rows.
pub struct SequenceMixers {
    /// Each layer's mixer state, in layer order.
    states: Vec<mixer::MixerState>,
    /// What the sequence's band sources leave for its band readers within a step.
    cell: BandCell,
    /// The sequence's token ids, which its engram layers hash.
    tokens: TokenLog,
}

/// Exchanges the state `mixer` holds with `held`.
///
/// Refused, with nothing exchanged, when `held` is not a state of `mixer`'s kind.
fn exchange_state(mixer: &mut mixer::Mixer, held: &mut mixer::MixerState) -> Result<()> {
    match (mixer, held) {
        (mixer::Mixer::LatentMqa(m), mixer::MixerState::LatentMqa(s)) => {
            std::mem::swap(&mut m.state, s)
        }
        (mixer::Mixer::CompressedBand(m), mixer::MixerState::CompressedBand(s)) => {
            std::mem::swap(&mut m.state, s)
        }
        _ => crate::tensor::bail!("a sequence's state is not of its layer's mixer kind"),
    }
    Ok(())
}
