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

use super::block_scaled_experts::BlockScaledExpert;
use super::compressed_band::BandCell;
use super::hyper::sinkhorn::SinkhornMixer;
use super::moe_weights::sqrt_softplus_choice;
use super::*;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use rayon::prelude::*;

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
        let onto_first = || {
            let mut row = vec![0f32; streams];
            if let Some(w) = row.first_mut() {
                *w = 1.0;
            }
            row
        };
        let pre = (0..rows).map(|_| onto_first()).collect();
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
}

impl RoutedFfn {
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
        let picks: Vec<(Vec<usize>, Vec<f32>)> = logits
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
}

impl WideSparseHead {
    /// The residual the rows of `hidden` `[1, seq, hidden]` at `index_pos..` start from: 1 copy
    /// of each row per stream.
    ///
    /// The carry and the band cell open for the forward over those rows.
    pub fn widen(&self, hidden: &Tensor, index_pos: usize) -> Result<Tensor> {
        let seq = hidden.dim(1)?;
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
