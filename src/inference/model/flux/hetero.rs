//! The FLUX transformer: the network a rectified-flow sampler asks, at each step, which way the
//! latent should move - wherever its blocks were placed.
//!
//! It carries the image and the text as two streams, each with its own weights, through
//! double-stream blocks that let them attend to one another; then it concatenates the two into
//! one sequence and carries that through single-stream blocks, which fuse attention and the
//! feed-forward into a single wide projection. Every block is modulated by one vector built from
//! the denoise step, the pooled text embedding and - where the checkpoint expects to be told one
//! - the guidance scale. Position enters as a rotation rather than an embedding, over three
//! axes: which image, which row, which column. The pieces all of that is built from are
//! [`super::common`].
//!
//! There is ONE set of blocks and two PLACEMENTS. [`HeteroFlux::whole`] puts the whole model on
//! one device; [`HeteroFlux::from_gguf`] splits the 19 double and 38 single blocks across CUDA,
//! OpenCL (Arc) and CPU so a model that does not fit one card still runs. The blocks compute the
//! same arithmetic either way - what the placement decides is how wide the intermediates may be
//! ([`Intermediates`]) and, when the blocks genuinely span devices, where the activation crosses.
//!
//! Across a denoise loop, four things do not change: the rotary table, which depends only on the
//! two id shapes, and the projections of the text, the pooled vector and the guidance scale,
//! which depend only on inputs the loop holds fixed. All four are cached. An entry keyed by a
//! tensor is keyed by the ADDRESS of its storage, so it also retains a clone of that tensor - a
//! freed one could otherwise hand its address to the next allocation and be read as a hit - and
//! compares shapes, since a reshaped view shares the storage it came from.
//!
//! The split placement adds: a single GGUF read to the host and one per card the plan uses;
//! pre-computed segment boundaries, so the forward does not test a device per block; and a
//! per-device conditioning cache, so `vec_` and `pe` are transferred once per denoise step
//! rather than once per block.

use crate::inference::model::flux::common;
use crate::inference::model::flux::common::{linear_b, FluxAdapters};
use crate::tensor::layer::qlinear::QLinear;
use crate::tensor::layer::LayerNorm;
use crate::tensor::quantized::QVarBuilder as VarBuilder;
use crate::tensor::{DType, Device, IndexOp, Module, Result, Tensor, D};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
use crate::inference::serve::pipeline::HeteroDevice;

#[cfg(feature = "opencl")]
use crate::inference::kernel::opencl::OpenCLPipelines;
#[cfg(feature = "opencl")]
use crate::inference::model::flux::opencl::{
    OclFluxScratch, OpenCLFluxDoubleBlock, OpenCLFluxSingleBlock,
};
#[cfg(feature = "opencl")]
use std::sync::Arc;

// --- What the placement decides ---
//
// The projections, the modulations and the head split are the family's shared ones. What
// this file holds of its own is everything a placement forces: crossing a device boundary,
// and how wide an intermediate may be.

/// Move a tensor to `next`, SYNCHRONISING both devices around the crossing.
///
/// A device-to-device copy is queued on a stream. Without draining the producing
/// device first, the copy can be issued before the kernels that wrote the source have
/// finished, and the destination reads memory nothing has written yet - which comes
/// out as a uniformly black image, with no error anywhere. It cost nothing while the
/// whole model sat on one card, because then the transfer was a no-op; the moment the
/// blocks genuinely spanned two cards it produced a 200 response carrying an empty
/// picture.
fn cross_to(t: &Tensor, from: &Device, next: &Device) -> Result<Tensor> {
    if from.same_device(next) {
        return Ok(t.clone());
    }
    from.synchronize()?;
    let moved = t.to_device(next)?;
    next.synchronize()?;
    Ok(moved)
}

/// How a block sizes the intermediates it materialises.
///
/// This is the ONE thing the two placements do not agree on. A block that owns its device knows
/// before the render starts how much room it has - nothing else competes for what is left - so
/// its widths are constants that were measured once. A block sharing a card with the rest of a
/// split model cannot state a constant: which card it landed on, and how much of that card is
/// left beside it, are decided by the plan rather than known here, so every width is derived
/// from the bytes actually free, per call.
///
/// Everything else - which norm, which modulation, what order the residuals are added in, the
/// arithmetic of every op - is the same either way, which is why one set of blocks serves both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intermediates {
    /// The block owns its device: the projections run whole, and attention tiles on a fixed
    /// width above a fixed threshold.
    Whole,
    /// The block shares its card: the projections run in slabs, and attention tiles on a width
    /// derived from a byte budget.
    Budgeted,
}

/// Attention with a FIXED tile above a fixed threshold - the policy a whole-device placement
/// can afford.
///
/// This side knows what it is running on before the render starts: one device, holding the
/// entire transformer, with no other block competing for what is left. The only thing whose
/// size is still in question is the score matrix, and a constant answers it - 512 query rows
/// once the sequence passes 2048 positions, which are the numbers measured to cap the peak
/// without costing anything at the lengths a whole-device placement reaches. Below the
/// threshold the single-shot path runs untouched, because there the tiling saves nothing.
///
/// A split placement cannot state its tile as a constant and derives a slab from a byte budget
/// instead, in [`sdpa_byte_budget`]. The two are deliberately not merged: a constant chosen
/// from a measurement and a width computed from free memory are different policies, and folding
/// them together would mean one placement silently adopting the other's.
fn sdpa_fixed_tile(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // MEASURED: running these two GEMMs in BF16 on tensor cores makes THIS engine SLOWER,
    // and by a lot - a 1024-square render went 26 s -> 100 s. The same switch is worth 1.96x
    // on the video DiT and 1.42x on Z-Image, so it is not the idea that is wrong, it is that
    // it does not transfer: those run one long sequence, this one runs many short tiles
    // where the per-call casts and the narrower kernels cost more than the arithmetic they
    // save. Boogu measured the same way, 17 s -> 57 s. Do not re-apply here without an A/B.
    let dim = q.dim(D::Minus1)?;
    let scale_factor = (1.0 / (dim as f64).sqrt()) as f32;
    let kt = k.transpose(D::Minus2, D::Minus1)?;
    let seq = q.dim(D::Minus2)?;
    // Query-tiling: the full [.., seq, seq] score matrix is the peak
    // activation and OOMs at 1024² (seq≈4608 -> the scores
    // tensor alone is ~1.6 GB). Each query's softmax is independent, so
    // processing queries in tiles caps the peak at [.., TILE, seq]  -
    // BIT-EXACT (identical per-query math). Only kicks in for large images;
    // <=512² (seq≲1300) runs the single-shot path unchanged.
    const TILE: usize = 512;
    if seq > 2048 {
        sdpa_query_tiled(q, &kt, v, scale_factor, TILE)
    } else {
        let attn_weights = q.matmul(&kt)?.scale(scale_factor)?;
        attn_weights.softmax_last_dim()?.matmul(v)
    }
}

/// Tiled-query SDPA body: `softmax(q.kt.scale).v` computed per query slab of
/// `tile` rows. Separate fn so the tiling can be unit-tested against the
/// single-shot path without a >2048-token input.
fn sdpa_query_tiled(
    q: &Tensor,
    kt: &Tensor,
    v: &Tensor,
    scale_factor: f32,
    tile: usize,
) -> Result<Tensor> {
    let seq = q.dim(D::Minus2)?;
    let mut outs = Vec::with_capacity(seq.div_ceil(tile));
    let mut off = 0;
    while off < seq {
        let t = (seq - off).min(tile);
        let qt = q.narrow(D::Minus2, off, t)?;
        let aw = qt.matmul(kt)?.scale(scale_factor)?;
        outs.push(aw.softmax_last_dim()?.matmul(v)?);
        off += t;
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    Tensor::cat(&refs, D::Minus2)
}

/// Attention with a slab DERIVED FROM A BYTE BUDGET - the policy a split placement is forced
/// into.
///
/// At 1536^2 the sequence is ~9.5k tokens, so the full `[.., seq, seq]` score matrix is
/// 8.6 GB in ONE allocation - it fails with 7 GB free on the card, which no amount of
/// re-planning or spilling can fix, because the problem is the size of a single tensor and
/// not how much of the model sits beside it. Processing queries in slabs caps the peak at
/// `[.., step, seq]`; each query's softmax depends only on that query's row, so the slabbing
/// is BIT-EXACT against the single-shot path below it - the arithmetic per output row is
/// identical, only the batching changes.
///
/// What this side cannot do is state `step` as a constant. A block here shares its card with
/// whatever else the plan put there, and how much is left beside it is not a property of the
/// sequence length - so the width is computed from the bytes actually available, per call.
/// The whole-device placement has no such question to answer and uses a fixed tile above a
/// fixed threshold, both measured, in [`sdpa_fixed_tile`]. Merging the two would mean one
/// placement adopting a width the other's circumstances chose.
fn sdpa_byte_budget(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale_factor = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let rank = q.rank();
    // Batch and heads are one axis for the duration: every head attends on its own, and one
    // leading axis is what the slab loop narrows behind.
    let mut out_dims: Vec<usize> = q.dims()[..rank - 2].to_vec();
    let fold = |t: &Tensor| t.flatten_to(rank - 3);
    let (q, k, v) = (fold(q)?, fold(k)?, fold(v)?);
    let seq = q.dim(D::Minus2)?;
    // A score row is `seq` wide per head, so the slab derives from that - not from a
    // token count, which would mean something different at every sequence length.
    let heads = q.dims().first().copied().unwrap_or(1);
    let step = slab_tokens(seq * heads, ACT_ELEM, seq);
    let attn_scores = if step < seq {
        let mut outs = Vec::with_capacity(seq.div_ceil(step));
        let mut off = 0usize;
        while off < seq {
            let t = (seq - off).min(step);
            let qt = q.narrow(D::Minus2, off, t)?;
            let aw = (qt.matmul_t(&k)? * scale_factor)?;
            outs.push(crate::tensor::ops::softmax_last_dim(&aw)?.matmul(&v)?);
            off += t;
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, D::Minus2)?
    } else {
        let attn_weights = (q.matmul_t(&k)? * scale_factor)?;
        crate::tensor::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?
    };
    // Back to the shape it arrived in: the folded axis unfolds, the two attention axes stay.
    out_dims.extend_from_slice(&attn_scores.dims()[attn_scores.rank() - 2..]);
    attn_scores.reshape(out_dims)
}

fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pe: &Tensor,
    sizing: Intermediates,
) -> Result<Tensor> {
    let q = super::common::apply_rope(q, pe)?.contiguous()?;
    let k = super::common::apply_rope(k, pe)?.contiguous()?;
    let x = match sizing {
        Intermediates::Whole => sdpa_fixed_tile(&q, &k, v)?,
        Intermediates::Budgeted => sdpa_byte_budget(&q, &k, v)?,
    };
    x.transpose(1, 2)?.flatten_from(2)
}

pub(crate) use super::common::timestep_embedding;

use super::common::layer_norm;

// ------------------------------------------------------------
// Adapters on the SPLIT model.
//
// `QLinear::apply_lora` above has always been able to take a delta; nothing ever
// called it. The projection knew how, the blocks and the model did not, so the
// engine had one adapter-capable variant - the single-device one - and answered
// every request on a split model with "this checkpoint cannot take adapters".
// A model that spans two cards because it does not fit on one is the NORMAL case
// on a multi-GPU host, so that message was most of the LoRA failures.
//
// The key scheme itself lives in `native_lora`, shared with the facade transformer:
// two hand-copies of it diverge silently, and the symptom is an adapter that matches
// on one path and nothing on the other.
// ------------------------------------------------------------

impl HeteroDoubleBlock {
    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
        i: usize,
    ) -> Result<usize> {
        let k = crate::inference::load::lora::double_block_keys(i);
        let mut n = 0;
        for (proj, key) in [
            (&mut self.img_attn.qkv, &k[0]),
            (&mut self.img_attn.proj, &k[1]),
            (&mut self.txt_attn.qkv, &k[2]),
            (&mut self.txt_attn.proj, &k[3]),
        ] {
            n += proj.apply_flux_lora(file, strength, &key.path, Some(&key.alt))?;
        }
        Ok(n)
    }

    fn clear_lora(&mut self) {
        for proj in [
            &mut self.img_attn.qkv,
            &mut self.img_attn.proj,
            &mut self.txt_attn.qkv,
            &mut self.txt_attn.proj,
        ] {
            proj.clear_lora();
        }
    }
}

impl HeteroSingleBlock {
    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
        i: usize,
    ) -> Result<usize> {
        let k = crate::inference::load::lora::single_block_keys(i);
        Ok(self
            .linear1
            .apply_flux_lora(file, strength, &k[0].path, Some(&k[0].alt))?
            + self
                .linear2
                .apply_flux_lora(file, strength, &k[1].path, Some(&k[1].alt))?)
    }

    fn clear_lora(&mut self) {
        self.linear1.clear_lora();
        self.linear2.clear_lora();
    }
}

impl HeteroFlux {
    /// Attach an adapter to every projection in the split DiT that it names.
    ///
    /// Blocks are numbered within their own kind, as the adapter files key them:
    /// `self.layers` runs the double blocks then the single ones, so the single
    /// index restarts at zero. Numbering them by position in `layers` would look
    /// right and match nothing past the first block.
    pub fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        let (mut di, mut si) = (0usize, 0usize);
        for layer in self.layers.iter_mut() {
            match &mut layer.block {
                HeteroFluxBlock::Double(b) => {
                    n += b.apply_lora(file, strength, di)?;
                    di += 1;
                }
                HeteroFluxBlock::Single(b) => {
                    n += b.apply_lora(file, strength, si)?;
                    si += 1;
                }
                // A block on a substrate whose projections cannot carry a delta would
                // render its own share of the steps UNADAPTED, which is not a partial
                // effect but a different model at every layer boundary. Say so instead.
                #[cfg(feature = "opencl")]
                HeteroFluxBlock::OclDouble(_) | HeteroFluxBlock::OclSingle(_) => {
                    return Err(crate::tensor::Error::msg(
                        "this model has blocks placed on an OpenCL device, which cannot \
                         carry adapters - unload it to re-place the blocks on CUDA/CPU"
                            .to_string(),
                    ));
                }
            }
        }
        self.adapters_attached += 1;
        Ok(n)
    }

    /// Drop every attached adapter, returning the split DiT to its checkpoint.
    pub fn clear_lora(&mut self) {
        for layer in self.layers.iter_mut() {
            match &mut layer.block {
                HeteroFluxBlock::Double(b) => b.clear_lora(),
                HeteroFluxBlock::Single(b) => b.clear_lora(),
                #[cfg(feature = "opencl")]
                HeteroFluxBlock::OclDouble(_) | HeteroFluxBlock::OclSingle(_) => {}
            }
        }
        self.adapters_attached = 0;
    }

    /// Arm repatriation for a render, with the headroom its activations need on a card.
    /// Pass 0 when the render ends, so a later free card cannot pull blocks off the host
    /// against a demand that no longer describes anything.
    pub fn set_render_headroom(&mut self, bytes: u64) {
        self.render_headroom = bytes;
        // A new render starts asking straight away: the last one's refusal says nothing
        // about the memory this one will find.
        self.repatriate_backoff = 0;
    }

    /// How many blocks are still running on the host.
    pub fn cpu_block_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.device == HeteroDevice::Cpu)
            .count()
    }

    /// Move ONE host-resident block back onto a card, if one now has room for it.
    ///
    /// A render placed under memory pressure spills blocks to the host, and they stay
    /// there for every remaining step even once the pressure lifts - the placement is
    /// decided at load and never revisited, so a model that had to share the cards with
    /// a peak that has since ended keeps paying host speed for the rest of the run.
    /// Called between denoise steps, this returns the model to the cards as they free.
    ///
    /// One block per call, so the cost is bounded and the next step re-measures instead
    /// of committing to a stale reading of free VRAM. `Ok(None)` means nothing moved -
    /// no host block, no room, or the upload failed - and is the normal quiet case.
    pub fn try_repatriate_block(&mut self) -> anyhow::Result<Option<(usize, usize)>> {
        if self.adapters_attached > 0 {
            return Ok(None);
        }
        let Some(pos) = self
            .layers
            .iter()
            .position(|l| l.device == HeteroDevice::Cpu)
        else {
            return Ok(None);
        };
        if self.bytes_per_block == 0 || self.render_headroom == 0 {
            return Ok(None);
        }
        if self.repatriate_backoff > 0 {
            self.repatriate_backoff -= 1;
            return Ok(None);
        }
        // The block's own bytes plus the room the render still needs, asked of the same
        // probe every placer here uses: fastest card first, minus what other subsystems
        // have declared they are about to allocate. Between two steps the activations are
        // at a low point, so free VRAM here overstates what the next step will leave -
        // which is exactly what the headroom covers.
        let need = self.bytes_per_block.saturating_add(self.render_headroom);
        let target = crate::inference::place::device_probe::probe_cuda_gpus_for("media", 1.0)
            .into_iter()
            .find(|g| self.cuda_devices.contains_key(&g.index) && g.stable_free > need);
        let Some(target) = target else {
            self.repatriate_backoff = REPATRIATE_BACKOFF_STEPS;
            return Ok(None);
        };
        let dev = self.cuda_devices[&target.index].clone();

        let is_double: Vec<bool> = self
            .layers
            .iter()
            .map(|l| matches!(l.block, HeteroFluxBlock::Double(_)))
            .collect();
        let name = checkpoint_block_name(&is_double, pos);
        let vb = VarBuilder::from_gguf(&self.checkpoint, &dev)
            .map_err(|e| anyhow::anyhow!("repatriate: VarBuilder on CUDA:{}: {e}", target.index))?;
        let rebuilt = match &self.layers[pos].block {
            HeteroFluxBlock::Double(_) => {
                HeteroDoubleBlock::new(&self.cfg, vb.pp(name)).map(HeteroFluxBlock::Double)
            }
            HeteroFluxBlock::Single(_) => {
                HeteroSingleBlock::new(&self.cfg, vb.pp(name)).map(HeteroFluxBlock::Single)
            }
            // A block on a substrate this path cannot rebuild stays where it is.
            #[cfg(feature = "opencl")]
            _ => return Ok(None),
        };
        // An upload that runs out is not an error here: the card filled between the
        // probe and the copy, and the block simply stays on the host.
        let Ok(rebuilt) = rebuilt else {
            // The card filled between the probe and the copy. Wait before asking again
            // rather than retrying the same allocation on the next step.
            self.repatriate_backoff = REPATRIATE_BACKOFF_STEPS;
            return Ok(None);
        };

        self.layers[pos] = HeteroFluxLayer {
            device: HeteroDevice::Cuda(target.index),
            block: rebuilt,
        };
        let (segments, mut unique) = build_segments(&self.layers, &self.cuda_devices);
        if !unique
            .iter()
            .any(|d| device_key(d) == device_key(&self.primary_device))
        {
            unique.push(self.primary_device.clone());
        }
        self.segments = segments;
        self.unique_devices = unique;
        // The per-device conditioning transfers are keyed by `pe.id()`, which is constant
        // for a whole render - so a map built before this block moved would be reused with
        // no entry for its new card, and the next step would fail looking one up. The
        // device set changed, so the map has to be rebuilt.
        *self
            .pe_per_device_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        info!(
            "repatriated block {pos} onto CUDA:{} ({} still on the host)",
            target.index,
            self.cpu_block_count()
        );
        Ok(Some((pos, target.index)))
    }
}

impl Module for common::MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

/// The largest intermediate a slabbed operation may materialise, in BYTES.
///
/// Set from the card at load, because a slab size stated in TOKENS is a claim about
/// one machine and one model: a model twice as wide doubles the buffer at the same
/// token count, and what a workstation shrugs at is fatal on a laptop. Expressing the
/// cap in bytes and deriving the token count from it makes the same code correct on a
/// 6 GB card and on a 48 GB one, at any width, dtype and resolution.
///
/// Zero means "not set yet" - the operations then run unslabbed, which is what they
/// did before any of this existed.
static SLAB_BUDGET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One sixteenth of what the card has free.
///
/// The FRACTION is the portable part; the byte count it produces is not. A sixteenth
/// leaves the slab small enough that a dozen of them in flight are still a minor share
/// of the card, and large enough that the loop is not launch-bound.
const SLAB_SHARE: u64 = 16;

/// Set the slab budget from the device the blocks were placed on.
///
/// Read once per load rather than per block: probing free VRAM costs a driver call,
/// and a denoise runs every block on every step - thousands of probes per render.
fn set_slab_budget(dev: &Device) {
    let free = match dev.location() {
        crate::tensor::DeviceLocation::Cuda { gpu_id } => {
            crate::inference::place::vram_manager::probe(0)
                .into_iter()
                .find(|(i, _, _)| *i == gpu_id)
                .map(|(_, free, _)| free)
                .unwrap_or(0)
        }
        // On the host the constraint is different in kind, and the allocator does not
        // behave the way this is guarding against.
        _ => 0,
    };
    SLAB_BUDGET_BYTES.store(free / SLAB_SHARE, std::sync::atomic::Ordering::Relaxed);
    if free > 0 {
        info!(
            "HeteroFlux: slabbing intermediates at {:.0} MB ({:.1} GB free / {SLAB_SHARE})",
            (free / SLAB_SHARE) as f64 / 1e6,
            free as f64 / 1e9
        );
    }
}

/// How many token rows fit in the budget for a tensor `width` elements wide.
///
/// Returns `total` when the whole thing fits, so the caller runs its single-shot path
/// and nothing changes for the small cases.
fn slab_tokens(width: usize, elem_bytes: u64, total: usize) -> usize {
    let budget = SLAB_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    if budget == 0 || width == 0 {
        return total;
    }
    let per_row = width as u64 * elem_bytes;
    if per_row == 0 {
        return total;
    }
    // At least one row: a budget below a single row still has to make progress.
    ((budget / per_row).max(1) as usize).min(total)
}

/// Bytes per element of the activations these blocks compute in.
const ACT_ELEM: u64 = 4;

/// The feed-forward run whole: a block that owns its device has no memory budget to answer
/// to, so the widening projection, the activation and the narrowing one are one call.
///
/// This is the step [`common::Stream::residuals`] asks its caller for; a block sharing a card
/// answers it with [`mlp_slabbed`] instead.
fn mlp_whole(mlp: &common::Mlp, xs: &Tensor) -> Result<Tensor> {
    mlp.forward(xs)
}

/// The feed-forward, computed in slabs of tokens.
///
/// This is the one thing a split block does differently from a whole one: it has a memory
/// budget to answer to, and the intermediate is the widest thing it holds. The projection
/// itself is the shared one.
fn mlp_slabbed(mlp: &common::Mlp, xs: &Tensor) -> Result<Tensor> {
    {
        let tokens = xs.dim(D::Minus2)?;
        let step = slab_tokens(mlp.hidden(), ACT_ELEM, tokens);
        if step >= tokens {
            return xs.apply(mlp.fc1())?.gelu()?.apply(mlp.fc2());
        }
        let mut outs = Vec::with_capacity(tokens.div_ceil(step));
        let mut off = 0usize;
        while off < tokens {
            let t = (tokens - off).min(step);
            let slab = xs.narrow(D::Minus2, off, t)?;
            outs.push(slab.apply(mlp.fc1())?.gelu()?.apply(mlp.fc2())?);
            off += t;
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, D::Minus2)
    }
}

// --- Block structs ---

#[derive(Debug)]
pub struct HeteroDoubleBlock {
    pub img_mod: common::Modulation2,
    pub img_norm1: LayerNorm,
    pub img_attn: common::SelfAttention,
    pub img_norm2: LayerNorm,
    pub img_mlp: common::Mlp,
    pub txt_mod: common::Modulation2,
    pub txt_norm1: LayerNorm,
    pub txt_attn: common::SelfAttention,
    pub txt_norm2: LayerNorm,
    pub txt_mlp: common::Mlp,
}

impl HeteroDoubleBlock {
    fn new(cfg: &crate::inference::model::flux::common::Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let heads = cfg.num_heads;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        // Field by field, in the order the tensors are read: a struct literal evaluates as it
        // is written, and the plan's VRAM ledger was measured against this order.
        Ok(Self {
            img_mod: common::Modulation2::new(h_sz, &vb.pp("img_mod"))?,
            img_norm1: layer_norm(h_sz, vb.device())?,
            img_attn: common::SelfAttention::new(h_sz, heads, cfg.qkv_bias, &vb.pp("img_attn"))?,
            img_norm2: layer_norm(h_sz, vb.device())?,
            img_mlp: common::mlp(h_sz, mlp_sz, &vb.pp("img_mlp"))?,
            txt_mod: common::Modulation2::new(h_sz, &vb.pp("txt_mod"))?,
            txt_norm1: layer_norm(h_sz, vb.device())?,
            txt_attn: common::SelfAttention::new(h_sz, heads, cfg.qkv_bias, &vb.pp("txt_attn"))?,
            txt_norm2: layer_norm(h_sz, vb.device())?,
            txt_mlp: common::mlp(h_sz, mlp_sz, &vb.pp("txt_mlp"))?,
        })
    }

    /// The image's share of the block.
    fn image(&self) -> common::Stream<'_> {
        common::Stream {
            norm1: &self.img_norm1,
            attn: &self.img_attn,
            norm2: &self.img_norm2,
            mlp: &self.img_mlp,
        }
    }

    /// The text's share of it.
    fn text(&self) -> common::Stream<'_> {
        common::Stream {
            norm1: &self.txt_norm1,
            attn: &self.txt_attn,
            norm2: &self.txt_norm2,
            mlp: &self.txt_mlp,
        }
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
        sizing: Intermediates,
    ) -> Result<(Tensor, Tensor)> {
        let (image, text) = (self.image(), self.text());
        // Each stream draws its own pair of modulations from the one conditioning vector: one
        // for the attention branch, one for the feed-forward.
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;
        let (img_q, img_k, img_v) = image.qkv(img, &img_mod1)?;
        let (txt_q, txt_k, txt_v) = text.qkv(txt, &txt_mod1)?;

        // This is what makes the block double rather than two blocks: the streams keep their
        // own weights but attend as one sequence, text in front of image, so each can read
        // the other.
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

        let attn = attention(&q, &k, &v, pe, sizing)?;
        // Each stream takes back the rows it put in.
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        // The feed-forward is the widest thing a block holds, so it is what a memory budget
        // has to be answered with - and the only step of the two residuals that the placement
        // gets a say in.
        let feed_forward: fn(&common::Mlp, &Tensor) -> Result<Tensor> = match sizing {
            Intermediates::Whole => mlp_whole,
            Intermediates::Budgeted => mlp_slabbed,
        };
        let img = image.residuals(img, &img_attn_out, &img_mod1, &img_mod2, feed_forward)?;
        let txt = text.residuals(txt, &txt_attn_out, &txt_mod1, &txt_mod2, feed_forward)?;
        Ok((img, txt))
    }
}

#[derive(Debug)]
pub struct HeteroSingleBlock {
    pub linear1: QLinear,
    pub linear2: QLinear,
    pub norm: crate::tensor::layer::QkNorm,
    pub pre_norm: LayerNorm,
    pub modulation: common::Modulation1,
    pub h_sz: usize,
    pub mlp_sz: usize,
    pub num_heads: usize,
}

impl HeteroSingleBlock {
    fn new(cfg: &crate::inference::model::flux::common::Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let heads = cfg.num_heads;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        // In the order the tensors are read - see the double block's note on the ledger.
        Ok(Self {
            linear1: linear_b(h_sz, h_sz * 3 + mlp_sz, true, &vb.pp("linear1"))?,
            linear2: linear_b(h_sz + mlp_sz, h_sz, true, &vb.pp("linear2"))?,
            norm: common::qk_norm(h_sz / heads, &vb.pp("norm"))?,
            pre_norm: layer_norm(h_sz, vb.device())?,
            modulation: common::Modulation1::new(h_sz, &vb.pp("modulation"))?,
            h_sz,
            mlp_sz,
            num_heads: heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
        sizing: Intermediates,
    ) -> Result<Tensor> {
        // One modulation for the block, since it has one branch to gate.
        let m = self.modulation.forward(vec_)?;
        let x_mod = m.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let tokens = x_mod.dim(D::Minus2)?;
        // The fused projection is the widest thing here, so it sets the slab. A block that
        // owns its device answers to no budget and takes the whole sequence in one step.
        let step = match sizing {
            Intermediates::Whole => tokens,
            Intermediates::Budgeted => slab_tokens(3 * self.h_sz + self.mlp_sz, ACT_ELEM, tokens),
        };
        let slabbed = step < tokens;

        // The fused projection is the widest tensor in the graph - three model widths
        // for the attention PLUS the feed-forward's, ~21.5k columns against a 3k-wide
        // model, and there are twice as many of these blocks as double ones.
        //
        // Both halves have to survive the projection: the attention needs every token
        // of q, k and v, and the feed-forward half is consumed after it. So their sum
        // is a floor and slabbing cannot go under it. What slabbing DOES avoid is
        // holding the fused tensor as well - splitting each slab as it is produced
        // means the two halves are the only things accumulated.
        let (qkv, mlp) = if slabbed {
            let mut qkv_parts = Vec::with_capacity(tokens.div_ceil(step));
            let mut mlp_parts = Vec::with_capacity(tokens.div_ceil(step));
            let mut off = 0usize;
            while off < tokens {
                let t = (tokens - off).min(step);
                let projected = x_mod.narrow(D::Minus2, off, t)?.apply(&self.linear1)?;
                qkv_parts.push(projected.narrow(D::Minus1, 0, 3 * self.h_sz)?);
                mlp_parts.push(projected.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?);
                off += t;
            }
            let qr: Vec<&Tensor> = qkv_parts.iter().collect();
            let mr: Vec<&Tensor> = mlp_parts.iter().collect();
            (Tensor::cat(&qr, D::Minus2)?, Tensor::cat(&mr, D::Minus2)?)
        } else {
            let projected = x_mod.apply(&self.linear1)?;
            let qkv = projected.narrow(D::Minus1, 0, 3 * self.h_sz)?;
            let mlp = projected.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
            (qkv, mlp)
        };

        let (q, k, v) = common::heads_qkv(&qkv, self.num_heads, &self.norm)?;
        let attn = attention(&q, &k, &v, pe, sizing)?;

        // Joining the attention output to the activated feed-forward makes a tensor as
        // wide as both, which at this sequence length is another half-gigabyte held
        // only to be consumed by the next projection. Per token slab it is a fraction
        // of that, and the result is the same: `linear2` is row-wise.
        let output = if slabbed {
            let mut outs = Vec::with_capacity(tokens.div_ceil(step));
            let mut off = 0usize;
            while off < tokens {
                let t = (tokens - off).min(step);
                let a = attn.narrow(D::Minus2, off, t)?;
                let m = mlp.narrow(D::Minus2, off, t)?.gelu()?;
                outs.push(Tensor::cat(&[&a, &m], 2)?.apply(&self.linear2)?);
                off += t;
            }
            let refs: Vec<&Tensor> = outs.iter().collect();
            Tensor::cat(&refs, D::Minus2)?
        } else {
            Tensor::cat(&[&attn, &mlp.gelu()?], 2)?.apply(&self.linear2)?
        };
        xs + m.gate(&output)
    }
}

#[derive(Debug)]
pub struct HeteroLastLayer {
    pub norm_final: LayerNorm,
    pub linear: QLinear,
    pub ada_ln_modulation: QLinear,
}

impl HeteroLastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.device())?;
        let linear = linear_b(h_sz, p_sz * p_sz * out_c, true, &vb.pp("linear"))?;
        let ada_ln_modulation = linear_b(h_sz, 2 * h_sz, true, &vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        // Two vectors out of one projection, shift then scale - the same modulation every
        // block applies, with nothing to gate because there is no branch left to gate.
        let chunks = super::common::vec_silu_cached(vec)?
            .apply(&self.ada_ln_modulation)?
            .chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let normed = xs.apply(&self.norm_final)?;
        common::scale_shift(&normed, &scale.unsqueeze(1)?, &shift.unsqueeze(1)?)?
            .apply(&self.linear)
    }
}

// --- HeteroFlux: multi-device Flux pipeline ---

pub enum HeteroFluxBlock {
    Double(HeteroDoubleBlock),
    Single(HeteroSingleBlock),
    #[cfg(feature = "opencl")]
    OclDouble(OpenCLFluxDoubleBlock),
    #[cfg(feature = "opencl")]
    OclSingle(OpenCLFluxSingleBlock),
}

pub struct HeteroFluxLayer {
    pub device: HeteroDevice,
    pub block: HeteroFluxBlock,
}

/// Pre-computed segment boundary for fast forward pass dispatch.
/// Avoids per-block device comparison - just iterate segments.
struct FluxSegment {
    device: HeteroDevice,
    tensor_device: Device,
    block_start: usize, // inclusive (into self.layers)
    block_end: usize,   // exclusive
}

/// How many steps to wait after a refusal before probing again. Small enough that a card
/// freeing mid-render is still noticed within a step or two of a typical schedule.
const REPATRIATE_BACKOFF_STEPS: u32 = 4;

/// The name a block carries in the checkpoint, from its position in `layers`.
///
/// Blocks are numbered WITHIN THEIR OWN KIND, as the files key them: `layers` runs the
/// double blocks then the single ones, so a single block's index restarts at zero.
/// Numbering by position would look right and name the wrong tensors past the first
/// single block - the same trap `apply_lora` documents.
fn checkpoint_block_name(is_double: &[bool], pos: usize) -> String {
    let doubles_before = is_double[..pos].iter().filter(|d| **d).count();
    if is_double[pos] {
        format!("double_blocks.{doubles_before}")
    } else {
        format!("single_blocks.{}", pos - doubles_before)
    }
}

/// Group consecutive same-device blocks into segments, and list the distinct devices.
///
/// Shared by the loader and by `try_repatriate_block`, which changes one block's device
/// mid-render: the segments describe where the activation has to cross a device boundary,
/// so a stale set would keep sending the tensor to the card the block just left.
fn segment_bounds(devices: &[HeteroDevice]) -> Vec<(HeteroDevice, usize, usize)> {
    let mut out = Vec::new();
    let mut seg_start = 0usize;
    while seg_start < devices.len() {
        let seg_device = &devices[seg_start];
        let mut seg_end = seg_start + 1;
        // Compare the device ITSELF, not its discriminant: `Cuda(0)` and `Cuda(1)`
        // share a discriminant, so grouping on that merged two cards' blocks into
        // one segment and ran them all on the first card's tensors.
        while seg_end < devices.len() && devices[seg_end] == *seg_device {
            seg_end += 1;
        }
        out.push((seg_device.clone(), seg_start, seg_end));
        seg_start = seg_end;
    }
    out
}

fn build_segments(
    layers: &[HeteroFluxLayer],
    cuda_devices: &std::collections::HashMap<usize, Device>,
) -> (Vec<FluxSegment>, Vec<Device>) {
    let mut segments = Vec::new();
    let devices: Vec<HeteroDevice> = layers.iter().map(|l| l.device.clone()).collect();
    for (seg_device, seg_start, seg_end) in segment_bounds(&devices) {
        // THE SAME DEVICE OBJECT THE WEIGHTS WERE LOADED WITH.
        //
        // `Device::new_cuda(idx)` names the same physical card but is not the same
        // handle: this path runs on a non-default stream, so a freshly made device
        // carries a different one. Weights uploaded through the loader's handle and
        // activations moved onto a new handle then live on two streams of one card,
        // and a kernel reading both races the copy that filled the second - which
        // comes out as NaN, and NaN through this network is a black image.
        //
        // It went unnoticed while every block sat on one card, because then the
        // move was a no-op and nothing ever crossed. The moment the blocks spanned
        // two cards, every render at that size came back black with no error
        // anywhere.
        let tensor_device = match &seg_device {
            HeteroDevice::Cuda(idx) => cuda_devices
                .get(idx)
                .cloned()
                .or_else(|| Device::new_cuda(*idx).ok())
                .unwrap_or(Device::Cpu),
            HeteroDevice::Cpu => Device::Cpu,
            _ => Device::Cpu,
        };
        segments.push(FluxSegment {
            device: seg_device,
            tensor_device,
            block_start: seg_start,
            block_end: seg_end,
        });
    }
    let mut unique_devices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for seg in &segments {
        if seen.insert(device_key(&seg.tensor_device)) {
            unique_devices.push(seg.tensor_device.clone());
        }
    }
    (segments, unique_devices)
}

/// Cached `pe` tensor with the (txt_ids.dims, img_ids.dims) shape pair
/// that produced it. Stored under a Mutex on `HeteroFlux::pe_cache`;
/// reused across all denoise steps within a single image generation.
type PeCacheEntry = (Vec<usize>, Vec<usize>, Tensor);

pub struct HeteroFlux {
    // Input embeddings (on primary device)
    img_in: QLinear,
    txt_in: QLinear,
    time_in: common::MlpEmbedder,
    vector_in: common::MlpEmbedder,
    guidance_in: Option<common::MlpEmbedder>,
    pe_embedder: crate::inference::model::flux::common::EmbedNd,

    // 57 blocks with per-device assignment
    pub layers: Vec<HeteroFluxLayer>,

    // Pre-computed segment boundaries for fast forward dispatch
    segments: Vec<FluxSegment>,

    // Final layer
    final_layer: HeteroLastLayer,

    // Device state
    pub plan: HeteroPlan,
    primary_device: Device,
    cfg: crate::inference::model::flux::common::Config,

    /// How the blocks size what they materialise - the one thing the two placements decide
    /// differently. Set at load, because it is a property of the placement and not of a step.
    intermediates: Intermediates,

    /// The checkpoint the blocks were read from, and THE device handles they were
    /// uploaded through. Both are what `try_repatriate_block` needs to rebuild a block
    /// on a card: a fresh `Device::new_cuda` carries a different stream (see
    /// `build_segments`), so the handle has to be the loader's own. A whole placement has
    /// no block to bring back and carries no path.
    checkpoint: std::path::PathBuf,
    cuda_devices: std::collections::HashMap<usize, Device>,
    /// Checkpoint bytes divided by the block count - what one block costs to place,
    /// derived from the file rather than assumed. ZERO stands repatriation down, which is
    /// what a whole placement wants: there is nothing on the host to move.
    bytes_per_block: u64,
    /// How many adapters are attached. A rebuilt block comes from the checkpoint, so it
    /// would arrive WITHOUT them and render its share of the steps as a different model.
    /// Repatriation stands down while any is attached rather than produce that silently.
    adapters_attached: usize,
    /// What the render in flight still needs free on a card, beyond the weights. Set by
    /// the engine from the same demand its placement used, because a block uploaded
    /// between two steps has to leave the NEXT step's activations room to allocate.
    /// Zero means no render is running and repatriation stands down.
    render_headroom: u64,
    /// Steps to skip before asking again after a refusal.
    ///
    /// The probe reads NVML and PRINTS a line per card, so asking at every step of a
    /// render that has nowhere to move a block to means two log lines per step and a
    /// driver round trip for an answer that has not changed. Backing off costs at most a
    /// few steps of a card being free before the block moves.
    repatriate_backoff: u32,

    // The distinct devices the segments use (for the conditioning tensor cache)
    unique_devices: Vec<Device>,

    // Cache for `pe` - the position embedding built from concat(txt_ids, img_ids)
    // and applied through pe_embedder. Constant across all denoise steps within
    // a single image generation; keyed by (txt_ids.dims, img_ids.dims) which
    // uniquely identifies its content for a given model config. Avoids
    // recomputing rope() per-axis on every step of `flux::sampling::denoise`.
    pe_cache: std::sync::Mutex<Option<PeCacheEntry>>,

    // Per-device pe transfers, keyed by pe.id(). For multi-GPU runs the
    // pe.to_device(dev) was redone every denoise step even though pe was
    // already cache-stable. Invalidates when pe.id changes (next image).
    pe_per_device_cache:
        std::sync::Mutex<Option<(crate::tensor::TensorId, HashMap<String, Tensor>)>>,

    // What a denoise loop asks for again at every step and gets the same answer to: the
    // projections of the text, the pooled vector and the guidance scale all depend only on
    // inputs the loop holds fixed. Each entry holds its KEY TENSOR beside the answer - an
    // entry keyed by a tensor is keyed by the ADDRESS of its storage, so a freed key could
    // otherwise hand its address to the next allocation and be read as a hit.
    txt_in_cache: std::sync::Mutex<Option<(Tensor, Tensor)>>,
    y_in_cache: std::sync::Mutex<Option<(Tensor, Tensor)>>,
    guidance_in_cache: std::sync::Mutex<Option<(Tensor, Tensor)>>,

    // OpenCL state (only present when Arc GPU segments exist)
    #[cfg(feature = "opencl")]
    pub opencl_pipelines: Option<Arc<OpenCLPipelines>>,
    #[cfg(feature = "opencl")]
    pub opencl_scratch: Option<std::sync::Mutex<OclFluxScratch>>,
}

/// Device key for HashMap caching (cheap to compute)
fn device_key(d: &Device) -> String {
    format!("{:?}", d)
}

/// Identity-keyed single-entry cache lookup.
///
/// A tensor's identity is the ADDRESS of its storage, so the entry RETAINS A CLONE of the key:
/// a freed key could otherwise hand its address to the next allocation and be read as a hit.
/// The shape is compared too, because a reshaped view shares the storage it came from.
fn cached_or(
    slot: &std::sync::Mutex<Option<(Tensor, Tensor)>>,
    key: &Tensor,
    compute: impl FnOnce() -> Result<Tensor>,
) -> Result<Tensor> {
    let mut g = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((ref k, ref t)) = *g {
        if k.storage_ptr_id() == key.storage_ptr_id() && k.dims() == key.dims() {
            return Ok(t.clone());
        }
    }
    let t = compute()?;
    *g = Some((key.clone(), t.clone()));
    Ok(t)
}

impl HeteroFlux {
    /// Load Flux model from GGUF with per-block device assignment.
    ///
    /// Loads GGUF to CPU first, then loads each CUDA-assigned block onto THE CARD THE
    /// PLAN GAVE IT. Input embeddings and final layer go to the primary device.
    ///
    /// `cuda_devices` maps a plan ordinal to its device. It used to be a single
    /// `Option<&Device>`: every block the plan marked `Cuda(anything)` was built from
    /// one VarBuilder on one card and tagged `Cuda(0)`, so a two-card host ran the
    /// model on one card and spilled the rest to the HOST while the other card sat
    /// empty. That is what turned a 1536x1536 render into 27 blocks on the CPU next to
    /// 16 GB of idle VRAM, and then into an OOM anyway.
    pub fn from_gguf(
        gguf_path: &std::path::Path,
        cfg: &crate::inference::model::flux::common::Config,
        plan: &HeteroPlan,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        #[cfg(feature = "opencl")] ocl_pipelines: Option<Arc<OpenCLPipelines>>,
    ) -> anyhow::Result<Self> {
        let has_cuda_segments = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Cuda(_)));
        #[cfg(feature = "opencl")]
        let has_ocl_segments = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::OpenCL(_)));
        let total_blocks = cfg.depth + cfg.depth_single_blocks; // 57

        info!(
            "HeteroFlux: loading from {} ({} blocks)",
            gguf_path.display(),
            total_blocks
        );

        // fp8-scaled safetensors (the Ray fine-tunes) are not GGUF: this builder
        // would fail on the magic ("not a GGUF file"), which is exactly how the
        // CPU-spill fallback died after a pressure eviction dropped the resident
        // model. The fp8 bridge already converts such a checkpoint to a GGUF
        // sidecar and caches it; point this loader at that file.
        let owned_gguf;
        let gguf_path = if gguf_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            owned_gguf = crate::inference::load::fp8_scaled::ensure_sidecar(
                gguf_path,
                crate::tensor::quantized::GgmlDType::Q8_0,
            )
            .map_err(|e| anyhow::anyhow!("fp8 sidecar for hetero load: {e}"))?;
            info!(
                "HeteroFlux: using converted sidecar {}",
                owned_gguf.display()
            );
            owned_gguf.as_path()
        } else {
            gguf_path
        };

        // Load GGUF to CPU (always needed for CPU segments and embeddings)
        let cpu_vb = VarBuilder::from_gguf(gguf_path, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("GGUF load to CPU: {e}"))?;

        // One VarBuilder PER CARD the plan actually uses. Building them lazily from the
        // plan's own ordinals is what makes the placement real: a block assigned to
        // card 1 is uploaded to card 1, not to whichever card happened to be passed in.
        let planned_ordinals: Vec<usize> = {
            let mut v: Vec<usize> = plan
                .segments
                .iter()
                .filter_map(|s| match s.kind {
                    DeviceKind::Cuda(i) => Some(i),
                    _ => None,
                })
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let mut cuda_vbs: std::collections::HashMap<usize, VarBuilder> =
            std::collections::HashMap::new();
        if has_cuda_segments {
            info!(
                "HeteroFlux: loading GGUF to {} CUDA device(s) for GPU-assigned blocks...",
                planned_ordinals.len()
            );
            for ord in &planned_ordinals {
                let Some(dev) = cuda_devices.get(ord) else {
                    continue;
                };
                match VarBuilder::from_gguf(gguf_path, dev) {
                    Ok(vb) => {
                        cuda_vbs.insert(*ord, vb);
                    }
                    // The plan asked for CUDA and the upload failed: the card was taken
                    // between the admission check and this upload (another engine is
                    // running). Silently putting EVERY block on the CPU used to look like
                    // a graceful degrade, but a 12 GB transformer on the host cannot
                    // finish a 1024^2 render inside any request timeout - the observed
                    // outcome was a 600 s hang that produced nothing while chat and audio
                    // requests sailed through. Fail the load instead, so the caller can
                    // report "retry" or re-plan against what is actually free.
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "HeteroFlux: the blocks assigned to CUDA:{ord} no longer fit ({e}); \
                             another engine took the card between planning and upload"
                        ));
                    }
                }
            }
        }

        // For OpenCL segments: parse GGUF content for raw tensor extraction
        #[cfg(feature = "opencl")]
        let gguf_mmap = if has_ocl_segments && ocl_pipelines.is_some() {
            let file = std::fs::File::open(gguf_path)
                .map_err(|e| anyhow::anyhow!("open GGUF for mmap: {e}"))?;
            // Arc, not a bare Mmap: the blocks below parse over these bytes and the
            // parsed Content has to be able to ADOPT the mapping, not just borrow it.
            Some(std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file)? }))
        } else {
            None
        };

        // Primary device: the FIRST card the plan uses, so the embeddings sit where the
        // first blocks are and the stem does not open with a transfer.
        let primary_device = planned_ordinals
            .first()
            .and_then(|o| cuda_devices.get(o))
            .filter(|_| !cuda_vbs.is_empty())
            .cloned()
            .unwrap_or(Device::Cpu);
        info!("HeteroFlux: primary device = {:?}", primary_device);
        // Size the slabs from THIS card, now that we know which one it is.
        set_slab_budget(&primary_device);

        // Embeddings + final layer on primary device
        let emb_vb = planned_ordinals
            .first()
            .and_then(|o| cuda_vbs.get(o))
            .unwrap_or(&cpu_vb);
        let img_in = linear_b(cfg.in_channels, cfg.hidden_size, true, &emb_vb.pp("img_in"))
            .map_err(|e| anyhow::anyhow!("img_in: {e}"))?;
        let txt_in = linear_b(
            cfg.context_in_dim,
            cfg.hidden_size,
            true,
            &emb_vb.pp("txt_in"),
        )
        .map_err(|e| anyhow::anyhow!("txt_in: {e}"))?;
        let time_in =
            common::MlpEmbedder::new(common::SCALAR_EMBED, cfg.hidden_size, &emb_vb.pp("time_in"))
                .map_err(|e| anyhow::anyhow!("time_in: {e}"))?;
        let vector_in =
            common::MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, &emb_vb.pp("vector_in"))
                .map_err(|e| anyhow::anyhow!("vector_in: {e}"))?;
        let guidance_in = if cfg.guidance_embed {
            Some(
                common::MlpEmbedder::new(
                    common::SCALAR_EMBED,
                    cfg.hidden_size,
                    &emb_vb.pp("guidance_in"),
                )
                .map_err(|e| anyhow::anyhow!("guidance_in: {e}"))?,
            )
        } else {
            None
        };
        let final_layer = HeteroLastLayer::new(
            cfg.hidden_size,
            1,
            cfg.in_channels,
            emb_vb.pp("final_layer"),
        )
        .map_err(|e| anyhow::anyhow!("final_layer: {e}"))?;

        // Build a block-to-device mapping from the plan
        let cuda_available = !cuda_vbs.is_empty();
        let mut block_device: Vec<DeviceKind> = vec![DeviceKind::Cpu; total_blocks];
        for seg in &plan.segments {
            // i is only used to index `block_device`; iterate the slice
            // mutably to drop the explicit index (clippy::needless_range_loop).
            let end = seg.layer_end.min(total_blocks);
            for slot in block_device[seg.layer_start..end].iter_mut() {
                match seg.kind {
                    DeviceKind::Cuda(_) if cuda_available => {
                        *slot = seg.kind;
                    }
                    #[cfg(feature = "opencl")]
                    DeviceKind::OpenCL(_) if ocl_pipelines.is_some() => {
                        *slot = seg.kind;
                    }
                    _ => {} // stay CPU
                }
            }
        }

        // Load blocks to their assigned devices
        let mut layers = Vec::with_capacity(total_blocks);
        let mut cuda_count = 0usize;
        let mut cpu_count = 0usize;
        #[cfg(feature = "opencl")]
        let mut ocl_count = 0usize;

        // idx is used both to index block_device AND in format!("double_blocks.{idx}")
        // and bookkeeping; the index loop is the right shape.
        #[allow(clippy::needless_range_loop)]
        for idx in 0..cfg.depth {
            match block_device[idx] {
                DeviceKind::Cuda(ord) => {
                    let vb = cuda_vbs.get(&ord).ok_or_else(|| {
                        anyhow::anyhow!("double_blocks.{idx}: no VarBuilder for CUDA:{ord}")
                    })?;
                    let block = HeteroDoubleBlock::new(cfg, vb.pp(format!("double_blocks.{idx}")))
                        .map_err(|e| anyhow::anyhow!("double_blocks.{idx}: {e}"))?;
                    cuda_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::Cuda(ord),
                        block: HeteroFluxBlock::Double(block),
                    });
                }
                #[cfg(feature = "opencl")]
                DeviceKind::OpenCL(ocl_idx) => {
                    let mmap = gguf_mmap.as_ref().unwrap();
                    let content = crate::tensor::quantized::gguf_file::Content::read_mapped(
                        &mut std::io::Cursor::new(mmap.as_ref().as_ref()),
                        mmap.clone(),
                    )
                    .map_err(|e| anyhow::anyhow!("GGUF re-parse for OCL block: {e}"))?;
                    let mut reader = std::io::Cursor::new(mmap.as_ref().as_ref());
                    let block = OpenCLFluxDoubleBlock::from_gguf(
                        &content,
                        &mut reader,
                        idx,
                        cfg,
                        ocl_pipelines.as_ref().unwrap(),
                    )?;
                    ocl_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::OpenCL(ocl_idx),
                        block: HeteroFluxBlock::OclDouble(block),
                    });
                }
                _ => {
                    let block =
                        HeteroDoubleBlock::new(cfg, cpu_vb.pp(format!("double_blocks.{idx}")))
                            .map_err(|e| anyhow::anyhow!("double_blocks.{idx}: {e}"))?;
                    cpu_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::Cpu,
                        block: HeteroFluxBlock::Double(block),
                    });
                }
            }
        }
        for idx in 0..cfg.depth_single_blocks {
            let block_idx = cfg.depth + idx;
            match block_device[block_idx] {
                DeviceKind::Cuda(ord) => {
                    let vb = cuda_vbs.get(&ord).ok_or_else(|| {
                        anyhow::anyhow!("single_blocks.{idx}: no VarBuilder for CUDA:{ord}")
                    })?;
                    let block = HeteroSingleBlock::new(cfg, vb.pp(format!("single_blocks.{idx}")))
                        .map_err(|e| anyhow::anyhow!("single_blocks.{idx}: {e}"))?;
                    cuda_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::Cuda(ord),
                        block: HeteroFluxBlock::Single(block),
                    });
                }
                #[cfg(feature = "opencl")]
                DeviceKind::OpenCL(ocl_idx) => {
                    let mmap = gguf_mmap.as_ref().unwrap();
                    let content = crate::tensor::quantized::gguf_file::Content::read_mapped(
                        &mut std::io::Cursor::new(mmap.as_ref().as_ref()),
                        mmap.clone(),
                    )
                    .map_err(|e| anyhow::anyhow!("GGUF re-parse for OCL block: {e}"))?;
                    let mut reader = std::io::Cursor::new(mmap.as_ref().as_ref());
                    let block = OpenCLFluxSingleBlock::from_gguf(
                        &content,
                        &mut reader,
                        idx,
                        cfg,
                        ocl_pipelines.as_ref().unwrap(),
                    )?;
                    ocl_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::OpenCL(ocl_idx),
                        block: HeteroFluxBlock::OclSingle(block),
                    });
                }
                _ => {
                    let block =
                        HeteroSingleBlock::new(cfg, cpu_vb.pp(format!("single_blocks.{idx}")))
                            .map_err(|e| anyhow::anyhow!("single_blocks.{idx}: {e}"))?;
                    cpu_count += 1;
                    layers.push(HeteroFluxLayer {
                        device: HeteroDevice::Cpu,
                        block: HeteroFluxBlock::Single(block),
                    });
                }
            }
        }
        drop(cpu_vb);
        drop(cuda_vbs);
        #[cfg(feature = "opencl")]
        drop(gguf_mmap);
        #[cfg(feature = "opencl")]
        info!(
            "HeteroFlux: {} blocks loaded ({} CUDA + {} OpenCL + {} CPU)",
            total_blocks, cuda_count, ocl_count, cpu_count
        );
        #[cfg(not(feature = "opencl"))]
        info!(
            "HeteroFlux: {} blocks loaded ({} CUDA + {} CPU)",
            total_blocks, cuda_count, cpu_count
        );

        // Build segments: consecutive runs of same-device blocks
        let (segments, mut unique_devices) = build_segments(&layers, cuda_devices);
        info!("HeteroFlux: {} segments", segments.len());
        for (i, seg) in segments.iter().enumerate() {
            info!(
                "  Segment {}: {:?} blocks [{}, {})",
                i, seg.device, seg.block_start, seg.block_end
            );
        }
        let mut seen_keys: std::collections::HashSet<String> =
            unique_devices.iter().map(device_key).collect();
        // Always include primary device
        {
            let pk = device_key(&primary_device);
            if seen_keys.insert(pk) {
                unique_devices.push(primary_device.clone());
            }
        }

        // Create OpenCL scratch buffers if we have OpenCL segments
        #[cfg(feature = "opencl")]
        let (opencl_pipelines_out, opencl_scratch) =
            if let Some(p) = ocl_pipelines.as_ref().filter(|_| ocl_count > 0) {
                let mlp_sz = (cfg.hidden_size as f64 * cfg.mlp_ratio) as usize;
                let head_dim = cfg.hidden_size / cfg.num_heads;
                // Max sequence lengths: img=1024 (32x32 patches), txt=512 (T5 max)
                let scratch = OclFluxScratch::new(
                    1024,
                    512,
                    cfg.hidden_size,
                    mlp_sz,
                    cfg.num_heads,
                    head_dim,
                    p,
                )?;
                info!("HeteroFlux: OpenCL scratch buffers allocated");
                (ocl_pipelines, Some(std::sync::Mutex::new(scratch)))
            } else {
                (None, None)
            };

        Ok(HeteroFlux {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            // Nothing is read for this one: the axes share out one head between them, so the
            // rotation is as wide as a head and the table is built from the config alone.
            pe_embedder: crate::inference::model::flux::common::EmbedNd::new(
                cfg.hidden_size / cfg.num_heads,
                cfg.theta,
                cfg.axes_dim.to_vec(),
            ),
            layers,
            segments,
            final_layer,
            plan: plan.clone(),
            primary_device,
            cfg: cfg.clone(),
            // Every block here shares its card with whatever else the plan put on it.
            intermediates: Intermediates::Budgeted,
            checkpoint: gguf_path.to_path_buf(),
            cuda_devices: cuda_devices.clone(),
            bytes_per_block: std::fs::metadata(gguf_path)
                .map(|m| m.len() / total_blocks.max(1) as u64)
                .unwrap_or(0),
            adapters_attached: 0,
            render_headroom: 0,
            repatriate_backoff: 0,
            unique_devices,
            pe_cache: std::sync::Mutex::new(None),
            pe_per_device_cache: std::sync::Mutex::new(None),
            txt_in_cache: std::sync::Mutex::new(None),
            y_in_cache: std::sync::Mutex::new(None),
            guidance_in_cache: std::sync::Mutex::new(None),
            #[cfg(feature = "opencl")]
            opencl_pipelines: opencl_pipelines_out,
            #[cfg(feature = "opencl")]
            opencl_scratch,
        })
    }

    /// The whole transformer on ONE device, from a builder the caller has already opened.
    ///
    /// Not every checkpoint is a GGUF. An fp8-scaled safetensors is read through the fp8
    /// bridge, which hands back a builder rather than a path and does it cancellably;
    /// [`Self::from_gguf`] converts such a file to a GGUF sidecar first, because a SPLIT load
    /// needs one builder PER CARD and a path is what it can reopen. A whole load needs one
    /// builder and is given it, so the sidecar - a full requantised copy written to disk - is
    /// not paid, and a load that the user cancels still stops.
    ///
    /// The tensors are read in the order a whole placement has always read them: the two input
    /// projections, the blocks, then the conditioning embedders and the final layer.
    /// `from_gguf` reads the embedders first instead, and the difference is deliberate - its
    /// order is what the plan's per-card VRAM ledger was measured against, and neither is free
    /// to adopt the other's.
    pub fn whole(
        cfg: &crate::inference::model::flux::common::Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let total_blocks = cfg.depth + cfg.depth_single_blocks;

        // The packed latent and the text sequence enter through a projection each, both to the
        // stream's width; from there on the transformer is one width throughout.
        let img_in = linear_b(cfg.in_channels, cfg.hidden_size, true, &vb.pp("img_in"))?;
        let txt_in = linear_b(cfg.context_in_dim, cfg.hidden_size, true, &vb.pp("txt_in"))?;

        // One device for every block, so the segment list below is a single entry and no
        // activation ever crosses a boundary.
        let placed = match device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => HeteroDevice::Cuda(gpu_id),
            _ => HeteroDevice::Cpu,
        };
        let mut layers = Vec::with_capacity(total_blocks);
        let vb_d = vb.pp("double_blocks");
        for i in 0..cfg.depth {
            layers.push(HeteroFluxLayer {
                device: placed.clone(),
                block: HeteroFluxBlock::Double(HeteroDoubleBlock::new(cfg, vb_d.pp(i))?),
            });
        }
        let vb_s = vb.pp("single_blocks");
        for i in 0..cfg.depth_single_blocks {
            layers.push(HeteroFluxLayer {
                device: placed.clone(),
                block: HeteroFluxBlock::Single(HeteroSingleBlock::new(cfg, vb_s.pp(i))?),
            });
        }

        // What every block is modulated by. The step and the guidance scale are scalars, so
        // they arrive as sinusoidal embeddings of a fixed width; the pooled text vector arrives
        // at whatever width the text encoder publishes. A checkpoint distilled without a
        // guidance scale has no projection for one, and is asked for none.
        let time_in =
            common::MlpEmbedder::new(common::SCALAR_EMBED, cfg.hidden_size, &vb.pp("time_in"))?;
        let vector_in =
            common::MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, &vb.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            Some(common::MlpEmbedder::new(
                common::SCALAR_EMBED,
                cfg.hidden_size,
                &vb.pp("guidance_in"),
            )?)
        } else {
            None
        };
        let final_layer =
            HeteroLastLayer::new(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?;

        let cuda_devices: HashMap<usize, Device> = match placed {
            HeteroDevice::Cuda(idx) => [(idx, device.clone())].into_iter().collect(),
            _ => HashMap::new(),
        };
        let (segments, unique_devices) = build_segments(&layers, &cuda_devices);

        Ok(HeteroFlux {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            // Nothing is read for this one: the axes share out one head between them, so the
            // rotation is as wide as a head and the table is built from the config alone.
            pe_embedder: crate::inference::model::flux::common::EmbedNd::new(
                cfg.hidden_size / cfg.num_heads,
                cfg.theta,
                cfg.axes_dim.to_vec(),
            ),
            layers,
            segments,
            final_layer,
            plan: HeteroPlan {
                segments: vec![HeteroSegment {
                    kind: match placed {
                        HeteroDevice::Cuda(idx) => DeviceKind::Cuda(idx),
                        _ => DeviceKind::Cpu,
                    },
                    layer_start: 0,
                    layer_end: total_blocks,
                    free_memory_bytes: 0,
                }],
                total_layers: total_blocks,
            },
            primary_device: device,
            cfg: cfg.clone(),
            // The blocks own the device, so nothing here answers to a memory budget.
            intermediates: Intermediates::Whole,
            // Repatriation has nothing to move: no block is on the host. A zero
            // bytes-per-block is what `try_repatriate_block` reads as "stand down", so the
            // checkpoint it would reopen is never needed and is not carried - which also
            // means an fp8 checkpoint, which that path could not reopen, cannot reach it.
            checkpoint: std::path::PathBuf::new(),
            cuda_devices,
            bytes_per_block: 0,
            adapters_attached: 0,
            render_headroom: 0,
            repatriate_backoff: 0,
            unique_devices,
            pe_cache: std::sync::Mutex::new(None),
            pe_per_device_cache: std::sync::Mutex::new(None),
            txt_in_cache: std::sync::Mutex::new(None),
            y_in_cache: std::sync::Mutex::new(None),
            guidance_in_cache: std::sync::Mutex::new(None),
            #[cfg(feature = "opencl")]
            opencl_pipelines: None,
            #[cfg(feature = "opencl")]
            opencl_scratch: None,
        })
    }

    /// Build per-device conditioning cache for vec_ and pe.
    /// Called once per denoise step (not per block), caching across all segments.
    /// pe is constant across denoise steps (cached at pe_cache layer) so we
    /// also cache the per-device transferred pe by pe.id() - only vec_
    /// needs a fresh per-device transfer each step.
    fn cache_conditioning(
        &self,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        // Look up cached per-device pe first; rebuild if pe.id() changed.
        let pe_id = pe.id();
        let pe_per_device: HashMap<String, Tensor> = {
            let mut guard = self
                .pe_per_device_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((cid, ref map)) = *guard {
                if cid == pe_id {
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                } else {
                    let mut m = HashMap::with_capacity(self.unique_devices.len());
                    for dev in &self.unique_devices {
                        if let std::collections::hash_map::Entry::Vacant(slot) =
                            m.entry(device_key(dev))
                        {
                            slot.insert(pe.to_device(dev)?);
                        }
                    }
                    *guard = Some((pe_id, m.clone()));
                    m
                }
            } else {
                let mut m = HashMap::with_capacity(self.unique_devices.len());
                for dev in &self.unique_devices {
                    if let std::collections::hash_map::Entry::Vacant(slot) =
                        m.entry(device_key(dev))
                    {
                        slot.insert(pe.to_device(dev)?);
                    }
                }
                *guard = Some((pe_id, m.clone()));
                m
            }
        };

        let mut cache = HashMap::with_capacity(self.unique_devices.len());
        for dev in &self.unique_devices {
            let key = device_key(dev);
            if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(key.clone()) {
                let v = vec_.to_device(dev)?;
                let p = pe_per_device.get(&key).cloned().ok_or_else(|| {
                    crate::tensor::Error::msg(format!("pe_per_device missing {key}"))
                })?;
                slot.insert((v, p));
            }
        }
        Ok(cache)
    }
}

/// Convert a facade Tensor to F32 CPU data, upload to an existing OpenCL buffer.
/// Tensor shape must be [1, seq, dim] or [seq, dim]; we flatten to F32.
#[cfg(feature = "opencl")]
fn tensor_to_ocl_buf(
    tensor: &Tensor,
    buf: &mut opencl3::memory::Buffer<u8>,
    pipelines: &OpenCLPipelines,
) -> Result<()> {
    let t = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    let byte_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    unsafe {
        pipelines
            .queue
            .enqueue_write_buffer(buf, opencl3::command_queue::CL_BLOCKING, 0, byte_slice, &[])
            .map_err(|e| crate::tensor::Error::msg(format!("tensor_to_ocl_buf: {e}")))?;
    }
    Ok(())
}

/// Read F32 data from an OpenCL buffer, create a facade Tensor.
/// Returns Tensor on CPU with given shape.
#[cfg(feature = "opencl")]
fn ocl_buf_to_tensor(
    buf: &opencl3::memory::Buffer<u8>,
    shape: &[usize],
    pipelines: &OpenCLPipelines,
) -> Result<Tensor> {
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f32; total];
    let byte_slice =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total * 4) };
    unsafe {
        pipelines
            .queue
            .enqueue_read_buffer(buf, opencl3::command_queue::CL_BLOCKING, 0, byte_slice, &[])
            .map_err(|e| crate::tensor::Error::msg(format!("ocl_buf_to_tensor: {e}")))?;
    }
    Tensor::from_vec(data, shape, &Device::Cpu)
}

impl crate::inference::model::flux::sampling::WithForward for HeteroFlux {
    /// Revisit the placement between steps: a block spilled to the host under pressure
    /// goes back onto a card as soon as one has room, instead of costing host speed for
    /// the rest of the render. One block per step - a failure here only means the block
    /// stays where it is, so the render carries on either way.
    fn between_steps(&mut self) {
        if let Err(e) = self.try_repatriate_block() {
            debug!("repatriation stood down: {e}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            crate::tensor::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            crate::tensor::bail!("unexpected shape for img {:?}", img.shape())
        }

        // Remember caller's device so we return the result on it
        let caller_device = img.device().clone();

        // Key the projection caches on the tensors the CALLER passed, before any to_device
        // which may hand back a fresh allocation when it transfers.
        let txt_key = txt.clone();
        let y_key = y.clone();
        let guidance_key = guidance.cloned();

        // Move all inputs to primary device (CPU in hetero mode) to match weight locations
        let img = img.to_device(&self.primary_device)?;
        let img_ids = img_ids.to_device(&self.primary_device)?;
        let txt = txt.to_device(&self.primary_device)?;
        let txt_ids = txt_ids.to_device(&self.primary_device)?;
        let timesteps = timesteps.to_device(&self.primary_device)?;
        let y = y.to_device(&self.primary_device)?;
        let guidance = match guidance {
            Some(g) => Some(g.to_device(&self.primary_device)?),
            None => None,
        };

        let dtype = img.dtype();

        // -- Compute embeddings on primary device (once per denoise step) --
        // `pe` depends only on (txt_ids, img_ids), both of which stay
        // constant across the entire denoise loop within one image. Cache
        // it by (txt_ids.dims, img_ids.dims) - for a given model config
        // those uniquely identify the contents.
        let txt_dims = txt_ids.dims().to_vec();
        let img_dims = img_ids.dims().to_vec();
        let pe = {
            let mut guard = self
                .pe_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((ref c_txt, ref c_img, ref c_pe)) = *guard {
                if *c_txt == txt_dims && *c_img == img_dims {
                    c_pe.clone()
                } else {
                    let ids = Tensor::cat(&[&txt_ids, &img_ids], 1)?;
                    let pe = ids.apply(&self.pe_embedder)?;
                    *guard = Some((txt_dims, img_dims, pe.clone()));
                    pe
                }
            } else {
                let ids = Tensor::cat(&[&txt_ids, &img_ids], 1)?;
                let pe = ids.apply(&self.pe_embedder)?;
                *guard = Some((txt_dims, img_dims, pe.clone()));
                pe
            }
        };
        let mut txt = cached_or(&self.txt_in_cache, &txt_key, || txt.apply(&self.txt_in))?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ =
            timestep_embedding(&timesteps, common::SCALAR_EMBED, dtype)?.apply(&self.time_in)?;
        let vec_ = match (
            self.guidance_in.as_ref(),
            guidance.as_ref(),
            guidance_key.as_ref(),
        ) {
            (Some(g_in), Some(guidance), Some(g_key)) => {
                let g_emb = cached_or(&self.guidance_in_cache, g_key, || {
                    timestep_embedding(guidance, common::SCALAR_EMBED, dtype)?.apply(g_in)
                })?;
                (vec_ + g_emb)?
            }
            _ => vec_,
        };
        let y_emb = cached_or(&self.y_in_cache, &y_key, || y.apply(&self.vector_in))?;
        let vec_ = (vec_ + y_emb)?;

        // -- Pre-cache conditioning tensors per device (transferred once, not per block) --
        let cond_cache = self.cache_conditioning(&vec_, &pe)?;

        // -- Process segments: double blocks first, then single blocks --
        let depth = self.cfg.depth;

        for seg in &self.segments {
            // -- OpenCL segment (double blocks only - single blocks handled separately) --
            #[cfg(feature = "opencl")]
            if matches!(seg.device, HeteroDevice::OpenCL(_)) && seg.block_start < depth {
                let p = self
                    .opencl_pipelines
                    .as_ref()
                    .ok_or_else(|| crate::tensor::Error::msg("OpenCL segment but no pipelines"))?;
                let mut scratch = self
                    .opencl_scratch
                    .as_ref()
                    .ok_or_else(|| crate::tensor::Error::msg("OpenCL segment but no scratch"))?
                    .lock()
                    .map_err(|e| crate::tensor::Error::msg(format!("scratch lock: {e}")))?;

                let img_seq = img.dim(1)?;
                let txt_seq = txt.dim(1)?;

                // Upload tensors to OpenCL buffers
                tensor_to_ocl_buf(&img, &mut scratch.img_buf, p)?;
                tensor_to_ocl_buf(&txt, &mut scratch.txt_buf, p)?;
                tensor_to_ocl_buf(&vec_, &mut scratch.vec_buf, p)?;
                tensor_to_ocl_buf(&pe, &mut scratch.pe_buf, p)?;
                // vec_buf changed - invalidate cached silu(vec_buf) so the
                // next block-forward in this segment computes it fresh.
                scratch.invalidate_silu_vec();

                for i in seg.block_start..seg.block_end.min(depth) {
                    let _timer = crate::inference::place::layer_perf::LayerTimer::start_for(
                        i,
                        "flux-schnell",
                    );
                    match &self.layers[i].block {
                        HeteroFluxBlock::OclDouble(block) => {
                            block
                                .forward(img_seq, txt_seq, &mut scratch, p)
                                .map_err(|e| {
                                    crate::tensor::Error::msg(format!("OclDouble {i}: {e}"))
                                })?;
                        }
                        _ => crate::tensor::bail!("expected OclDouble block at index {i}"),
                    }
                }

                // Download results back to CPU tensors
                let img_shape = img.dims().to_vec();
                let txt_shape = txt.dims().to_vec();
                img = ocl_buf_to_tensor(&scratch.img_buf, &img_shape, p)?;
                txt = ocl_buf_to_tensor(&scratch.txt_buf, &txt_shape, p)?;

                // Check if this segment also has single blocks
                if seg.block_end > depth {
                    let txt_len = txt.dim(1)?;
                    let mut seq = Tensor::cat(&[&txt, &img], 1)?;
                    seq = self.process_single_blocks(seq, txt_len, depth, &cond_cache)?;
                    let from = seq.device().clone();
                    let seq = cross_to(&seq, &from, &self.primary_device)?;
                    let pk = device_key(&self.primary_device);
                    let (vec_primary, _) = cond_cache.get(&pk).ok_or_else(|| {
                        crate::tensor::Error::msg("no cached conditioning for primary")
                    })?;
                    let result = seq.i((.., txt_len..))?;
                    let out = self.final_layer.forward(&result, vec_primary)?;
                    let from = out.device().clone();
                    return cross_to(&out, &from, &caller_device);
                }
                continue;
            }

            // -- substrate segment (CUDA or CPU) --
            let dk = device_key(&seg.tensor_device);
            let (vec_dev, pe_dev) = cond_cache.get(&dk).ok_or_else(|| {
                crate::tensor::Error::msg(format!("no cached conditioning for {:?}", seg.device))
            })?;

            // Transfer the mutable hidden states to this segment's device, draining
            // the producing device first - see `cross_to`.
            let from = img.device().clone();
            img = cross_to(&img, &from, &seg.tensor_device)?;
            txt = cross_to(&txt, &from, &seg.tensor_device)?;

            for i in seg.block_start..seg.block_end {
                if i < depth {
                    // Double block
                    let _timer = crate::inference::place::layer_perf::LayerTimer::start_for(
                        i,
                        "flux-schnell",
                    );
                    match &self.layers[i].block {
                        HeteroFluxBlock::Double(block) => {
                            let (new_img, new_txt) =
                                block.forward(&img, &txt, vec_dev, pe_dev, self.intermediates)?;
                            img = new_img;
                            txt = new_txt;
                        }
                        _ => crate::tensor::bail!("expected double block at index {i}"),
                    }
                } else {
                    // Single block - merge txt+img on first single block
                    if i == depth {
                        let txt_len = txt.dim(1)?;
                        let mut seq = Tensor::cat(&[&txt, &img], 1)?;

                        // Process all remaining single blocks in this and subsequent segments
                        seq = self.process_single_blocks(seq, txt_len, i, &cond_cache)?;

                        // Final layer on primary device
                        let from = seq.device().clone();
                        let seq = cross_to(&seq, &from, &self.primary_device)?;
                        let pk = device_key(&self.primary_device);
                        let (vec_primary, _) = cond_cache.get(&pk).ok_or_else(|| {
                            crate::tensor::Error::msg("no cached conditioning for primary")
                        })?;
                        let result = seq.i((.., txt_len..))?;
                        let out = self.final_layer.forward(&result, vec_primary)?;
                        let from = out.device().clone();
                        return cross_to(&out, &from, &caller_device);
                    }
                }
            }
        }

        // Fallback: if all blocks are double (shouldn't happen for Flux)
        crate::tensor::bail!("forward completed without processing single blocks")
    }
}

impl HeteroFlux {
    /// Process all single blocks (from block index `start_idx` onward) using segment dispatch.
    /// Returns the final merged sequence tensor.
    fn process_single_blocks(
        &self,
        mut seq: Tensor,
        _txt_len: usize,
        start_idx: usize,
        cond_cache: &HashMap<String, (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        for seg in &self.segments {
            if seg.block_end <= start_idx {
                continue;
            }
            let seg_start = seg.block_start.max(start_idx);
            let seg_end = seg.block_end;
            if seg_start >= seg_end {
                continue;
            }

            // -- OpenCL single block segment --
            #[cfg(feature = "opencl")]
            if matches!(seg.device, HeteroDevice::OpenCL(_)) {
                let p = self
                    .opencl_pipelines
                    .as_ref()
                    .ok_or_else(|| crate::tensor::Error::msg("OpenCL segment but no pipelines"))?;
                let mut scratch = self
                    .opencl_scratch
                    .as_ref()
                    .ok_or_else(|| crate::tensor::Error::msg("OpenCL segment but no scratch"))?
                    .lock()
                    .map_err(|e| crate::tensor::Error::msg(format!("scratch lock: {e}")))?;

                let seq_len = seq.dim(1)?;
                let dim = self.cfg.hidden_size;

                // Upload seq to merged_buf
                tensor_to_ocl_buf(&seq, &mut scratch.merged_buf, p)?;

                // Upload vec_ and pe to dedicated buffers
                let ppk = device_key(&self.primary_device);
                let (vec_ref, pe_ref) = cond_cache
                    .get(&ppk)
                    .or_else(|| cond_cache.get(&device_key(&Device::Cpu)))
                    .ok_or_else(|| {
                        crate::tensor::Error::msg("no cached conditioning for OpenCL single blocks")
                    })?;
                tensor_to_ocl_buf(vec_ref, &mut scratch.vec_buf, p)?;
                tensor_to_ocl_buf(pe_ref, &mut scratch.pe_buf, p)?;
                scratch.invalidate_silu_vec();

                for i in seg_start..seg_end {
                    let _timer = crate::inference::place::layer_perf::LayerTimer::start_for(
                        i,
                        "flux-schnell",
                    );
                    match &self.layers[i].block {
                        HeteroFluxBlock::OclSingle(block) => {
                            block.forward(seq_len, &mut scratch, p).map_err(|e| {
                                crate::tensor::Error::msg(format!("OclSingle {i}: {e}"))
                            })?;
                        }
                        _ => crate::tensor::bail!("expected OclSingle block at index {i}"),
                    }
                }

                // Download back to tensor
                seq = ocl_buf_to_tensor(&scratch.merged_buf, &[1, seq_len, dim], p)?;
                continue;
            }

            // -- substrate segment --
            let dk = device_key(&seg.tensor_device);
            let (vec_dev, pe_dev) = cond_cache.get(&dk).ok_or_else(|| {
                crate::tensor::Error::msg(format!("no cached conditioning for {:?}", seg.device))
            })?;

            let from = seq.device().clone();
            seq = cross_to(&seq, &from, &seg.tensor_device)?;

            for i in seg_start..seg_end {
                let _timer =
                    crate::inference::place::layer_perf::LayerTimer::start_for(i, "flux-schnell");
                match &self.layers[i].block {
                    HeteroFluxBlock::Single(block) => {
                        seq = block.forward(&seq, vec_dev, pe_dev, self.intermediates)?;
                    }
                    _ => crate::tensor::bail!("expected single block at index {i}"),
                }
            }
        }
        Ok(seq)
    }
}

#[cfg(test)]
mod attention_tiling_tests {
    use super::*;

    /// The token slabs must cover the sequence exactly once, whatever its length.
    ///
    /// The step is fixed here on purpose: the production one is derived from the card,
    /// and what is under test is the LOOP, which is identical either way.
    ///
    /// Slabbing is exact only if the slabs partition the tokens: an off-by-one in the
    /// offset or the width silently drops or duplicates rows, which comes out as a
    /// garbled image rather than an error. The loop bounds are where that bug would
    /// live, so they are checked directly - including the lengths that are not a
    /// multiple of the slab, which is where such a bug hides.
    #[test]
    fn token_slabs_partition_the_sequence() {
        const STEP: usize = 2048;
        for tokens in [1usize, STEP - 1, STEP, STEP + 1, 2 * STEP, 9472] {
            let mut covered = vec![0u8; tokens];
            let mut off = 0usize;
            while off < tokens {
                let t = (tokens - off).min(STEP);
                assert!(t > 0, "a zero-width slab would loop forever at {tokens}");
                for c in covered[off..off + t].iter_mut() {
                    *c += 1;
                }
                off += t;
            }
            assert_eq!(off, tokens, "the slabs overran {tokens}");
            assert!(
                covered.iter().all(|c| *c == 1),
                "the slabs do not partition {tokens} tokens"
            );
        }
    }

    /// Tiling the queries must not change a single output value.
    ///
    /// The whole justification for tiling is that a query's softmax depends only on
    /// that query's own row, so slicing the queries changes the batching and nothing
    /// else. If that were even slightly false the split path would render a different
    /// image from the single-device path at the same seed - a difference no error
    /// would report, and one only a pixel comparison would ever surface.
    ///
    /// Run at a sequence that crosses the threshold, so the tiled branch is the one
    /// under test, against the same input through the single-shot branch.
    #[test]
    fn query_tiling_is_bit_exact() {
        // A length that is not a multiple of anything, with the budget forced low
        // enough that the slabbed branch is the one under test.
        SLAB_BUDGET_BYTES.store(4096, std::sync::atomic::Ordering::Relaxed);
        let (heads, seq, dim) = (2usize, 2177usize, 8usize);
        // Deterministic, spread over a range where softmax is not saturated.
        let gen = |n: usize, salt: u64| -> Vec<f32> {
            let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 11) as f32 / (1u64 << 53) as f32) - 0.5
                })
                .collect()
        };
        let shape = vec![heads, seq, dim];
        let q = Tensor::from_vec_f32(gen(heads * seq * dim, 1), shape.clone()).unwrap();
        let k = Tensor::from_vec_f32(gen(heads * seq * dim, 2), shape.clone()).unwrap();
        let v = Tensor::from_vec_f32(gen(heads * seq * dim, 3), shape).unwrap();

        let tiled = sdpa_byte_budget(&q, &k, &v).unwrap();

        // The single-shot branch, spelled out here so the comparison does not depend
        // on being able to reach it through the function's own threshold.
        let scale = 1.0 / (dim as f64).sqrt();
        let expected = {
            let w = (q.matmul_t(&k).unwrap() * scale).unwrap();
            crate::tensor::ops::softmax_last_dim(&w)
                .unwrap()
                .matmul(&v)
                .unwrap()
        };

        SLAB_BUDGET_BYTES.store(0, std::sync::atomic::Ordering::Relaxed);
        let a = tiled.flatten_all().unwrap().to_vec1_f32().unwrap();
        let b = expected.flatten_all().unwrap().to_vec1_f32().unwrap();
        assert_eq!(a.len(), b.len(), "tiling changed the output shape");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "value {i} differs: {x} vs {y}");
        }
    }
}

#[cfg(test)]
mod fixed_tile_tests {
    use super::*;
    use crate::inference::model::flux::common::EmbedNd;
    use crate::tensor::Module as _;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Query-tiled SDPA must be bit-exact vs the single-shot path (same
    /// per-query math). Exercises uneven tail tiles (seq=10, tile=3).
    #[test]
    fn sdpa_query_tiling_is_bit_exact() {
        let (b, h, l, d) = (1usize, 2usize, 10usize, 8usize);
        let q = Tensor::from_vec_f32(data(b * h * l * d, 11), vec![b, h, l, d]).unwrap();
        let k = Tensor::from_vec_f32(data(b * h * l * d, 12), vec![b, h, l, d]).unwrap();
        let v = Tensor::from_vec_f32(data(b * h * l * d, 13), vec![b, h, l, d]).unwrap();
        let single = sdpa_fixed_tile(&q, &k, &v).unwrap();
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let kt = k.transpose(D::Minus2, D::Minus1).unwrap();
        for tile in [1usize, 3, 5, 10, 16] {
            let tiled = sdpa_query_tiled(&q, &kt, &v, scale, tile).unwrap();
            assert_eq!(tiled.dims(), single.dims());
            assert_eq!(
                tiled.to_vec_f32(),
                single.to_vec_f32(),
                "tile={tile} not bit-exact"
            );
        }
    }

    /// Attention held against what it is, not against a second copy of itself.
    ///
    /// Every position is given the id zero, so each rotation is by an angle of zero and the
    /// rope step is the identity; what must come back is then exactly `softmax(QKᵀ/√d).V`,
    /// computed below in plain arithmetic with no tensor op in the reference. A second pass
    /// with real positions must NOT match that, or the rotation is not being applied at all.
    #[test]
    fn attention_is_the_softmaxed_scores_times_the_values() {
        let (h, l, d) = (2usize, 4usize, 8usize);
        let (qv, kv, vv) = (data(h * l * d, 1), data(h * l * d, 2), data(h * l * d, 3));
        let shaped = |v: &Vec<f32>| Tensor::from_vec_f32(v.clone(), vec![1, h, l, d]).unwrap();
        let table = |ids: Vec<f32>| {
            EmbedNd::new(d / 2, 10_000, vec![d])
                .forward(&Tensor::from_vec_f32(ids, vec![1, l, 1]).unwrap())
                .unwrap()
        };

        let unrotated = attention(
            &shaped(&qv),
            &shaped(&kv),
            &shaped(&vv),
            &table(vec![0.0; l]),
            Intermediates::Whole,
        )
        .unwrap();
        assert_eq!(unrotated.dims(), &[1, l, h * d]);
        let got = unrotated.to_vec_f32();

        let scale = 1.0 / (d as f32).sqrt();
        for head in 0..h {
            let at = |t: &Vec<f32>, pos: usize, i: usize| t[(head * l + pos) * d + i];
            for qi in 0..l {
                let scores: Vec<f32> = (0..l)
                    .map(|kj| (0..d).map(|i| at(&qv, qi, i) * at(&kv, kj, i)).sum::<f32>() * scale)
                    .collect();
                let top = scores.iter().cloned().fold(f32::MIN, f32::max);
                let exp: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = exp.iter().sum();
                for i in 0..d {
                    let want: f32 = (0..l).map(|kj| exp[kj] / total * at(&vv, kj, i)).sum();
                    let have = got[qi * h * d + head * d + i];
                    assert!(
                        (have - want).abs() < 1e-4,
                        "head {head} query {qi} channel {i}: {have} is not {want}"
                    );
                }
            }
        }

        // And the rotation is not a no-op the reference happens to agree with.
        let rotated = attention(
            &shaped(&qv),
            &shaped(&kv),
            &shaped(&vv),
            &table((0..l).map(|i| i as f32).collect()),
            Intermediates::Whole,
        )
        .unwrap()
        .to_vec_f32();
        assert!(
            rotated.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-3),
            "positions changed and the answer did not - the rotation is not being applied"
        );
    }

    /// The two placements compute the same attention, and NOT to the last bit.
    ///
    /// A model that had to be split must not render a different picture from the same seed,
    /// so the two policies are held together here. What they agree on is the STATEMENT -
    /// `softmax(QKᵀ/√d).V`, per query row, whatever the slab width - and each is separately
    /// held bit-exact against its OWN single-shot path, which is where a slabbing bug would
    /// show. What they do not agree on to the last bit is the arithmetic that reaches it: the
    /// whole placement transposes K and takes the ordinary product, the split one issues the
    /// NT product against K untransposed, and those are different kernels with different
    /// accumulation orders. Measured here at 2 ULP, on values around 0.1.
    ///
    /// So this bounds a DRIFT IN MEANING and cannot bound a drift in the last bits. A change
    /// that alters which of the two a placement uses is a change to the pixels it renders, and
    /// only a fixed-seed render can say by how much.
    #[test]
    fn the_two_policies_agree() {
        let (b, h, l, d) = (1usize, 3usize, 37usize, 8usize);
        let q = Tensor::from_vec_f32(data(b * h * l * d, 31), vec![b, h, l, d]).unwrap();
        let k = Tensor::from_vec_f32(data(b * h * l * d, 32), vec![b, h, l, d]).unwrap();
        let v = Tensor::from_vec_f32(data(b * h * l * d, 33), vec![b, h, l, d]).unwrap();
        // A budget low enough that the split policy really slabs, against the whole one's
        // single-shot path at this length.
        SLAB_BUDGET_BYTES.store(4096, std::sync::atomic::Ordering::Relaxed);
        let budgeted = sdpa_byte_budget(&q, &k, &v).unwrap();
        SLAB_BUDGET_BYTES.store(0, std::sync::atomic::Ordering::Relaxed);
        let whole = sdpa_fixed_tile(&q, &k, &v).unwrap();
        assert_eq!(budgeted.dims(), whole.dims());
        // Two orders of magnitude tighter than the measured gap, and far below anything a
        // wrong slab bound, a dropped row or a lost scale could hide under.
        const TOL: f32 = 1e-6;
        for (i, (a, b)) in budgeted
            .to_vec_f32()
            .iter()
            .zip(whole.to_vec_f32().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < TOL,
                "value {i}: the split policy says {a}, the whole one {b}"
            );
        }
    }
}

#[cfg(test)]
mod repatriation_tests {
    use super::{checkpoint_block_name, segment_bounds, HeteroDevice};

    /// Segments are where the activation crosses a device boundary. After a block moves
    /// they have to be rebuilt: a stale set keeps sending the tensor to the card the
    /// block just left, which is silent - the weights are not there and the read races
    /// whatever is, which is how a split render came back black with no error.
    #[test]
    fn a_moved_block_is_regrouped_with_its_new_neighbours() {
        use HeteroDevice::{Cpu, Cuda};
        let before = vec![Cuda(0), Cuda(0), Cpu, Cpu];
        assert_eq!(segment_bounds(&before), vec![(Cuda(0), 0, 2), (Cpu, 2, 4)]);
        // block 2 goes back onto the card its neighbour is on: one segment, not three.
        let after = vec![Cuda(0), Cuda(0), Cuda(0), Cpu];
        assert_eq!(segment_bounds(&after), vec![(Cuda(0), 0, 3), (Cpu, 3, 4)]);
    }

    /// Two cards must NEVER share a segment: they hold different tensors, and running
    /// one card's blocks on the other's handle is the documented black-image failure.
    #[test]
    fn two_cards_are_never_merged_into_one_segment() {
        use HeteroDevice::Cuda;
        let devs = vec![Cuda(0), Cuda(1), Cuda(1)];
        assert_eq!(
            segment_bounds(&devs),
            vec![(Cuda(0), 0, 1), (Cuda(1), 1, 3)]
        );
    }

    /// Every block belongs to exactly one segment, whatever the placement - the property
    /// the forward relies on to visit them all once.
    #[test]
    fn the_segments_cover_every_block_once() {
        use HeteroDevice::{Cpu, Cuda};
        let devs = vec![Cpu, Cuda(1), Cpu, Cpu, Cuda(0), Cuda(0)];
        let segs = segment_bounds(&devs);
        let mut at = 0;
        for (_, start, end) in &segs {
            assert_eq!(*start, at);
            at = *end;
        }
        assert_eq!(at, devs.len());
    }

    /// A model with 3 double blocks then 2 single ones: the single indices restart at
    /// zero, which is what the checkpoint keys and what rebuilding a block depends on.
    fn layout() -> Vec<bool> {
        vec![true, true, true, false, false]
    }

    #[test]
    fn double_blocks_keep_their_position() {
        let l = layout();
        assert_eq!(checkpoint_block_name(&l, 0), "double_blocks.0");
        assert_eq!(checkpoint_block_name(&l, 2), "double_blocks.2");
    }

    #[test]
    fn single_blocks_restart_at_zero() {
        let l = layout();
        assert_eq!(checkpoint_block_name(&l, 3), "single_blocks.0");
        assert_eq!(checkpoint_block_name(&l, 4), "single_blocks.1");
    }

    /// Every position names a DISTINCT tensor group. Numbering by position in `layers`
    /// would pass the two tests above and still collide here.
    #[test]
    fn no_two_positions_name_the_same_block() {
        let l = layout();
        let names: std::collections::HashSet<String> =
            (0..l.len()).map(|i| checkpoint_block_name(&l, i)).collect();
        assert_eq!(names.len(), l.len());
    }
}
