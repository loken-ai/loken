use crate::tensor::layer::qlinear::QLinear;
use crate::tensor::layer::{LayerNorm, QkNorm, RmsNorm};
use crate::tensor::quantized::QVarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor, D};

/// What the transformer is shaped by.
#[derive(Debug, Clone)]
pub struct Config {
    /// The packed latent's width: sixteen VAE channels times the four pixels of a 2x2 patch.
    pub in_channels: usize,
    /// The two conditionings' widths - a pooled vector from CLIP, and a sequence from T5.
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    /// The stream's width, how much the feed-forward widens it, and across how many heads.
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    /// Blocks that carry the image and the text as two streams, then blocks that carry them
    /// concatenated as one.
    pub depth: usize,
    pub depth_single_blocks: usize,
    /// How the head is split between the three position axes - image index, row, column - and
    /// the base the rotary frequencies are drawn from.
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    /// Whether the attention's input projection carries a bias.
    pub qkv_bias: bool,
    /// Whether this checkpoint expects to be told a guidance scale. It is the one field the two
    /// published sizes disagree on.
    pub guidance_embed: bool,
}

/// How wide a scalar conditioning is made before it is embedded: the denoise step and the
/// guidance scale are each one number, spread over this many sines and cosines so that two
/// nearby steps are told apart at every frequency the transformer can read.
pub(crate) const SCALAR_EMBED: usize = 256;

impl Config {
    /// The two published checkpoints are the same transformer. `dev` was distilled with a
    /// guidance scale it expects to be told at inference, and `schnell` was not - that one
    /// flag is the whole difference, and spelling the other eleven fields out twice was an
    /// invitation for them to drift apart.
    fn shared(guidance_embed: bool) -> Self {
        // The head is the unit the transformer is stated in. One is 128 channels wide, split by
        // the rotary table between the image index and the two pixel coordinates; the stream is
        // twenty-four of them side by side. The single-stream blocks are twice as many as the
        // double-stream ones, which is what makes the second half of the depth the cheap half.
        let axes_dim = vec![16, 56, 56];
        let heads = 24;
        let two_stream = 19;
        Self {
            // Sixteen VAE channels folded over the four pixels of a 2x2 patch.
            in_channels: 16 * 4,
            // CLIP's pooled vector, then T5's sequence.
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: heads * axes_dim.iter().sum::<usize>(),
            mlp_ratio: 4.0,
            num_heads: heads,
            depth: two_stream,
            depth_single_blocks: 2 * two_stream,
            axes_dim,
            theta: 10_000,
            qkv_bias: true,
            guidance_embed,
        }
    }

    pub fn dev() -> Self {
        Self::shared(true)
    }

    pub fn schnell() -> Self {
        Self::shared(false)
    }
}

/// FLUX normalises without learning anything: `elementwise_affine=False`, so the weight is
/// ones and there is no bias. The ones live on the model's device, because the kernel reads
/// them in place and a host-resident weight would be uploaded on every call.
pub(crate) fn layer_norm(dim: usize, device: &Device) -> Result<LayerNorm> {
    crate::tensor::layer::layer_norm_no_affine(dim, 1e-6, device)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        crate::tensor::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    // `inv_freq` depends only on (device, dim, theta, dtype). Reused
    // every denoise step across 2-3 axes per call - caching saves the
    // from_vec + to_dtype + alloc per call.
    let inv_freq = rope_inv_freq_cached(&dev, dim, theta, pos.dtype())?;
    // One angle per position and per pair of channels: how far this position turns that pair.
    let angles = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let (cos, sin) = (angles.cos()?, angles.sin()?);
    // The angle is stored already turned into the plane rotation it stands for, as the two rows
    // of [[cos, -sin], [sin, cos]] on the last two axes: the attention then reads a position's
    // rotation without a trigonometric call of its own.
    let upper = Tensor::stack(&[&cos, &sin.neg()?], 3)?;
    let lower = Tensor::stack(&[&sin, &cos], 3)?;
    Tensor::stack(&[&upper, &lower], 3)
}

pub(crate) fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    // The channels are read two at a time - each pair is a point in a plane - turned by the
    // rotation this position holds, and laid back out flat afterwards.
    let flat = x.dims().to_vec();
    let (b, heads, seq, width) = x.dims4()?;
    let pairs = x.reshape((b, heads, seq, width / 2, 2))?;
    let (first, second) = (
        pairs.narrow(D::Minus1, 0, 1)?,
        pairs.narrow(D::Minus1, 1, 1)?,
    );
    // The rotation is taken a column at a time: one column says where a unit of the pair's
    // first channel lands and the other where a unit of its second lands, so the turn is two
    // broadcast multiplies and a sum rather than a matrix product per position.
    let (from_first, from_second) = rope_split_cached(freq_cis)?;
    (from_first.broadcast_mul(&first)? + from_second.broadcast_mul(&second)?)?.reshape(flat)
}

/// Cache freq_cis.get_on_dim(D::Minus1, 0/1) by freq_cis.id().
/// pe is itself cached across denoise steps (pe_cache), so within one
/// image gen the same `freq_cis` tensor flows through every attention()
/// call across every block. With ~57 blocks x ~4 slice ops on pe per
/// step in Flux schnell, this saves ~228 per-step Tensor_ allocations.
pub(crate) fn rope_split_cached(freq_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<Mutex<HashMap<crate::tensor::TensorId, (Tensor, Tensor, Tensor)>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let id = freq_cis.id();
    if let Ok(g) = cache.lock() {
        if let Some((key, fr0, fr1)) = g.get(&id) {
            // The identity is the storage ADDRESS, so a freed tensor can hand its address to
            // the next one. The entry holds the key tensor, which keeps the storage alive and
            // makes the address unrepeatable while the entry lives; the shape is checked
            // anyway, because a view of the same storage is a different tensor.
            if key.dims() == freq_cis.dims() {
                return Ok((fr0.clone(), fr1.clone()));
            }
        }
    }
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    if let Ok(mut g) = cache.lock() {
        // Bounded: the positional encoding is rebuilt once per generation, so distinct keys
        // are rare and clearing wholesale costs nothing worth measuring.
        if g.len() >= 32 {
            g.clear();
        }
        g.insert(id, (freq_cis.clone(), fr0.clone(), fr1.clone()));
    }
    Ok((fr0, fr1))
}

pub(crate) fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    if dim % 2 == 1 {
        crate::tensor::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    // Widened before the factor is applied, not after: a step arriving in a narrow dtype has
    // few enough mantissa bits that multiplying by a thousand in it loses the fraction that
    // distinguishes one denoise step from the next.
    let t = t.to_dtype(DType::F32)?.scale(TIME_FACTOR as f32)?;
    // `freqs` depends only on (device, half), so it is cached across denoise steps: a four- to
    // nine-step run would otherwise launch an arange, a cast, a multiply and an exp per step
    // to rebuild the same slim `[1, half]` row.
    let freqs = timestep_freqs_cached(&dev, half)?;
    let args = t.unsqueeze(1)?.broadcast_mul(&freqs)?;
    Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)?.to_dtype(dtype)
}

fn rope_inv_freq_cached(
    dev: &crate::tensor::Device,
    dim: usize,
    theta: usize,
    dtype: DType,
) -> Result<Tensor> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<
        Mutex<HashMap<(crate::tensor::DeviceLocation, usize, usize, DType), Tensor>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (dev.location(), dim, theta, dtype);
    if let Ok(g) = cache.lock() {
        if let Some(t) = g.get(&key) {
            return Ok(t.clone());
        }
    }
    let inv_freq = crate::inference::model::rope::inverse_frequencies_f64(dim, theta as f64);
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?.to_dtype(dtype)?;
    if let Ok(mut g) = cache.lock() {
        g.insert(key, inv_freq.clone());
    }
    Ok(inv_freq)
}

pub(crate) fn timestep_freqs_cached(dev: &crate::tensor::Device, half: usize) -> Result<Tensor> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(crate::tensor::DeviceLocation, usize), Tensor>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (dev.location(), half);
    if let Ok(g) = cache.lock() {
        if let Some(t) = g.get(&key) {
            return Ok(t.clone());
        }
    }
    const MAX_PERIOD: f64 = 10000.;
    let arange = Tensor::arange(0.0, half as f32)?
        .to_device(dev)?
        .to_dtype(crate::tensor::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?
        .exp()?
        .unsqueeze(0)?;
    if let Ok(mut g) = cache.lock() {
        g.insert(key, freqs.clone());
    }
    Ok(freqs)
}

#[derive(Debug, Clone)]
pub struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    pub fn new(_dim: usize, theta: usize, axes_dim: Vec<usize>) -> Self {
        Self { theta, axes_dim }
    }
}

impl crate::tensor::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        // A position is not one number but one per axis - which image, which row, which column -
        // and each axis turns its own share of the head. The shares are laid end to end, so a
        // head's rotations read as the axes in order, and the result carries a head axis of one
        // for the attention to broadcast over.
        let axes = ids.dim(D::Minus1)?;
        let per_axis = (0..axes)
            .map(|axis| {
                let pos = ids.get_on_dim(D::Minus1, axis)?;
                rope(&pos, self.axes_dim[axis], self.theta)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&per_axis.iter().collect::<Vec<_>>(), 2)?.unsqueeze(1)
    }
}

// ---------------------------------------------------------------------------------------
// The pieces every FLUX block is built from.
//
// Both placements - the whole model on one card, and the model split across several - read the
// same checkpoint and compute the same thing here. They were written out twice, once with an
// `H` in front of every name.

/// `{prefix}.weight` quantised `[out, in]`, and `{prefix}.bias` when the checkpoint has one.
///
/// Nothing in that is FLUX's - it is how every quantised projection in the tree is read - so
/// the reader is the shared one, under the name this family calls it by.
pub(crate) use crate::tensor::layer::qlinear::qlinear_b as linear_b;

/// What a modulation hands a block: where to move the activations, how far to stretch them,
/// and how much of the branch's output to keep.
#[derive(Debug)]
pub(crate) struct ModulationOut {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Tensor,
}

impl ModulationOut {
    pub(crate) fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        scale_shift(xs, &self.scale, &self.shift)
    }

    pub(crate) fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

/// Stretch activations by a scale and move them by a shift.
///
/// The scale is stated as a DEVIATION from one, so that a zero-initialised checkpoint starts
/// as the identity. The final layer is modulated the same way with nothing to gate, so it
/// reaches here directly rather than through a [`ModulationOut`] with a gate it would ignore.
pub(crate) fn scale_shift(xs: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(&scale.affine(1.0, 1.0)?)?
        .broadcast_add(shift)
}

/// Read `N` modulations out of one projection of the conditioning vector.
///
/// A block modulates once per branch it has - twice for a double-stream block, once for a
/// single-stream one - and the checkpoint publishes all of them as one wide projection, laid
/// out branch by branch and, within a branch, shift then scale then gate. The activation is
/// the conditioning's alone, so every block in a denoise step shares one cached copy of it.
///
/// Three vectors per branch is what this reads, and a gate is one of the three. The final
/// layer's projection publishes TWO vectors in total, because it has no branch to gate: it
/// cannot be asked for here at any `N`, and takes [`scale_shift`] directly instead. That is
/// why this stays private - the only callers it can have are the two modulations below.
fn modulate<const N: usize>(lin: &QLinear, vec_: &Tensor) -> Result<[ModulationOut; N]> {
    let wanted = 3 * N;
    let ys = vec_silu_cached(vec_)?
        .apply(lin)?
        .unsqueeze(1)?
        .chunk(wanted, D::Minus1)?;
    if ys.len() != wanted {
        return Err(Error(format!(
            "modulation: chunk gave {} of {wanted}",
            ys.len()
        )));
    }
    Ok(std::array::from_fn(|branch| ModulationOut {
        shift: ys[3 * branch].clone(),
        scale: ys[3 * branch + 1].clone(),
        gate: ys[3 * branch + 2].clone(),
    }))
}

/// Single-entry cache for silu(vec_) across all block modulations within one
/// Flux::forward step (~96 silu calls/step share one result). When vec_
/// changes (next denoise step) the first call replaces the entry, which holds
/// a clone of its key so the storage it is identified by cannot be freed and
/// its address handed to a different tensor while the entry lives.
pub(crate) fn vec_silu_cached(vec_: &Tensor) -> Result<Tensor> {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Option<(Tensor, Tensor)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut g = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((ref key, ref t)) = *g {
        if key.storage_ptr_id() == vec_.storage_ptr_id() && key.dims() == vec_.dims() {
            return Ok(t.clone());
        }
    }
    let s = vec_.silu()?;
    *g = Some((vec_.clone(), s.clone()));
    Ok(s)
}

/// One modulation: three vectors out of the conditioning.
#[derive(Debug)]
pub struct Modulation1 {
    pub lin: QLinear,
}

impl Modulation1 {
    pub(crate) fn new(dim: usize, vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            lin: linear_b(dim, 3 * dim, true, &vb.pp("lin"))?,
        })
    }

    pub(crate) fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let [branch]: [ModulationOut; 1] = modulate(&self.lin, vec_)?;
        Ok(branch)
    }
}

/// Two modulations from one projection - a block that modulates before and after its branch.
#[derive(Debug)]
pub struct Modulation2 {
    pub lin: QLinear,
}

impl Modulation2 {
    pub(crate) fn new(dim: usize, vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            lin: linear_b(dim, 6 * dim, true, &vb.pp("lin"))?,
        })
    }

    pub(crate) fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let [attn, mlp]: [ModulationOut; 2] = modulate(&self.lin, vec_)?;
        Ok((attn, mlp))
    }
}

/// The two-layer feed-forward a block ends with.
///
/// FLUX publishes it as a sequential pair, so `0` and `2` are the only part of it this family
/// names; the shape is the shared quantised one.
pub(crate) type Mlp = crate::tensor::layer::qlinear::QMlp;

pub(crate) fn mlp(in_sz: usize, mlp_sz: usize, vb: &QVarBuilder) -> Result<Mlp> {
    Ok(Mlp::new(
        linear_b(in_sz, mlp_sz, true, &vb.pp("0"))?,
        crate::tensor::ops::Activation::GeluPytorchTanh,
        linear_b(mlp_sz, in_sz, true, &vb.pp("2"))?,
    ))
}

/// The two-layer embedder the timestep and the pooled text vector go through.
#[derive(Debug)]
pub(crate) struct MlpEmbedder {
    pub in_layer: QLinear,
    pub out_layer: QLinear,
}

impl MlpEmbedder {
    pub(crate) fn new(in_sz: usize, h_sz: usize, vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            in_layer: linear_b(in_sz, h_sz, true, &vb.pp("in_layer"))?,
            out_layer: linear_b(h_sz, h_sz, true, &vb.pp("out_layer"))?,
        })
    }
}

/// Flux's adapter-name resolution, over the shared projection.
///
/// The projection knows how to hold an adapter; which adapter belongs to it is a question
/// about Flux's publishing conventions. They are in the kohya layout, whose keys are
/// `lora_unet_` plus the checkpoint path with dots replaced by underscores, and a folder will
/// hold both that and the layout-specific spelling. Trying only one is how an adapter loads,
/// matches nothing, and renders the base model in silence.
pub(crate) trait FluxAdapters {
    fn apply_flux_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
        path: &str,
        alt_key: Option<&str>,
    ) -> Result<usize>;
}

impl FluxAdapters for QLinear {
    fn apply_flux_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
        path: &str,
        alt_key: Option<&str>,
    ) -> Result<usize> {
        let d = match file.delta_for(path, strength)? {
            Some(d) => Some(d),
            None => match alt_key {
                Some(k) => file.delta_for_key(k, strength)?,
                None => None,
            },
        };
        let Some(d) = d else { return Ok(0) };
        self.add_lora(d)?;
        Ok(1)
    }
}

/// The query and key norms, read from the checkpoint.
/// Read a FLUX block's qk-norm scales.
pub(crate) fn qk_norm(dim: usize, vb: &QVarBuilder) -> Result<QkNorm> {
    Ok(QkNorm::new(
        RmsNorm::new(vb.get_f32(dim, "query_norm.scale")?, 1e-6),
        RmsNorm::new(vb.get_f32(dim, "key_norm.scale")?, 1e-6),
    ))
}

#[derive(Debug)]
pub struct SelfAttention {
    pub qkv: QLinear,
    pub norm: QkNorm,
    pub proj: QLinear,
    pub num_heads: usize,
}

impl SelfAttention {
    pub(crate) fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        vb: &QVarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear_b(dim, dim * 3, qkv_bias, &vb.pp("qkv"))?;
        let norm = qk_norm(head_dim, &vb.pp("norm"))?;
        let proj = linear_b(dim, dim, true, &vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    pub(crate) fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        heads_qkv(&self.qkv.forward(xs)?, self.num_heads, &self.norm)
    }
}

/// Split a projection's packed output into the three per-head sequences attention reads.
///
/// The queries, keys and values come back side by side on the last axis, so they are read as
/// `[batch, length, 3, heads, head width]` and taken one at a time; the head axis then moves
/// in front of the length, which is what lets every head attend on its own. The query and key
/// norms are applied here because a head is only meaningful once it has been separated out.
///
/// A single-stream block projects its feed-forward alongside the three, so it narrows the
/// attention part off first and arrives here with the same packing as a two-stream one.
pub(crate) fn heads_qkv(
    packed: &Tensor,
    num_heads: usize,
    norm: &QkNorm,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, len, width) = packed.shape().dims3()?;
    let split = packed.reshape(vec![b, len, 3, num_heads, width / (3 * num_heads)])?;
    let take = |which: usize| -> Result<Tensor> { split.get_on_dim(2, which)?.transpose(1, 2) };
    let (q, k, v) = (take(0)?, take(1)?, take(2)?);
    let (q, k) = norm.forward(&q, &k)?;
    Ok((q, k, v))
}

/// One of the two streams a double-stream block carries.
///
/// The image and the text go through the same shape of block - normalise, modulate, attend,
/// then a modulated feed-forward - and differ only in which weights they read. This borrows
/// one stream's share of a block so that shape is written once and run twice, by whichever
/// placement holds the block.
pub(crate) struct Stream<'a> {
    pub norm1: &'a LayerNorm,
    pub attn: &'a SelfAttention,
    pub norm2: &'a LayerNorm,
    pub mlp: &'a Mlp,
}

impl Stream<'_> {
    /// Normalise and modulate the stream, then split it into what attention reads.
    pub(crate) fn qkv(&self, xs: &Tensor, m: &ModulationOut) -> Result<(Tensor, Tensor, Tensor)> {
        self.attn.qkv(&m.scale_shift(&self.norm1.forward(xs)?)?)
    }

    /// The two residual branches. Attention has already run over both streams at once, so
    /// what arrives is this stream's rows of that joint result: they are projected and gated
    /// back into the stream, then the feed-forward runs over a second modulation of the sum.
    ///
    /// `feed_forward` is asked for rather than chosen here, and it is the one step the two
    /// placements do not agree on: a block that owns its card runs the projection whole,
    /// while a block sharing a card with the rest of a split model runs it in slabs to stay
    /// inside a memory budget. Everything around it - which norm, which modulation, which
    /// order the two residuals are added in - is the same either way, and is stated once.
    pub(crate) fn residuals(
        &self,
        xs: &Tensor,
        attn: &Tensor,
        attn_mod: &ModulationOut,
        mlp_mod: &ModulationOut,
        feed_forward: impl FnOnce(&Mlp, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let xs = xs.add(&attn_mod.gate(&self.attn.proj.forward(attn)?)?)?;
        let ffn = feed_forward(self.mlp, &mlp_mod.scale_shift(&self.norm2.forward(&xs)?)?)?;
        xs.add(&mlp_mod.gate(&ffn)?)
    }
}

// ---------------------------------------------------------------------------
// Tests: each piece held against its DEFINITION, never against a second copy of itself.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Module as _;

    pub(super) fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The identity-keyed silu cache: hit on the same tensor, miss (and
    /// correct recompute) on a different one - including the ABA-style case
    /// where the old key is dropped before the new tensor is created.
    #[test]
    fn vec_silu_cache_correctness() {
        let check = |t: &Tensor| {
            let got = vec_silu_cached(t).unwrap().to_vec_f32();
            let want = t.silu().unwrap().to_vec_f32();
            assert_eq!(got, want);
        };
        let a = Tensor::from_vec_f32(data(16, 21), vec![2, 8]).unwrap();
        check(&a);
        check(&a); // cached hit, same values
        drop(a); // cache entry still pins the old storage (no address reuse)
        let b = Tensor::from_vec_f32(data(16, 22), vec![2, 8]).unwrap();
        check(&b); // different content -> recompute
                   // metadata view shares storage AND dims-compare guards the hit shape
        let c = b.reshape(vec![16]).unwrap();
        check(&c);
    }

    /// The embedding held against its definition, not against another copy of itself.
    ///
    /// It used to be checked against a second implementation, which is now the only one - a
    /// test that compares a function with itself passes whatever either does. What the
    /// function IS: position `i` of a step `t` carries `cos(1000.t.w_i)` and, half a row
    /// later, `sin` of the same, with `w_i = exp(-ln(10000).i/half)`.
    #[test]
    fn a_timestep_carries_the_cosines_and_sines_its_definition_says() {
        const HALF: usize = 16;
        let steps = [0.0f32, 0.25, 1.0];
        let got = timestep_embedding(
            &Tensor::from_vec_f32(steps.to_vec(), 3).unwrap(),
            2 * HALF,
            DType::F32,
        )
        .unwrap();
        assert_eq!(got.dims(), &[3, 2 * HALF]);
        let got = got.to_vec_f32();

        for (row, t) in steps.iter().enumerate() {
            for i in 0..HALF {
                let w = (-(10_000f32.ln()) * i as f32 / HALF as f32).exp();
                let arg = 1000.0 * t * w;
                for (half, want) in [(0, arg.cos()), (HALF, arg.sin())] {
                    let at = row * 2 * HALF + half + i;
                    assert!(
                        (got[at] - want).abs() < 1e-5,
                        "step {t} position {i}: {} is not {want}",
                        got[at]
                    );
                }
            }
        }
    }

    /// The rotary table held against its definition.
    ///
    /// There used to be two builders of one, checked against each other; there is one now, and
    /// this checks what it produces rather than that a pair agreed.
    ///
    /// An axis of width `d` contributes `d/2` rotations to each position: pair `i` turns by
    /// `p . theta^(-2i/d)`, laid out as the 2x2 `[[cos, -sin], [sin, cos]]`. Axes are
    /// concatenated in order, so a position's row is as many pairs as the axes' widths sum to
    /// halved. Comparing the two implementations would pass whatever the pair agreed on.
    #[test]
    fn a_rope_table_holds_one_rotation_per_position_and_pair() {
        let (n, theta) = (5usize, 10_000usize);
        let axes = [4usize, 6, 6];
        let pairs: usize = axes.iter().map(|d| d / 2).sum();
        let ids: Vec<f32> = (0..n * axes.len()).map(|i| (i % 7) as f32).collect();
        let width: usize = axes.iter().sum();

        let ids_t = Tensor::from_vec_f32(ids.clone(), vec![1, n, axes.len()]).unwrap();
        let got = EmbedNd::new(width / 2, theta, axes.to_vec())
            .forward(&ids_t)
            .unwrap();
        assert_eq!(got.dims(), &[1, 1, n, pairs, 2, 2]);
        let got = got.to_vec_f32();

        for pos in 0..n {
            let mut pair = 0;
            for (axis, &d) in axes.iter().enumerate() {
                let p = ids[pos * axes.len() + axis] as f64;
                for i in 0..d / 2 {
                    let w = 1f64 / (theta as f64).powf((2 * i) as f64 / d as f64);
                    let (c, sn) = ((p * w).cos() as f32, (p * w).sin() as f32);
                    let at = ((pos * pairs + pair) * 2) * 2;
                    for (off, want) in [(0, c), (1, -sn), (2, sn), (3, c)] {
                        assert!(
                            (got[at + off] - want).abs() < 1e-5,
                            "position {pos} axis {axis} pair {i}: {} is not {want}",
                            got[at + off]
                        );
                    }
                    pair += 1;
                }
            }
        }
    }
}
