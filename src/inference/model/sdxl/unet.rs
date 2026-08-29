//! SDXL UNet: the epsilon-prediction denoiser, on the native substrate.
//!
//! Structure read from the checkpoint, not assumed (Raymnants 6.2):
//! ```text
//!   time_embed  320 -> 1280 -> 1280            label_emb  2816 -> 1280 -> 1280
//!   input  [conv_in, Res, Res, Down, Res+Attn(2), Res+Attn(2), Down,
//!           Res+Attn(10), Res+Attn(10)]
//!   middle [Res, Attn(10), Res]
//!   output [Res+Attn(10)]x2, Res+Attn(10)+Up, [Res+Attn(2)]x2, Res+Attn(2)+Up,
//!          Res, Res, Res
//!   out    GroupNorm(320) + conv3x3 320 -> 4
//! ```
//! Channels 320 / 640 / 1280; NO attention at the 320 level; attention head dim 64;
//! cross-attention context 2048 (the two CLIP towers concatenated).
//!
//! Details that change the image WITHOUT failing, so they were read off a working
//! reference rather than guessed:
//! - the sinusoidal embedding concatenates `[cos, sin]` in that order (the LDM
//!   convention; the diffusers one is the opposite and yields a plausible-but-wrong
//!   image);
//! - SDXL's spatial transformers use LINEAR `proj_in`/`proj_out`, not 1x1 convs;
//! - the feed-forward is GEGLU (`ff.net.0.proj` is twice as wide as the hidden);
//! - the micro-conditioning vector is `[pooled(1280), sinusoid_256 x 6]` = 2816, and
//!   it is ADDED to the timestep embedding.

use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention};
use crate::tensor::layer::{
    conv2d, group_norm, layer_norm, Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear,
};
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// Where the UNet lives inside an SDXL single-file checkpoint.
pub const CHECKPOINT_PREFIX: &str = "model.diffusion_model";

/// Latent channels in and out (SD's 4-wide latent).
pub const LATENT_CHANNELS: usize = 4;
/// Base channel count; the three stages are 1x, 2x, 4x.
pub(crate) const BASE: usize = 320;
/// Width of the timestep/label embedding the blocks are conditioned on.
pub(crate) const EMB_DIM: usize = 4 * BASE;
/// Attention head dimension is fixed; the head COUNT follows the stage width.
pub(crate) const HEAD_DIM: usize = 64;
/// Group count for every normalisation in the UNet.
const GROUPS: usize = 32;
/// Cross-attention context width: CLIP-L (768) + bigG (1280).
pub const CONTEXT_DIM: usize = 2048;
/// Width of each sinusoid in the size/crop micro-conditioning.
const MICRO_FREQ_DIM: usize = 256;
/// pooled(1280) + 6 sinusoids of 256.
pub const LABEL_DIM: usize = 1280 + 6 * MICRO_FREQ_DIM;

/// `[cos(t*freqs), sin(t*freqs)]`, the LDM order.
///
/// Used for BOTH the timestep and the six micro-conditioning integers, which is why
/// it takes the width as an argument.
pub fn sinusoidal(values: &[f32], dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let mut out = Vec::with_capacity(values.len() * dim);
    for v in values {
        let mut cos = Vec::with_capacity(half);
        let mut sin = Vec::with_capacity(half);
        for i in 0..half {
            let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = v * freq;
            cos.push(a.cos());
            sin.push(a.sin());
        }
        out.extend(cos);
        out.extend(sin);
    }
    Tensor::from_vec_f32(out, vec![values.len(), dim])
}

/// A residual block: two normed convolutions with the embedding injected between.
pub(crate) struct ResBlock {
    norm_in: GroupNorm,
    conv_in: Conv2d,
    emb_proj: Linear,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    /// 1x1 projection, present only when the channel count changes.
    skip: Option<Conv2d>,
    out_channels: usize,
    /// This block's path in the checkpoint, for locating its adapter.
    path: String,
}

impl ResBlock {
    fn new(in_c: usize, out_c: usize, vb: &VarBuilder) -> Result<Self> {
        let pad = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm_in: group_norm(GROUPS, in_c, 1e-5, &vb.pp("in_layers").pp("0"))?,
            conv_in: conv2d(in_c, out_c, 3, pad, &vb.pp("in_layers").pp("2"))?,
            emb_proj: linear_wb(out_c, EMB_DIM, &vb.pp("emb_layers").pp("1"))?,
            norm_out: group_norm(GROUPS, out_c, 1e-5, &vb.pp("out_layers").pp("0"))?,
            conv_out: conv2d(out_c, out_c, 3, pad, &vb.pp("out_layers").pp("3"))?,
            skip: if in_c == out_c {
                None
            } else {
                Some(conv2d(
                    in_c,
                    out_c,
                    1,
                    Default::default(),
                    &vb.pp("skip_connection"),
                )?)
            },
            out_channels: out_c,
            path: vb.prefix().to_string(),
        })
    }

    /// The same block, loaded from the DIFFUSERS names.
    ///
    /// ControlNet checkpoints are published in that layout while our UNet reads the
    /// single-file LDM one. The two describe the same computation - norm, conv, a
    /// per-channel time embedding, norm, conv, and a 1x1 shortcut when the width
    /// changes - so what differs is only where each tensor is stored. Rather than write
    /// a second block, this constructor fills the same struct from the other names, and
    /// the forward below stays the single implementation of the math.
    pub(crate) fn new_diffusers(in_c: usize, out_c: usize, vb: &VarBuilder) -> Result<Self> {
        let pad = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm_in: group_norm(GROUPS, in_c, 1e-5, &vb.pp("norm1"))?,
            conv_in: conv2d(in_c, out_c, 3, pad, &vb.pp("conv1"))?,
            emb_proj: linear_wb(out_c, EMB_DIM, &vb.pp("time_emb_proj"))?,
            norm_out: group_norm(GROUPS, out_c, 1e-5, &vb.pp("norm2"))?,
            conv_out: conv2d(out_c, out_c, 3, pad, &vb.pp("conv2"))?,
            skip: if in_c == out_c {
                None
            } else {
                Some(conv2d(
                    in_c,
                    out_c,
                    1,
                    Default::default(),
                    &vb.pp("conv_shortcut"),
                )?)
            },
            out_channels: out_c,
            path: vb.prefix().to_string(),
        })
    }

    /// Adapt the time-embedding projection.
    ///
    /// The only LINEAR weight in a residual block; its two convolutions carry adapters
    /// of their own that this does not attach - a conv delta is a different shape and
    /// merging one is separate work. The count returned is what was ACTUALLY attached,
    /// so the caller can say how much of the file went unused instead of implying all
    /// of it landed.
    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        if let Some(diffusers) = ldm_to_diffusers(&format!("{}.time_emb_proj", self.path)) {
            if let Some(d) = file.delta_for(&diffusers, strength)? {
                self.emb_proj.add_lora(d)?;
                n += 1;
            }
        }
        let convs: [(&str, Option<&mut Conv2d>); 3] = [
            ("conv1", Some(&mut self.conv_in)),
            ("conv2", Some(&mut self.conv_out)),
            ("conv_shortcut", self.skip.as_mut()),
        ];
        for (leaf, conv) in convs {
            let Some(conv) = conv else { continue };
            let Some(diffusers) = ldm_to_diffusers(&format!("{}.{leaf}", self.path)) else {
                continue;
            };
            if let Some(d) = file.conv_delta_for(&diffusers, strength)? {
                conv.add_lora(d)?;
                n += 1;
            }
        }
        Ok(n)
    }

    fn clear_lora(&mut self) {
        self.emb_proj.clear_lora();
        self.conv_in.clear_lora();
        self.conv_out.clear_lora();
        if let Some(s) = self.skip.as_mut() {
            s.clear_lora();
        }
    }

    /// `xs`: `[b, c, h, w]`, `emb`: `[b, EMB_DIM]`.
    pub(crate) fn forward(&self, xs: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let h = self.conv_in.forward(&self.norm_in.forward(xs)?.silu()?)?;
        // The embedding is projected per output channel and broadcast over h, w.
        let (b, _, _, _) = h.shape().dims4()?;
        let e = self
            .emb_proj
            .forward(&emb.silu()?)?
            .reshape(vec![b, self.out_channels, 1, 1])?;
        let h = h.broadcast_add(&e)?;
        let h = self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)?;
        match &self.skip {
            Some(s) => s.forward(xs)?.add(&h),
            None => xs.add(&h),
        }
    }
}

/// One attention over flattened spatial tokens, read off this checkpoint's names.
///
/// `ctx_dim` is the width of whatever the keys and values are read from: the tokens
/// themselves for self-attention, and the text context for cross-attention, which is the
/// only thing that separates the two here. Nothing is kept between calls - a render's
/// context changes with every prompt and with each side of the guidance pair - so both
/// operands are projected on each forward.
fn attention(dim: usize, ctx_dim: usize, vb: &VarBuilder) -> Result<MultiHeadAttention> {
    let heads = dim / HEAD_DIM;
    MultiHeadAttention::new(
        // q/k/v carry NO bias in this architecture; only the output does.
        linear_no_bias(dim, dim, &vb.pp("to_q"))?,
        linear_no_bias(dim, ctx_dim, &vb.pp("to_k"))?,
        linear_no_bias(dim, ctx_dim, &vb.pp("to_v"))?,
        // The output projection is stored one level down, as `to_out.0`.
        linear_wb(dim, dim, &vb.pp("to_out").pp("0"))?,
        dim,
        heads,
        heads,
        Kv::None,
    )
}

/// GEGLU feed-forward: one projection twice as wide, gated by its own second half.
struct FeedForward {
    proj: Linear,
    out: Linear,
    hidden: usize,
    /// This module's path in the checkpoint, recorded at load so an adapter can find
    /// its two projections. The feed-forward is a THIRD of an SDXL adapter's linear
    /// entries and was not wired at all: attention alone is 560 of the 788 pairs in a
    /// stock LCM adapter, and applying part of a distillation is not applying it.
    path: String,
}

impl FeedForward {
    fn new(dim: usize, vb: &VarBuilder) -> Result<Self> {
        // `ff.net.0.proj` is [2*hidden, dim]; SDXL uses hidden = 4*dim.
        let hidden = 4 * dim;
        Ok(Self {
            proj: linear_wb(2 * hidden, dim, &vb.pp("net").pp("0").pp("proj"))?,
            out: linear_wb(dim, hidden, &vb.pp("net").pp("2"))?,
            hidden,
            path: vb.prefix().to_string(),
        })
    }

    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        for (suffix, lin) in [("net.0.proj", &mut self.proj), ("net.2", &mut self.out)] {
            let Some(diffusers) = ldm_to_diffusers(&format!("{}.{suffix}", self.path)) else {
                continue;
            };
            if let Some(d) = file.delta_for(&diffusers, strength)? {
                lin.add_lora(d)?;
                n += 1;
            }
        }
        Ok(n)
    }

    fn clear_lora(&mut self) {
        self.proj.clear_lora();
        self.out.clear_lora();
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(xs)?;
        let x = h.narrow(2, 0, self.hidden)?;
        let gate = h.narrow(2, self.hidden, self.hidden)?;
        self.out.forward(&x.mul(&gate.gelu()?)?)
    }
}

/// One transformer block: self-attention, cross-attention, feed-forward, each
/// pre-normed and residual.
struct TransformerBlock {
    norm1: LayerNorm,
    attn1: MultiHeadAttention,
    norm2: LayerNorm,
    attn2: MultiHeadAttention,
    norm3: LayerNorm,
    ff: FeedForward,
    /// This block's path in the checkpoint (`..._blocks.N.M.transformer_blocks.K`),
    /// recorded at load so an adapter can find the projections underneath it without
    /// reconstructing names.
    path: String,
}

impl TransformerBlock {
    /// The checkpoint leaf of each attention projection, in the order the attention hands
    /// its projections over. `to_out` is stored as `to_out.0`, and adapters follow the
    /// checkpoint, not the field name.
    const ATTN_LEAVES: [&'static str; 4] = ["to_q", "to_k", "to_v", "to_out_0"];

    fn new(dim: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-5, &vb.pp("norm1"))?,
            attn1: attention(dim, dim, &vb.pp("attn1"))?,
            norm2: layer_norm(dim, 1e-5, &vb.pp("norm2"))?,
            attn2: attention(dim, CONTEXT_DIM, &vb.pp("attn2"))?,
            norm3: layer_norm(dim, 1e-5, &vb.pp("norm3"))?,
            ff: FeedForward::new(dim, &vb.pp("ff"))?,
            path: vb.prefix().to_string(),
        })
    }

    fn forward(&self, xs: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        // Every token sees every other one, in both attentions: an image has no order to
        // read in, and the text context is whole before the first step.
        let h = self
            .attn1
            .forward_stateless(&self.norm1.forward(xs)?, None, Mask::All)?;
        let xs = xs.add(&h)?;
        let h = self
            .attn2
            .forward_stateless(&self.norm2.forward(&xs)?, Some(ctx), Mask::All)?;
        let xs = xs.add(&h)?;
        let h = self.ff.forward(&self.norm3.forward(&xs)?)?;
        xs.add(&h)
    }

    /// Attach a LoRA's corrections to this block's two attentions and its feed-forward.
    ///
    /// Returns how many matched. A LoRA trained for another architecture matches nothing,
    /// and silence there is the failure people actually hit - the render just looks
    /// unchanged - so the count travels back up to be reported.
    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        for (leaf, attn) in [("attn1", &mut self.attn1), ("attn2", &mut self.attn2)] {
            let Some(diffusers) = ldm_to_diffusers(&format!("{}.{leaf}", self.path)) else {
                continue;
            };
            for (suffix, lin) in Self::ATTN_LEAVES.into_iter().zip(attn.projections_mut()) {
                if let Some(d) = file.delta_for(&format!("{diffusers}.{suffix}"), strength)? {
                    lin.add_lora(d)?;
                    n += 1;
                }
            }
        }
        Ok(n + self.ff.apply_lora(file, strength)?)
    }

    fn clear_lora(&mut self) {
        for attn in [&mut self.attn1, &mut self.attn2] {
            for lin in attn.projections_mut() {
                lin.clear_lora();
            }
        }
        self.ff.clear_lora();
    }
}

/// A stack of transformer blocks wrapped in the spatial<->token reshape.
pub(crate) struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<TransformerBlock>,
    proj_out: Linear,
    /// This stack's path in the checkpoint: `proj_in` and `proj_out` are adapted too,
    /// and they are keyed on the stack rather than on any block inside it.
    path: String,
}

impl SpatialTransformer {
    pub(crate) fn new(dim: usize, depth: usize, vb: &VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);
        let vb_b = vb.pp("transformer_blocks");
        for i in 0..depth {
            blocks.push(TransformerBlock::new(dim, &vb_b.pp(i.to_string()))?);
        }
        Ok(Self {
            norm: group_norm(GROUPS, dim, 1e-6, &vb.pp("norm"))?,
            // SDXL stores these as LINEAR projections (`use_linear_in_transformer`).
            proj_in: linear_wb(dim, dim, &vb.pp("proj_in"))?,
            blocks,
            proj_out: linear_wb(dim, dim, &vb.pp("proj_out"))?,
            path: vb.prefix().to_string(),
        })
    }

    fn forward(&self, xs: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.shape().dims4()?;
        let residual = xs;
        let t = self.norm.forward(xs)?;
        // [b, c, h, w] -> [b, h*w, c]
        let t = t
            .reshape(vec![b, c, h * w])?
            .transpose(1, 2)?
            .contiguous()?;
        let mut t = self.proj_in.forward(&t)?;
        for blk in &self.blocks {
            t = blk.forward(&t, ctx)?;
        }
        let t = self.proj_out.forward(&t)?;
        let t = t.transpose(1, 2)?.contiguous()?.reshape(vec![b, c, h, w])?;
        residual.add(&t)
    }

    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        for b in &mut self.blocks {
            n += b.apply_lora(file, strength)?;
        }
        for (suffix, lin) in [
            ("proj_in", &mut self.proj_in),
            ("proj_out", &mut self.proj_out),
        ] {
            let Some(diffusers) = ldm_to_diffusers(&format!("{}.{suffix}", self.path)) else {
                continue;
            };
            if let Some(d) = file.delta_for(&diffusers, strength)? {
                lin.add_lora(d)?;
                n += 1;
            }
        }
        Ok(n)
    }

    fn clear_lora(&mut self) {
        for b in &mut self.blocks {
            b.clear_lora();
        }
        self.proj_in.clear_lora();
        self.proj_out.clear_lora();
    }
}

/// What a level of the UNet does to its input, in order.
pub(crate) enum Layer {
    Res(ResBlock),
    Attn(SpatialTransformer),
    /// Stride-2 3x3 convolution.
    Down(Conv2d),
    /// Nearest-neighbour 2x upsample followed by a 3x3 convolution.
    Up(Conv2d),
}

impl Layer {
    pub(crate) fn forward(&self, xs: &Tensor, emb: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        match self {
            Self::Res(r) => r.forward(xs, emb),
            Self::Attn(a) => a.forward(xs, ctx),
            Self::Down(c) => c.forward(xs),
            Self::Up(c) => {
                let (_, _, h, w) = xs.shape().dims4()?;
                c.forward(&xs.upsample_nearest2d(2 * h, 2 * w)?)
            }
        }
    }
    fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        match self {
            Layer::Attn(t) => t.apply_lora(file, strength),
            Layer::Res(r) => r.apply_lora(file, strength),
            // A resampler is one convolution, and diffusers names it `conv` under its
            // own list whatever the checkpoint called it (`op` on the way down).
            Layer::Down(c) | Layer::Up(c) => {
                let Some((parent, _)) = c.path().rsplit_once('.') else {
                    return Ok(0);
                };
                let Some(diffusers) = ldm_to_diffusers(&format!("{parent}.conv")) else {
                    return Ok(0);
                };
                match file.conv_delta_for(&diffusers, strength)? {
                    Some(d) => {
                        c.add_lora(d)?;
                        Ok(1)
                    }
                    None => Ok(0),
                }
            }
        }
    }

    fn clear_lora(&mut self) {
        match self {
            Layer::Attn(t) => t.clear_lora(),
            Layer::Res(r) => r.clear_lora(),
            Layer::Down(c) | Layer::Up(c) => c.clear_lora(),
        }
    }
}

/// The SDXL denoiser.
pub struct SdxlUnet {
    time_1: Linear,
    time_2: Linear,
    label_1: Linear,
    label_2: Linear,
    conv_in: Conv2d,
    /// Each entry is one `input_blocks.N`, whose output is pushed on the skip stack.
    input_blocks: Vec<Vec<Layer>>,
    middle: Vec<Layer>,
    /// Each entry is one `output_blocks.N`; it pops one skip and concatenates it.
    output_blocks: Vec<Vec<Layer>>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    device: Device,
    /// One device per STAGE, in forward order: the input blocks, then the middle, then
    /// the output blocks. The stem and the tail live on `device`.
    ///
    /// A UNet that only ever loads onto one card cannot use a second one, and when it
    /// fits none it falls to the host entirely - the worst of the three outcomes. This
    /// is the same contract the rest of the fleet follows, and its absence here was an
    /// omission, not a property of the architecture.
    stage_devices: Vec<Device>,
    /// The dtype the WEIGHTS are resident in. Everything handed to `forward` has to
    /// arrive in it: the sampler works in F32 on the host, so the latent, the
    /// timestep embedding and the micro-conditioning all need converting, and a
    /// mismatch surfaces as a bare "dtype mismatch" from deep inside a matmul.
    dtype: crate::tensor::DType,
}

/// Resident bytes of each STAGE, in forward order, from the specs that define them.
///
/// Stages are wildly uneven - a 1280-wide attention stage is two orders of magnitude
/// heavier than a 320-wide residual one - so packing them needs their real sizes.
/// Derived from the same tables the loader builds from, so the two cannot drift.
fn stage_weight_bytes(dtype: crate::tensor::DType) -> Vec<u64> {
    let elem = if dtype == crate::tensor::DType::F32 {
        4u64
    } else {
        2
    };
    // A residual block: two 3x3 convolutions, the per-channel time projection, and a
    // 1x1 shortcut when the width changes.
    let res = |in_c: usize, out_c: usize| -> u64 {
        let (i, o) = (in_c as u64, out_c as u64);
        9 * i * o + 9 * o * o + o * EMB_DIM as u64 + if i != o { i * o } else { 0 }
    };
    // A spatial transformer: the two projections, then per block a self-attention, a
    // cross-attention against the text context, and a GEGLU feed-forward.
    let attn = |dim: usize, depth: usize| -> u64 {
        let d = dim as u64;
        2 * d * d + depth as u64 * (18 * d * d + 2 * d * CONTEXT_DIM as u64)
    };
    let conv3 = |c: usize| -> u64 { 9 * (c as u64) * (c as u64) };

    let mut out = Vec::with_capacity(INPUT_SPEC.len() + 1 + SKIP_SPEC.len());
    let mut width = BASE;
    for (in_c, out_c, depth) in INPUT_SPEC.iter() {
        if *in_c == 0 && *out_c == 0 && *depth == 0 {
            out.push(conv3(width) * elem);
            continue;
        }
        out.push((res(*in_c, *out_c) + attn(*out_c, *depth)) * elem);
        width = *out_c;
    }
    let mid = 4 * BASE;
    out.push((2 * res(mid, mid) + attn(mid, 10)) * elem);
    for (skip_c, in_c, depth, up) in SKIP_SPEC.iter() {
        let mut b = res(skip_c + in_c, *in_c) + attn(*in_c, *depth);
        if *up {
            b += conv3(*in_c);
        }
        out.push(b * elem);
    }
    out
}

/// Greedily pack the stages onto `devices`, in order, falling to the host only for the
/// stages that fit nowhere.
///
/// In forward order on purpose: consecutive stages that share a card cost no transfer, so
/// staying on the card the previous stage used yields the fewest crossings for a given
/// split. With no budgets everything lands on the first device, which reproduces the
/// single-device placement exactly.
///
/// THE HOST IS FOR STAGES NO CARD HOLDS, and only those. This walked a cursor that never
/// went back: the first stage too large for the card it was on advanced the cursor, and
/// once the cursor ran off the end it stayed there - so a single oversized stage sent
/// every stage AFTER it to the processor, including the small ones, with gigabytes still
/// free on the cards the cursor had already passed. The deepest levels of this UNet are
/// twenty times the shallowest, so "one stage does not fit here" is the normal case, not
/// the exception. A stage is now offered the previous card first and then every card,
/// fastest first, before the host is considered at all.
fn plan_stages(weights: &[u64], devices: &[Device], budgets: &[u64]) -> Vec<Device> {
    let fallback = devices.first().cloned().unwrap_or(Device::Cpu);
    if budgets.is_empty() || devices.is_empty() {
        return vec![fallback; weights.len()];
    }
    plan_stage_slots(weights, budgets)
        .into_iter()
        .map(|slot| match slot {
            Some(d) => devices.get(d).cloned().unwrap_or(Device::Cpu),
            None => Device::Cpu,
        })
        .collect()
}

/// The packing decision alone: which card each stage lands on, by index into the
/// fastest-first budget list, or `None` for the host.
///
/// THE BODY MOVED, and only the body: this is now the one placement rule the repository
/// has, so a UNet stage and a transformer block are packed by the same code against the
/// same invariant. It was the general form already - a weight per element, a budget per
/// card - which is why it is the one that survived; what it lacked was callers.
pub(crate) fn plan_stage_slots(weights: &[u64], budgets: &[u64]) -> Vec<Option<usize>> {
    crate::inference::place::plan::place(weights, budgets)
}

/// Residual blocks per resolution level, the constant that sets how the checkpoint's
/// flat block numbering groups into levels. Both `INPUT_SPEC` and `SKIP_SPEC` are
/// written with this grouping; `the_specs_group_by_level` holds them to it.
const RES_PER_LEVEL: usize = 2;

/// Translate an attention module path from the single-file (LDM) layout this UNet loads
/// into the diffusers name that LoRA files are keyed on.
///
/// The two layouts name the same graph differently: the checkpoint numbers every block
/// in one flat sequence (`input_blocks.4.1`), diffusers groups them by resolution level
/// (`down_blocks.1.attentions.0`). Adapters are published against the diffusers names
/// whatever layout the base checkpoint uses, so a server that loads single-file
/// checkpoints has to bridge the two or match nothing at all - a LoRA that appears to
/// load and changes nothing.
///
/// The grouping is mechanical: each level holds `RES_PER_LEVEL` residual blocks plus one
/// resampler, so a flat index divides into `(level, slot)`. Derived rather than
/// tabulated, so the two cannot drift apart when the topology tables change.
fn ldm_to_diffusers(path: &str) -> Option<String> {
    let stride = RES_PER_LEVEL + 1;
    // Everything before the block marker is the checkpoint wrapper prefix, which the
    // diffusers name does not carry.
    let (marker, rest) = ["input_blocks.", "middle_block.", "output_blocks."]
        .into_iter()
        .find_map(|m| path.find(m).map(|i| (m, &path[i + m.len()..])))?;
    if marker == "middle_block." {
        let (sub, tail) = rest.split_once('.')?;
        // The middle block is resnet, attention, resnet.
        return Some(match sub {
            "0" => format!("mid_block.resnets.0.{tail}"),
            "1" => format!("mid_block.attentions.0.{tail}"),
            "2" => format!("mid_block.resnets.1.{tail}"),
            _ => return None,
        });
    }
    let (idx, rest) = rest.split_once('.')?;
    let n: usize = idx.parse().ok()?;
    // Sub-index 0 is the residual block, 1 the attention stack. Diffusers splits what
    // LDM numbers together into separate named lists, which is the whole of the
    // difference between the two conventions.
    let (sub, tail) = rest.split_once('.')?;
    let (group, flat) = if marker == "input_blocks." {
        // Index 0 is the input convolution, so the levels start one later.
        ("down", n.checked_sub(1)?)
    } else {
        ("up", n)
    };
    let (level, slot) = (flat / stride, flat % stride);
    let addressed = match sub {
        // A down level is residual, residual, resample - so its LAST slot is the
        // resampler, which diffusers keeps in a list of its own.
        "0" if marker == "input_blocks." && slot == stride - 1 => {
            format!("{group}_blocks.{level}.downsamplers.0")
        }
        "0" => format!("{group}_blocks.{level}.resnets.{slot}"),
        "1" => format!("{group}_blocks.{level}.attentions.{slot}"),
        // An up level resamples AFTER its residual and attention, so the resampler is a
        // third sub-index rather than the last slot.
        "2" if marker == "output_blocks." => format!("{group}_blocks.{level}.upsamplers.0"),
        _ => return None,
    };
    Some(format!("{addressed}.{tail}"))
}

/// `(in, out)` channel pairs and attention depth per level, as read from the
/// checkpoint. Kept as data so the shape of the net is inspectable in one place.
pub(crate) const INPUT_SPEC: [(usize, usize, usize); 8] = [
    // (in, out, attn depth); depth 0 = no attention at this level
    (BASE, BASE, 0),
    (BASE, BASE, 0),
    (0, 0, 0), // downsample marker
    (BASE, 2 * BASE, 2),
    (2 * BASE, 2 * BASE, 2),
    (0, 0, 0), // downsample marker
    (2 * BASE, 4 * BASE, 10),
    (4 * BASE, 4 * BASE, 10),
];

/// Every attention stage: `(level index, width, transformer depth)`.
///
/// A UNet with spatial transformers in it is not sized by its feature maps alone -
/// at the resolutions this runs at the ATTENTION dominates, by an order of
/// magnitude. Sizing on the convolution stack only asked for a fraction of what a
/// render takes, which is the under-reserve this whole exercise exists to prevent.
pub fn attention_levels() -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut level = 0usize;
    for (in_c, out_c, depth) in INPUT_SPEC.iter() {
        if *in_c == 0 && *out_c == 0 && *depth == 0 {
            level += 1;
            continue;
        }
        if *depth > 0 {
            out.push((level, *out_c, *depth));
        }
    }
    out
}

/// The channel width of each RESOLUTION LEVEL, outermost first.
///
/// The skip list repeats a width once per block at that level; what sizing a feature
/// map needs is the distinct widths in depth order, since each level's map is half
/// the side of the one before it. Derived from the same walk so a change to the
/// architecture cannot leave the two disagreeing.
pub fn level_channels() -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for w in skip_widths() {
        if out.last() != Some(&w) {
            out.push(w);
        }
    }
    out
}

/// The widths the nine skips carry, in PUSH order.
///
/// Derived from `INPUT_SPEC` - the DOWN path, which is what pushes them - and not from
/// `SKIP_SPEC`. That was my first guess and it is wrong: `SKIP_SPEC` records what each
/// OUTPUT block concatenates, which is not the reverse of the push order, because a
/// block at a resolution boundary takes its skip from the level above. The checkpoint
/// settled it: a real ControlNet's nine residual projections are
/// 320,320,320,320,640,640,640,1280,1280, which is this walk and not that reversal.
///
/// ControlNet adds one residual per skip and must agree with this exactly, so it is
/// computed once here rather than written down twice.
pub fn skip_widths() -> Vec<usize> {
    // conv_in pushes first, at the base width.
    let mut v = vec![BASE];
    let mut cur = BASE;
    for (in_c, out_c, depth) in INPUT_SPEC.iter() {
        if *in_c == 0 && *out_c == 0 && *depth == 0 {
            // A downsample: it pushes, at the width it was already carrying.
            v.push(cur);
        } else {
            cur = *out_c;
            v.push(cur);
        }
    }
    // NINE, and no pop. Every input block pushes, and so does conv_in before them; the
    // middle block READS the last skip rather than replacing it. I trimmed one here on
    // the assumption that the final output fed the middle instead of the stack, which
    // left eight against the checkpoint's nine - a shape mismatch that would have
    // silently paired every residual with the wrong stage.
    v
}

/// Skip channel widths the output blocks concatenate, in pop order.
const SKIP_SPEC: [(usize, usize, usize, bool); 9] = [
    // (skip_c, in_c, attn depth, has upsample)
    (4 * BASE, 4 * BASE, 10, false),
    (4 * BASE, 4 * BASE, 10, false),
    (2 * BASE, 4 * BASE, 10, true),
    (4 * BASE, 2 * BASE, 2, false),
    (2 * BASE, 2 * BASE, 2, false),
    (BASE, 2 * BASE, 2, true),
    (2 * BASE, BASE, 0, false),
    (BASE, BASE, 0, false),
    (BASE, BASE, 0, false),
];

impl SdxlUnet {
    /// Attach a LoRA to every attention in the net.
    ///
    /// Returns the number of projections that matched, so a caller can refuse an
    /// adapter that matched NOTHING - the common failure (an SD1.5 LoRA on SDXL), and
    /// one that is otherwise completely silent: the image simply comes out unchanged
    /// and the user concludes the feature does not work.
    pub fn apply_lora(
        &mut self,
        file: &crate::inference::load::lora::LoraFile,
        strength: f32,
    ) -> Result<usize> {
        let mut n = 0;
        for blk in self
            .input_blocks
            .iter_mut()
            .chain(std::iter::once(&mut self.middle))
        {
            for l in blk.iter_mut() {
                n += l.apply_lora(file, strength)?;
            }
        }
        for blk in self.output_blocks.iter_mut() {
            for l in blk.iter_mut() {
                n += l.apply_lora(file, strength)?;
            }
        }
        Ok(n)
    }

    /// Drop every attached adapter, returning the net to the base checkpoint.
    pub fn clear_lora(&mut self) {
        for blk in self
            .input_blocks
            .iter_mut()
            .chain(std::iter::once(&mut self.middle))
        {
            for l in blk.iter_mut() {
                l.clear_lora();
            }
        }
        for blk in self.output_blocks.iter_mut() {
            for l in blk.iter_mut() {
                l.clear_lora();
            }
        }
    }

    pub fn load(checkpoint: &str, device: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        Self::load_planned(checkpoint, std::slice::from_ref(device), &[], dtype)
    }

    /// Load with the stages SPREAD over `devices`, fastest first.
    ///
    /// `budgets` is the usable bytes on each of those devices, in the same order; an
    /// empty slice means "put everything on the first one", which is what the
    /// single-device door above asks for and is bit-identical to the old behaviour.
    /// Any stage that does not fit the cards falls to the host - a stage at a time,
    /// not the whole network.
    pub fn load_planned(
        checkpoint: &str,
        devices: &[Device],
        budgets: &[u64],
        dtype: crate::tensor::DType,
    ) -> Result<Self> {
        let primary = devices.first().cloned().unwrap_or(Device::Cpu);
        let weights = stage_weight_bytes(dtype);
        let stage_devices = plan_stages(&weights, devices, budgets);
        {
            let mut counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for d in &stage_devices {
                *counts.entry(format!("{:?}", d.location())).or_default() += 1;
            }
            tracing::info!(
                "SDXL UNet: {} stages placed as {:?}",
                stage_devices.len(),
                counts
            );
        }
        // One VarBuilder per distinct device, built once and reused by every stage
        // that landed on it.
        let mut builders: Vec<(Device, VarBuilder)> = Vec::new();
        let mut builder_for = |dev: &Device| -> Result<VarBuilder> {
            if let Some((_, vb)) = builders.iter().find(|(d, _)| d.same_device(dev)) {
                return Ok(vb.pp(CHECKPOINT_PREFIX));
            }
            let vb = unsafe { VarBuilder::from_files(&[checkpoint], dtype, dev) }?;
            builders.push((dev.clone(), vb));
            Ok(builders.last().unwrap().1.pp(CHECKPOINT_PREFIX))
        };
        let device = &primary;
        let vb = builder_for(device)?;
        let pad = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let mut input_blocks: Vec<Vec<Layer>> = Vec::with_capacity(8);
        for (i, (in_c, out_c, depth)) in INPUT_SPEC.iter().enumerate() {
            let vb_in = builder_for(&stage_devices[i])?.pp("input_blocks");
            // The block index in the file is offset by one: index 0 is conv_in.
            let vb_b = vb_in.pp((i + 1).to_string());
            if *depth == 0 && *in_c == 0 {
                // Downsample level: a single strided conv under `.0.op`.
                let c = if i < 3 { BASE } else { 2 * BASE };
                let stride2 = Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                input_blocks.push(vec![Layer::Down(conv2d(
                    c,
                    c,
                    3,
                    stride2,
                    &vb_b.pp("0").pp("op"),
                )?)]);
                continue;
            }
            let mut layers = vec![Layer::Res(ResBlock::new(*in_c, *out_c, &vb_b.pp("0"))?)];
            if *depth > 0 {
                layers.push(Layer::Attn(SpatialTransformer::new(
                    *out_c,
                    *depth,
                    &vb_b.pp("1"),
                )?));
            }
            input_blocks.push(layers);
        }

        let vb_mid = builder_for(&stage_devices[INPUT_SPEC.len()])?.pp("middle_block");
        let middle = vec![
            Layer::Res(ResBlock::new(4 * BASE, 4 * BASE, &vb_mid.pp("0"))?),
            Layer::Attn(SpatialTransformer::new(4 * BASE, 10, &vb_mid.pp("1"))?),
            Layer::Res(ResBlock::new(4 * BASE, 4 * BASE, &vb_mid.pp("2"))?),
        ];

        let mut output_blocks: Vec<Vec<Layer>> = Vec::with_capacity(9);
        for (i, (skip_c, in_c, depth, up)) in SKIP_SPEC.iter().enumerate() {
            let vb_out = builder_for(&stage_devices[INPUT_SPEC.len() + 1 + i])?.pp("output_blocks");
            let vb_b = vb_out.pp(i.to_string());
            let mut layers = vec![Layer::Res(ResBlock::new(
                skip_c + in_c,
                *in_c,
                &vb_b.pp("0"),
            )?)];
            if *depth > 0 {
                layers.push(Layer::Attn(SpatialTransformer::new(
                    *in_c,
                    *depth,
                    &vb_b.pp("1"),
                )?));
            }
            if *up {
                // The upsample conv sits at index 2 when there is attention, else 1.
                let at = if *depth > 0 { "2" } else { "1" };
                layers.push(Layer::Up(conv2d(
                    *in_c,
                    *in_c,
                    3,
                    pad,
                    &vb_b.pp(at).pp("conv"),
                )?));
            }
            output_blocks.push(layers);
        }

        Ok(Self {
            stage_devices,
            time_1: linear_wb(EMB_DIM, BASE, &vb.pp("time_embed").pp("0"))?,
            time_2: linear_wb(EMB_DIM, EMB_DIM, &vb.pp("time_embed").pp("2"))?,
            label_1: linear_wb(EMB_DIM, LABEL_DIM, &vb.pp("label_emb").pp("0").pp("0"))?,
            label_2: linear_wb(EMB_DIM, EMB_DIM, &vb.pp("label_emb").pp("0").pp("2"))?,
            conv_in: conv2d(
                LATENT_CHANNELS,
                BASE,
                3,
                pad,
                &vb.pp("input_blocks").pp("0").pp("0"),
            )?,
            input_blocks,
            middle,
            output_blocks,
            norm_out: group_norm(GROUPS, BASE, 1e-5, &vb.pp("out").pp("0"))?,
            conv_out: conv2d(BASE, LATENT_CHANNELS, 3, pad, &vb.pp("out").pp("2"))?,
            device: device.clone(),
            dtype,
        })
    }

    /// SDXL's micro-conditioning vector: the bigG pooled embedding followed by
    /// sinusoids of (original h, original w, crop top, crop left, target h, target w).
    pub fn label_vector(&self, pooled: &Tensor, sizes: [f32; 6]) -> Result<Tensor> {
        let micro = sinusoidal(&sizes, MICRO_FREQ_DIM)?
            .reshape(vec![1, 6 * MICRO_FREQ_DIM])?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let pooled = pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        Tensor::cat(&[&pooled, &micro], 1)
    }

    /// The dtype callers must hand this UNet.
    pub fn dtype(&self) -> crate::tensor::DType {
        self.dtype
    }

    /// One denoise step: predict the noise in `x` at timestep `t`.
    ///
    /// `x`: `[1, 4, h/8, w/8]`, `ctx`: `[1, 77, 2048]`, `label`: `[1, 2816]`.
    pub fn forward(&self, x: &Tensor, t: f32, ctx: &Tensor, label: &Tensor) -> Result<Tensor> {
        self.forward_controlled(x, t, ctx, label, None)
    }

    /// The same forward with STRUCTURAL residuals added into the skips.
    ///
    /// A prompt says what to draw and cannot say where the limbs are, which is why
    /// "too many legs" survives more steps and more guidance - nothing in the
    /// conditioning constrains geometry. A ControlNet reads a pose image and emits one
    /// residual per skip; adding them is what imposes the structure.
    ///
    /// `control` is `None` for an ordinary render, and then this is the forward above
    /// with no arithmetic added - the zero-initialised projections would contribute
    /// nothing anyway, but not running them at all is cheaper and clearer.
    pub fn forward_controlled(
        &self,
        x: &Tensor,
        t: f32,
        ctx: &Tensor,
        label: &Tensor,
        control: Option<&crate::inference::model::sdxl::controlnet::ControlResiduals>,
    ) -> Result<Tensor> {
        let t_emb = sinusoidal(&[t], BASE)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let emb = self.time_2.forward(&self.time_1.forward(&t_emb)?.silu()?)?;
        let ctx = ctx.to_device(&self.device)?.to_dtype(self.dtype)?;
        let label = label.to_device(&self.device)?.to_dtype(self.dtype)?;
        let y = self
            .label_2
            .forward(&self.label_1.forward(&label)?.silu()?)?;
        let emb = emb.add(&y)?;

        let mut h = self
            .conv_in
            .forward(&x.to_device(&self.device)?.to_dtype(self.dtype)?)?;
        // The skip stack: conv_in's output first, then one per input block. Each skip
        // is kept ON THE CARD THAT PRODUCED IT and moved when it is consumed - the
        // output block that pops it can be on another device, and it is the only
        // tensor here whose producer and consumer are decided independently.
        let mut skips = vec![h.clone()];
        // The conditioning is small; carrying it to each stage costs a few hundred
        // kilobytes and keeps the block code device-agnostic.
        let stage = |i: usize| -> &Device { &self.stage_devices[i] };
        for (i, block) in self.input_blocks.iter().enumerate() {
            let dev = stage(i);
            h = h.to_device(dev)?;
            let (e, c) = (emb.to_device(dev)?, ctx.to_device(dev)?);
            for layer in block {
                h = layer.forward(&h, &e, &c)?;
            }
            skips.push(h.clone());
        }
        // ADD THE RESIDUALS to the skips they belong to, each on the card that will
        // consume it. The stack is built here, and the output blocks that pop it may
        // sit on another device - moving the residual at injection keeps the addition
        // where its operands already are.
        if let Some(c) = control {
            if c.down.len() != skips.len() {
                return Err(crate::tensor::Error(format!(
                    "controlnet produced {} residuals for {} skips",
                    c.down.len(),
                    skips.len()
                )));
            }
            for (skip, res) in skips.iter_mut().zip(c.down.iter()) {
                let dev = skip.device().clone();
                *skip = skip.add(&res.to_device(&dev)?)?;
            }
        }
        {
            let dev = stage(self.input_blocks.len());
            h = h.to_device(dev)?;
            let (e, c) = (emb.to_device(dev)?, ctx.to_device(dev)?);
            for layer in &self.middle {
                h = layer.forward(&h, &e, &c)?;
            }
            if let Some(ctrl) = control {
                let hd = h.device().clone();
                h = h.add(&ctrl.mid.to_device(&hd)?)?;
            }
        }
        let out_base = self.input_blocks.len() + 1;
        for (i, block) in self.output_blocks.iter().enumerate() {
            let dev = stage(out_base + i);
            let skip = skips
                .pop()
                .ok_or_else(|| crate::tensor::Error("sdxl unet: skip stack underflow".into()))?;
            h = h.to_device(dev)?;
            h = Tensor::cat(&[&h, &skip.to_device(dev)?], 1)?;
            let (e, c) = (emb.to_device(dev)?, ctx.to_device(dev)?);
            for layer in block {
                h = layer.forward(&h, &e, &c)?;
            }
        }
        // The tail lives with the stem, so the caller always gets its result back on
        // the device it handed the latent in on.
        let h = h.to_device(&self.device)?;
        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// `{prefix}.weight` `[out, in]` + `{prefix}.bias`.
pub(crate) fn linear_wb(out_dim: usize, in_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = vb.get(out_dim, "bias")?;
    Linear::new(w, Some(b))
}

/// `{prefix}.weight` only - the attention q/k/v projections carry no bias here.
fn linear_no_bias(out_dim: usize, in_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    Linear::new(vb.get((out_dim, in_dim), "weight")?, None)
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Every attention stack this UNet owns, mapped to the diffusers name a LoRA is
    /// keyed on. The expected column is the reference's own derivation
    /// (its `unet_to_diffusers` walk over the SDXL topology), transcribed rather than
    /// re-derived here - so this test compares two independent constructions.
    ///
    /// A wrong entry does not fail loudly: the adapter loads, matches fewer modules
    /// than it should, and renders an image that is subtly not what the LoRA trains.
    #[test]
    fn every_attention_maps_to_its_diffusers_name() {
        let expected = [
            ("input_blocks.4.1", "down_blocks.1.attentions.0"),
            ("input_blocks.5.1", "down_blocks.1.attentions.1"),
            ("input_blocks.7.1", "down_blocks.2.attentions.0"),
            ("input_blocks.8.1", "down_blocks.2.attentions.1"),
            ("middle_block.1", "mid_block.attentions.0"),
            ("output_blocks.0.1", "up_blocks.0.attentions.0"),
            ("output_blocks.1.1", "up_blocks.0.attentions.1"),
            ("output_blocks.2.1", "up_blocks.0.attentions.2"),
            ("output_blocks.3.1", "up_blocks.1.attentions.0"),
            ("output_blocks.4.1", "up_blocks.1.attentions.1"),
            ("output_blocks.5.1", "up_blocks.1.attentions.2"),
        ];
        for (ldm, diffusers) in expected {
            let got = ldm_to_diffusers(&format!(
                "{CHECKPOINT_PREFIX}.{ldm}.transformer_blocks.0.attn1"
            ));
            assert_eq!(
                got.as_deref(),
                Some(format!("{diffusers}.transformer_blocks.0.attn1").as_str()),
                "{ldm}"
            );
        }
    }

    /// Residual blocks address the OTHER diffusers list, and the middle block holds one
    /// on each side of its attention.
    #[test]
    fn residual_blocks_map_to_their_own_list() {
        for (ldm, diffusers) in [
            ("input_blocks.1.0", "down_blocks.0.resnets.0"),
            ("input_blocks.2.0", "down_blocks.0.resnets.1"),
            ("input_blocks.4.0", "down_blocks.1.resnets.0"),
            ("output_blocks.0.0", "up_blocks.0.resnets.0"),
            ("output_blocks.3.0", "up_blocks.1.resnets.0"),
            ("middle_block.0", "mid_block.resnets.0"),
            ("middle_block.2", "mid_block.resnets.1"),
        ] {
            let got = ldm_to_diffusers(&format!("{CHECKPOINT_PREFIX}.{ldm}.time_emb_proj"));
            assert_eq!(
                got.as_deref(),
                Some(format!("{diffusers}.time_emb_proj").as_str()),
                "{ldm}"
            );
        }
    }

    /// The resamplers live in lists of their own, and the two directions place them
    /// differently: last slot of a down level, third sub-index of an up one. Getting
    /// this wrong produces a plausible `resnets.2` that matches nothing.
    #[test]
    fn resamplers_map_to_their_own_lists() {
        for (ldm, diffusers) in [
            ("input_blocks.3.0.op", "down_blocks.0.downsamplers.0"),
            ("input_blocks.6.0.op", "down_blocks.1.downsamplers.0"),
            ("output_blocks.2.2", "up_blocks.0.upsamplers.0"),
            ("output_blocks.5.2", "up_blocks.1.upsamplers.0"),
        ] {
            // The checkpoint leaf is dropped by the caller; the diffusers leaf is `conv`.
            let ldm = ldm.strip_suffix(".op").unwrap_or(ldm);
            let got = ldm_to_diffusers(&format!("{CHECKPOINT_PREFIX}.{ldm}.conv"));
            assert_eq!(
                got.as_deref(),
                Some(format!("{diffusers}.conv").as_str()),
                "{ldm}"
            );
        }
    }

    /// Anything outside the block lists has no diffusers address at all. Inside them the
    /// mapping translates the ADDRESS and trusts the leaf: a leaf that names nothing
    /// simply matches no adapter key, which is not a case worth inventing a name for.
    #[test]
    fn paths_outside_the_blocks_map_to_nothing() {
        for p in [
            "model.diffusion_model.out.0",
            "model.diffusion_model.time_embed.0",
        ] {
            assert_eq!(ldm_to_diffusers(p), None, "{p}");
        }
    }

    /// The level grouping `ldm_to_diffusers` divides by must be the grouping the
    /// topology tables are actually written in: resamplers land in the last slot of
    /// each level. If a table changes shape, the division silently starts producing
    /// wrong names - so tie them together.
    #[test]
    fn the_specs_group_by_level() {
        let stride = RES_PER_LEVEL + 1;
        for (i, (in_c, out_c, depth)) in INPUT_SPEC.iter().enumerate() {
            let is_resampler = *in_c == 0 && *out_c == 0 && *depth == 0;
            assert_eq!(is_resampler, i % stride == stride - 1, "INPUT_SPEC[{i}]");
        }
        for (i, (.., up)) in SKIP_SPEC.iter().enumerate() {
            // The last level has no upsample: there is nothing after it.
            let expect_up = i % stride == stride - 1 && i + 1 < SKIP_SPEC.len();
            assert_eq!(*up, expect_up, "SKIP_SPEC[{i}]");
        }
    }

    /// The label vector's width is fixed by the architecture: getting it wrong makes
    /// `label_emb` reject its input, which is the cheap failure. Pin it.
    #[test]
    fn the_label_vector_is_2816_wide() {
        assert_eq!(LABEL_DIM, 2816);
        assert_eq!(CONTEXT_DIM, 2048);
    }

    /// The LDM sinusoid puts COSINE first. A sin-first embedding is the diffusers
    /// convention and produces a plausible but wrong image, so assert the order.
    #[test]
    fn the_sinusoid_is_cosine_first() {
        let e = sinusoidal(&[0.0], 8).unwrap().to_vec_f32();
        // At t = 0 every cosine is 1 and every sine is 0.
        assert_eq!(
            &e[..4],
            &[1.0, 1.0, 1.0, 1.0],
            "the first half must be cosines"
        );
        assert_eq!(
            &e[4..],
            &[0.0, 0.0, 0.0, 0.0],
            "the second half must be sines"
        );
    }

    /// Frequencies must DECREASE across the half-width (exp(-log(1e4) i / half)).
    #[test]
    fn the_sinusoid_frequencies_decay() {
        let e = sinusoidal(&[1.0], 16).unwrap().to_vec_f32();
        // sin(t*freq) for decreasing freq at t=1: the first is the fastest.
        let sines = &e[8..];
        assert!(sines[0].abs() > 0.0);
        // The last frequency is ~exp(-log(1e4)*7/8) ~ 1.3e-3, so its sine is tiny.
        assert!(
            sines[7].abs() < 0.01,
            "the slowest frequency should barely move"
        );
    }

    /// The skip stack must be consumed exactly: 1 (conv_in) + 8 input blocks = 9
    /// pushes for 9 output blocks.
    #[test]
    fn the_skip_stack_balances() {
        assert_eq!(INPUT_SPEC.len() + 1, SKIP_SPEC.len());
    }

    /// Each output block's ResBlock takes `skip + in` channels; a wrong pairing
    /// fails the LOAD, so pin the table against the widths the checkpoint stores.
    #[test]
    fn output_block_input_widths_match_the_checkpoint() {
        let expected = [
            (2560, 1280),
            (2560, 1280),
            (1920, 1280),
            (1920, 640),
            (1280, 640),
            (960, 640),
            (960, 320),
            (640, 320),
            (640, 320),
        ];
        for (i, (skip, inc, _, _)) in SKIP_SPEC.iter().enumerate() {
            assert_eq!(
                (skip + inc, *inc),
                expected[i],
                "output block {i} channel pairing"
            );
        }
    }

    /// THE gate: load the whole UNet from a real checkpoint and run one forward.
    ///
    /// Every channel pairing in the tables above is checked by the LOAD (a wrong
    /// width fails with a shape mismatch naming the tensor), and the forward proves
    /// the skip stack, the reshapes and the embedding arithmetic line up. Values are
    /// not compared to a reference here - that needs the ref-diff harness - so this
    /// asserts shape and finiteness, which is what a plumbing gate can honestly claim.
    #[test]
    #[ignore = "needs an SDXL checkpoint under the configured models dir"]
    fn loads_and_runs_one_forward_on_a_real_checkpoint() {
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let ckpt = std::fs::read_dir(dir.join("raymnants"))
            .expect("raymnants dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .expect("an SDXL .safetensors");
        let dev = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .next()
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
        let t0 = std::time::Instant::now();
        // F32 for the gate: it is the arithmetic that is under test here, and the
        // native ops mix dtypes badly enough that a BF16 resident needs its own pass.
        let unet = SdxlUnet::load(ckpt.to_str().unwrap(), &dev, crate::tensor::DType::F32)
            .expect("UNet load (a wrong channel pairing surfaces here)");
        println!(
            "SDXL UNet loaded in {:.1}s on {:?}",
            t0.elapsed().as_secs_f32(),
            dev.location()
        );

        // 512^2 -> a 64x64 latent.
        let (lh, lw) = (64usize, 64);
        let x = Tensor::from_vec_f32(
            vec![0.1f32; LATENT_CHANNELS * lh * lw],
            vec![1, LATENT_CHANNELS, lh, lw],
        )
        .unwrap();
        let ctx = Tensor::from_vec_f32(vec![0.02f32; 77 * CONTEXT_DIM], vec![1, 77, CONTEXT_DIM])
            .unwrap();
        let pooled = Tensor::from_vec_f32(vec![0.03f32; 1280], vec![1, 1280]).unwrap();
        let label = unet
            .label_vector(&pooled, [512.0, 512.0, 0.0, 0.0, 512.0, 512.0])
            .expect("label vector");
        assert_eq!(label.dims(), &[1, LABEL_DIM]);

        let t1 = std::time::Instant::now();
        let out = unet.forward(&x, 999.0, &ctx, &label).expect("forward");
        println!(
            "SDXL UNet forward: {:?} in {:.2}s",
            out.dims(),
            t1.elapsed().as_secs_f32()
        );
        assert_eq!(
            out.dims(),
            &[1, LATENT_CHANNELS, lh, lw],
            "the denoiser must return the latent's shape"
        );
        let v = out.to_device(&Device::Cpu).unwrap().to_vec_f32();
        assert!(
            v.iter().all(|q| q.is_finite()),
            "forward produced non-finite values"
        );
        // A dead net would return a constant slab; the prediction must vary.
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), q| (l.min(*q), h.max(*q)));
        println!("SDXL UNet output range: [{lo:.4}, {hi:.4}]");
        assert!(hi - lo > 1e-3, "the prediction is flat ({lo}..{hi})");
    }
}
