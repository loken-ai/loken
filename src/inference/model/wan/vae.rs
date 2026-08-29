//! Wan 2.1 video Wan-VAE 3D causal decoder (full-Rust, `crate::tensor`).
//!
//! Stage 1 (#?? video): load the Wan-VAE decoder from `Wan2.1_VAE.pth` and decode a
//! latent `[16, T, H, W]` -> RGB frames `[3, 4.(T-1)+1, 8.H, 8.W]`, clamped to [-1,1].
//!
//! ## Net-new primitive: `CausalConv3d`
//! Conv3d with CAUSAL temporal padding (left-pad the time axis by `kt-1`, symmetric pad
//! on H/W, no right/future pad). We have conv1d/conv2d but no conv3d, so we decompose:
//! the running activation is carried as `[T, C, H, W]` (time folded into the batch dim),
//! and a `(kt, kh, kw)` 3D conv is a sum over the `kt` temporal taps - each tap is a 2D
//! conv over every frame, then shifted forward in time by `(kt-1-j)` and accumulated.
//! For tap `j`, output frame `t` reads input frame `t - (kt-1) + j` (zero when < 0), i.e.
//! a standard causal conv: only past/current frames contribute.
//!
//! ## Temporal handling - whole-clip, cache-free, but mathematically exact
//! The reference (`wan/modules/vae.py`) decodes frame-by-frame with a 2-frame feature
//! cache (`CACHE_T=2`), and the temporal upsampling lives ENTIRELY inside that cached
//! path. Tracing the streaming algorithm shows it is exactly equivalent to a whole-clip
//! decode in which: (a) every `CausalConv3d` is a plain causal conv over all `T` frames
//! at once (streaming with the 2-frame cache == causal conv over the full sequence), and
//! (b) each temporal `Resample` (`upsample3d`) splits off frame 0 (which the reference's
//! "Rep" first-frame branch passes through WITHOUT temporal conv), applies the causal
//! `time_conv` (kt=3) to frames `1..` with its own zero history, and interleaves the
//! doubled output. So one temporal-upsample turns `T -> 1 + 2.(T-1)`; two stages give
//! `4.(T-1)+1`. We therefore decode the whole clip at once (no streaming cache) and get
//! the correct temporal length; the per-frame cache would only be needed to bound memory
//! for very long clips (deferred to a later stage).

use crate::tensor::pth::read_pt;
use crate::tensor::{Device as ND, Result, Tensor as NT};
use std::collections::HashMap;

/// Resolve a file `rel` from an HF-cache `repo` snapshot dir (mirrors `ezaudio_pt` /
/// `acestep_gguf`; never a hardcoded path / env var). Searches every snapshot dir of the
/// repo for `rel`, falling back to `snapshots/main/<rel>` when none is materialized yet.
pub fn wan_file_in(repo: &str, rel: &str) -> std::path::PathBuf {
    crate::inference::cache::hf::file(repo, rel)
}

/// Resolve a file from the Wan 1.3B repo snapshot (the umT5 + Wan-VAE + 1.3B DiT live here).
pub fn wan_file(rel: &str) -> std::path::PathBuf {
    wan_file_in("models--Wan-AI--Wan2.1-T2V-1.3B", rel)
}

// Hardcoded per-channel latent de-normalization (16 channels), from `wan/modules/vae.py`
// (`mean` / `std`). decode applies `z = z / (1/std) + mean = z.std + mean`.
const LATENT_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENT_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

fn get<'a>(m: &'a HashMap<String, NT>, name: &str) -> Result<&'a NT> {
    m.get(name)
        .ok_or_else(|| crate::tensor::Error(format!("wan-vae: missing tensor `{name}`")))
}

// -- temporal streaming cache (mirrors the reference `feat_cache`) --------------
// CACHE_T frames of left-context per `CausalConv3d` is exactly what a kt=3 causal
// conv needs (kt-1 = 2 past frames). The reference `wan/modules/vae.py` decode()
// threads one cache slot per CausalConv3d through a per-latent-frame loop; carrying
// these last-(kt-1) input frames makes each chunk's causal-conv output bit-identical
// to the whole-clip decode while bounding peak memory to a single latent frame's
// worth of activations (instead of the whole clip -> ~15 GB at 384²/33f -> OOM).
const CACHE_T: usize = 2;

/// One temporal-cache slot. `Frames` holds the last up-to-`CACHE_T` input frames of
/// the previous chunk (the conv's left context); `Rep` is the upsample3d first-frame
/// marker (frame 0 passes through with NO temporal conv - reference "Rep").
enum CacheEntry {
    Rep,
    Frames(NT),
}

/// Per-decode temporal cache: one slot per `CausalConv3d`, consumed in fixed execution
/// order. `idx` is reset to 0 at the start of every latent-frame chunk; slots persist
/// across chunks so each conv sees the previous chunk's trailing frames as left context.
struct FeatCache {
    slots: Vec<Option<CacheEntry>>,
    idx: usize,
}

impl FeatCache {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            idx: 0,
        }
    }
    fn reset(&mut self) {
        self.idx = 0;
    }
    /// Claim the next cache slot (extending on first encounter) and advance.
    fn next(&mut self) -> usize {
        let i = self.idx;
        if i >= self.slots.len() {
            self.slots.push(None);
        }
        self.idx += 1;
        i
    }
}

// -- primitives ------------------------------------------------------------

/// Causal 3D convolution decomposed into per-temporal-tap 2D convs (see module docs).
/// Operates on `[T, C, H, W]` (time in the batch position); stride 1, temporal output
/// length == input length (the only stride-2 conv is encoder-side `downsample`).
struct CausalConv3d {
    taps: Vec<NT>, // kt kernels, each [c_out, c_in, kh, kw]
    bias: NT,      // [1, c_out, 1, 1]
    kt: usize,
    ph: usize,
    pw: usize,
}

impl CausalConv3d {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        let w = get(m, &format!("{prefix}.weight"))?;
        let d = w.dims(); // [c_out, c_in, kt, kh, kw]
        let (c_out, c_in, kt, kh, kw) = (d[0], d[1], d[2], d[3], d[4]);
        let ph = (kh - 1) / 2;
        let pw = (kw - 1) / 2;
        let mut taps = Vec::with_capacity(kt);
        for j in 0..kt {
            let t = w
                .narrow(2, j, 1)?
                .contiguous()?
                .reshape((c_out, c_in, kh, kw))?;
            taps.push(t);
        }
        let bias = get(m, &format!("{prefix}.bias"))?.reshape((1, c_out, 1, 1))?;
        Ok(Self {
            taps,
            bias,
            kt,
            ph,
            pw,
        })
    }

    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            taps: self
                .taps
                .iter()
                .map(|t| t.to_device(d))
                .collect::<Result<_>>()?,
            bias: self.bias.to_device(d)?,
            kt: self.kt,
            ph: self.ph,
            pw: self.pw,
        })
    }

    /// Whole-clip causal conv (zero temporal history). Equivalent to `forward_left(x, None)`.
    fn forward(&self, x: &NT) -> Result<NT> {
        self.forward_left(x, None)
    }

    /// Causal conv over `x` (`[T,C,H,W]`) with an optional explicit `left` temporal
    /// context of up to `kt-1` past frames. The effective input is
    /// `[zeros(kt-1-L), left(L), x]` (length `T+kt-1`); a valid conv over it yields `T`
    /// causal outputs. `left=None` ⇒ whole-clip start (full zero history). Identical
    /// arithmetic to the previous shift-and-pad form (zero frames contribute nothing),
    /// so it is bit-for-bit the same as the whole-clip decode.
    fn forward_left(&self, x: &NT, left: Option<&NT>) -> Result<NT> {
        let _ = self.pw; // symmetric padding uses ph for square kernels in this VAE
        let t = x.dim(0)?;
        if self.kt == 1 {
            // No temporal context (1x1x1 shortcut / post-quant): plain per-frame conv.
            return x
                .conv2d(&self.taps[0], self.ph, 1, 1, 1)?
                .broadcast_add(&self.bias);
        }
        // Assemble `[zeros(kt-1-L), left(L), x]`. `left` is clamped to the last kt-1 frames.
        let (lx, l) = match left {
            Some(c) => {
                let cl = c.dim(0)?;
                let keep = (self.kt - 1).min(cl);
                let c = if cl > keep {
                    c.narrow(0, cl - keep, keep)?
                } else {
                    c.clone()
                };
                (NT::cat(&[&c, x], 0)?, keep)
            }
            None => (x.clone(), 0),
        };
        let zpad = (self.kt - 1) - l;
        let full = if zpad > 0 {
            lx.pad_with_zeros(0, zpad, 0)?
        } else {
            lx
        };
        // out[o] = Σ_j W_j . full[o+j], o in 0..t (standard valid conv over the causal seq).
        let mut acc: Option<NT> = None;
        for j in 0..self.kt {
            let slice = full.narrow(0, j, t)?;
            let conv = slice.conv2d(&self.taps[j], self.ph, 1, 1, 1)?;
            acc = Some(match acc {
                None => conv,
                Some(a) => a.add(&conv)?,
            });
        }
        acc.unwrap().broadcast_add(&self.bias)
    }

    /// Streaming causal conv: use the previous chunk's trailing frames (this conv's
    /// cache slot) as left context, then store this chunk's trailing frames for the
    /// next chunk. Bit-identical to the whole-clip causal conv over the full sequence.
    fn forward_cached(&self, x: &NT, fc: &mut FeatCache) -> Result<NT> {
        let i = fc.next();
        let prev = fc.slots[i].take();
        let left: Option<NT> = match &prev {
            Some(CacheEntry::Frames(p)) => Some(p.clone()),
            _ => None,
        };
        // Cache the last up-to-CACHE_T frames of THIS input; if fewer than CACHE_T and
        // a previous cache exists, prepend its last frame (reference behaviour).
        let t = x.dim(0)?;
        let take = CACHE_T.min(t);
        let mut cache_x = x.narrow(0, t - take, take)?.contiguous()?;
        if cache_x.dim(0)? < CACHE_T {
            if let Some(CacheEntry::Frames(p)) = &prev {
                let pt = p.dim(0)?;
                cache_x = NT::cat(&[&p.narrow(0, pt - 1, 1)?, &cache_x], 0)?;
            }
        }
        let out = self.forward_left(x, left.as_ref())?;
        fc.slots[i] = Some(CacheEntry::Frames(cache_x));
        Ok(out)
    }
}

/// Plain per-frame 2D conv (`[T,C,H,W]`), used by the spatial `Resample` and attention 1x1s.
struct Conv2dW {
    w: NT,
    bias: NT, // [1, c_out, 1, 1]
    pad: usize,
}

impl Conv2dW {
    fn load(m: &HashMap<String, NT>, prefix: &str, pad: usize) -> Result<Self> {
        let w = get(m, &format!("{prefix}.weight"))?.clone();
        let c_out = w.dims()[0];
        let bias = get(m, &format!("{prefix}.bias"))?.reshape((1, c_out, 1, 1))?;
        Ok(Self { w, bias, pad })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            w: self.w.to_device(d)?,
            bias: self.bias.to_device(d)?,
            pad: self.pad,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        x.conv2d(&self.w, self.pad, 1, 1, 1)?
            .broadcast_add(&self.bias)
    }
}

/// Channel-wise RMS norm: `F.normalize(x, dim=C) . sqrt(C) . gamma` (== `x / rms_C(x) . gamma`).
/// NOT the LLM token-RMSNorm - the reduction is over the channel axis. `gamma` is `[1,C,1,1]`.
/// RMS normalisation over the CHANNEL axis of an `[N, C, H, W]` activation.
///
/// NOT the substrate's `RmsNorm`, which normalises the last axis against a 1-D weight. This one
/// divides by `sqrt(sum over C)` and scales by a `[1, C, 1, 1]` gamma - a different operator
/// that wore the same name, which is how a reader concludes there are two implementations of
/// one thing.
struct ChannelRmsNorm {
    gamma: NT,
    scale: f32,
}

impl ChannelRmsNorm {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        let g = get(m, &format!("{prefix}.gamma"))?;
        let c = g.elem_count();
        let gamma = g.reshape((1, c, 1, 1))?;
        Ok(Self {
            gamma,
            scale: (c as f32).sqrt(),
        })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            gamma: self.gamma.to_device(d)?,
            scale: self.scale,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        // denom = sqrt(sum_C x²); tiny eps avoids div-by-zero on all-zero columns
        // (the reference clamps the L2 norm to 1e-12 - equivalent for finite inputs).
        let ss = x.mul(x)?.sum_keepdim(1)?; // [T,1,H,W]
        let inv = ss.affine(1.0, 1e-12)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv)?
            .scale(self.scale)?
            .broadcast_mul(&self.gamma)
    }
}

fn silu(x: &NT) -> Result<NT> {
    x.silu()
}

// -- blocks ------------------------------------------------------------

/// RMS_norm->SiLU->CausalConv3d->RMS_norm->SiLU->CausalConv3d + (1x1x1 shortcut if C changes).
struct ResidualBlock {
    norm1: ChannelRmsNorm,
    conv1: CausalConv3d,
    norm2: ChannelRmsNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        // residual.{0:RMS, 2:Conv, 3:RMS, 6:Conv}; optional `shortcut` when in!=out.
        let shortcut = if m.contains_key(&format!("{prefix}.shortcut.weight")) {
            Some(CausalConv3d::load(m, &format!("{prefix}.shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: ChannelRmsNorm::load(m, &format!("{prefix}.residual.0"))?,
            conv1: CausalConv3d::load(m, &format!("{prefix}.residual.2"))?,
            norm2: ChannelRmsNorm::load(m, &format!("{prefix}.residual.3"))?,
            conv2: CausalConv3d::load(m, &format!("{prefix}.residual.6"))?,
            shortcut,
        })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            norm1: self.norm1.to_device(d)?,
            conv1: self.conv1.to_device(d)?,
            norm2: self.norm2.to_device(d)?,
            conv2: self.conv2.to_device(d)?,
            shortcut: self.shortcut.as_ref().map(|c| c.to_device(d)).transpose()?,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        let res = self.shortcut.as_ref().map(|c| c.forward(x)).transpose()?;
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        match res {
            Some(r) => h.add(&r),
            None => h.add(x),
        }
    }

    /// Streaming variant: the two residual `CausalConv3d`s thread the temporal cache;
    /// the 1x1x1 shortcut is purely per-frame (no cache slot - matches the reference,
    /// where `shortcut` is applied outside the cached residual loop).
    fn forward_cached(&self, x: &NT, fc: &mut FeatCache) -> Result<NT> {
        let res = self.shortcut.as_ref().map(|c| c.forward(x)).transpose()?;
        let h = self
            .conv1
            .forward_cached(&silu(&self.norm1.forward(x)?)?, fc)?;
        let h = self
            .conv2
            .forward_cached(&silu(&self.norm2.forward(&h)?)?, fc)?;
        match res {
            Some(r) => h.add(&r),
            None => h.add(x),
        }
    }
}

/// Per-frame 2D spatial single-head self-attention (RMS_norm -> qkv 1x1 -> sdpa -> proj 1x1 + res).
struct AttnBlock {
    norm: ChannelRmsNorm,
    to_qkv: Conv2dW,
    proj: Conv2dW,
}

impl AttnBlock {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: ChannelRmsNorm::load(m, &format!("{prefix}.norm"))?,
            to_qkv: Conv2dW::load(m, &format!("{prefix}.to_qkv"), 0)?,
            proj: Conv2dW::load(m, &format!("{prefix}.proj"), 0)?,
        })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            norm: self.norm.to_device(d)?,
            to_qkv: self.to_qkv.to_device(d)?,
            proj: self.proj.to_device(d)?,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        let (t, c, h, w) = x.shape().dims4()?;
        let qkv = self.to_qkv.forward(&self.norm.forward(x)?)?; // [T, 3C, H, W]
        let hw = h * w;
        // each of q,k,v -> [T, HW(tokens), C(dim)]
        let to_tokens =
            |chunk: NT| -> Result<NT> { chunk.reshape((t, c, hw))?.transpose(1, 2)?.contiguous() };
        let q = to_tokens(qkv.narrow(1, 0, c)?)?;
        let k = to_tokens(qkv.narrow(1, c, c)?)?;
        let v = to_tokens(qkv.narrow(1, 2 * c, c)?)?;
        let scale = 1.0 / (c as f32).sqrt();
        let att = q
            .matmul(&k.transpose(1, 2)?)?
            .scale(scale)?
            .softmax_last_dim()?; // [T,HW,HW]
        let o = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((t, c, h, w))?
            .contiguous()?;
        self.proj.forward(&o)?.add(x)
    }
}

/// Spatial x2 nearest upsample + 3x3 conv (`dim -> dim/2`); for `upsample3d` it is preceded
/// by a temporal upsample (causal `time_conv` kt=3 over frames `1..`, then 2x interleave).
struct Resample {
    time_conv: Option<CausalConv3d>, // present ⇒ upsample3d (temporal+spatial)
    conv: Conv2dW,                   // dim -> dim/2, 3x3 pad 1
}

impl Resample {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        let time_conv = if m.contains_key(&format!("{prefix}.time_conv.weight")) {
            Some(CausalConv3d::load(m, &format!("{prefix}.time_conv"))?)
        } else {
            None
        };
        Ok(Self {
            time_conv,
            conv: Conv2dW::load(m, &format!("{prefix}.resample.1"), 1)?,
        })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            time_conv: self
                .time_conv
                .as_ref()
                .map(|c| c.to_device(d))
                .transpose()?,
            conv: self.conv.to_device(d)?,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        let x = if let Some(tc) = &self.time_conv {
            let (t, c, h, w) = x.shape().dims4()?;
            if t > 1 {
                let f0 = x.narrow(0, 0, 1)?; // frame 0 passes through temporally (reference "Rep")
                let rest = x.narrow(0, 1, t - 1)?;
                let y = tc.forward(&rest)?; // [T-1, 2C, H, W]
                let a = y.narrow(1, 0, c)?.unsqueeze(1)?; // [T-1,1,C,H,W]
                let b = y.narrow(1, c, c)?.unsqueeze(1)?;
                // interleave along time: [a0,b0,a1,b1,...] -> 2.(T-1) frames
                let inter = NT::cat(&[&a, &b], 1)?.reshape((2 * (t - 1), c, h, w))?;
                NT::cat(&[&f0, &inter], 0)? // 1 + 2.(T-1) frames
            } else {
                x.clone()
            }
        } else {
            x.clone()
        };
        let (_, _, h, w) = x.shape().dims4()?;
        self.conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)
    }

    /// Interleave a `time_conv` output `[T, 2C, H, W]` into `2T` frames: for each input
    /// time t emit channels `[0..C)` then `[C..2C)` (reference `reshape->stack->reshape`).
    fn interleave(y: &NT, c: usize) -> Result<NT> {
        let (t, _, h, w) = y.shape().dims4()?;
        let a = y.narrow(1, 0, c)?.unsqueeze(1)?; // [T,1,C,H,W]
        let b = y.narrow(1, c, c)?.unsqueeze(1)?;
        NT::cat(&[&a, &b], 1)?.reshape((2 * t, c, h, w))
    }

    /// Streaming variant. The temporal `upsample3d` threads its `time_conv` cache so the
    /// first chunk (latent frame 0) passes through untouched (reference "Rep") and every
    /// later chunk's doubled output continues bit-identically from the previous chunk.
    fn forward_cached(&self, x: &NT, fc: &mut FeatCache) -> Result<NT> {
        let x = if let Some(tc) = &self.time_conv {
            let i = fc.next();
            let (t, c, _, _) = x.shape().dims4()?;
            match fc.slots[i].take() {
                None => {
                    // First chunk: frame 0 passes through (no temporal conv, no doubling).
                    fc.slots[i] = Some(CacheEntry::Rep);
                    x.clone()
                }
                Some(entry) => {
                    let take = CACHE_T.min(t);
                    let mut cache_x = x.narrow(0, t - take, take)?.contiguous()?;
                    let left: Option<NT> = match &entry {
                        CacheEntry::Frames(p) => Some(p.clone()),
                        CacheEntry::Rep => None, // first post-Rep chunk: zero history
                    };
                    if cache_x.dim(0)? < CACHE_T {
                        match &entry {
                            CacheEntry::Frames(p) => {
                                let pt = p.dim(0)?;
                                cache_x = NT::cat(&[&p.narrow(0, pt - 1, 1)?, &cache_x], 0)?;
                            }
                            CacheEntry::Rep => {
                                // a zero frame of matching shape/device (reference `zeros_like`)
                                let zf = cache_x.affine(0.0, 0.0)?;
                                cache_x = NT::cat(&[&zf, &cache_x], 0)?;
                            }
                        }
                    }
                    let y = tc.forward_left(x, left.as_ref())?; // [T, 2C, H, W]
                    fc.slots[i] = Some(CacheEntry::Frames(cache_x));
                    Self::interleave(&y, c)?
                }
            }
        } else {
            x.clone()
        };
        let (_, _, h, w) = x.shape().dims4()?;
        self.conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)
    }
}

enum UpLayer {
    Res(ResidualBlock),
    Resample(Resample),
}

impl UpLayer {
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(match self {
            UpLayer::Res(r) => UpLayer::Res(r.to_device(d)?),
            UpLayer::Resample(r) => UpLayer::Resample(r.to_device(d)?),
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        match self {
            UpLayer::Res(r) => r.forward(x),
            UpLayer::Resample(r) => r.forward(x),
        }
    }
    fn forward_cached(&self, x: &NT, fc: &mut FeatCache) -> Result<NT> {
        match self {
            UpLayer::Res(r) => r.forward_cached(x, fc),
            UpLayer::Resample(r) => r.forward_cached(x, fc),
        }
    }
}

// -- decoder ------------------------------------------------------------

pub struct WanVaeDecoder {
    mean: NT, // [1,16,1,1]
    std: NT,  // [1,16,1,1]
    post_quant: CausalConv3d,
    conv1: CausalConv3d,
    mid0: ResidualBlock,
    mid_attn: AttnBlock,
    mid2: ResidualBlock,
    upsamples: Vec<UpLayer>,
    head_norm: ChannelRmsNorm,
    head_conv: CausalConv3d,
}

/// Load the Wan-VAE decoder (and the top-level post-quant conv) from a `Wan2.1_VAE.pth`.
pub fn load_wan_vae_decoder(pth: &str) -> Result<WanVaeDecoder> {
    decoder_from_map(read_pt(pth)?)
}

/// Load the Wan-VAE decoder from a diffusers->Wan remapped safetensors - e.g. the Qwen-Image
/// VAE (same Wan 2.1 architecture) run through scripts/qwen_vae_remap.py.
pub fn load_wan_vae_decoder_safetensors(path: &str) -> Result<WanVaeDecoder> {
    let ld = unsafe { crate::tensor::safetensors_io::SafeTensorsLoader::multi(&[path])? };
    let mut m = HashMap::new();
    for name in ld.names() {
        let n = name.to_string();
        // Qwen ships the VAE in BF16; the Wan-VAE ops want F32 CPU storage.
        let t = ld.load(&n)?.to_dtype(crate::tensor::DType::F32)?;
        m.insert(n, t);
    }
    decoder_from_map(m)
}

fn decoder_from_map(m: HashMap<String, NT>) -> Result<WanVaeDecoder> {
    let mk = |v: &[f32]| -> Result<NT> { NT::from_vec_f32(v.to_vec(), (1, 16, 1, 1)) };

    // walk the `upsamples` ModuleList by index; each entry is a ResidualBlock (has
    // `residual.*`) or a Resample (has `resample.1.*`, plus `time_conv.*` for upsample3d).
    let mut upsamples = Vec::new();
    let mut i = 0usize;
    loop {
        let p = format!("decoder.upsamples.{i}");
        if m.contains_key(&format!("{p}.residual.2.weight")) {
            upsamples.push(UpLayer::Res(ResidualBlock::load(&m, &p)?));
        } else if m.contains_key(&format!("{p}.resample.1.weight")) {
            upsamples.push(UpLayer::Resample(Resample::load(&m, &p)?));
        } else {
            break;
        }
        i += 1;
    }

    Ok(WanVaeDecoder {
        mean: mk(&LATENT_MEAN)?,
        std: mk(&LATENT_STD)?,
        post_quant: CausalConv3d::load(&m, "conv2")?,
        conv1: CausalConv3d::load(&m, "decoder.conv1")?,
        mid0: ResidualBlock::load(&m, "decoder.middle.0")?,
        mid_attn: AttnBlock::load(&m, "decoder.middle.1")?,
        mid2: ResidualBlock::load(&m, "decoder.middle.2")?,
        upsamples,
        head_norm: ChannelRmsNorm::load(&m, "decoder.head.0")?,
        head_conv: CausalConv3d::load(&m, "decoder.head.2")?,
    })
}

impl WanVaeDecoder {
    /// Device of the decoder weights (the post-quant conv bias is representative).
    pub fn device(&self) -> ND {
        self.post_quant.bias.device()
    }

    /// Move every retained weight tensor onto `dev` (CPU↔CUDA). Only the decoder
    /// tensors actually held are copied - the encoder weights from the `.pth` are
    /// already dropped - so the GPU footprint stays small. Used to place the decode
    /// on the (now-free) GPU, and to fall BACK to CPU on a CUDA OOM.
    pub fn to_device(&self, dev: &ND) -> Result<Self> {
        Ok(WanVaeDecoder {
            mean: self.mean.to_device(dev)?,
            std: self.std.to_device(dev)?,
            post_quant: self.post_quant.to_device(dev)?,
            conv1: self.conv1.to_device(dev)?,
            mid0: self.mid0.to_device(dev)?,
            mid_attn: self.mid_attn.to_device(dev)?,
            mid2: self.mid2.to_device(dev)?,
            upsamples: self
                .upsamples
                .iter()
                .map(|u| u.to_device(dev))
                .collect::<Result<_>>()?,
            head_norm: self.head_norm.to_device(dev)?,
            head_conv: self.head_conv.to_device(dev)?,
        })
    }

    /// Decode a latent `[16, T, H, W]` -> RGB frames `[3, 4.(T-1)+1, 8.H, 8.W]`, clamped [-1,1].
    ///
    /// Temporal-streaming by default ([`decode_chunked`]) so peak memory stays bounded and
    /// a 384²/33f clip decodes on the GPU instead of OOMing into a CPU fallback. Set
    /// `WAN_VAE_WHOLE=1` to force the original whole-clip path (used by the parity test);
    /// the two produce bit-identical output.
    pub fn decode(&self, latent: &NT) -> Result<NT> {
        self.decode_cancellable(latent, None)
    }

    /// [`decode`] with client-disconnect support: the per-frame chunk loop
    /// checks the token so a dead request stops burning CPU/GPU mid-decode
    /// (the denoise loop already cancels; the decode used to run to the end).
    pub fn decode_cancellable(
        &self,
        latent: &NT,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<NT> {
        self.decode_reporting(latent, cancel, None)
    }

    /// The decode, reporting each finished frame.
    ///
    /// It walks the clip one latent frame at a time, so it always KNEW how far along it was
    /// - it just never said. On a long clip that is minutes during which the only thing
    /// anyone can see is the word "decoding", which is indistinguishable from a hang and was
    /// reported as one. The callback is given (done, total) in latent frames.
    pub fn decode_reporting(
        &self,
        latent: &NT,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
        on_frame: Option<&dyn Fn(usize, usize)>,
    ) -> Result<NT> {
        // Always chunked: `decode_whole` allocates every frame's activations at
        // once and OOMs on any real clip. It stays as the reference the chunked
        // path is tested against, not as a runtime choice.
        self.decode_chunked_cancellable(latent, cancel, on_frame)
    }

    /// Whole-clip decode: every `CausalConv3d` runs over all `T` frames at once. Simple
    /// but allocates the full clip's activations at each layer (~15.6 GB at 384²/33f -> OOM).
    pub fn decode_whole(&self, latent: &NT) -> Result<NT> {
        // [16,T,H,W] -> [T,16,H,W] (time in the batch position throughout).
        let z = latent.transpose(0, 1)?.contiguous()?;
        // de-normalize: z.std + mean, then the post-quant 1x1x1 conv.
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut x = self.post_quant.forward(&z)?;

        x = self.conv1.forward(&x)?;
        x = self.mid0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid2.forward(&x)?;
        for layer in &self.upsamples {
            x = layer.forward(&x)?;
        }
        x = self
            .head_conv
            .forward(&silu(&self.head_norm.forward(&x)?)?)?; // [Tout, 3, 8H, 8W]
        x = clamp_unit(&x)?;
        // -> [3, Tout, 8H, 8W]
        x.transpose(0, 1)?.contiguous()
    }

    /// Temporal-streaming decode (reference `wan/modules/vae.py` `decode`): the post-quant
    /// 1x1x1 conv (no temporal mixing) runs whole, then the decoder body runs ONE latent
    /// frame at a time, threading a per-`CausalConv3d` temporal cache so each chunk's causal
    /// output is bit-identical to the whole-clip decode. Peak memory is bounded by a single
    /// latent frame's activations + the (kt-1)-frame caches, so large clips stay on the GPU.
    ///
    /// The DECODED frames are moved to the host as they are produced. Bounding the
    /// activations was only half of it: keeping every finished frame on the card meant the
    /// output alone grew with the clip, and the concatenation and transpose that followed
    /// each allocated another copy of it - three gigabytes apiece at a minute of 512-square
    /// video, on top of a resident denoiser. A 60 s clip OOMed, fell back to a CPU decode
    /// and turned a fast path into a very slow one. These frames are bound for the host
    /// regardless (`pack_frames` reads them straight back), which is the same argument
    /// already written for the spatially tiled decode below.
    pub fn decode_chunked(&self, latent: &NT) -> Result<NT> {
        self.decode_chunked_cancellable(latent, None, None)
    }

    fn decode_chunked_cancellable(
        &self,
        latent: &NT,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
        on_frame: Option<&dyn Fn(usize, usize)>,
    ) -> Result<NT> {
        // [16,T,H,W] -> [T,16,H,W], de-normalize, post-quant 1x1x1 conv (whole; kt=1).
        let z = latent.transpose(0, 1)?.contiguous()?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let x = self.post_quant.forward(&z)?;
        let f = x.dim(0)?;

        let mut fc = FeatCache::new();
        // Host-side, one finished chunk at a time: the device holds the activations of the
        // frame being decoded and nothing that is already done.
        let mut outs: Vec<Vec<f32>> = Vec::with_capacity(f);
        let mut geom: Option<(usize, usize, usize)> = None; // (channels, out_h, out_w)
        for i in 0..f {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return Err(crate::tensor::Error(
                    "wan decode cancelled (client disconnected)".into(),
                ));
            }
            fc.reset();
            let xi = x.narrow(0, i, 1)?; // [1, 16, H, W]
            let mut h = self.conv1.forward_cached(&xi, &mut fc)?;
            h = self.mid0.forward_cached(&h, &mut fc)?;
            h = self.mid_attn.forward(&h)?; // per-frame self-attention (no temporal cache)
            h = self.mid2.forward_cached(&h, &mut fc)?;
            for layer in &self.upsamples {
                h = layer.forward_cached(&h, &mut fc)?;
            }
            h = self
                .head_conv
                .forward_cached(&silu(&self.head_norm.forward(&h)?)?, &mut fc)?;
            let o = clamp_unit(&h)?; // [chunk_T_out, 3, 8H, 8W]
            let (_, c, oh, ow) = o.dims4()?;
            geom = Some((c, oh, ow));
            outs.push(o.flatten_all()?.to_vec_f32());
            if let Some(f_cb) = on_frame {
                f_cb(i + 1, f);
            }
        }
        // Assemble straight into the caller's [3, Tout, 8H, 8W] order rather than
        // concatenating and then transposing: the transpose of a whole clip is another full
        // copy of it, and here it is only an index.
        let (c, oh, ow) =
            geom.ok_or_else(|| crate::tensor::Error("wan decode: empty latent".into()))?;
        let plane = oh * ow;
        let t_out: usize = outs.iter().map(|o| o.len() / (c * plane)).sum();
        let mut flat = vec![0f32; c * t_out * plane];
        let mut at = 0usize;
        for chunk in &outs {
            let tc = chunk.len() / (c * plane);
            for ti in 0..tc {
                for ci in 0..c {
                    let src = (ti * c + ci) * plane;
                    let dst = (ci * t_out + at + ti) * plane;
                    flat[dst..dst + plane].copy_from_slice(&chunk[src..src + plane]);
                }
            }
            at += tc;
        }
        NT::from_vec(flat, (c, t_out, oh, ow), &ND::Cpu)
    }
    /// Spatially tiled decode, for images too large to decode whole.
    ///
    /// The decoder's last stages hold `[96, 8H, 8W]` activations AND the 3x3 im2col of
    /// the same tensor, so a 1024^2 decode peaks past 6 GB - a card that comfortably
    /// holds the 243 MB decoder still cannot run it, and the op-level host fallback
    /// turns that into minutes of PCIe ping-pong. Decoding overlapping LATENT tiles
    /// bounds the peak by the tile instead of the image.
    ///
    /// Blending follows the standard tiled-decode recipe: every tile edge gets a linear
    /// feather of the overlap width, tiles accumulate `value * weight` and `weight`, and
    /// the sum is divided at the end - so the interior is an exact partition of unity and
    /// the image borders normalize themselves. Accumulation is host-side: the result is
    /// bound for PNG encoding on the CPU anyway, and it keeps device memory at one tile.
    ///
    /// `tile` and `overlap` are in LATENT pixels (the decoder upscales 8x).
    pub fn decode_spatial_tiled(&self, latent: &NT, tile: usize, overlap: usize) -> Result<NT> {
        let (_c, _t, h, w) = latent.dims4()?;
        if h <= tile && w <= tile {
            return self.decode(latent);
        }
        let (acc, tout) = tiled_reassemble(h, w, tile, overlap, |py, px, lh, lw| {
            let sub = latent.narrow(2, py, lh)?.narrow(3, px, lw)?.contiguous()?;
            let ps = self.decode(&sub)?; // [3, tout, 8lh, 8lw]
            let tout = ps.dim(1)?;
            Ok((
                ps.to_device(&crate::tensor::Device::Cpu)?.to_vec_f32(),
                tout,
            ))
        })?;
        NT::from_vec_f32(acc, (3, tout, h * 8, w * 8))
    }
}

/// Linear feather: 1 in the interior, ramping to `1/feather` at each edge.
fn feather_ramp(i: usize, n: usize, feather: usize) -> f32 {
    let mut m = 1.0f32;
    if i < feather {
        m *= (i + 1) as f32 / feather as f32;
    }
    if i + feather >= n {
        m *= (n - i) as f32 / feather as f32;
    }
    m
}

/// Tile start positions along one axis: full tiles stepping by `tile - overlap`,
/// the last one clamped so it still covers the edge.
fn tile_starts(n: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if n <= tile {
        return vec![0];
    }
    let step = tile.saturating_sub(overlap).max(1);
    let mut v = Vec::new();
    let mut i = 0;
    while i < n - overlap.min(n) {
        v.push(i.min(n - tile));
        if v.last() == Some(&(n - tile)) {
            break;
        }
        i += step;
    }
    if v.last() != Some(&(n - tile)) {
        v.push(n - tile);
    }
    v
}

/// Decode every tile and blend the results into the full image, host-side.
///
/// `decode_tile(py, px, lh, lw)` returns that tile's pixels as `[3, tout, 8lh, 8lw]`
/// (row-major) plus `tout`. Weights accumulate alongside the values and divide at the
/// end, so the interior is an exact partition of unity and the borders normalize
/// themselves - reassembly is exact wherever the tiles agree.
#[allow(clippy::type_complexity)]
fn tiled_reassemble(
    h: usize,
    w: usize,
    tile: usize,
    overlap: usize,
    mut decode_tile: impl FnMut(usize, usize, usize, usize) -> Result<(Vec<f32>, usize)>,
) -> Result<(Vec<f32>, usize)> {
    let (oh, ow) = (h * 8, w * 8);
    let mut acc: Vec<f32> = Vec::new();
    let mut wsum: Vec<f32> = Vec::new();
    let mut tout = 0usize;
    for py in tile_starts(h, tile, overlap) {
        for px in tile_starts(w, tile, overlap) {
            let lh = tile.min(h - py);
            let lw = tile.min(w - px);
            let (v, t) = decode_tile(py, px, lh, lw)?;
            if acc.is_empty() {
                tout = t;
                acc = vec![0.0; 3 * tout * oh * ow];
                wsum = vec![0.0; 3 * tout * oh * ow];
            }
            let (th, tw) = (lh * 8, lw * 8);
            let feather = (overlap * 8).min(th).min(tw).max(1);
            let (oy, ox) = (py * 8, px * 8);
            for ch in 0..3 {
                for f in 0..tout {
                    for y in 0..th {
                        let wy = feather_ramp(y, th, feather);
                        let src_row = ((ch * tout + f) * th + y) * tw;
                        let dst_row = ((ch * tout + f) * oh + (oy + y)) * ow + ox;
                        for x in 0..tw {
                            let g = wy * feather_ramp(x, tw, feather);
                            acc[dst_row + x] += v[src_row + x] * g;
                            wsum[dst_row + x] += g;
                        }
                    }
                }
            }
        }
    }
    for (a, s) in acc.iter_mut().zip(wsum.iter()) {
        if *s > 0.0 {
            *a /= *s;
        }
    }
    Ok((acc, tout))
}

/// Clamp to [-1, 1] using two ReLU folds (`min(x,1)=1-relu(1-x)`, `max(.,-1)=relu(.+1)-1`).
fn clamp_unit(x: &NT) -> Result<NT> {
    let min1 = x.affine(-1.0, 1.0)?.relu()?.affine(-1.0, 1.0)?; // min(x, 1)
    min1.affine(1.0, 1.0)?.relu()?.affine(1.0, -1.0) // max(., -1)
}

// -- encoder (net-new: Wan is decode-only; Qwen-Image-Edit needs the ref-image latents) ------

/// Spatial x½ downsample: `ZeroPad2d((0,1,0,1))` + stride-2 3x3 conv (`resample.1`). For a
/// single image (T=1) the reference's temporal `time_conv` downsample is a no-op (a lone
/// frame can't be halved), so the image encoder implements the spatial path only.
struct DownResample {
    w: NT,
    bias: NT,
}
impl DownResample {
    fn load(m: &HashMap<String, NT>, prefix: &str) -> Result<Self> {
        let w = get(m, &format!("{prefix}.resample.1.weight"))?.clone();
        let c_out = w.dims()[0];
        let bias = get(m, &format!("{prefix}.resample.1.bias"))?.reshape((1, c_out, 1, 1))?;
        Ok(Self { w, bias })
    }
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            w: self.w.to_device(d)?,
            bias: self.bias.to_device(d)?,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        // ZeroPad2d((left0,right1,top0,bottom1)) on [T,C,H,W]: H is dim 2, W is dim 3.
        let x = x.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        x.conv2d(&self.w, 0, 2, 1, 1)?.broadcast_add(&self.bias)
    }
}

enum DownLayer {
    Res(ResidualBlock),
    Down(DownResample),
}
impl DownLayer {
    fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(match self {
            DownLayer::Res(r) => DownLayer::Res(r.to_device(d)?),
            DownLayer::Down(r) => DownLayer::Down(r.to_device(d)?),
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        match self {
            DownLayer::Res(r) => r.forward(x),
            DownLayer::Down(r) => r.forward(x),
        }
    }
}

pub struct WanVaeEncoder {
    mean: NT,
    std: NT,
    conv1: CausalConv3d, // encoder.conv1 (conv_in)
    downsamples: Vec<DownLayer>,
    mid0: ResidualBlock,
    mid_attn: AttnBlock,
    mid2: ResidualBlock,
    head_norm: ChannelRmsNorm,
    head_conv: CausalConv3d,
    quant: CausalConv3d, // top-level conv1 (quant_conv)
    z_dim: usize,
}

/// Load the Wan-VAE encoder from the diffusers->Wan remapped safetensors (Qwen-Image VAE).
pub fn load_wan_vae_encoder_safetensors(path: &str) -> Result<WanVaeEncoder> {
    load_wan_vae_encoder_safetensors_on(path, &crate::tensor::Device::Cpu)
}

/// The encoder, built on a caller-chosen device.
///
/// It had no device at all, so it was always built on the host and the reference frame was
/// encoded there - two passes over a full-resolution image through the whole down-path.
/// That is not a rounding cost: it was measured at thirty-two seconds, the largest single
/// item in a render outside the clip itself, while both cards sat idle. The decoder next to
/// it has taken a device since it was written; this is the same network run backwards.
pub fn load_wan_vae_encoder_safetensors_on(
    path: &str,
    dev: &crate::tensor::Device,
) -> Result<WanVaeEncoder> {
    let ld = unsafe { crate::tensor::safetensors_io::SafeTensorsLoader::multi(&[path])? };
    let mut m = HashMap::new();
    for name in ld.names() {
        let n = name.to_string();
        let t = ld
            .load(&n)?
            .to_dtype(crate::tensor::DType::F32)?
            .to_device(dev)?;
        m.insert(n, t);
    }
    let mk =
        |v: &[f32]| -> Result<NT> { NT::from_vec_f32(v.to_vec(), (1, 16, 1, 1))?.to_device(dev) };
    let mut downsamples = Vec::new();
    let mut i = 0usize;
    loop {
        let p = format!("encoder.downsamples.{i}");
        if m.contains_key(&format!("{p}.residual.2.weight")) {
            downsamples.push(DownLayer::Res(ResidualBlock::load(&m, &p)?));
        } else if m.contains_key(&format!("{p}.resample.1.weight")) {
            downsamples.push(DownLayer::Down(DownResample::load(&m, &p)?));
        } else {
            break;
        }
        i += 1;
    }
    Ok(WanVaeEncoder {
        mean: mk(&LATENT_MEAN)?,
        std: mk(&LATENT_STD)?,
        conv1: CausalConv3d::load(&m, "encoder.conv1")?,
        downsamples,
        mid0: ResidualBlock::load(&m, "encoder.middle.0")?,
        mid_attn: AttnBlock::load(&m, "encoder.middle.1")?,
        mid2: ResidualBlock::load(&m, "encoder.middle.2")?,
        head_norm: ChannelRmsNorm::load(&m, "encoder.head.0")?,
        head_conv: CausalConv3d::load(&m, "encoder.head.2")?,
        quant: CausalConv3d::load(&m, "conv1")?,
        z_dim: 16,
    })
}

impl WanVaeEncoder {
    pub fn to_device(&self, d: &ND) -> Result<Self> {
        Ok(Self {
            mean: self.mean.to_device(d)?,
            std: self.std.to_device(d)?,
            conv1: self.conv1.to_device(d)?,
            downsamples: self
                .downsamples
                .iter()
                .map(|l| l.to_device(d))
                .collect::<Result<_>>()?,
            mid0: self.mid0.to_device(d)?,
            mid_attn: self.mid_attn.to_device(d)?,
            mid2: self.mid2.to_device(d)?,
            head_norm: self.head_norm.to_device(d)?,
            head_conv: self.head_conv.to_device(d)?,
            quant: self.quant.to_device(d)?,
            z_dim: self.z_dim,
        })
    }

    /// Encode an image `[3, 1, H, W]` (RGB in [-1,1]) -> normalized latent `[16, 1, H/8, W/8]`.
    /// Takes the DiagonalGaussian mean (deterministic) and normalizes to the DiT latent space
    /// (the inverse of decode's `z.std + mean`), so encode∘decode round-trips.
    pub fn encode(&self, image: &NT) -> Result<NT> {
        let mut x = self.conv1.forward(&image.transpose(0, 1)?.contiguous()?)?; // [1,3,H,W]->
        for layer in &self.downsamples {
            x = layer.forward(&x)?;
        }
        x = self.mid0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid2.forward(&x)?;
        x = self
            .head_conv
            .forward(&silu(&self.head_norm.forward(&x)?)?)?; // [1, 2.z, h, w]
        x = self.quant.forward(&x)?;
        let mean_z = x.narrow(1, 0, self.z_dim)?.contiguous()?; // DiagonalGaussian mean [1,z,h,w]
        let neg_mean = self.mean.affine(-1.0, 0.0)?;
        let z = mean_z.broadcast_add(&neg_mean)?.broadcast_div(&self.std)?;
        z.transpose(0, 1)?.contiguous() // [z,1,h,w]
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    #[test]
    #[ignore]
    fn inspect_wan_vae_keys() {
        let p = wan_file("Wan2.1_VAE.pth");
        eprintln!("loading {}", p.display());
        let m = read_pt(p.to_str().unwrap()).expect("read_pt");
        eprintln!("total tensors = {}", m.len());
        let mut keys: Vec<_> = m.keys().cloned().collect();
        keys.sort();
        for k in &keys {
            let t = &m[k];
            if k.starts_with("decoder.") || !k.starts_with("encoder.") {
                eprintln!("{:60}  {:?}  {:?}", k, t.dims(), t.dtype());
            }
        }
    }

    #[test]
    #[ignore]
    fn decode_sanity() {
        let p = wan_file("Wan2.1_VAE.pth");
        let dec = load_wan_vae_decoder(p.to_str().unwrap()).expect("load decoder");

        for (tt, hh, ww) in [(1usize, 16usize, 16usize), (2, 8, 8)] {
            // pseudo-random but deterministic small latent [16, T, H, W]
            let n = 16 * tt * hh * ww;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5) * 2.0)
                .collect();
            let latent = NT::from_vec_f32(data, (16, tt, hh, ww)).unwrap();

            let out = dec.decode(&latent).expect("decode");
            let dims = out.dims().to_vec();
            let expect_t = 4 * (tt - 1) + 1;
            eprintln!(
                "latent [16,{tt},{hh},{ww}] -> frames {:?} (expect [3,{expect_t},{},{}])",
                dims,
                hh * 8,
                ww * 8
            );
            assert_eq!(dims, vec![3, expect_t, hh * 8, ww * 8], "output shape");

            let v = out.to_vec_f32();
            assert!(v.iter().all(|x| x.is_finite()), "all finite");
            let (mn, mx) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            eprintln!("  range [{mn:.4}, {mx:.4}], n={}", v.len());
            assert!(mn >= -1.0001 && mx <= 1.0001, "clamped to [-1,1]");
        }
    }

    /// Correctness bar: the temporal-streaming decode must reproduce the whole-clip decode
    /// (the per-`CausalConv3d` cache makes every causal output bit-identical). Multiple T so
    /// both temporal upsamples (`Rep` first-frame + cached continuation) are exercised.
    #[test]
    #[ignore]
    fn chunked_matches_whole() {
        let p = wan_file("Wan2.1_VAE.pth");
        let dec = load_wan_vae_decoder(p.to_str().unwrap()).expect("load decoder");

        for (tt, hh, ww) in [(1usize, 16usize, 16usize), (3, 16, 16), (5, 32, 32)] {
            let n = 16 * tt * hh * ww;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5) * 2.0)
                .collect();
            let latent = NT::from_vec_f32(data, (16, tt, hh, ww)).unwrap();

            let whole = dec.decode_whole(&latent).expect("decode_whole");
            let chunk = dec.decode_chunked(&latent).expect("decode_chunked");
            assert_eq!(whole.dims(), chunk.dims(), "shape T={tt}");

            let (a, b) = (whole.to_vec_f32(), chunk.to_vec_f32());
            let max_abs = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            eprintln!(
                "T={tt} {hh}x{ww}: frames {:?}, max|whole-chunk| = {max_abs:.3e}",
                whole.dims()
            );
            assert!(
                max_abs < 1e-4,
                "chunked decode diverges from whole-clip: {max_abs}"
            );
        }
    }
}

#[cfg(test)]
mod tiling_tests {
    use super::{feather_ramp, tile_starts, tiled_reassemble};

    /// Ground truth: a per-pixel value function the "decoder" reproduces exactly for
    /// any crop. Reassembly must then return that same image - if the tile offsets or
    /// the blend weights are wrong, the interior shows it immediately.
    fn truth(ch: usize, y: usize, x: usize) -> f32 {
        (ch as f32) * 100.0 + (y as f32) * 0.5 - (x as f32) * 0.25
    }

    fn reassemble(h: usize, w: usize, tile: usize, overlap: usize) -> Vec<f32> {
        let (out, tout) = tiled_reassemble(h, w, tile, overlap, |py, px, lh, lw| {
            let (th, tw) = (lh * 8, lw * 8);
            let (oy, ox) = (py * 8, px * 8);
            let mut v = vec![0.0f32; 3 * th * tw];
            for ch in 0..3 {
                for y in 0..th {
                    for x in 0..tw {
                        v[(ch * th + y) * tw + x] = truth(ch, oy + y, ox + x);
                    }
                }
            }
            Ok((v, 1))
        })
        .unwrap();
        assert_eq!(tout, 1);
        out
    }

    fn assert_matches_truth(h: usize, w: usize, tile: usize, overlap: usize) {
        let out = reassemble(h, w, tile, overlap);
        let (oh, ow) = (h * 8, w * 8);
        assert_eq!(out.len(), 3 * oh * ow);
        let mut worst = 0.0f32;
        for ch in 0..3 {
            for y in 0..oh {
                for x in 0..ow {
                    let got = out[(ch * oh + y) * ow + x];
                    worst = worst.max((got - truth(ch, y, x)).abs());
                }
            }
        }
        // Weights are a partition of unity after the division, so agreeing tiles
        // reassemble to the original up to f32 rounding of the weighted sum.
        assert!(
            worst < 1e-2,
            "tile={tile} overlap={overlap}: worst diff {worst}"
        );
    }

    #[test]
    fn reassembly_is_exact_for_the_production_tiling() {
        // 1024^2 image = 128^2 latent, the case that OOMed when decoded whole.
        assert_matches_truth(128, 128, 64, 8);
    }

    #[test]
    fn reassembly_is_exact_for_non_multiple_and_lopsided_shapes() {
        assert_matches_truth(100, 128, 64, 8);
        assert_matches_truth(96, 40, 32, 4);
        assert_matches_truth(65, 65, 64, 8);
    }

    #[test]
    fn a_latent_smaller_than_one_tile_is_a_single_tile() {
        assert_eq!(tile_starts(40, 64, 8), vec![0]);
    }

    #[test]
    fn tiles_cover_every_row_and_reach_the_far_edge() {
        for n in [65usize, 100, 128, 129, 200] {
            let starts = tile_starts(n, 64, 8);
            assert_eq!(*starts.last().unwrap(), n - 64, "n={n} must reach the edge");
            // No gap: each tile starts before the previous one ends.
            for pair in starts.windows(2) {
                assert!(pair[1] <= pair[0] + 64, "gap between tiles at n={n}");
            }
        }
    }

    #[test]
    fn the_feather_ramps_at_the_edges_and_is_flat_inside() {
        let (n, f) = (64usize, 8usize);
        assert!((feather_ramp(0, n, f) - 0.125).abs() < 1e-6);
        assert!((feather_ramp(n - 1, n, f) - 0.125).abs() < 1e-6);
        assert!((feather_ramp(n / 2, n, f) - 1.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod decode_cost_probe {
    //! Where the video decode's time actually goes.
    //!
    //! After the denoiser was made linear in duration, the VAE decode became the other half
    //! of a render - 249 s of a 507 s minute-long clip - and it runs at 94-98% GPU
    //! utilisation, so nothing is idle. Utilisation says kernels are resident, not that they
    //! are efficient, and the arithmetic did not add up: the convolutions in this decoder
    //! total a few hundred GFLOP per output frame, which should be a few milliseconds.
    //!
    //! So measure one convolution at the decoder's widest shape before assuming anything
    //! about the rest.
    //!
    //!   cargo test -p loken --release decode_cost_probe -- --ignored --nocapture
    use crate::tensor::{Device, Tensor};

    #[test]
    #[ignore]
    fn one_full_resolution_convolution() {
        let dev = match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(d) => Device::Cuda(d),
            Err(_) => {
                eprintln!("no cuda; skip");
                return;
            }
        };
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
                })
                .collect()
        };
        // The last decoder stage: full output resolution, its widest channel count there.
        for (c, hw) in [(96usize, 512usize), (192, 256), (384, 128)] {
            let x = Tensor::from_vec(mk(c * hw * hw), (1, c, hw, hw), &dev).unwrap();
            let k = Tensor::from_vec(mk(c * c * 9), (c, c, 3, 3), &dev).unwrap();
            let _ = x.conv2d(&k, 1, 1, 1, 1).unwrap();
            let _ = dev.synchronize();
            let t0 = std::time::Instant::now();
            const N: usize = 5;
            for _ in 0..N {
                let _ = x.conv2d(&k, 1, 1, 1, 1).unwrap();
            }
            let _ = dev.synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / N as f64;
            let gflop = 2.0 * (c * c * 9) as f64 * (hw * hw) as f64 / 1e9;
            eprintln!(
                "conv2d [1,{c},{hw},{hw}] x [{c},{c},3,3]: {ms:7.2} ms  \
                 {:6.1} GFLOP  -> {:6.1} TFLOP/s",
                gflop,
                gflop / ms
            );
        }
    }
}
