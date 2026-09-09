//! Stable Audio Open (stabilityai, 44.1 kHz stereo, up to ~47 s) - native port.
//!
//! Architecture (model_config.json): Oobleck VAE (latent 64 @ sample_rate/2048)
//! + t5-base text conditioning + timing conditioners + a 24x1536 continuous-
//! transformer DiT (v-objective diffusion). This module starts with the VAE
//! DECODER - the first parity milestone; DiT + conditioning follow.
//!
//! Weight-normed convs are FOLDED at load (w = g * v / ||v||, per out channel),
//! and the SnakeBeta activations pre-exponentiate their log-scale alpha/beta,
//! so the forward is plain convs + a fused-able `x + sin^2(a x)/b` pointwise.

use crate::tensor::safetensors_io::SafeTensorsLoader;
use crate::tensor::{DType, Device, Error, Result, Tensor};

const LATENT_DIM: usize = 64;
const CHANNELS: usize = 128;
const C_MULTS: [usize; 5] = [1, 2, 4, 8, 16];
const STRIDES: [usize; 5] = [2, 4, 4, 8, 8];

/// SnakeBeta with log-scale parameters, pre-exponentiated:
/// `y = x + (1/beta) * sin(alpha * x)^2`, per channel. Runs as device
/// tensor ops so the decoder can live on GPU end-to-end.
struct SnakeBeta {
    /// `exp(alpha)` as `[C, 1]` (broadcasts over time).
    alpha: Tensor,
    /// `1 / (exp(beta) + 1e-9)` as `[C, 1]`.
    inv_beta: Tensor,
}

impl SnakeBeta {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [C, T] (channel-major, batch-free). Pointwise per channel.
        x.broadcast_mul(&self.alpha)?
            .sin()?
            .sqr()?
            .broadcast_mul(&self.inv_beta)?
            .add(x)
    }
}

/// A weight-norm-folded 1-D convolution (regular or transposed).
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    transposed: bool,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Tensor conv ops want [B, C, T].
        let (c, t) = x.shape().dims2()?;
        let xb = x.reshape((1, c, t))?;
        let y = if self.transposed {
            xb.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?
        } else {
            xb.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?
        };
        let (_, oc, ot) = y.shape().dims3()?;
        let mut y = y.reshape((oc, ot))?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&b.reshape((oc, 1))?)?;
        }
        Ok(y)
    }
}

struct ResidualUnit {
    snake1: SnakeBeta,
    conv1: Conv,
    snake2: SnakeBeta,
    conv2: Conv,
}

impl ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.snake1.forward(x)?;
        h = self.conv1.forward(&h)?;
        h = self.snake2.forward(&h)?;
        h = self.conv2.forward(&h)?;
        &h + x
    }
}

struct DecoderBlock {
    snake: SnakeBeta,
    upsample: Conv,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.snake.forward(x)?;
        h = self.upsample.forward(&h)?;
        for r in &self.res {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

pub struct OobleckDecoder {
    head: Conv,
    blocks: Vec<DecoderBlock>,
    tail_snake: SnakeBeta,
    tail: Conv,
}

impl OobleckDecoder {
    /// Decode a latent `[64, T]` into audio `[2, T * 2048]` (f32, ~[-1, 1]).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let (c, _) = latent.shape().dims2()?;
        if c != LATENT_DIM {
            return Err(Error::msg(format!(
                "oobleck decoder expects [{LATENT_DIM}, T] latents, got {c} channels"
            )));
        }
        let mut h = self.head.forward(latent)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.tail_snake.forward(&h)?;
        self.tail.forward(&h)
    }
}

/// Read a tensor as f32 host data.
fn get_f32(loader: &SafeTensorsLoader, name: &str) -> Result<Tensor> {
    loader
        .load(name)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)
}

/// Load a weight-normed conv's folded weight (+ optional bias). Handles both
/// checkpoint spellings: `weight_g`/`weight_v` and torch>=2.1's
/// `parametrizations.weight.original0/original1`.
fn load_wn_conv(
    loader: &SafeTensorsLoader,
    dev: &Device,
    prefix: &str,
    stride: usize,
    padding: usize,
    dilation: usize,
    transposed: bool,
) -> Result<Conv> {
    let (g_name, v_name) = if loader.contains(&format!("{prefix}.weight_g")) {
        (format!("{prefix}.weight_g"), format!("{prefix}.weight_v"))
    } else {
        (
            format!("{prefix}.parametrizations.weight.original0"),
            format!("{prefix}.parametrizations.weight.original1"),
        )
    };
    let g = get_f32(loader, &g_name)?; // [O,1,1] (or [1,I,1] for transposed? torch: dim=0 default)
    let v = get_f32(loader, &v_name)?; // conv: [O,I,K]; transposed: [I,O,K]
    let dims = v.dims().to_vec();
    let gv = g.to_vec_f32();
    let vv = v.to_vec_f32();
    // Weight-norm dim=0: the norm runs over all dims but the FIRST. That is the
    // out-channel axis for a regular conv and the IN-channel axis for a
    // transposed conv (torch keeps dim=0 in both).
    let d0 = dims[0];
    let rest: usize = dims[1..].iter().product();
    if gv.len() != d0 {
        return Err(Error(format!(
            "{prefix}: weight_g has {} entries, dim0 is {d0}",
            gv.len()
        )));
    }
    let mut w = vec![0f32; vv.len()];
    for i in 0..d0 {
        let seg = &vv[i * rest..(i + 1) * rest];
        let norm = seg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let scale = gv[i] / norm;
        for (j, s) in seg.iter().enumerate() {
            w[i * rest + j] = s * scale;
        }
    }
    let weight = Tensor::from_vec_f32(w, dims)?.to_device(dev)?;
    let bias_name = format!("{prefix}.bias");
    let bias = if loader.contains(&bias_name) {
        Some(get_f32(loader, &bias_name)?.to_device(dev)?)
    } else {
        None
    };
    Ok(Conv {
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
    })
}

fn load_snake(loader: &SafeTensorsLoader, dev: &Device, prefix: &str) -> Result<SnakeBeta> {
    let a = get_f32(loader, &format!("{prefix}.alpha"))?.to_vec_f32();
    let b = get_f32(loader, &format!("{prefix}.beta"))?.to_vec_f32();
    let c = a.len();
    let alpha: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let inv_beta: Vec<f32> = b.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect();
    Ok(SnakeBeta {
        alpha: Tensor::from_vec_f32(alpha, (c, 1usize))?.to_device(dev)?,
        inv_beta: Tensor::from_vec_f32(inv_beta, (c, 1usize))?.to_device(dev)?,
    })
}

fn load_residual_unit(
    loader: &SafeTensorsLoader,
    dev: &Device,
    prefix: &str,
    dilation: usize,
) -> Result<ResidualUnit> {
    let pad = (dilation * 6) / 2;
    Ok(ResidualUnit {
        snake1: load_snake(loader, dev, &format!("{prefix}.layers.0"))?,
        conv1: load_wn_conv(
            loader,
            dev,
            &format!("{prefix}.layers.1"),
            1,
            pad,
            dilation,
            false,
        )?,
        snake2: load_snake(loader, dev, &format!("{prefix}.layers.2"))?,
        conv2: load_wn_conv(loader, dev, &format!("{prefix}.layers.3"), 1, 0, 1, false)?,
    })
}

/// Load the decoder from the bundled Stable Audio checkpoint. `prefix` is the
/// checkpoint's decoder root (`pretransform.model.decoder` in the release file).
pub fn load_decoder(path: &str, prefix: &str) -> Result<OobleckDecoder> {
    load_decoder_on(path, prefix, &Device::Cpu)
}

/// Load the decoder onto a specific device (GPU decode is ~2 orders of
/// magnitude faster than CPU for the long upsampled stages).
pub fn load_decoder_on(path: &str, prefix: &str, dev: &Device) -> Result<OobleckDecoder> {
    // SAFETY: read-only mmap of the checkpoint; tensors are copied out below.
    let loader = unsafe { SafeTensorsLoader::multi(&[path]) }?;
    let c_mults: Vec<usize> = std::iter::once(1).chain(C_MULTS).collect();
    let depth = c_mults.len();
    let head = load_wn_conv(&loader, dev, &format!("{prefix}.layers.0"), 1, 3, 1, false)?;
    let mut blocks = Vec::with_capacity(depth - 1);
    let mut li = 1usize;
    for i in (1..depth).rev() {
        let stride = STRIDES[i - 1];
        let bp = format!("{prefix}.layers.{li}");
        let _cin = c_mults[i] * CHANNELS;
        blocks.push(DecoderBlock {
            snake: load_snake(&loader, dev, &format!("{bp}.layers.0"))?,
            upsample: load_wn_conv(
                &loader,
                dev,
                &format!("{bp}.layers.1"),
                stride,
                stride.div_ceil(2),
                1,
                true,
            )?,
            res: [
                load_residual_unit(&loader, dev, &format!("{bp}.layers.2"), 1)?,
                load_residual_unit(&loader, dev, &format!("{bp}.layers.3"), 3)?,
                load_residual_unit(&loader, dev, &format!("{bp}.layers.4"), 9)?,
            ],
        });
        li += 1;
    }
    let tail_snake = load_snake(&loader, dev, &format!("{prefix}.layers.{li}"))?;
    let tail = load_wn_conv(
        &loader,
        dev,
        &format!("{prefix}.layers.{}", li + 1),
        1,
        3,
        1,
        false,
    )?;
    Ok(OobleckDecoder {
        head,
        blocks,
        tail_snake,
        tail,
    })
}

// ===================== Oobleck encoder (audio -> latent) =====================
//
// Mirror of the decoder: k7 head, then per stage 3 ResidualUnits (dilations
// 1/3/9) -> SnakeBeta -> strided conv (k = 2*stride), then snake + k3 conv to
// 2*latent_dim channels (VAE mean|scale). Used for audio-to-audio variations.

struct EncoderBlock {
    res: [ResidualUnit; 3],
    snake: SnakeBeta,
    downsample: Conv,
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.res {
            h = r.forward(&h)?;
        }
        h = self.snake.forward(&h)?;
        self.downsample.forward(&h)
    }
}

pub struct OobleckEncoder {
    head: Conv,
    blocks: Vec<EncoderBlock>,
    tail_snake: SnakeBeta,
    tail: Conv,
}

impl OobleckEncoder {
    /// Encode stereo audio `[2, N]` (N a multiple of 2048) into the VAE
    /// `mean` latent `[64, N/2048]` (the deterministic center; the `scale`
    /// half of the bottleneck output is dropped).
    pub fn encode_mean(&self, audio: &Tensor) -> Result<Tensor> {
        let out = self.encode_raw(audio)?;
        out.narrow(0, 0, LATENT_DIM)?.contiguous()
    }

    /// Full bottleneck input `[128, T]` = `mean | scale` stacked.
    pub fn encode_raw(&self, audio: &Tensor) -> Result<Tensor> {
        let mut h = self.head.forward(audio)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.tail_snake.forward(&h)?;
        self.tail.forward(&h)
    }
}

/// Load the encoder (`pretransform.model.encoder`) onto a device.
pub fn load_encoder_on(path: &str, prefix: &str, dev: &Device) -> Result<OobleckEncoder> {
    // SAFETY: read-only mmap of the checkpoint; tensors are copied out below.
    let loader = unsafe { SafeTensorsLoader::multi(&[path]) }?;
    let c_mults: Vec<usize> = std::iter::once(1).chain(C_MULTS).collect();
    let depth = c_mults.len();
    let head = load_wn_conv(&loader, dev, &format!("{prefix}.layers.0"), 1, 3, 1, false)?;
    let mut blocks = Vec::with_capacity(depth - 1);
    for i in 0..depth - 1 {
        let stride = STRIDES[i];
        let bp = format!("{prefix}.layers.{}", i + 1);
        blocks.push(EncoderBlock {
            res: [
                load_residual_unit(&loader, dev, &format!("{bp}.layers.0"), 1)?,
                load_residual_unit(&loader, dev, &format!("{bp}.layers.1"), 3)?,
                load_residual_unit(&loader, dev, &format!("{bp}.layers.2"), 9)?,
            ],
            snake: load_snake(&loader, dev, &format!("{bp}.layers.3"))?,
            downsample: load_wn_conv(
                &loader,
                dev,
                &format!("{bp}.layers.4"),
                stride,
                stride.div_ceil(2),
                1,
                false,
            )?,
        });
    }
    let tail_snake = load_snake(&loader, dev, &format!("{prefix}.layers.{depth}"))?;
    let tail = load_wn_conv(
        &loader,
        dev,
        &format!("{prefix}.layers.{}", depth + 1),
        1,
        1,
        1,
        false,
    )?;
    Ok(OobleckEncoder {
        head,
        blocks,
        tail_snake,
        tail,
    })
}

// ===================== DiT (continuous transformer) =====================
//
// DiffusionTransformer in "prepend" global-cond mode (no adaLN): the global
// embedding (number conditioners + timestep Fourier features) becomes one
// prepended token; 24 pre-norm blocks of self-attention (partial rotary on
// the first 32 of 64 head dims), cross-attention over the conditioning
// tokens (12 kv heads, GQA-repeated to 24 q heads), and a SwiGLU FFN.

const DIT_DIM: usize = 1536;
const DIT_HEADS: usize = 24;
const DIT_HEAD_DIM: usize = 64;
const DIT_COND_DIM: usize = 768;
const DIT_DEPTH: usize = 24;
const DIT_FF_INNER: usize = 6144;
const ROT_DIM: usize = 32; // rotated prefix of each head (half-rotary)

use crate::tensor::layer::{LayerNorm, Linear};

struct SaoBlock {
    pre_norm: LayerNorm,
    to_qkv: Linear,
    self_out: Linear,
    cross_norm: LayerNorm,
    cross_q: Linear,
    cross_kv: Linear,
    cross_out: Linear,
    ff_norm: LayerNorm,
    ff_proj: Linear,
    ff_out: Linear,
}

pub struct SaoDit {
    /// FourierFeatures weight `[128]` (in_features = 1), host-side.
    timestep_w: Vec<f32>,
    to_timestep_a: Linear,
    to_timestep_b: Linear,
    to_cond_a: Linear,
    to_cond_b: Linear,
    to_global_a: Linear,
    to_global_b: Linear,
    preprocess: Linear,
    project_in: Linear,
    project_out: Linear,
    blocks: Vec<SaoBlock>,
    /// Where each block's weights live, one entry per block.
    ///
    /// A stack that threads a single device through its block loop cannot be given a second
    /// card, whatever the loader in front of it decides: the fleet rule has nowhere to land.
    /// Measured before this existed - 7.4 GB of this DiT on the first card while the second
    /// held its bare 238 MiB of context - and on a box whose first card is busy that is not a
    /// placement, it is a spill to the host or a failure to run.
    ///
    /// All the same device is the ONE-SEGMENT PLAN and stays bit-identical: the moves below
    /// are no-ops, because a tensor already on a device is returned unchanged.
    block_devices: Vec<Device>,
    /// Rotary inverse frequencies `[16]`, host-side.
    inv_freq: Vec<f32>,
    postprocess: Linear,
    device: Device,
}

impl SaoDit {
    /// One denoise evaluation. `x` `[T, 64]` (time-major latent), `t` in
    /// (0, 1), `cross` `[S, 768]` conditioning tokens, `glob` `[1, 1536]`.
    /// Returns the v-prediction `[T, 64]`.
    pub fn forward(&self, x: &Tensor, t: f32, cross: &Tensor, glob: &Tensor) -> Result<Tensor> {
        let (seq_x, _) = x.shape().dims2()?;
        let cond = self
            .to_cond_b
            .forward(&self.to_cond_a.forward(cross)?.silu()?)?;
        let mut g = self
            .to_global_b
            .forward(&self.to_global_a.forward(glob)?.silu()?)?;

        // Timestep Fourier features: [cos(2*pi*t*w), sin(2*pi*t*w)] -> MLP.
        let two_pi_t = 2.0 * std::f32::consts::PI * t;
        let mut feats = Vec::with_capacity(self.timestep_w.len() * 2);
        for w in &self.timestep_w {
            feats.push((two_pi_t * w).cos());
        }
        for w in &self.timestep_w {
            feats.push((two_pi_t * w).sin());
        }
        let n_feats = feats.len();
        let feats = Tensor::from_vec_f32(feats, (1usize, n_feats))?.to_device(&self.device)?;
        let ts = self
            .to_timestep_b
            .forward(&self.to_timestep_a.forward(&feats)?.silu()?)?;
        g = g.add(&ts)?;

        let mut h = x.add(&self.preprocess.forward(x)?)?;
        h = self.project_in.forward(&h)?;
        h = Tensor::cat(&[&g, &h], 0)?; // prepend the global token
        let seq = seq_x + 1;

        // Rotary tables for the rotated 32-dim prefix: cos/sin [1, seq, 16],
        // duplicated across the two 16-dim halves at apply time.
        let half = ROT_DIM / 2;
        let (mut cos_v, mut sin_v) = (vec![0f32; seq * half], vec![0f32; seq * half]);
        for i in 0..seq {
            for (j, f) in self.inv_freq.iter().enumerate() {
                let a = i as f32 * f;
                cos_v[i * half + j] = a.cos();
                sin_v[i * half + j] = a.sin();
            }
        }
        let cos = Tensor::from_vec_f32(cos_v, (1usize, seq, half))?.to_device(&self.device)?;
        let sin = Tensor::from_vec_f32(sin_v, (1usize, seq, half))?.to_device(&self.device)?;

        // What crosses a block boundary: the stream, the conditioning it attends over, and
        // the rotary tables. They follow the blocks rather than being rebuilt per card,
        // because they are the same numbers on either side and a copy is cheaper than a
        // second construction. On a one-segment plan every move below is a no-op.
        dbg_stage_hash("dit_h_pre", &h);
        dbg_stage_hash("dit_cond", &cond);
        let (mut cond, mut cos, mut sin) = (cond, cos, sin);
        let mut here = self.device.clone();

        for (blk, want) in self.blocks.iter().zip(self.block_devices.iter()) {
            crate::inference::place::plan::cross_to(
                &mut here,
                want,
                &mut [&mut h, &mut cond, &mut cos, &mut sin],
            )?;
            // Self-attention (pre-norm, rotary, 24 heads).
            let y = blk.pre_norm.forward(&h)?;
            let qkv = blk.to_qkv.forward(&y)?;
            let split_heads = |t: &Tensor, off: usize| -> Result<Tensor> {
                t.narrow(1, off, DIT_DIM)?
                    .reshape((seq, DIT_HEADS, DIT_HEAD_DIM))?
                    .transpose(0, 1)?
                    .contiguous() // [H, seq, hd]
            };
            let q = self.rope(&split_heads(&qkv, 0)?, &cos, &sin)?;
            let k = self.rope(&split_heads(&qkv, DIT_DIM)?, &cos, &sin)?;
            let v = split_heads(&qkv, 2 * DIT_DIM)?;
            let ctx = attn(&q, &k, &v)?; // [H, seq, hd]
            let ctx = ctx.transpose(0, 1)?.contiguous()?.reshape((seq, DIT_DIM))?;
            h = h.add(&blk.self_out.forward(&ctx)?)?;

            // Cross-attention over the conditioning tokens (no rotary; the 12
            // kv heads are repeated to match the 24 q heads).
            let (s_c, _) = cond.shape().dims2()?;
            let y = blk.cross_norm.forward(&h)?;
            let q = blk
                .cross_q
                .forward(&y)?
                .reshape((seq, DIT_HEADS, DIT_HEAD_DIM))?
                .transpose(0, 1)?
                .contiguous()?;
            let kv = blk.cross_kv.forward(&cond)?; // [S, 2*768]
            let kv_heads = DIT_COND_DIM / DIT_HEAD_DIM; // 12
            let take = |off: usize| -> Result<Tensor> {
                let heads = kv
                    .narrow(1, off, DIT_COND_DIM)?
                    .reshape((s_c, kv_heads, DIT_HEAD_DIM))?
                    .transpose(0, 1)?
                    .contiguous()?;
                crate::tensor::ops::repeat_kv_unbatched(&heads, DIT_HEADS / kv_heads)
            };
            let ctx = attn(&q, &take(0)?, &take(DIT_COND_DIM)?)?;
            let ctx = ctx.transpose(0, 1)?.contiguous()?.reshape((seq, DIT_DIM))?;
            h = h.add(&blk.cross_out.forward(&ctx)?)?;

            // SwiGLU FFN: proj -> [x | gate] -> x * silu(gate) -> out.
            let y = blk.ff_norm.forward(&h)?;
            let p = blk.ff_proj.forward(&y)?;
            let a = p.narrow(1, 0, DIT_FF_INNER)?;
            let gate = p.narrow(1, DIT_FF_INNER, DIT_FF_INNER)?;
            h = h.add(&blk.ff_out.forward(&a.mul(&gate.silu()?)?)?)?;
        }

        // The tail rides on the primary device, so a split stack comes home before it
        // projects out - the same place a single-device stack already was.
        crate::inference::place::plan::cross_to(&mut here, &self.device, &mut [&mut h])?;
        let out = self
            .project_out
            .forward(&h)?
            .narrow(0, 1, seq_x)?
            .contiguous()?;
        out.add(&self.postprocess.forward(&out)?)
    }

    /// Partial rotary (GPT-J half-split on the 32-dim prefix): the first 16
    /// and second 16 dims form the rotation pair; the remaining 32 pass through.
    fn rope(&self, t: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let hd = t.shape().dims3()?.2;
        let half = ROT_DIM / 2;
        let a = t.narrow(2, 0, half)?.contiguous()?;
        let b = t.narrow(2, half, half)?.contiguous()?;
        let pass = t.narrow(2, ROT_DIM, hd - ROT_DIM)?.contiguous()?;
        let ra = a.broadcast_mul(cos)?.sub(&b.broadcast_mul(sin)?)?;
        let rb = b.broadcast_mul(cos)?.add(&a.broadcast_mul(sin)?)?;
        Tensor::cat(&[&ra, &rb, &pass], 2)
    }
}

/// Plain softmax attention `[H, Sq, hd] x [H, Sk, hd] -> [H, Sq, hd]`.
fn attn(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_, _, hd) = q.shape().dims3()?;
    let scale = 1.0 / (hd as f32).sqrt();
    let scores = q
        .matmul(&k.transpose(1, 2)?.contiguous()?)?
        .affine(scale, 0.0)?;
    scores.softmax_last_dim()?.matmul(v)
}

/// Load the DiT from the checkpoint (`model.model.*`).
pub fn load_dit(path: &str, device: &Device) -> Result<SaoDit> {
    load_dit_planned(path, &vec![device.clone(); DIT_DEPTH], device)
}

/// [`load_dit`] with a device per block, which is what lets this stack span cards.
///
/// `block_devices` must hold one entry per block; everything outside the stack - the
/// embedders, the projections, the rotary tables - rides on `primary`, which is where the
/// forward starts and ends. Passing the same device for every block reproduces the
/// single-device load exactly, so the plan with one segment is the old path and not a
/// second one.
pub fn load_dit_planned(path: &str, block_devices: &[Device], primary: &Device) -> Result<SaoDit> {
    if block_devices.len() != DIT_DEPTH {
        return Err(crate::tensor::Error::msg(format!(
            "stable-audio: a placement of {} blocks for a stack of {DIT_DEPTH}",
            block_devices.len(),
        )));
    }
    let device = primary;
    let loader = unsafe { SafeTensorsLoader::multi(&[path]) }?;
    let pfx = "model.model";
    let get_on = |name: String, d: &Device| -> Result<Tensor> {
        loader.load(&name)?.to_dtype(DType::F32)?.to_device(d)
    };
    let lin_on = |name: String, d: &Device| -> Result<Linear> {
        Linear::new(get_on(format!("{name}.weight"), d)?, None)
    };
    let lin_b_on = |name: String, d: &Device| -> Result<Linear> {
        Linear::new(
            get_on(format!("{name}.weight"), d)?,
            Some(get_on(format!("{name}.bias"), d)?),
        )
    };
    let norm_on = |name: String, d: &Device| -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            get_on(format!("{name}.gamma"), d)?,
            Some(get_on(format!("{name}.beta"), d)?),
            1e-5,
        ))
    };
    let get = |name: String| -> Result<Tensor> {
        loader.load(&name)?.to_dtype(DType::F32)?.to_device(device)
    };
    let lin =
        |name: String| -> Result<Linear> { Linear::new(get(format!("{name}.weight"))?, None) };
    let lin_b = |name: String| -> Result<Linear> {
        Linear::new(
            get(format!("{name}.weight"))?,
            Some(get(format!("{name}.bias"))?),
        )
    };
    // 1x1 convs are plain linears over the channel dim.
    let conv1 = |name: String| -> Result<Linear> {
        let w = get(format!("{name}.weight"))?;
        let (o, i, _) = w.shape().dims3()?;
        Linear::new(w.reshape((o, i))?, None)
    };

    let timestep_w = get(format!("{pfx}.timestep_features.weight"))?.to_vec_f32();
    let tr = format!("{pfx}.transformer");
    let inv_freq = get(format!("{tr}.rotary_pos_emb.inv_freq"))?.to_vec_f32();
    let mut blocks = Vec::with_capacity(DIT_DEPTH);
    for (i, d) in block_devices.iter().enumerate() {
        let b = format!("{tr}.layers.{i}");
        blocks.push(SaoBlock {
            pre_norm: norm_on(format!("{b}.pre_norm"), d)?,
            to_qkv: lin_on(format!("{b}.self_attn.to_qkv"), d)?,
            self_out: lin_on(format!("{b}.self_attn.to_out"), d)?,
            cross_norm: norm_on(format!("{b}.cross_attend_norm"), d)?,
            cross_q: lin_on(format!("{b}.cross_attn.to_q"), d)?,
            cross_kv: lin_on(format!("{b}.cross_attn.to_kv"), d)?,
            cross_out: lin_on(format!("{b}.cross_attn.to_out"), d)?,
            ff_norm: norm_on(format!("{b}.ff_norm"), d)?,
            ff_proj: lin_b_on(format!("{b}.ff.ff.0.proj"), d)?,
            ff_out: lin_b_on(format!("{b}.ff.ff.2"), d)?,
        });
    }
    Ok(SaoDit {
        timestep_w,
        to_timestep_a: lin_b(format!("{pfx}.to_timestep_embed.0"))?,
        to_timestep_b: lin_b(format!("{pfx}.to_timestep_embed.2"))?,
        to_cond_a: lin(format!("{pfx}.to_cond_embed.0"))?,
        to_cond_b: lin(format!("{pfx}.to_cond_embed.2"))?,
        to_global_a: lin(format!("{pfx}.to_global_embed.0"))?,
        to_global_b: lin(format!("{pfx}.to_global_embed.2"))?,
        preprocess: conv1(format!("{pfx}.preprocess_conv"))?,
        project_in: lin(format!("{tr}.project_in"))?,
        project_out: lin(format!("{tr}.project_out"))?,
        blocks,
        block_devices: block_devices.to_vec(),
        inv_freq,
        postprocess: conv1(format!("{pfx}.postprocess_conv"))?,
        device: device.clone(),
    })
}

/// NumberConditioner: clamp+normalize to [0,1], learned Fourier features
/// `[x, sin(2*pi*x*w), cos(2*pi*x*w)]`, then a Linear to 768.
pub struct SaoNumberConditioner {
    weights: Vec<f32>, // [128]
    proj: Linear,
    min_val: f32,
    max_val: f32,
    device: Device,
}

impl SaoNumberConditioner {
    pub fn load(path: &str, which: &str, device: &Device) -> Result<Self> {
        let loader = unsafe { SafeTensorsLoader::multi(&[path]) }?;
        let p = format!("conditioner.conditioners.{which}.embedder.embedding");
        let get = |name: String| -> Result<Tensor> {
            loader.load(&name)?.to_dtype(DType::F32)?.to_device(device)
        };
        let weights = get(format!("{p}.0.weights"))?.to_vec_f32();
        let proj = Linear::new(
            get(format!("{p}.1.weight"))?,
            Some(get(format!("{p}.1.bias"))?),
        )?;
        Ok(Self {
            weights,
            proj,
            min_val: 0.0,
            max_val: 512.0,
            device: device.clone(),
        })
    }

    /// Embed one value -> `[1, 768]`.
    pub fn forward(&self, value: f32) -> Result<Tensor> {
        let x = ((value.clamp(self.min_val, self.max_val)) - self.min_val)
            / (self.max_val - self.min_val);
        let mut feats = Vec::with_capacity(1 + self.weights.len() * 2);
        feats.push(x);
        let two_pi_x = 2.0 * std::f32::consts::PI * x;
        for w in &self.weights {
            feats.push((two_pi_x * w).sin());
        }
        for w in &self.weights {
            feats.push((two_pi_x * w).cos());
        }
        let n = feats.len();
        let feats = Tensor::from_vec_f32(feats, (1usize, n))?.to_device(&self.device)?;
        self.proj.forward(&feats)
    }
}

// ===================== v-diffusion sampler =====================
//
// k-diffusion recipe used by stable-audio-tools: VDenoiser scalings around
// the v-objective DiT, a polyexponential sigma schedule, and the DPM++(2M)
// multistep update (deterministic).

/// Polyexponential sigma schedule (rho=1 -> log-linear), with the trailing 0.
pub fn sigmas_polyexponential(steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(steps + 1);
    for i in 0..steps {
        let ramp = (1.0 - i as f32 / (steps - 1).max(1) as f32).powf(rho);
        sigmas.push((ramp * (sigma_max.ln() - sigma_min.ln()) + sigma_min.ln()).exp());
    }
    sigmas.push(0.0);
    sigmas
}

/// One VDenoiser evaluation: `denoised = model(x*c_in, t(sigma)) * c_out + x*c_skip`.
/// `model` maps `(x_scaled, t)` to the v-prediction.
fn v_denoise(
    model: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    x: &Tensor,
    sigma: f32,
) -> Result<Tensor> {
    let c_skip = 1.0 / (sigma * sigma + 1.0);
    let c_out = -sigma / (sigma * sigma + 1.0).sqrt();
    let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
    let t = sigma.atan() / std::f32::consts::PI * 2.0;
    let v = model(&x.affine(c_in, 0.0)?, t)?;
    v.affine(c_out, 0.0)?.add(&x.affine(c_skip, 0.0)?)
}

/// DPM++(2M) over the VDenoiser: deterministic multistep sampling.
/// `noise` must be unit-variance; it is scaled by `sigmas[0]` internally.
pub fn sample_dpmpp_2m(
    model: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    noise: &Tensor,
    sigmas: &[f32],
    mut on_step: impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<Tensor> {
    let mut x = noise.affine(sigmas[0], 0.0)?;
    let t_fn = |s: f32| -> f32 { -(s.ln()) };
    let mut old_denoised: Option<Tensor> = None;
    let n = sigmas.len() - 1;
    for i in 0..n {
        let denoised = v_denoise(model, &x, sigmas[i])?;
        on_step(crate::inference::serve::progress::phase::DENOISE, i + 1, n)?;
        if sigmas[i + 1] == 0.0 {
            x = denoised;
            continue;
        }
        let (t, t_next) = (t_fn(sigmas[i]), t_fn(sigmas[i + 1]));
        let h = t_next - t;
        let ratio = sigmas[i + 1] / sigmas[i];
        let em = -((-h).exp_m1());
        let d = match &old_denoised {
            None => denoised.clone(),
            Some(old) => {
                let h_last = t - t_fn(sigmas[i - 1]);
                let r = h_last / h;
                denoised
                    .affine(1.0 + 1.0 / (2.0 * r), 0.0)?
                    .add(&old.affine(-1.0 / (2.0 * r), 0.0)?)?
            }
        };
        x = x.affine(ratio, 0.0)?.add(&d.affine(em, 0.0)?)?;
        old_denoised = Some(denoised);
    }
    Ok(x)
}

// ===================== end-to-end pipeline =====================

pub const SAMPLE_RATE: u32 = 44100;
/// Audio samples per latent frame (Oobleck stride product 2*4*4*8*8).
const LATENT_STRIDE: usize = 2048;
/// Training window: 2_097_152 samples = 1024 latent frames (~47.55 s).
const MAX_LATENT_T: usize = 1024;

/// Resolve the stable-audio model directory from the configured HF models dir.
pub fn model_dir() -> Option<std::path::PathBuf> {
    let hf = crate::config::hf_models_dir();
    let base = if hf.file_name().is_some_and(|n| n == "hub") {
        hf.parent().map(|p| p.to_path_buf()).unwrap_or(hf)
    } else {
        hf
    };
    let dir = base.join("stable-audio");
    dir.join("model.safetensors").is_file().then_some(dir)
}

/// The loaded pipeline components, kept resident between requests (the
/// reloads dominated warm-request latency: ~2.5 s vs 1.5 s of denoising).
struct SaoResident {
    t5_dir: String,
    t5: crate::inference::model::t5::flan::T5Encoder,
    dit: SaoDit,
    decoder: OobleckDecoder,
    /// Lazily loaded on the first audio-to-audio request (variations only);
    /// lives and dies with the resident, so the reclaim hook frees it too.
    encoder: std::sync::Mutex<Option<std::sync::Arc<OobleckEncoder>>>,
    nc_start: SaoNumberConditioner,
    nc_total: SaoNumberConditioner,
    ckpt_mtime: Option<std::time::SystemTime>,
    device: Device,
}

impl SaoResident {
    fn encoder(&self) -> Result<std::sync::Arc<OobleckEncoder>> {
        let mut slot = self.encoder.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = slot.as_ref() {
            return Ok(e.clone());
        }
        let ckpt_path = model_dir()
            .ok_or_else(|| Error::msg("stable-audio weights missing".to_string()))?
            .join("model.safetensors");
        let ckpt = ckpt_path
            .to_str()
            .ok_or_else(|| Error::msg("bad checkpoint path".to_string()))?;
        let enc = std::sync::Arc::new(load_encoder_on(
            ckpt,
            "pretransform.model.encoder",
            &self.device,
        )?);
        *slot = Some(enc.clone());
        Ok(enc)
    }
}

fn resident_slot() -> &'static std::sync::Mutex<Option<std::sync::Arc<SaoResident>>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<std::sync::Arc<SaoResident>>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

/// The parts of the resident pipeline and where each sits; none when nothing is resident.
pub fn resident_parts() -> Option<crate::inference::serve::progress::placement::Parts> {
    let slot = resident_slot().try_lock().ok()?;
    let r = slot.as_ref()?;
    let mut parts = vec![
        ("text-encoder".to_string(), r.t5.placement()),
        ("dit".to_string(), r.dit.placement()),
        ("vae".to_string(), r.decoder.placement()),
    ];
    if let Ok(enc) = r.encoder.try_lock() {
        if let Some(enc) = enc.as_ref() {
            parts.push(("vae-encoder".to_string(), enc.placement()));
        }
    }
    Some(parts)
}

/// Is the pipeline resident right now?
///
/// The unload contract has to be able to ask before it answers: a request naming
/// stable-audio when nothing is loaded must not be reported as having freed it.
pub fn is_resident() -> bool {
    resident_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

/// Drop the resident pipeline (vram_manager reclaim hook). Returns 1 if
/// something was actually released.
pub fn unload_resident() -> usize {
    let dropped = resident_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .is_some();
    usize::from(dropped)
}

/// On-disk bytes of everything the resident keeps loaded (DiT + decoder
/// checkpoint + t5-base) - the figure the pressure protocol should demand
/// before a stable-audio request places its pipeline.
pub fn resident_bytes() -> u64 {
    let Some(dir) = model_dir() else { return 0 };
    let sz = |p: std::path::PathBuf| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    sz(dir.join("model.safetensors")) + sz(dir.join("t5-base/model.safetensors"))
}

fn resident() -> Result<std::sync::Arc<SaoResident>> {
    let dir = model_dir().ok_or_else(|| {
        Error::msg(
            "stable-audio weights not found (expected <hf models dir>/stable-audio)".to_string(),
        )
    })?;
    let ckpt_path = dir.join("model.safetensors");
    let ckpt = ckpt_path
        .to_str()
        .ok_or_else(|| Error::msg("bad checkpoint path".to_string()))?;
    let mtime = std::fs::metadata(&ckpt_path)
        .and_then(|m| m.modified())
        .ok();
    let mut slot = resident_slot().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = slot.as_ref() {
        if r.ckpt_mtime == mtime {
            return Ok(r.clone());
        }
    }
    let t5_dir = dir.join("t5-base");
    let t5_dir = t5_dir
        .to_str()
        .ok_or_else(|| Error::msg("bad t5 path".to_string()))?
        .to_string();
    let sz = std::fs::metadata(&ckpt_path)
        .map(|m| m.len())
        .unwrap_or(5 << 30);
    let device = crate::inference::model::acestep::vae::vae_best_device(sz);
    let r = std::sync::Arc::new(SaoResident {
        t5: crate::inference::model::t5::flan::T5Encoder::from_dir(&t5_dir)?,
        t5_dir,
        dit: load_dit(ckpt, &device)?,
        decoder: load_decoder_on(ckpt, "pretransform.model.decoder", &device)?,
        encoder: std::sync::Mutex::new(None),
        nc_start: SaoNumberConditioner::load(ckpt, "seconds_start", &device)?,
        nc_total: SaoNumberConditioner::load(ckpt, "seconds_total", &device)?,
        ckpt_mtime: mtime,
        device,
    });
    *slot = Some(r.clone());
    Ok(r)
}

/// Render `prompt` to stereo 44.1 kHz audio. Returns interleaved L/R samples.
///
/// `seconds` conditions the model (seconds_total) and sets the latent length
/// (up to the ~47.5 s training window); `cfg` > 1 enables classifier-free
/// guidance (uncond = zeroed conditioning tokens, or the encoded
/// `negative_prompt` when non-empty). `on_step(done, total)` reports progress and
/// may ABORT the render by returning Err - a diffusion sampler runs in
/// spawn_blocking, so a dropped request cannot stop it any other way. Callers that
/// have a cancel token must check it here.
/// FNV of a tensor's f32s at debug level - the stage probe a nondeterminism hunt
/// bisects with. Costs a device download, so it only runs when debug logging is on.
fn dbg_stage_hash(tag: &str, t: &Tensor) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let host = match t.to_device(&Device::Cpu) {
        Ok(h) => h,
        Err(_) => return,
    };
    let v = host.to_vec_f32();
    let mut h: u64 = 0xcbf29ce484222325;
    for x in &v {
        for b in x.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    tracing::debug!("sa stage[{tag}]: {h:016x} ({} elems)", v.len());
}

/// Pins the process-global F32 GEMM precision to FULL for a scope, restoring the prior
/// state on drop. The switch is flipped by whichever engine loaded last, and a seed must
/// not answer different audio because an LLM load happened in between - the last face of
/// the "same seed, different audio depending on what else is resident" defect.
struct FullPrecisionGuard(bool);

impl FullPrecisionGuard {
    fn pin() -> Self {
        #[cfg(feature = "cuda")]
        {
            let was = crate::tensor::cuda::gemm_reduced_precision_f32();
            crate::tensor::cuda::set_gemm_reduced_precision_f32(false);
            Self(was)
        }
        #[cfg(not(feature = "cuda"))]
        Self(false)
    }
}

impl Drop for FullPrecisionGuard {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        crate::tensor::cuda::set_gemm_reduced_precision_f32(self.0);
    }
}

pub fn render(
    prompt: &str,
    negative_prompt: &str,
    seconds: f32,
    steps: usize,
    cfg: f32,
    seed: u64,
    on_step: impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<(Vec<f32>, u32)> {
    render_with_init(
        prompt,
        negative_prompt,
        seconds,
        steps,
        cfg,
        seed,
        None,
        on_step,
    )
}

/// Audio-to-audio: like [`render`], but the diffusion starts from the encoded
/// `init` audio (interleaved stereo 44.1 kHz) noised to `noise_level` (the
/// schedule's sigma_max - ~1 keeps the source close, ~10+ reinterprets it).
pub fn render_with_init(
    prompt: &str,
    negative_prompt: &str,
    seconds: f32,
    steps: usize,
    cfg: f32,
    seed: u64,
    init: Option<(&[f32], f32)>,
    mut on_step: impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<(Vec<f32>, u32)> {
    let _full_precision = FullPrecisionGuard::pin();
    let res = resident()?;
    crate::inference::place::vram_manager::touch("stable-audio");
    let device = res.device.clone();

    // With an init clip the length comes from the clip itself.
    let seconds = match init {
        Some((pcm, _)) => ((pcm.len() / 2) as f32 / SAMPLE_RATE as f32).clamp(1.0, 47.0),
        None => seconds.clamp(1.0, 47.0),
    };
    let t_latent = (((seconds as f64) * SAMPLE_RATE as f64 / LATENT_STRIDE as f64).ceil() as usize)
        .clamp(1, MAX_LATENT_T);

    // Encode the init clip to its latent (VAE mean), padded to whole frames.
    let init_latent = match init {
        Some((pcm, _)) => {
            let frames = (pcm.len() / 2).min(t_latent * LATENT_STRIDE);
            let n = t_latent * LATENT_STRIDE;
            let mut planar = vec![0f32; 2 * n];
            for (i, fr) in pcm.chunks_exact(2).take(frames).enumerate() {
                planar[i] = fr[0];
                planar[n + i] = fr[1];
            }
            let audio = Tensor::from_vec_f32(planar, (2usize, n))?.to_device(&device)?;
            let enc = res.encoder()?;
            let lat = enc.encode_mean(&audio)?; // [64, T]
            Some(lat.transpose(0, 1)?.contiguous()?) // [T, 64]
        }
        None => None,
    };

    // Conditioning: t5 prompt tokens + the two number conditioners, both as
    // cross-attention tokens and (numbers only) as the global embedding.
    let ids = crate::inference::model::t5::flan::t5_tokenize(&res.t5_dir, prompt)?;
    let prompt_emb = res.t5.encode(&ids)?.to_device(&device)?;
    dbg_stage_hash("prompt_emb", &prompt_emb);
    let neg_emb = if cfg > 1.0 && !negative_prompt.trim().is_empty() {
        let nids = crate::inference::model::t5::flan::t5_tokenize(&res.t5_dir, negative_prompt)?;
        Some(res.t5.encode(&nids)?.to_device(&device)?)
    } else {
        None
    };

    let nc_start = &res.nc_start;
    let nc_total = &res.nc_total;
    let e_start = nc_start.forward(0.0)?;
    let e_total = nc_total.forward(seconds)?;
    let cross = Tensor::cat(&[&prompt_emb, &e_start, &e_total], 0)?;
    let cross_uncond = match &neg_emb {
        Some(n) => Some(Tensor::cat(&[n, &e_start, &e_total], 0)?),
        None if cfg > 1.0 => Some(cross.zeros_like()?),
        None => None,
    };
    let glob = Tensor::cat(&[&e_start, &e_total], 1)?;

    let dit = &res.dit;

    // Seeded unit noise [T, 64] (LCG + Box-Muller).
    let mut rng = seed.max(1);
    let mut u01 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
    };
    let noise: Vec<f32> = (0..t_latent * LATENT_DIM)
        .map(|_| {
            let (u1, u2) = (u01(), u01());
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect();
    let noise = Tensor::from_vec_f32(noise, (t_latent, LATENT_DIM))?.to_device(&device)?;
    dbg_stage_hash("noise", &noise);

    // Variations shrink the schedule to the init noise level (the reference
    // sample_k semantics: sigma_max = init_noise_level, x0 = init + noise*s0).
    let sigma_max = match init {
        Some((_, level)) => level.clamp(0.31, 500.0),
        None => 500.0,
    };
    let sigmas = sigmas_polyexponential(steps.max(1), 0.3, sigma_max, 1.0);
    let model = |x: &Tensor, t: f32| -> Result<Tensor> {
        let v_cond = dit.forward(x, t, &cross, &glob)?;
        match &cross_uncond {
            Some(cu) if cfg != 1.0 => {
                let v_un = dit.forward(x, t, cu, &glob)?;
                v_un.add(&v_cond.sub(&v_un)?.affine(cfg, 0.0)?)
            }
            _ => Ok(v_cond),
        }
    };
    // The reference recipe: DPM++(3M) SDE. (2M stays as the deterministic
    // parity-tested reference path.)
    let latent = sample_dpmpp_3m_sde_from(
        &model,
        &noise,
        init_latent.as_ref(),
        &sigmas,
        seed,
        |ph, i, n| on_step(ph, i, n),
    )?;

    // Decode: [T, 64] -> [64, T] -> [2, T*2048], trim to the requested length.
    dbg_stage_hash("latent", &latent);
    let latent = latent.transpose(0, 1)?.contiguous()?;
    let audio = match res.decoder.decode(&latent) {
        // Device out of memory even after the allocator-level reclaim: the
        // decode transients no longer fit next to whatever else moved onto
        // the card. Re-plan on the CURRENT best device (free-VRAM-gated, so
        // it sees the pressure and may pick another GPU or the CPU) and
        // decode there instead of failing the render.
        Err(e) if e.is_oom() => {
            let ckpt_path = model_dir()
                .ok_or_else(|| Error::msg("stable-audio weights missing".to_string()))?
                .join("model.safetensors");
            let ckpt = ckpt_path
                .to_str()
                .ok_or_else(|| Error::msg("bad checkpoint path".to_string()))?;
            let sz = std::fs::metadata(&ckpt_path)
                .map(|m| m.len())
                .unwrap_or(5 << 30);
            let alt = crate::inference::model::acestep::vae::vae_best_device(sz);
            tracing::warn!(
                "stable-audio decode OOM on {:?}; re-planning decode on {alt:?}",
                device
            );
            let fallback = load_decoder_on(ckpt, "pretransform.model.decoder", &alt)?;
            match fallback.decode(&latent.to_device(&alt)?) {
                Err(e2) if e2.is_oom() => {
                    tracing::warn!("stable-audio decode OOM again on {alt:?}; decoding on CPU");
                    let cpu_dec =
                        load_decoder_on(ckpt, "pretransform.model.decoder", &Device::Cpu)?;
                    cpu_dec.decode(&latent.to_device(&Device::Cpu)?)?
                }
                r => r?,
            }
        }
        r => r?,
    };
    let (ch, n) = audio.shape().dims2()?;
    let pcm = audio.to_vec_f32();
    let want = ((seconds as f64 * SAMPLE_RATE as f64) as usize).min(n);
    // Peak-normalize like the reference pipeline (raw decodes can exceed 1.0
    // at high CFG; a hard clamp would distort the transients instead).
    if tracing::enabled!(tracing::Level::DEBUG) {
        let mut h: u64 = 0xcbf29ce484222325;
        for x in &pcm {
            for b in x.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        tracing::debug!("sa stage[pcm]: {h:016x} ({} elems)", pcm.len());
    }
    let peak = pcm.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let gain = if peak > 1.0 { 1.0 / peak } else { 1.0 };
    let mut out = Vec::with_capacity(want * 2);
    for i in 0..want {
        for c in 0..ch.min(2) {
            out.push((pcm[c * n + i] * gain).clamp(-1.0, 1.0));
        }
    }
    Ok((out, SAMPLE_RATE))
}

/// DPM-Solver++(3M) SDE over the VDenoiser - the reference pipeline's default
/// sampler (eta = s_noise = 1). Third-order multistep with per-step noise
/// injection; `noise` unit-variance, scaled by `sigmas[0]` internally. The
/// per-step noise comes from the seeded generator (plain gaussian - the
/// Brownian-tree sampler only matters for cross-step-count reproducibility).
pub fn sample_dpmpp_3m_sde(
    model: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    noise: &Tensor,
    sigmas: &[f32],
    seed: u64,
    on_step: impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<Tensor> {
    sample_dpmpp_3m_sde_from(model, noise, None, sigmas, seed, on_step)
}

/// [`sample_dpmpp_3m_sde`] with an optional init latent: `x0 = init +
/// noise * sigmas[0]` (audio-to-audio variations start partway down the
/// schedule instead of from pure noise).
pub fn sample_dpmpp_3m_sde_from(
    model: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    noise: &Tensor,
    init: Option<&Tensor>,
    sigmas: &[f32],
    seed: u64,
    mut on_step: impl FnMut(&str, usize, usize) -> Result<()>,
) -> Result<Tensor> {
    let mut x = noise.affine(sigmas[0], 0.0)?;
    if let Some(init) = init {
        x = x.add(init)?;
    }
    dbg_stage_hash("x_init", &x);
    let elems = x.elem_count();
    let device = x.device().clone();
    let shape = x.dims().to_vec();
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).max(1);
    let mut gauss = move || -> Vec<f32> {
        let mut u01 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
        };
        (0..elems)
            .map(|_| {
                let (u1, u2) = (u01(), u01());
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect()
    };
    let (mut den_1, mut den_2): (Option<Tensor>, Option<Tensor>) = (None, None);
    let (mut h_1, mut h_2): (Option<f32>, Option<f32>) = (None, None);
    let n = sigmas.len() - 1;
    for i in 0..n {
        let denoised = v_denoise(model, &x, sigmas[i])?;
        if i == 0 {
            dbg_stage_hash("denoised0", &denoised);
        }
        on_step(crate::inference::serve::progress::phase::DENOISE, i + 1, n)?;
        if sigmas[i + 1] == 0.0 {
            x = denoised;
            continue;
        }
        let (t, s) = (-(sigmas[i].ln()), -(sigmas[i + 1].ln()));
        let h = s - t;
        let h_eta = h * 2.0; // h * (eta + 1), eta = 1
        let em = -((-h_eta).exp_m1());
        x = x
            .affine((-h_eta).exp(), 0.0)?
            .add(&denoised.affine(em, 0.0)?)?;
        if let (Some(hp), Some(hpp), Some(d1p), Some(d2p)) = (h_1, h_2, &den_1, &den_2) {
            let (r0, r1) = (hp / h, hpp / h);
            let d1_0 = denoised.sub(d1p)?.affine(1.0 / r0, 0.0)?;
            let d1_1 = d1p.sub(d2p)?.affine(1.0 / r1, 0.0)?;
            let d1 = d1_0.add(&d1_0.sub(&d1_1)?.affine(r0 / (r0 + r1), 0.0)?)?;
            let d2 = d1_0.sub(&d1_1)?.affine(1.0 / (r0 + r1), 0.0)?;
            let phi_2 = (-h_eta).exp_m1() / h_eta + 1.0;
            let phi_3 = phi_2 / h_eta - 0.5;
            x = x
                .add(&d1.affine(phi_2, 0.0)?)?
                .sub(&d2.affine(phi_3, 0.0)?)?;
        } else if let (Some(hp), Some(d1p)) = (h_1, &den_1) {
            let r = hp / h;
            let d = denoised.sub(d1p)?.affine(1.0 / r, 0.0)?;
            let phi_2 = (-h_eta).exp_m1() / h_eta + 1.0;
            x = x.add(&d.affine(phi_2, 0.0)?)?;
        }
        // SDE noise injection: sigma_next * sqrt(1 - exp(-2h)) fresh gaussian.
        let amp = sigmas[i + 1] * (-((-2.0 * h).exp_m1())).sqrt();
        let nz = Tensor::from_vec_f32(gauss(), shape.clone())?.to_device(&device)?;
        x = x.add(&nz.affine(amp, 0.0)?)?;
        (h_2, h_1) = (h_1, Some(h));
        (den_2, den_1) = (den_1.take(), Some(denoised));
    }
    Ok(x)
}

#[cfg(test)]
mod forward_determinism {
    use super::*;

    /// Four FULL renders (T5, sampler, VAE - the whole pipeline) at one seed in a bare
    /// process: distinct-output count. The DiT-only probe below stays clean while the
    /// served pipeline diverges, so the widest in-process net is the discriminator
    /// between "the pipeline races" and "the server environment interferes".
    #[test]
    #[ignore = "hardware probe: needs the stable-audio weights and a CUDA card"]
    fn four_full_renders_one_output() {
        if model_dir().is_none() {
            return;
        }
        // The stage hashes are debug-gated; give the test a subscriber so they print.
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let (pcm, _rate) = render(
                "gentle rain on a tin roof",
                "",
                8.0,
                4,
                7.0,
                42,
                |_, _, _| Ok(()),
            )
            .unwrap();
            let mut h: u64 = 0xcbf29ce484222325;
            for f in &pcm {
                for b in f.to_le_bytes() {
                    h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
                }
            }
            seen.insert(h);
        }
        println!(
            "empreintes distinctes sur 4 rendus complets: {}",
            seen.len()
        );
        assert_eq!(
            seen.len(),
            1,
            "the full pipeline is nondeterministic in a bare process"
        );
    }

    /// Ten forwards of the REAL DiT on one input: the count of distinct outputs is the
    /// race detector the op-level harness could not be (every op alone was clean).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "hardware probe: needs the stable-audio weights and a CUDA card"]
    fn ten_identical_forwards_one_output() {
        let Some(dir) = model_dir() else { return };
        let ckpt = dir.join("model.safetensors");
        let Ok(dev) = Device::new_cuda(0) else { return };
        // The server disables per-slice event tracking on its devices; the test must
        // run under the same discipline or it probes a different machine.
        if let Device::Cuda(cd) = &dev {
            unsafe { cd.context().disable_event_tracking() };
        }
        let dit = load_dit(ckpt.to_str().unwrap(), &dev).unwrap();
        let mut st = 0x2545F4914F6CDD1Du64;
        let mut xs = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / 8388608.0) - 1.0
        };
        let x: Vec<f32> = (0..346 * 64).map(|_| xs()).collect();
        let cross: Vec<f32> = (0..130 * DIT_COND_DIM).map(|_| xs()).collect();
        let glob: Vec<f32> = (0..2 * DIT_COND_DIM).map(|_| xs()).collect();
        let x = Tensor::from_vec_f32(x, (346usize, 64usize))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let cross = Tensor::from_vec_f32(cross, (130usize, DIT_COND_DIM))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let glob = Tensor::from_vec_f32(glob, (1usize, 2 * DIT_COND_DIM))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let out = dit.forward(&x, 0.5, &cross, &glob).unwrap();
            let v = out.to_device(&Device::Cpu).unwrap().to_vec_f32();
            let mut h: u64 = 0xcbf29ce484222325;
            for f in &v {
                for b in f.to_le_bytes() {
                    h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
                }
            }
            seen.insert(h);
        }
        println!("empreintes distinctes sur 10 forwards: {}", seen.len());
        assert_eq!(seen.len(), 1, "the forward is nondeterministic");
    }
}

#[cfg(test)]
mod op_determinism {
    use crate::tensor::{Device, Tensor};

    fn xs(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 40) as f32 / 8388608.0) - 1.0
    }

    fn t(dev: &Device, dims: &[usize], seed: u64) -> Tensor {
        let mut st = seed;
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n).map(|_| xs(&mut st)).collect();
        Tensor::from_vec_f32(v, dims.to_vec())
            .unwrap()
            .to_device(dev)
            .unwrap()
    }

    fn twice(tag: &str, f: impl Fn() -> Tensor) {
        let a = f().to_device(&Device::Cpu).unwrap().to_vec_f32();
        let b = f().to_device(&Device::Cpu).unwrap().to_vec_f32();
        let diff = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        println!(
            "{tag}: {}",
            if diff == 0 {
                "deterministe".into()
            } else {
                format!("NON-DETERMINISTE ({diff}/{} bits changes)", a.len())
            }
        );
    }

    /// The op-level harness the stable-audio hunt narrowed to: same inputs, twice,
    /// bit-compared. Whichever line says so is the nondeterministic primitive.
    #[test]
    #[ignore = "hardware probe: runs on the first CUDA card"]
    fn which_op_is_nondeterministic_on_gpu() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let q = t(&dev, &[24, 348, 64], 1);
        let k = t(&dev, &[24, 348, 64], 2);
        let v = t(&dev, &[24, 348, 348], 3);
        let x = t(&dev, &[348, 1536], 4);
        let w = t(&dev, &[1536, 1536], 5);
        twice("matmul3d qk^t", || {
            q.matmul(&k.transpose(1, 2).unwrap().contiguous().unwrap())
                .unwrap()
        });
        twice("softmax_last_dim", || v.softmax_last_dim().unwrap());
        twice("matmul3d probs*v", || v.matmul(&k).unwrap());
        twice("matmul2d lineaire", || x.matmul(&w).unwrap());
        twice("silu", || x.silu().unwrap());
        twice("affine+add", || {
            x.affine(0.5, 0.0).unwrap().add(&x).unwrap()
        });
        let a1 = t(&dev, &[348, 16], 6);
        twice("cat dim1", || Tensor::cat(&[&a1, &a1, &a1], 1).unwrap());
        let big = t(&dev, &[348, 3072], 7);
        twice("narrow strided", || {
            big.narrow(1, 512, 1024).unwrap().contiguous().unwrap()
        });
    }
}

#[cfg(test)]
mod tests;

impl OobleckDecoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(
            &self.head.weight.device(),
            self.blocks.len(),
        )
    }
}

impl OobleckEncoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(
            &self.head.weight.device(),
            self.blocks.len(),
        )
    }
}

impl SaoDit {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::runs(
            self.block_devices.iter().map(|d| d.location()),
            0,
        )
    }
}
