//! FLUX.2 Klein DiT - the 3.9B dual-stream + parallel-single MMDiT behind FLUX.2-klein.
//!
//! Ported against the diffusers reference (`tmp/flux2_ref/transformer_flux2.py`) and diffed
//! stage-by-stage against the f32 oracle it dumps, not judged on a finished image.
//!
//! ## What it shares with the FLUX.1 port
//! Dual-stream blocks with joint attention, parallel (ViT-22B style) single blocks with a fused
//! QKV+MLP projection, `SiLU -> linear -> chunk(shift, scale, gate)` modulation, affine-free
//! LayerNorms, and RMSNorm on q/k.
//!
//! ## What is different, and why a FLUX.1 block would not have loaded
//! - **Modulation lives on the MODEL, not on the block.** There are exactly three modulation
//!   projections for all 25 blocks (`double_stream_modulation_img/txt` with two parameter sets
//!   each, `single_stream_modulation` with one), where FLUX.1 gives every block its own. So the
//!   modulation is computed ONCE per sampling step and every block reads the same tensor.
//! - **The feedforward is SwiGLU at mlp_ratio 3**, with the gate fused into `linear_in`
//!   (`[2*inner, dim]` -> `silu(h[..:inner]) * h[inner..]`), where FLUX.1 is GELU at ratio 4.
//! - **RoPE has FOUR axes** `(T, H, W, L)` at 32 each: text tokens vary along L, latent tokens
//!   along H/W, and additional reference images are separated along T. The rotation itself is the
//!   same interleaved-pair form the other DiTs use, so `apply_rope` is shared.
//! - **No biases anywhere** - every projection in the checkpoint is weight-only.
//!
//! ## Shapes (verified against the checkpoint header, not inferred from the config)
//! dim 3072 = 24 heads x 128; `x_embedder [3072,128]`, `context_embedder [3072,7680]`,
//! `ff.linear_in [18432,3072]`, `ff.linear_out [3072,9216]`,
//! single `to_qkv_mlp_proj [27648,3072]` = 3x3072 qkv + 2x9216 gated MLP,
//! single `to_out [3072,12288]` = 3072 attn + 9216 MLP, `norm_out.linear [6144,3072]`,
//! `proj_out [128,3072]`.

use crate::inference::model::qwen_image::dit::apply_rope;
use crate::tensor::quantized::QVarBuilder;
use crate::tensor::{Device, Result, Tensor as NT};

/// FLUX.2 Klein DiT config. Field names mirror `transformer/config.json` so a new checkpoint of
/// the family can be read straight off its own config rather than guessed.
#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Dual-stream blocks.
    pub num_layers: usize,
    /// Parallel single-stream blocks.
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: f64,
    /// `(T, H, W, L)` - sums to `attention_head_dim`.
    pub axes_dims_rope: [usize; 4],
    pub rope_theta: f32,
    /// Sinusoidal width feeding the timestep MLP.
    pub timestep_guidance_channels: usize,
    pub eps: f64,
}

impl Config {
    /// Hidden width (`num_attention_heads * attention_head_dim`).
    pub fn dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    /// SwiGLU inner width. The gate doubles it in `linear_in` only.
    pub fn mlp_hidden(&self) -> usize {
        (self.dim() as f64 * self.mlp_ratio) as usize
    }
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            num_layers: 5,
            num_single_layers: 20,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 7680,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            timestep_guidance_channels: 256,
            eps: 1e-6,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::klein_4b()
    }
}

/// `y = x * w^T`. Every FLUX.2 projection is bias-free, so there is no bias arm to carry.
pub enum Linear {
    F32 {
        w: NT,
    },
    Quant {
        qm: crate::tensor::quantized::QKernelMatMul,
    },
}

impl Linear {
    fn forward(&self, x: &NT) -> Result<NT> {
        match self {
            Linear::F32 { w } => x.matmul(w),
            Linear::Quant { qm } => {
                // Decode-width rows go through MMVQ (int32 accumulation); wider calls dequantize
                // the weight once on-device and run a single parallel F32 GEMM, which is both
                // fewer launches and more accurate than chunking the vector path. This DiT's
                // activations peak around 1e3 (measured on the oracle), well inside F32.
                if x.shape().dims()[0] <= 5 {
                    qm.forward(x)
                } else {
                    qm.forward_dequant_gpu(x)
                }
            }
        }
    }
}

/// Four-axis RoPE. Each axis contributes `axes_dims_rope[i]/2` frequency pairs, concatenated in
/// axis order to `head_dim/2`; the rotation applied downstream is the interleaved-pair form, which
/// is exactly what the reference's `repeat_interleave_real` cos/sin expresses.
pub struct Flux2Rope {
    axes_half: [usize; 4],
    theta: f32,
}

impl Flux2Rope {
    pub fn new(cfg: &Config) -> Self {
        let a = cfg.axes_dims_rope;
        Self {
            axes_half: [a[0] / 2, a[1] / 2, a[2] / 2, a[3] / 2],
            theta: cfg.rope_theta,
        }
    }

    /// `(cos, sin)` of shape `[n, head_dim/2]` for `ids[n][4]` given as `(t, h, w, l)` rows.
    pub fn forward(&self, ids: &[[f32; 4]]) -> Result<(NT, NT)> {
        let n = ids.len();
        let half: usize = self.axes_half.iter().sum();
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for (p, id) in ids.iter().enumerate() {
            let mut off = 0;
            for (axis, &ah) in self.axes_half.iter().enumerate() {
                let dim = (2 * ah) as f32;
                for j in 0..ah {
                    // inv_freq[j] = theta^(-2j/dim); computed in f64 because the reference
                    // builds its frequencies in float64 and a 32-bit power drifts at high j.
                    let inv = (self.theta as f64).powf(-(2.0 * j as f64) / dim as f64);
                    let a = id[axis] as f64 * inv;
                    cos[p * half + off + j] = a.cos() as f32;
                    sin[p * half + off + j] = a.sin() as f32;
                }
                off += ah;
            }
        }
        Ok((
            NT::from_vec_f32(cos, (n, half))?,
            NT::from_vec_f32(sin, (n, half))?,
        ))
    }
}

/// The `(T,H,W,L)` coordinates of a latent grid: `T=0`, `H` and `W` over the grid, `L=0`.
pub fn latent_ids(gh: usize, gw: usize) -> Vec<[f32; 4]> {
    let mut v = Vec::with_capacity(gh * gw);
    for h in 0..gh {
        for w in 0..gw {
            v.push([0.0, h as f32, w as f32, 0.0]);
        }
    }
    v
}

/// The `(T,H,W,L)` coordinates of a text sequence: only `L` advances.
pub fn text_ids(len: usize) -> Vec<[f32; 4]> {
    (0..len).map(|i| [0.0, 0.0, 0.0, i as f32]).collect()
}

/// LayerNorm over the last dim with no learnable affine.
fn layernorm_noaffine(x: &NT, eps: f64) -> Result<NT> {
    let d = x.shape().dims().len() - 1;
    let mean = x.mean_keepdim(d)?;
    let c = x.broadcast_sub(&mean)?;
    let var = c.sqr()?.mean_keepdim(d)?;
    c.broadcast_div(&var.affine(1.0, eps as f32)?.sqrt()?)
}

/// SwiGLU over a `[.., 2*inner]` activation: `silu(h[..inner]) * h[inner..]`.
fn swiglu(h: &NT) -> Result<NT> {
    if let Some(y) = crate::tensor::ops::split_silu_mul_f32(h)? {
        return Ok(y);
    }
    let last = h.shape().dims().len() - 1;
    let inner = h.shape().dims()[last] / 2;
    let gate = h.narrow(last, 0, inner)?;
    let up = h.narrow(last, inner, inner)?;
    crate::tensor::ops::silu(&gate)?.mul(&up)
}

/// `x * (1 + scale) + shift`, both modulation params `[1, dim]` broadcasting over the sequence.
fn modulate(x: &NT, shift: &NT, scale: &NT) -> Result<NT> {
    x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
        .broadcast_add(shift)
}

/// One `(shift, scale, gate)` set `i` out of a `[1, 3*sets*dim]` modulation row.
fn mod_set(m: &NT, dim: usize, i: usize) -> Result<(NT, NT, NT)> {
    let base = 3 * i * dim;
    Ok((
        m.narrow(1, base, dim)?,
        m.narrow(1, base + dim, dim)?,
        m.narrow(1, base + 2 * dim, dim)?,
    ))
}

/// RMSNorm over the head dimension of a `[seq, heads, head_dim]` tensor.
fn head_rmsnorm(x: &NT, w: &NT, eps: f64) -> Result<NT> {
    let d = x.shape().dims().to_vec();
    let flat = x.reshape((d[0] * d[1], d[2]))?;
    let y = crate::tensor::ops::rms_norm(&flat, w, eps as f32)?;
    y.reshape((d[0], d[1], d[2]))
}

/// Scaled dot-product attention over `[seq, heads, head_dim]`, returned flattened to `[seq, dim]`.
fn attend(q: &NT, k: &NT, v: &NT, head_dim: usize) -> Result<NT> {
    let d = q.shape().dims().to_vec();
    let (seq, heads) = (d[0], d[1]);
    // ops::sdpa wants [B, H, S, D].
    let t = |x: &NT| -> Result<NT> {
        x.reshape((1, seq, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    };
    // The last argument is the soft-cap DIVISOR, and 1.0 is what disables it. Passing 0.0
    // does not mean "no capping": it makes every score `tanh(score/0) * 0` = 0, i.e. uniform
    // attention returning the mean of V - which stays finite and merely renders wrongly.
    let o = crate::inference::model::acestep::ops::sdpa(
        &t(q)?,
        &t(k)?,
        &t(v)?,
        None,
        false,
        1.0 / (head_dim as f32).sqrt(),
        1.0,
    )?;
    o.transpose(1, 2)?
        .contiguous()?
        .reshape((seq, heads * head_dim))
}

/// Dual-stream block: joint attention over `[txt ; img]`, then a gated SwiGLU feedforward per
/// stream. Carries no modulation of its own - it is handed the model-level rows.
pub struct DoubleBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    norm_q: NT,
    norm_k: NT,
    norm_added_q: NT,
    norm_added_k: NT,
    to_out: Linear,
    to_add_out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    ff_ctx_in: Linear,
    ff_ctx_out: Linear,
    heads: usize,
    head_dim: usize,
    eps: f64,
}

impl DoubleBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &NT,
        txt: &NT,
        mod_img: &NT,
        mod_txt: &NT,
        cos: &NT,
        sin: &NT,
    ) -> Result<(NT, NT)> {
        let dim = self.heads * self.head_dim;
        let (si, st) = (img.shape().dims()[0], txt.shape().dims()[0]);
        let (sh_i, sc_i, g_i) = mod_set(mod_img, dim, 0)?;
        let (sh_i2, sc_i2, g_i2) = mod_set(mod_img, dim, 1)?;
        let (sh_t, sc_t, g_t) = mod_set(mod_txt, dim, 0)?;
        let (sh_t2, sc_t2, g_t2) = mod_set(mod_txt, dim, 1)?;

        let ni = modulate(&layernorm_noaffine(img, self.eps)?, &sh_i, &sc_i)?;
        let nt = modulate(&layernorm_noaffine(txt, self.eps)?, &sh_t, &sc_t)?;

        let split = |x: &NT, s: usize| -> Result<NT> { x.reshape((s, self.heads, self.head_dim)) };
        // q/k are RMSNormed per stream with their OWN weights, BEFORE the streams are joined.
        let qi = head_rmsnorm(
            &split(&self.to_q.forward(&ni)?, si)?,
            &self.norm_q,
            self.eps,
        )?;
        let ki = head_rmsnorm(
            &split(&self.to_k.forward(&ni)?, si)?,
            &self.norm_k,
            self.eps,
        )?;
        let vi = split(&self.to_v.forward(&ni)?, si)?;
        let qt = head_rmsnorm(
            &split(&self.add_q.forward(&nt)?, st)?,
            &self.norm_added_q,
            self.eps,
        )?;
        let kt = head_rmsnorm(
            &split(&self.add_k.forward(&nt)?, st)?,
            &self.norm_added_k,
            self.eps,
        )?;
        let vt = split(&self.add_v.forward(&nt)?, st)?;

        // TEXT FIRST - the reference concatenates encoder before image, and the rope ids that
        // accompany this call are built in the same order.
        let q = apply_rope(&NT::cat(&[&qt, &qi], 0)?, cos, sin)?;
        let k = apply_rope(&NT::cat(&[&kt, &ki], 0)?, cos, sin)?;
        let v = NT::cat(&[&vt, &vi], 0)?;
        let a = attend(&q, &k, &v, self.head_dim)?;
        let a_txt = a.narrow(0, 0, st)?;
        let a_img = a.narrow(0, st, si)?;

        let img = img.add(&self.to_out.forward(&a_img)?.broadcast_mul(&g_i)?)?;
        let ff = self.ff_out.forward(&swiglu(&self.ff_in.forward(&modulate(
            &layernorm_noaffine(&img, self.eps)?,
            &sh_i2,
            &sc_i2,
        )?)?)?)?;
        let img = img.add(&ff.broadcast_mul(&g_i2)?)?;

        let txt = txt.add(&self.to_add_out.forward(&a_txt)?.broadcast_mul(&g_t)?)?;
        let ff_c = self
            .ff_ctx_out
            .forward(&swiglu(&self.ff_ctx_in.forward(&modulate(
                &layernorm_noaffine(&txt, self.eps)?,
                &sh_t2,
                &sc_t2,
            )?)?)?)?;
        let txt = txt.add(&ff_c.broadcast_mul(&g_t2)?)?;
        Ok((txt, img))
    }
}

/// Parallel single-stream block: one projection produces QKV **and** the gated MLP, and one
/// projection consumes the attention output concatenated with the MLP output.
pub struct SingleBlock {
    qkv_mlp: Linear,
    to_out: Linear,
    norm_q: NT,
    norm_k: NT,
    heads: usize,
    head_dim: usize,
    mlp_hidden: usize,
    eps: f64,
}

impl SingleBlock {
    /// `x` is the already-joined `[txt ; img]` stream.
    pub fn forward(&self, x: &NT, m: &NT, cos: &NT, sin: &NT) -> Result<NT> {
        let dim = self.heads * self.head_dim;
        let seq = x.shape().dims()[0];
        let (shift, scale, gate) = mod_set(m, dim, 0)?;
        let n = modulate(&layernorm_noaffine(x, self.eps)?, &shift, &scale)?;
        let h = self.qkv_mlp.forward(&n)?;
        let qkv = h.narrow(1, 0, 3 * dim)?;
        let mlp = h.narrow(1, 3 * dim, 2 * self.mlp_hidden)?;

        let split = |o: usize| -> Result<NT> {
            qkv.narrow(1, o * dim, dim)?
                .reshape((seq, self.heads, self.head_dim))
        };
        let q = apply_rope(&head_rmsnorm(&split(0)?, &self.norm_q, self.eps)?, cos, sin)?;
        let k = apply_rope(&head_rmsnorm(&split(1)?, &self.norm_k, self.eps)?, cos, sin)?;
        let a = attend(&q, &k, &split(2)?, self.head_dim)?;

        let joined = NT::cat(&[&a, &swiglu(&mlp)?], 1)?;
        x.add(&self.to_out.forward(&joined)?.broadcast_mul(&gate)?)
    }
}

/// Sinusoidal timestep embedding, `flip_sin_to_cos` with zero downscale shift (diffusers
/// `Timesteps`): the cosine half leads.
fn timestep_proj(t: f32, channels: usize) -> Result<NT> {
    let half = channels / 2;
    let mut v = vec![0f32; channels];
    for i in 0..half {
        let f = (-(i as f64) * (10_000f64).ln() / half as f64).exp();
        let a = t as f64 * f;
        v[i] = a.cos() as f32;
        v[half + i] = a.sin() as f32;
    }
    NT::from_vec_f32(v, (1, channels))
}

pub struct Model {
    x_embedder: Linear,
    context_embedder: Linear,
    time1: Linear,
    time2: Linear,
    mod_double_img: Linear,
    mod_double_txt: Linear,
    mod_single: Linear,
    doubles: Vec<DoubleBlock>,
    singles: Vec<SingleBlock>,
    norm_out: Linear,
    proj_out: Linear,
    rope: Flux2Rope,
    cfg: Config,
    input_device: Device,
}

impl Model {
    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn input_device(&self) -> &Device {
        &self.input_device
    }

    /// One velocity prediction. `img [Si, in_channels]`, `txt [St, joint_attention_dim]`,
    /// `t` the timestep in `[0, 1000]`, and the latent grid `gh x gw`.
    pub fn forward(&self, img: &NT, txt: &NT, t: f32, gh: usize, gw: usize) -> Result<NT> {
        self.forward_tapped(img, txt, t, gh, gw, &mut None)
    }

    /// As `forward`, but recording each block's output under the same names the python oracle
    /// dumps (`dit_double_<i>_s0/_s1`, `dit_single_<i>`). Diffing a whole stack on its final
    /// tensor only says THAT it is wrong; this says WHERE.
    pub fn forward_tapped(
        &self,
        img: &NT,
        txt: &NT,
        t: f32,
        gh: usize,
        gw: usize,
        taps: &mut Option<Vec<(String, NT)>>,
    ) -> Result<NT> {
        let dev = &self.input_device;
        let st = txt.shape().dims()[0];
        // Rope ids follow the SAME text-then-image order the attention concatenates in.
        let mut ids = text_ids(st);
        ids.extend(latent_ids(gh, gw));
        let (cos, sin) = self.rope.forward(&ids)?;
        let (cos, sin) = (cos.to_device(dev)?, sin.to_device(dev)?);

        let temb = {
            let p = timestep_proj(t, self.cfg.timestep_guidance_channels)?.to_device(dev)?;
            self.time2
                .forward(&crate::tensor::ops::silu(&self.time1.forward(&p)?)?)?
        };
        // Modulation is computed ONCE for the whole stack - the three projections are model-level.
        let act = crate::tensor::ops::silu(&temb)?;
        let m_img = self.mod_double_img.forward(&act)?;
        let m_txt = self.mod_double_txt.forward(&act)?;
        let m_single = self.mod_single.forward(&act)?;

        let mut img = self.x_embedder.forward(img)?;
        let mut txt = self.context_embedder.forward(txt)?;
        if let Some(v) = taps.as_mut() {
            v.push(("dit_x_embed".into(), img.clone()));
            v.push(("dit_ctx_embed".into(), txt.clone()));
            v.push(("dit_temb".into(), temb.clone()));
            v.push(("dit_mod_img".into(), m_img.clone()));
            v.push(("dit_mod_txt".into(), m_txt.clone()));
            v.push(("dit_mod_single".into(), m_single.clone()));
        }
        for (i, b) in self.doubles.iter().enumerate() {
            let (t2, i2) = b.forward(&img, &txt, &m_img, &m_txt, &cos, &sin)?;
            txt = t2;
            img = i2;
            if let Some(v) = taps.as_mut() {
                v.push((format!("dit_double_{i}_s0"), txt.clone()));
                v.push((format!("dit_double_{i}_s1"), img.clone()));
            }
        }
        let mut x = NT::cat(&[&txt, &img], 0)?;
        for (i, b) in self.singles.iter().enumerate() {
            x = b.forward(&x, &m_single, &cos, &sin)?;
            if let Some(v) = taps.as_mut() {
                v.push((format!("dit_single_{i}"), x.clone()));
            }
        }
        let img = x.narrow(0, st, x.shape().dims()[0] - st)?;

        // AdaLayerNormContinuous: SiLU(temb) -> linear -> chunk into (SCALE, SHIFT) - in that
        // order, which is the reverse of the (shift, scale, gate) the block modulations use.
        // Reading it the other way round is not a shape error and still renders.
        let dim = self.cfg.dim();
        let ada = self.norm_out.forward(&crate::tensor::ops::silu(&temb)?)?;
        let scale = ada.narrow(1, 0, dim)?;
        let shift = ada.narrow(1, dim, dim)?;
        let normed = modulate(&layernorm_noaffine(&img, self.cfg.eps)?, &shift, &scale)?;
        self.proj_out.forward(&normed)
    }
}

// -- loader -------------------------------------------------------------------------------------

impl Model {
    /// Load the DiT, placing each block on the device the `plan` assigns (adaptive multi-GPU +
    /// CPU, fastest-first, spill - never a hardcoded device). A `.safetensors` checkpoint goes
    /// through the fp8/bf16 bridge (decode -> fold -> block-quantize, cached as a sidecar GGUF);
    /// anything else is read as a GGUF.
    pub fn load_hetero(
        path: &str,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Self> {
        // Q8_0, not Q4_K, and the difference is MEASURED (see `quantization_cost_over_the_stack`):
        // against the f32 oracle this stack costs 0.2741 relative RMS at Q4_K but 0.0275 at Q8_0 -
        // ten times less error for 1.9 GB more resident. That trade is only obvious because the
        // DiT is 3.9B: at 20B parameters the same choice would not fit a card at all. Do not carry
        // a block dtype over from another family; measure it per model.
        Self::load_hetero_as(
            path,
            cuda_devices,
            plan,
            cancel,
            crate::tensor::quantized::GgmlDType::Q8_0,
        )
    }

    /// As `load_hetero`, but with the block dtype the safetensors bridge re-quantizes the 2-D
    /// projections to. Q4_K is what production wants; `F32` exists so a parity run can separate
    /// "the port computes the wrong thing" from "4-bit weights cost this much over 25 blocks" -
    /// a distinction no amount of staring at a single error figure can make.
    pub fn load_hetero_as(
        path: &str,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
        weight_dtype: crate::tensor::quantized::GgmlDType,
    ) -> Result<Self> {
        let cfg = Config::klein_4b();
        let is_st = std::path::Path::new(path)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        let vb = if is_st {
            unsafe {
                crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                    &[path],
                    weight_dtype,
                    &Device::Cpu,
                    cancel,
                )?
            }
        } else {
            crate::inference::cache::qvb::from_gguf_cached(path, &Device::Cpu)?
        };

        let n_blocks = cfg.num_layers + cfg.num_single_layers;
        let block_dev =
            crate::inference::model::qwen_image::dit::plan_block_devices(plan, cuda_devices);
        if block_dev.len() != n_blocks {
            return Err(crate::tensor::Error(format!(
                "flux2 DiT: plan has {} blocks, expected {n_blocks}",
                block_dev.len()
            )));
        }
        let input_device = block_dev.first().cloned().unwrap_or(Device::Cpu);
        let idev = &input_device;
        let dim = cfg.dim();
        let mlp_hidden = cfg.mlp_hidden();

        let lin =
            |vb: &QVarBuilder, name: &str, ind: usize, outd: usize, d: &Device| -> Result<Linear> {
                Ok(Linear::Quant {
                    qm: vb.qmatmul_on(ind, outd, &format!("{name}.weight"), d)?,
                })
            };

        let x_embedder = lin(&vb, "x_embedder", cfg.in_channels, dim, idev)?;
        let context_embedder = lin(&vb, "context_embedder", cfg.joint_attention_dim, dim, idev)?;
        let time1 = lin(
            &vb,
            "time_guidance_embed.timestep_embedder.linear_1",
            cfg.timestep_guidance_channels,
            dim,
            idev,
        )?;
        let time2 = lin(
            &vb,
            "time_guidance_embed.timestep_embedder.linear_2",
            dim,
            dim,
            idev,
        )?;
        let mod_double_img = lin(
            &vb,
            "double_stream_modulation_img.linear",
            dim,
            6 * dim,
            idev,
        )?;
        let mod_double_txt = lin(
            &vb,
            "double_stream_modulation_txt.linear",
            dim,
            6 * dim,
            idev,
        )?;
        let mod_single = lin(&vb, "single_stream_modulation.linear", dim, 3 * dim, idev)?;
        let norm_out = lin(&vb, "norm_out.linear", dim, 2 * dim, idev)?;
        let proj_out = lin(&vb, "proj_out", dim, cfg.out_channels, idev)?;

        let mut doubles = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let d = &block_dev[i];
            let p = format!("transformer_blocks.{i}");
            let g = |n: &str| vb.get_f32_auto_on(&format!("{p}.{n}.weight"), d);
            doubles.push(DoubleBlock {
                to_q: lin(&vb, &format!("{p}.attn.to_q"), dim, dim, d)?,
                to_k: lin(&vb, &format!("{p}.attn.to_k"), dim, dim, d)?,
                to_v: lin(&vb, &format!("{p}.attn.to_v"), dim, dim, d)?,
                add_q: lin(&vb, &format!("{p}.attn.add_q_proj"), dim, dim, d)?,
                add_k: lin(&vb, &format!("{p}.attn.add_k_proj"), dim, dim, d)?,
                add_v: lin(&vb, &format!("{p}.attn.add_v_proj"), dim, dim, d)?,
                norm_q: g("attn.norm_q")?,
                norm_k: g("attn.norm_k")?,
                norm_added_q: g("attn.norm_added_q")?,
                norm_added_k: g("attn.norm_added_k")?,
                to_out: lin(&vb, &format!("{p}.attn.to_out.0"), dim, dim, d)?,
                to_add_out: lin(&vb, &format!("{p}.attn.to_add_out"), dim, dim, d)?,
                ff_in: lin(&vb, &format!("{p}.ff.linear_in"), dim, 2 * mlp_hidden, d)?,
                ff_out: lin(&vb, &format!("{p}.ff.linear_out"), mlp_hidden, dim, d)?,
                ff_ctx_in: lin(
                    &vb,
                    &format!("{p}.ff_context.linear_in"),
                    dim,
                    2 * mlp_hidden,
                    d,
                )?,
                ff_ctx_out: lin(
                    &vb,
                    &format!("{p}.ff_context.linear_out"),
                    mlp_hidden,
                    dim,
                    d,
                )?,
                heads: cfg.num_attention_heads,
                head_dim: cfg.attention_head_dim,
                eps: cfg.eps,
            });
        }

        let mut singles = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let d = &block_dev[cfg.num_layers + i];
            let p = format!("single_transformer_blocks.{i}");
            singles.push(SingleBlock {
                qkv_mlp: lin(
                    &vb,
                    &format!("{p}.attn.to_qkv_mlp_proj"),
                    dim,
                    3 * dim + 2 * mlp_hidden,
                    d,
                )?,
                to_out: lin(&vb, &format!("{p}.attn.to_out"), dim + mlp_hidden, dim, d)?,
                norm_q: vb.get_f32_auto_on(&format!("{p}.attn.norm_q.weight"), d)?,
                norm_k: vb.get_f32_auto_on(&format!("{p}.attn.norm_k.weight"), d)?,
                heads: cfg.num_attention_heads,
                head_dim: cfg.attention_head_dim,
                mlp_hidden,
                eps: cfg.eps,
            });
        }

        let rope = Flux2Rope::new(&cfg);
        Ok(Model {
            x_embedder,
            context_embedder,
            time1,
            time2,
            mod_double_img,
            mod_double_txt,
            mod_single,
            doubles,
            singles,
            norm_out,
            proj_out,
            rope,
            cfg,
            input_device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_shapes_match_the_checkpoint_header() {
        let c = Config::klein_4b();
        assert_eq!(c.dim(), 3072);
        assert_eq!(c.mlp_hidden(), 9216);
        // ff.linear_in is [2*inner, dim] because the SwiGLU gate is fused into it.
        assert_eq!(2 * c.mlp_hidden(), 18432);
        // The single block's fused projection: 3 x qkv, then the gated MLP.
        assert_eq!(3 * c.dim() + 2 * c.mlp_hidden(), 27648);
        // ...and its output projection consumes attention ++ MLP.
        assert_eq!(c.dim() + c.mlp_hidden(), 12288);
        // The four rope axes must tile the head dimension exactly.
        assert_eq!(c.axes_dims_rope.iter().sum::<usize>(), c.attention_head_dim);
    }

    /// Text advances only on L, latents only on H/W. Mixing the two up is not a shape error -
    /// it silently gives every token the wrong position.
    #[test]
    fn position_ids_use_the_documented_axes() {
        let t = text_ids(3);
        assert_eq!(
            t,
            vec![
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 2.0]
            ]
        );
        let l = latent_ids(2, 2);
        assert_eq!(
            l,
            vec![
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0]
            ]
        );
    }

    #[test]
    fn rope_is_identity_at_the_origin() -> Result<()> {
        let cfg = Config::klein_4b();
        let rope = Flux2Rope::new(&cfg);
        let (cos, sin) = rope.forward(&[[0.0, 0.0, 0.0, 0.0]])?;
        assert_eq!(cos.shape().dims(), &[1, cfg.attention_head_dim / 2]);
        for c in cos.to_vec_f32() {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in sin.to_vec_f32() {
            assert!(s.abs() < 1e-6);
        }
        Ok(())
    }

    /// The gate is the FIRST half - swapping the halves still type-checks and still renders,
    /// just wrongly.
    #[test]
    fn swiglu_gates_on_the_first_half() -> Result<()> {
        let h = NT::from_vec_f32(vec![0.0, 2.0, 1.0, 3.0], (1usize, 4usize))?;
        let y = swiglu(&h)?.to_vec_f32();
        // silu(0)*1 = 0 ; silu(2)*3 = 3 * 2/(1+e^-2)
        assert!((y[0] - 0.0).abs() < 1e-6);
        let want = 3.0 * (2.0 / (1.0 + (-2.0f32).exp()));
        assert!((y[1] - want).abs() < 1e-5, "got {} want {want}", y[1]);
        Ok(())
    }
}
