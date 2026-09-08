//! Qwen-Image DiT (`general.architecture = "qwen_image"`) - the 20B dual-stream MMDiT that
//! denoises the 16-channel VAE latents for Qwen-Image / Qwen-Image-Edit.
//!
//! Increment 2 of the Qwen-Image-Edit port (increment 1, the VAE = our Wan 2.1 VAE, is done +
//! roundtrip-validated). This file is the FOUNDATION: the exact config + the port spec captured
//! from the diffusers reference (`transformer_qwenimage.py`). The block/attn/rope/loader land as
//! subsequent increments - each shape-validatable against the reference.
//!
//! ## Architecture (from QwenImageTransformer2DModel)
//! - patch_size 2, in_channels 64 (= 16*2*2 patchified latent), out_channels 16, num_layers 60,
//!   num_attention_heads 24, attention_head_dim 128, joint_attention_dim 3584 (Qwen2.5-VL text),
//!   axes_dims_rope (16, 56, 56)  (sums to head_dim 128).
//!
//! ## Dual-stream block (QwenImageTransformerBlock - ~= our flux DoubleStreamBlock)
//! - `img_mod = SiLU -> Linear(dim, 6*dim)` -> 6 chunks (shift1,scale1,gate1,shift2,scale2,gate2).
//! - `img_norm1/2 = LayerNorm(elementwise_affine=false, eps)`.
//! - joint `Attention(qk_norm=rms_norm)` - img+txt share ONE attention (concat q/k/v); txt has
//!   NO separate attention.
//! - `img_mlp = FeedForward(gelu)`. `txt_mod / txt_norm1/2 / txt_mlp` mirror the img stream.
//! - `_modulate(x, [shift,scale,gate]) = x*(1+scale)+shift`, then the residual is `gate`-scaled.
//! - Forward per stream: `h = _modulate(norm1(x), mod[:3]); (img_a, txt_a) = attn(img_h, txt_h,
//!   rope); x = x + gate1*a; x = x + gate2*mlp(_modulate(norm2(x), mod[3:]))`.
//!
//! ## 3D RoPE (QwenEmbedRope, theta=10000, axes_dim=[16,56,56])
//! - `rope_params(index, dim) = polar(1, outer(index, theta^(-arange(0,dim,2)/dim)))`
//!   -> per position, dim/2 (cos,sin) pairs.
//! - image token at grid (f,h,w) gets `cat(rope_f[16], rope_h[56], rope_w[56])` = 128 = head_dim;
//!   text tokens get 1D positions `0..max_txt_seq_len`.
//!
//! ## Timestep (QwenTimestepProjEmbeddings)
//! - `Timesteps(256, flip_sin_to_cos)` sinusoidal -> `TimestepEmbedding` MLP(256->dim) - mirrors
//!   our flux `timestep_embedding` + MlpEmbedder.
//!
//! ## Top level
//! - `img_in = Linear(in_channels, dim)`, `txt_in = Linear(joint_attention_dim, dim)`,
//!   `time_text_embed` (timestep), 60 blocks, `norm_out = AdaLayerNormContinuous`,
//!   `proj_out = Linear(dim, out_channels)`. GGUF loader: qmatmul like flux_quantized_model.

/// Qwen-Image DiT config (diffusers defaults for the 20B checkpoint).
#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub theta: f32,
    pub eps: f64,
}

impl Config {
    /// The dimensionality of the hidden stream (`num_attention_heads * attention_head_dim`).
    pub fn dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 3584,
            axes_dims_rope: [16, 56, 56],
            theta: 10000.0,
            eps: 1e-6,
        }
    }
}

use crate::tensor::{Result, Tensor as NT};

/// 3D RoPE (`QwenEmbedRope`, scale_rope=True): image tokens at grid `(frame,h,w)` get
/// `cat(rope_frame[axes_half0], rope_h[axes_half1], rope_w[axes_half2])` = head_dim/2 complex
/// freqs; text tokens get 1D positions offset past the image. Stored as real (cos,sin) per axis
/// for positions `0..4096` (pos) and `-4096..-1` (neg, for the centered H/W scheme).
pub struct QwenEmbedRope {
    pos_cos: [NT; 3],
    pos_sin: [NT; 3],
    neg_cos: [NT; 3],
    neg_sin: [NT; 3],
    axes_half: [usize; 3],
    scale_rope: bool,
}

/// `angle[p,j] = index[p] * theta^(-2j/dim)` with `dim = 2*half` -> `theta^(-j/half)`.
fn rope_angles(indices: &[f32], half: usize, theta: f32) -> Result<NT> {
    let dim = (2 * half) as f32;
    let inv: Vec<f32> = (0..half)
        .map(|j| theta.powf(-(2.0 * j as f32) / dim))
        .collect();
    let n = indices.len();
    let mut a = vec![0f32; n * half];
    for (p, &ix) in indices.iter().enumerate() {
        for j in 0..half {
            a[p * half + j] = ix * inv[j];
        }
    }
    NT::from_vec_f32(a, (n, half))
}

impl QwenEmbedRope {
    const MAXPOS: usize = 4096;

    pub fn new(theta: f32, axes_dim: [usize; 3], scale_rope: bool) -> Result<Self> {
        let axes_half = [axes_dim[0] / 2, axes_dim[1] / 2, axes_dim[2] / 2];
        let pos_index: Vec<f32> = (0..Self::MAXPOS).map(|i| i as f32).collect();
        // neg_index = flip(arange(4096))*-1 - 1  ->  [-4096, -4095, ..., -1]
        let neg_index: Vec<f32> = (0..Self::MAXPOS)
            .map(|i| -((Self::MAXPOS - i) as f32))
            .collect();
        let mk = |idx: &[f32]| -> Result<([NT; 3], [NT; 3])> {
            let mut cos: Vec<NT> = Vec::new();
            let mut sin: Vec<NT> = Vec::new();
            for &half in &axes_half {
                let ang = rope_angles(idx, half, theta)?;
                cos.push(ang.cos()?);
                sin.push(ang.sin()?);
            }
            Ok((
                [cos[0].clone(), cos[1].clone(), cos[2].clone()],
                [sin[0].clone(), sin[1].clone(), sin[2].clone()],
            ))
        };
        let (pos_cos, pos_sin) = mk(&pos_index)?;
        let (neg_cos, neg_sin) = mk(&neg_index)?;
        Ok(Self {
            pos_cos,
            pos_sin,
            neg_cos,
            neg_sin,
            axes_half,
            scale_rope,
        })
    }

    /// One axis' centered H/W freqs `[len, half]`: first `len - len/2` rows from `neg[-(...):]`,
    /// last `len/2` from `pos[:len/2]` (scale_rope); else `pos[:len]`.
    fn axis_hw(&self, axis: usize, len: usize) -> Result<(NT, NT)> {
        if self.scale_rope {
            let lo = len - len / 2;
            let hi = len / 2;
            let (nc, ns) = (&self.neg_cos[axis], &self.neg_sin[axis]);
            let (pc, ps) = (&self.pos_cos[axis], &self.pos_sin[axis]);
            let nrow = nc.dim(0)?;
            let cos = NT::cat(&[&nc.narrow(0, nrow - lo, lo)?, &pc.narrow(0, 0, hi)?], 0)?;
            let sin = NT::cat(&[&ns.narrow(0, nrow - lo, lo)?, &ps.narrow(0, 0, hi)?], 0)?;
            Ok((cos, sin))
        } else {
            Ok((
                self.pos_cos[axis].narrow(0, 0, len)?,
                self.pos_sin[axis].narrow(0, 0, len)?,
            ))
        }
    }

    /// RoPE for one image `(frame=f, h, w)` + text of length `txt_len`. Returns
    /// `(img_cos,img_sin)` `[f*h*w, head_dim/2]` and `(txt_cos,txt_sin)` `[txt_len, head_dim/2]`.
    pub fn forward(
        &self,
        f: usize,
        h: usize,
        w: usize,
        txt_len: usize,
    ) -> Result<((NT, NT), (NT, NT))> {
        let [hf, hh, hw] = self.axes_half;
        // frame axis: pos[0][0:f] -> [f,hf] -> [f,h,w,hf]
        let fr_c = self.pos_cos[0]
            .narrow(0, 0, f)?
            .reshape((f, 1, 1, hf))?
            .broadcast_as((f, h, w, hf))?;
        let fr_s = self.pos_sin[0]
            .narrow(0, 0, f)?
            .reshape((f, 1, 1, hf))?
            .broadcast_as((f, h, w, hf))?;
        let (hc, hs) = self.axis_hw(1, h)?; // [h,hh]
        let hc = hc.reshape((1, h, 1, hh))?.broadcast_as((f, h, w, hh))?;
        let hs = hs.reshape((1, h, 1, hh))?.broadcast_as((f, h, w, hh))?;
        let (wc, ws) = self.axis_hw(2, w)?; // [w,hw]
        let wc = wc.reshape((1, 1, w, hw))?.broadcast_as((f, h, w, hw))?;
        let ws = ws.reshape((1, 1, w, hw))?.broadcast_as((f, h, w, hw))?;
        let n = f * h * w;
        let img_cos = NT::cat(&[&fr_c, &hc, &wc], 3)?
            .reshape((n, hf + hh + hw))?
            .contiguous()?;
        let img_sin = NT::cat(&[&fr_s, &hs, &ws], 3)?
            .reshape((n, hf + hh + hw))?
            .contiguous()?;
        // text: pos[max_vid_index : +txt_len] over the concatenated-axes freqs.
        let max_vid = if self.scale_rope {
            (h / 2).max(w / 2)
        } else {
            h.max(w)
        };
        let tc = self.cat_axes_pos(max_vid, txt_len, false)?;
        let ts = self.cat_axes_pos(max_vid, txt_len, true)?;
        Ok(((img_cos, img_sin), (tc, ts)))
    }

    /// Text freqs: `cat_over_axes(pos_freqs)[off:off+len]` - the concatenated per-axis freqs
    /// share the same position index for text (1D), so slice each axis then cat.
    fn cat_axes_pos(&self, off: usize, len: usize, sin: bool) -> Result<NT> {
        let src = if sin { &self.pos_sin } else { &self.pos_cos };
        let parts: Vec<NT> = (0..3)
            .map(|a| src[a].narrow(0, off, len))
            .collect::<Result<_>>()?;
        NT::cat(&[&parts[0], &parts[1], &parts[2]], 1)?.contiguous()
    }
}

/// Apply Qwen 3D RoPE to `x` `[seq, heads, head_dim]` given `(cos,sin)` `[seq, head_dim/2]`
/// (`apply_rotary_emb_qwen`, use_real=False -> complex multiply of consecutive pairs, i.e. the
/// interleaved/GPT-J convention): for each pair `(x0,x1)`, `out0=x0*cos-x1*sin`,
/// `out1=x0*sin+x1*cos`.
pub fn apply_rope(x: &NT, cos: &NT, sin: &NT) -> Result<NT> {
    let d = x.shape().dims().to_vec();
    let (seq, heads, hd) = (d[0], d[1], d[2]);
    let half = hd / 2;
    // x -> [seq, heads, half, 2]; split the pair dim.
    let xp = x.reshape((seq, heads, half, 2))?;
    let x0 = xp.narrow(3, 0, 1)?.reshape((seq, heads, half))?;
    let x1 = xp.narrow(3, 1, 1)?.reshape((seq, heads, half))?;
    // cos/sin [seq, half] -> [seq, 1, half] to broadcast over heads.
    let cos = cos.reshape((seq, 1, half))?;
    let sin = sin.reshape((seq, 1, half))?;
    let o0 = x0
        .broadcast_mul(&cos)?
        .add(&x1.broadcast_mul(&sin)?.affine(-1.0, 0.0)?)?;
    let o1 = x0.broadcast_mul(&sin)?.add(&x1.broadcast_mul(&cos)?)?;
    // re-interleave: stack on a new last dim then flatten back to head_dim.
    let o0 = o0.reshape((seq, heads, half, 1))?;
    let o1 = o1.reshape((seq, heads, half, 1))?;
    NT::cat(&[&o0, &o1], 3)?
        .reshape((seq, heads, hd))?
        .contiguous()
}

// -- dual-stream block (increment 2d) ---------------------------------------------------------

/// `y = x*w (+b)`. F32 (w stored `[in,out]`, for tests) or Quant (GGUF QKernelMatMul, real weights).
pub enum Linear {
    F32 {
        w: NT,
        b: Option<NT>,
    },
    Quant {
        qm: crate::tensor::quantized::QKernelMatMul,
        b: Option<NT>,
    },
}
impl Linear {
    /// Dense BF16 weight for a caller that runs several GEMMs against it.
    #[cfg(feature = "cuda")]
    fn dequant_bf16(&self) -> Result<Option<NT>> {
        match self {
            Linear::F32 { w, .. } => Ok(Some(w.to_dtype(crate::tensor::DType::BF16)?)),
            Linear::Quant { qm, .. } => qm.dequant_weight_bf16(),
        }
    }

    /// Apply this linear's bias (if any) to an already-computed product.
    fn add_bias(&self, y: &NT) -> Result<NT> {
        let b = match self {
            Linear::F32 { b, .. } => b,
            Linear::Quant { b, .. } => b,
        };
        match b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y.clone()),
        }
    }

    fn forward(&self, x: &NT) -> Result<NT> {
        let (y, b) = match self {
            Linear::F32 { w, b } => (x.matmul(w)?, b),
            Linear::Quant { qm, b } => {
                // The DiT's activations reach fp16-overflow magnitudes (AdaLN scale ~477) that the
                // The DiT's activations reach magnitudes that overflow a 16-bit accumulation, so
                // the width of the call decides how it runs. At decode width the mat-vec path
                // accumulates in int32 and is both correct and fastest. Wider, the weight is
                // dequantised on the device and one cuBLAS F32 GEMM does the work: fewer launches
                // than chunking the mat-vec, and more accurate besides (cosine 0.926 against the F32
                // reference, where chunked mat-vec reaches 0.910).
                let s = x.shape().dims()[0];
                let y = if s <= 5 {
                    qm.forward(x)?
                } else {
                    qm.forward_dequant_gpu(x)?
                };
                (y, b)
            }
        };
        match b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// LayerNorm over the last dim, `elementwise_affine=False` (fused GPU kernel). The all-ones
/// weight is CACHED per (device, dim): rebuilding + uploading it on every call put a host
/// alloc + H2D copy on the hot path ~hundreds of times per denoise step.
fn layernorm_noaffine(x: &NT, eps: f64) -> Result<NT> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static ONES: OnceLock<Mutex<HashMap<(crate::tensor::DeviceLocation, usize), NT>>> =
        OnceLock::new();
    let dn = *x.shape().dims().last().unwrap();
    let key = (x.device().location(), dn);
    let ones = {
        let mut map = ONES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(t) = map.get(&key) {
            t.clone()
        } else {
            let t = NT::from_vec_f32(vec![1.0f32; dn], (dn,))?.to_device(&x.device())?;
            map.insert(key, t.clone());
            t
        }
    };
    x.layer_norm(&ones, None, eps as f32)
}

/// SiLU / swish: `x*sigmoid(x) = x/(1+e^-x)`.
fn silu(x: &NT) -> Result<NT> {
    let neg = x.affine(-1.0, 0.0)?.exp()?.affine(1.0, 1.0)?; // 1 + e^-x
    x.broadcast_div(&neg)
}

/// RMSNorm over the last dim with a learnable `gamma` `[dim]` (the attn q/k norm).
fn rmsnorm(x: &NT, gamma: &NT, eps: f64) -> Result<NT> {
    // Fused GPU rms_norm kernel - the manual sum_keepdim path round-trips to CPU via reduce_dim.
    x.rms_norm(gamma, eps as f32)
}

/// One QwenImageTransformerBlock (dual-stream, joint attention). All weights F32 here; the
/// GGUF/qmatmul loader is increment 2e. Kept structurally faithful to the reference so the
/// forward wiring (6-chunk modulation, q/k RMSNorm, rope, cat[txt,img] joint SDPA, gated
/// residuals, gelu MLP) can be shape-validated before the real weights land.
pub struct Block {
    img_mod: Linear,
    txt_mod: Linear,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    norm_q: NT,
    norm_k: NT,
    norm_add_q: NT,
    norm_add_k: NT,
    to_out: Linear,
    to_add_out: Linear,
    img_mlp: (Linear, Linear),
    txt_mlp: (Linear, Linear),
    heads: usize,
    head_dim: usize,
    eps: f64,
    /// Device this block's weights + its slice of the forward live on (assigned by the HeteroPlan).
    device: crate::tensor::Device,
}

impl Block {
    /// Load a block from a name->tensor map (F32 path - the qmatmul/GGUF variant reuses the
    /// SAME name mapping). GGUF/diffusers stores Linear weights `[out,in]`; we transpose to
    /// `[in,out]` for `Linear::forward`. Names: transformer_blocks.{i}.attn.{to_q,to_k,to_v,
    /// add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out}, .attn.{norm_q,norm_k,
    /// norm_added_q,norm_added_k}, .{img_mod,txt_mod}.1, .{img_mlp,txt_mlp}.net.{0.proj,2}.
    pub fn load_f32(
        m: &std::collections::HashMap<String, NT>,
        i: usize,
        heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        use crate::tensor::Error;
        let p = format!("transformer_blocks.{i}");
        let get = |n: String| -> Result<NT> {
            m.get(&n)
                .cloned()
                .ok_or_else(|| Error(format!("qwen_image DiT: missing tensor {n}")))
        };
        let lin = |name: &str| -> Result<Linear> {
            let w = get(format!("{p}.{name}.weight"))?
                .transpose(0, 1)?
                .contiguous()?;
            let b = m.get(&format!("{p}.{name}.bias")).cloned();
            Ok(Linear::F32 { w, b })
        };
        let gam = |name: &str| -> Result<NT> { get(format!("{p}.{name}.weight")) };
        Ok(Block {
            img_mod: lin("img_mod.1")?,
            txt_mod: lin("txt_mod.1")?,
            to_q: lin("attn.to_q")?,
            to_k: lin("attn.to_k")?,
            to_v: lin("attn.to_v")?,
            add_q: lin("attn.add_q_proj")?,
            add_k: lin("attn.add_k_proj")?,
            add_v: lin("attn.add_v_proj")?,
            norm_q: gam("attn.norm_q")?,
            norm_k: gam("attn.norm_k")?,
            norm_add_q: gam("attn.norm_added_q")?,
            norm_add_k: gam("attn.norm_added_k")?,
            to_out: lin("attn.to_out.0")?,
            to_add_out: lin("attn.to_add_out")?,
            img_mlp: (lin("img_mlp.net.0.proj")?, lin("img_mlp.net.2")?),
            txt_mlp: (lin("txt_mlp.net.0.proj")?, lin("txt_mlp.net.2")?),
            heads,
            head_dim,
            eps,
            device: crate::tensor::Device::Cpu,
        })
    }

    /// `[shift,scale,gate]` from a `[1, 3*dim]` slice; returns `(x*(1+scale)+shift, gate)`,
    /// each modulation param `[1,dim]` broadcasting over the sequence.
    fn modulate(x: &NT, mod3: &NT, dim: usize) -> Result<(NT, NT)> {
        let shift = mod3.narrow(1, 0, dim)?;
        let scale = mod3.narrow(1, dim, dim)?;
        let gate = mod3.narrow(1, 2 * dim, dim)?;
        let y = x
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?;
        Ok((y, gate))
    }

    fn heads_split(&self, x: &NT, seq: usize) -> Result<NT> {
        x.reshape((seq, self.heads, self.head_dim))
    }

    /// forward. `img [Si,dim]`, `txt [St,dim]`, `temb [6... ]` via mods, rope (cos,sin) per stream.
    /// `mlp.1(gelu(mlp.0(x)))` computed in token chunks so the 4x-wide hidden
    /// activation never materializes for the whole sequence at once. Pointwise
    /// over tokens, so chunking changes nothing mathematically; the chunk width
    /// is chosen so one chunk's hidden buffer stays a few hundred MB.
    fn mlp_token_chunked(l0: &Linear, l1: &Linear, x: &NT) -> Result<NT> {
        let s = x.shape().dims()[0];
        // ~2048 tokens keeps the hidden buffer near 100 MB at 4x3072 F32; below that
        // the launch overhead is not worth it.
        const CHUNK: usize = 2048;
        if s <= CHUNK {
            return l1.forward(&l0.forward(x)?.gelu_erf()?);
        }
        // Dequantize each weight ONCE for the whole chunk loop. The per-call path
        // re-dequantizes the full weight for every chunk (151 MB F32 + 75 MB BF16
        // per pass on the 4x-wide MLP), which both churns the pool and dominates
        // the cost of chunking; hoisting it makes the chunk loop pure GEMM.
        #[cfg(feature = "cuda")]
        let dense = match (l0.dequant_bf16()?, l1.dequant_bf16()?) {
            (Some(w0), Some(w1)) => Some((w0, w1)),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let dense: Option<(NT, NT)> = None;
        let mut parts: Vec<NT> = Vec::with_capacity(s.div_ceil(CHUNK));
        let mut off = 0;
        while off < s {
            let n = (s - off).min(CHUNK);
            let xi = x.narrow(0, off, n)?;
            let y = match &dense {
                Some((w0, w1)) => {
                    let xb = xi.to_dtype(crate::tensor::DType::BF16)?;
                    let h = xb.matmul_t(w0)?;
                    let h = l0.add_bias(&h)?;
                    let h = h.to_dtype(crate::tensor::DType::F32)?.gelu_erf()?;
                    let hb = h.to_dtype(crate::tensor::DType::BF16)?;
                    let o = hb.matmul_t(w1)?;
                    l1.add_bias(&o.to_dtype(crate::tensor::DType::F32)?)?
                }
                None => l1.forward(&l0.forward(&xi)?.gelu_erf()?)?,
            };
            parts.push(y);
            off += n;
        }
        let refs: Vec<&NT> = parts.iter().collect();
        NT::cat(&refs, 0)
    }

    pub fn forward(
        &self,
        img: &NT,
        txt: &NT,
        temb: &NT,
        img_cos: &NT,
        img_sin: &NT,
        txt_cos: &NT,
        txt_sin: &NT,
    ) -> Result<(NT, NT)> {
        let dim = self.heads * self.head_dim;
        let (si, st) = (img.shape().dims()[0], txt.shape().dims()[0]);
        // img_mod/txt_mod are Sequential(SiLU, Linear) in the reference - the SiLU on
        // temb is essential: it zeroes the negative temb components; without it they
        // pass through and the AdaLN scale/gate explode.
        let smod = silu(temb)?;
        let img_mod = self.img_mod.forward(&smod)?; // [1, 6*dim]
        let txt_mod = self.txt_mod.forward(&smod)?;
        let (im1, im2) = (
            img_mod.narrow(1, 0, 3 * dim)?,
            img_mod.narrow(1, 3 * dim, 3 * dim)?,
        );
        let (tm1, tm2) = (
            txt_mod.narrow(1, 0, 3 * dim)?,
            txt_mod.narrow(1, 3 * dim, 3 * dim)?,
        );

        let (img_m, img_g1) = Self::modulate(&layernorm_noaffine(img, self.eps)?, &im1, dim)?;
        let (txt_m, txt_g1) = Self::modulate(&layernorm_noaffine(txt, self.eps)?, &tm1, dim)?;

        // qkv + head split
        let iq = rmsnorm(
            &self.heads_split(&self.to_q.forward(&img_m)?, si)?,
            &self.norm_q,
            self.eps,
        )?;
        let ik = rmsnorm(
            &self.heads_split(&self.to_k.forward(&img_m)?, si)?,
            &self.norm_k,
            self.eps,
        )?;
        let iv = self.heads_split(&self.to_v.forward(&img_m)?, si)?;
        let tq = rmsnorm(
            &self.heads_split(&self.add_q.forward(&txt_m)?, st)?,
            &self.norm_add_q,
            self.eps,
        )?;
        let tk = rmsnorm(
            &self.heads_split(&self.add_k.forward(&txt_m)?, st)?,
            &self.norm_add_k,
            self.eps,
        )?;
        let tv = self.heads_split(&self.add_v.forward(&txt_m)?, st)?;
        let iq = apply_rope(&iq, img_cos, img_sin)?;
        let ik = apply_rope(&ik, img_cos, img_sin)?;
        let tq = apply_rope(&tq, txt_cos, txt_sin)?;
        let tk = apply_rope(&tk, txt_cos, txt_sin)?;
        // joint cat [txt, img] on the sequence dim
        let q = NT::cat(&[&tq, &iq], 0)?; // [S, H, D]
        let k = NT::cat(&[&tk, &ik], 0)?;
        let v = NT::cat(&[&tv, &iv], 0)?;
        // per-head SDPA: [H, S, D]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // Query-tiled attention: the full [H,S,S] scores are 485 MB at 2048 tokens, and
        // materialising them leaves the GPU idle churning allocations. Tiling is bit-exact to
        // the direct form and 3.9x faster there.
        let out = {
            // Query tile: 512 is the MEASURED optimum for this DiT (the commit that
            // introduced tiling clocked 3.9x over the untiled path at 2048 tokens).
            // A memory-adaptive tile was tried and REVERTED: shrinking it to fit a
            // 128 MB score budget took the tile to 144 queries at edit-sized
            // sequences - 64 tiles instead of 18, and profiling showed attention
            // ballooning to 74% of the forward (123 s of a 166 s render). The score
            // buffer is accounted for in the engine's reserve instead; trading a
            // placement decision for a 3.5x slower attention is the wrong trade.
            const QUERY_TILE: usize = 512;
            let kv_len = k.shape().dims()[1];
            let tile = QUERY_TILE.min(kv_len.max(1));
            // BF16 GEMMs on tensor cores, F32 softmax: attention measured 71% of a
            // 1024^2 edit's forward time as an F32 GEMM. BF16 keeps F32's exponent
            // range (this DiT's activations reach ~1e9, which F16 could not hold) and
            // cuBLAS accumulates in F32.
            crate::inference::model::acestep::ops::sdpa_tiled_dt(
                &q, &k, &v, None, false, scale, 1.0, tile, true,
            )?
            .transpose(0, 1)?
            .contiguous()?
        };
        let out = out.reshape((st + si, dim))?;
        let txt_a = self.to_add_out.forward(&out.narrow(0, 0, st)?)?;
        let img_a = self.to_out.forward(&out.narrow(0, st, si)?)?;
        // gated residual + MLP
        let img = img.add(&img_a.broadcast_mul(&img_g1)?)?;
        let txt = txt.add(&txt_a.broadcast_mul(&txt_g1)?)?;
        let (img_m2, img_g2) = Self::modulate(&layernorm_noaffine(&img, self.eps)?, &im2, dim)?;
        let (txt_m2, txt_g2) = Self::modulate(&layernorm_noaffine(&txt, self.eps)?, &tm2, dim)?;
        // The MLP is the widest tensor in the graph: its hidden width is 4x dim, so at
        // edit-sized sequences (image tokens doubled by the [noise ; clean] pair) the
        // up-projection and its GELU alone hold hundreds of megabytes of F32 - the
        // term that pushes a large edit past a card and forces a slower cross-GPU split.
        // Chunking over TOKENS is exact (the MLP is pointwise across the sequence) and
        // caps that peak; only sequences long enough to matter pay the extra launches.
        let img_ff = Self::mlp_token_chunked(&self.img_mlp.0, &self.img_mlp.1, &img_m2)?;
        let txt_ff = Self::mlp_token_chunked(&self.txt_mlp.0, &self.txt_mlp.1, &txt_m2)?;
        let img = img.add(&img_ff.broadcast_mul(&img_g2)?)?;
        let txt = txt.add(&txt_ff.broadcast_mul(&txt_g2)?)?;
        Ok((img, txt))
    }
}

// -- top-level model (increment 2f) ------------------------------------------------------------

/// Sinusoidal timestep features `[n, 256]` (diffusers Timesteps: flip_sin_to_cos=True,
/// downscale_freq_shift=0) -> `emb[i, j] = cos(t*f_j)`(first half) || `sin(t*f_j)`(second),
/// `f_j = exp(-ln(10000)*j/(half))`.
fn timestep_features(t: f32, dim: usize) -> Result<NT> {
    let half = dim / 2;
    let mut v = vec![0f32; dim];
    for j in 0..half {
        let f = (-(10000f32.ln()) * j as f32 / half as f32).exp();
        v[j] = (t * f).cos();
        v[half + j] = (t * f).sin();
    }
    NT::from_vec_f32(v, (1usize, dim))
}

/// The Qwen-Image DiT (`QwenImageTransformer2DModel`). Weights are F32 `Linear` here (the
/// GGUF/qmatmul loader swaps to QLinear); the forward wiring is what this validates.
pub struct Model {
    img_in: Linear, // in_channels -> dim
    txt_norm: NT,   // RMSNorm gamma [joint_attention_dim]
    txt_in: Linear, // joint_attention_dim -> dim
    time1: Linear,  // 256 -> dim   (timestep_embedder.linear_1)
    time2: Linear,  // dim -> dim   (timestep_embedder.linear_2)
    blocks: Vec<Block>,
    norm_out: Linear, // dim -> 2*dim (AdaLayerNormContinuous.linear, from SiLU(temb))
    proj_out: Linear, // dim -> out_channels*patch^2
    cfg: Config,
    rope: QwenEmbedRope,
    /// First block's device: img_in/txt_in/time/norm_out/proj_out live here, the forward starts and
    /// ends here (output_device == input_device).
    input_device: crate::tensor::Device,
}

/// Flatten a `HeteroPlan` into one `Device` per block (the plan owns placement - NEVER hardcode a
/// device). `DeviceKind::Cuda(i)` resolves via the probed `cuda_devices` map; CPU/OpenCL -> CPU.
pub(crate) fn plan_block_devices(
    plan: &crate::inference::place::layer_executor::HeteroPlan,
    cuda_devices: &std::collections::HashMap<usize, crate::tensor::Device>,
) -> Vec<crate::tensor::Device> {
    use crate::inference::place::layer_executor::DeviceKind;
    let mut out = Vec::with_capacity(plan.total_layers);
    for seg in &plan.segments {
        let dev = match seg.kind {
            DeviceKind::Cuda(i) => cuda_devices
                .get(&i)
                .cloned()
                .unwrap_or(crate::tensor::Device::Cpu),
            _ => crate::tensor::Device::Cpu,
        };
        for _ in seg.layer_start..seg.layer_end {
            out.push(dev.clone());
        }
    }
    out
}

impl Model {
    fn silu(x: &NT) -> Result<NT> {
        // x*sigmoid(x) = x/(1+e^-x)
        let neg = x.affine(-1.0, 0.0)?.exp()?.affine(1.0, 1.0)?; // 1+e^-x
        x.broadcast_div(&neg)
    }

    /// forward: patchified latent tokens `img [S_img, in_channels]`, text embeds
    /// `txt [S_txt, joint_attention_dim]`, scalar `timestep`, grid `(f,h,w)` (h,w in patch units).
    /// Returns the velocity `[S_img, out_channels*patch^2]` (unpatchify downstream).
    pub fn forward(
        &self,
        img: &NT,
        txt: &NT,
        timestep: f32,
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<NT> {
        use crate::tensor::DeviceLocation;
        let dim = self.cfg.dim();
        let idev = &self.input_device;
        // Inputs + the I/O linears live on the input device; the block stack may then hop devices.
        let mut hs = self.img_in.forward(&img.to_device(idev)?)?; // [S_img, dim]
        let mut eh = self.txt_in.forward(&rmsnorm(
            &txt.to_device(idev)?,
            &self.txt_norm,
            self.cfg.eps,
        )?)?; // [S_txt, dim]
        let tf = timestep_features(timestep, 256)?.to_device(idev)?;
        let temb_in = self
            .time2
            .forward(&Self::silu(&self.time1.forward(&tf)?)?)?; // [1, dim] on idev
                                                                // rope/temb are small; pre-materialize them on EACH distinct block device (cache), so a
                                                                // block only reads tensors already on its own device.
        let ((ic0, is0), (tc0, ts0)) = self.rope.forward(f, h, w, txt.shape().dims()[0])?; // on CPU
        let mut cache: std::collections::HashMap<DeviceLocation, (NT, NT, NT, NT, NT)> =
            std::collections::HashMap::new();
        for b in &self.blocks {
            let loc = b.device.location();
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(loc) {
                e.insert((
                    temb_in.to_device(&b.device)?,
                    ic0.to_device(&b.device)?,
                    is0.to_device(&b.device)?,
                    tc0.to_device(&b.device)?,
                    ts0.to_device(&b.device)?,
                ));
            }
        }
        // Walk the blocks; move BOTH streams to a block's device only when it differs (single-device
        // plan -> every guard is a no-op -> numerically identical to the single-device path). The
        // residual stream legitimately reaches ~1e9; our f32/MMVQ path holds it, so no fp16 clip.
        let mut cur = idev.clone();
        for b in &self.blocks {
            if b.device.location() != cur.location() {
                cur.synchronize()?;
                hs = hs.to_device(&b.device)?;
                eh = eh.to_device(&b.device)?;
                b.device.synchronize()?;
                cur = b.device.clone();
            }
            let (temb, ic, is, tc, ts) = &cache[&b.device.location()];
            let (nhs, neh) = b.forward(&hs, &eh, temb, ic, is, tc, ts)?;
            hs = nhs;
            eh = neh;
        }
        // Back to the input device for norm_out + proj_out (temb_in already lives there).
        if cur.location() != idev.location() {
            cur.synchronize()?;
            hs = hs.to_device(idev)?;
            idev.synchronize()?;
        }
        // AdaLayerNormContinuous: LN(hs)*(1+scale)+shift, (scale,shift)=norm_out(SiLU(temb)).
        let ss = self.norm_out.forward(&Self::silu(&temb_in)?)?; // [1, 2*dim]
        let scale = ss.narrow(1, 0, dim)?;
        let shift = ss.narrow(1, dim, dim)?;
        let normed = layernorm_noaffine(&hs, self.cfg.eps)?
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?;
        self.proj_out.forward(&normed) // [S_img, out_channels*patch^2]
    }
}

// -- GGUF loader (increment 2g) ---------------------------------------------------------------

use crate::tensor::quantized::QVarBuilder;

/// Dequantize a GGUF tensor (norm gamma / bias - small) to a native F32 tensor on `device`.
impl Block {
    /// Load a block from the GGUF: quantized linears (QKernelMatMul, kept quantized on-device) +
    /// dequantized biases + norm gammas. Reuses the exact names validated by `load_f32`.
    pub fn load_gguf(
        vb: &QVarBuilder,
        device: &crate::tensor::Device,
        i: usize,
        cfg: &Config,
    ) -> Result<Self> {
        let p = format!("transformer_blocks.{i}");
        let dim = cfg.dim();
        let qlin = |name: &str, ind: usize, outd: usize| -> Result<Linear> {
            let qm = vb.qmatmul_on(ind, outd, &format!("{p}.{name}.weight"), device)?;
            let b = vb.get_f32_auto_on(&format!("{p}.{name}.bias"), device).ok();
            Ok(Linear::Quant { qm, b })
        };
        let gam = |name: &str| vb.get_f32_auto_on(&format!("{p}.{name}.weight"), device);
        Ok(Block {
            img_mod: qlin("img_mod.1", dim, 6 * dim)?,
            txt_mod: qlin("txt_mod.1", dim, 6 * dim)?,
            to_q: qlin("attn.to_q", dim, dim)?,
            to_k: qlin("attn.to_k", dim, dim)?,
            to_v: qlin("attn.to_v", dim, dim)?,
            add_q: qlin("attn.add_q_proj", dim, dim)?,
            add_k: qlin("attn.add_k_proj", dim, dim)?,
            add_v: qlin("attn.add_v_proj", dim, dim)?,
            norm_q: gam("attn.norm_q")?,
            norm_k: gam("attn.norm_k")?,
            norm_add_q: gam("attn.norm_added_q")?,
            norm_add_k: gam("attn.norm_added_k")?,
            to_out: qlin("attn.to_out.0", dim, dim)?,
            to_add_out: qlin("attn.to_add_out", dim, dim)?,
            img_mlp: (
                qlin("img_mlp.net.0.proj", dim, 4 * dim)?,
                qlin("img_mlp.net.2", 4 * dim, dim)?,
            ),
            txt_mlp: (
                qlin("txt_mlp.net.0.proj", dim, 4 * dim)?,
                qlin("txt_mlp.net.2", 4 * dim, dim)?,
            ),
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps,
            device: device.clone(),
        })
    }
}

impl Model {
    /// Load the Qwen-Image DiT from the GGUF, placing each of the 60 blocks on the device the
    /// `plan` assigns (adaptive multi-GPU+CPU, fastest-first, spill - NO hardcoded device). The
    /// GGUF is parsed once; QHostTensor blobs are device-independent, so `qmatmul_on(.., dev)` /
    /// `get_f32_auto_on(.., dev)` place each block per its planned device. Top-level I/O lives on the first
    /// block's device (the forward starts and ends there).
    pub fn load_hetero(
        path: &str,
        cuda_devices: &std::collections::HashMap<usize, crate::tensor::Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Self> {
        let cfg = Config::default();
        let dim = cfg.dim();
        // Builder source is CHECKPOINT-AWARE: a .safetensors checkpoint is an fp8-scaled Ray
        // drop-in (RayQwest) - decode fp8 (e4m3/e5m2), fold any weight_scale, re-quantize the
        // 2-D projections to Q8_0 (same resident kernels as the GGUF path); anything else is
        // the regular GGUF. The walk below is IDENTICAL for both (same canonical names).
        let is_safetensors = std::path::Path::new(path)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        let vb = if is_safetensors {
            // Q4_K, not Q8_0: this DiT is 20B params, so Q8_0 (1 B/w) is ~20.6 GB resident,
            // which fits no mainstream card whole and leaves a two-card split no activation
            // headroom. Q4_K (~0.56 B/w) lands ~11.6 GB - and it is the SAME quant level as
            // the base checkpoint this fine-tune drops into (dit-q4km.gguf). Tensors whose
            // in-dim does not tile the 256-wide K-quant blocks stay dense F32 automatically.
            unsafe {
                crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                    &[path],
                    crate::tensor::quantized::GgmlDType::Q4K,
                    &crate::tensor::Device::Cpu,
                    cancel,
                )?
            }
        } else {
            crate::inference::cache::qvb::from_gguf_cached(path, &crate::tensor::Device::Cpu)?
        };
        let block_dev = plan_block_devices(plan, cuda_devices);
        if block_dev.len() != cfg.num_layers {
            return Err(crate::tensor::Error(format!(
                "qwen-image DiT: plan has {} blocks, expected {}",
                block_dev.len(),
                cfg.num_layers
            )));
        }
        let input_device = block_dev
            .first()
            .cloned()
            .unwrap_or(crate::tensor::Device::Cpu);
        let idev = &input_device;
        let top = |name: &str, ind: usize, outd: usize| -> Result<Linear> {
            let qm = vb.qmatmul_on(ind, outd, &format!("{name}.weight"), idev)?;
            let b = vb.get_f32_auto_on(&format!("{name}.bias"), idev).ok();
            Ok(Linear::Quant { qm, b })
        };
        let img_in = top("img_in", cfg.in_channels, dim)?;
        let txt_in = top("txt_in", cfg.joint_attention_dim, dim)?;
        let txt_norm = vb.get_f32_auto_on("txt_norm.weight", idev)?;
        let time1 = top("time_text_embed.timestep_embedder.linear_1", 256, dim)?;
        let time2 = top("time_text_embed.timestep_embedder.linear_2", dim, dim)?;
        let norm_out = top("norm_out.linear", dim, 2 * dim)?;
        let proj_out = top(
            "proj_out",
            dim,
            cfg.out_channels * cfg.patch_size * cfg.patch_size,
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if let Some(c) = cancel {
                c.bail()?;
            }
            blocks.push(Block::load_gguf(&vb, &block_dev[i], i, &cfg)?);
        }
        let rope = QwenEmbedRope::new(cfg.theta, cfg.axes_dims_rope, true)?;
        Ok(Model {
            img_in,
            txt_norm,
            txt_in,
            time1,
            time2,
            blocks,
            norm_out,
            proj_out,
            cfg,
            rope,
            input_device,
        })
    }

    /// Single-device compat wrapper (smoke/render bins): all blocks on `device`.
    pub fn load_gguf(path: &str, device: &crate::tensor::Device) -> Result<Self> {
        let n = Config::default().num_layers;
        let mut map = std::collections::HashMap::new();
        let plan = match device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => {
                map.insert(gpu_id, device.clone());
                crate::inference::place::layer_executor::HeteroPlan::forced_gpu(n, n, gpu_id)
            }
            _ => {
                crate::inference::place::layer_executor::HeteroPlan::calculate(n, 0, &[], &[], 1.0)
            }
        };
        Self::load_hetero(path, &map, &plan, None)
    }

    /// First block's device (where the forward's inputs/outputs live).
    /// True when the blocks do NOT all live on one device: the forward then pays a
    /// cross-device transfer per boundary per step (and a host bounce when the pair
    /// has no P2P). Used by the engine to decide whether a re-plan is worth it once
    /// VRAM frees up.
    pub fn is_split(&self) -> bool {
        let first = self.blocks.first().map(|b| b.device.location());
        self.blocks
            .iter()
            .any(|b| Some(b.device.location()) != first)
    }

    pub fn input_device(&self) -> &crate::tensor::Device {
        &self.input_device
    }
    /// Output lands on the input device (norm_out/proj_out run there).
    pub fn output_device(&self) -> &crate::tensor::Device {
        &self.input_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_shapes_are_consistent() {
        let c = Config::default();
        // hidden dim = heads * head_dim
        assert_eq!(c.dim(), 3072);
        // the 3 rope axes tile the head dim exactly
        assert_eq!(c.axes_dims_rope.iter().sum::<usize>(), c.attention_head_dim);
        // patchified latent channels = latent_channels * patch^2
        assert_eq!(c.out_channels * c.patch_size * c.patch_size, c.in_channels);
    }

    #[test]
    fn rope_shapes_and_identity_at_origin() {
        let c = Config::default();
        let rope = QwenEmbedRope::new(c.theta, c.axes_dims_rope, true).unwrap();
        let (f, h, w, txt) = (1usize, 4usize, 6usize, 5usize);
        let ((ic, is), (tc, ts)) = rope.forward(f, h, w, txt).unwrap();
        let half = c.attention_head_dim / 2; // 64
        assert_eq!(ic.shape().dims(), &[f * h * w, half]);
        assert_eq!(is.shape().dims(), &[f * h * w, half]);
        assert_eq!(tc.shape().dims(), &[txt, half]);
        assert_eq!(ts.shape().dims(), &[txt, half]);
        // token 0 = grid (0,0,0): the FRAME axis (first axes_half0 dims) is at position 0 ->
        // angle 0 -> cos=1, sin=0. (H/W are centered so not necessarily 0 there.)
        let c0 = ic.to_vec_f32();
        let s0 = is.to_vec_f32();
        let hf = c.axes_dims_rope[0] / 2; // 8 frame dims
        for j in 0..hf {
            assert!((c0[j] - 1.0).abs() < 1e-5, "frame cos[{j}]={} != 1", c0[j]);
            assert!(s0[j].abs() < 1e-5, "frame sin[{j}]={} != 0", s0[j]);
        }
    }

    #[test]
    fn apply_rope_identity_on_frame_axis_at_origin() {
        let c = Config::default();
        let rope = QwenEmbedRope::new(c.theta, c.axes_dims_rope, true).unwrap();
        let ((ic, is), _) = rope.forward(1, 4, 6, 3).unwrap();
        let (seq, heads, hd) = (24usize, 2usize, c.attention_head_dim);
        let q: Vec<f32> = (0..seq * heads * hd)
            .map(|i| ((i % 7) as f32) - 3.0)
            .collect();
        let qt = NT::from_vec_f32(q.clone(), (seq, heads, hd)).unwrap();
        let out = apply_rope(&qt, &ic, &is).unwrap().to_vec_f32();
        // token 0, frame axis (first 8 complex pairs = first 16 dims): cos=1,sin=0 -> unchanged.
        for h in 0..heads {
            for k in 0..2 * (c.axes_dims_rope[0] / 2) {
                let idx = h * hd + k;
                assert!(
                    (out[idx] - q[idx]).abs() < 1e-4,
                    "rope changed origin frame dim {k}"
                );
            }
        }
    }

    #[test]
    fn block_forward_shapes_and_finite() {
        let c = Config::default();
        let hd = c.attention_head_dim;
        let heads = 2usize;
        let dim = heads * hd; // 256
        let rope = QwenEmbedRope::new(c.theta, c.axes_dims_rope, true).unwrap();
        let ((ic, is), (tc, ts)) = rope.forward(1, 4, 6, 3).unwrap();
        let lin = |inn: usize, out: usize, seed: f32| {
            let v: Vec<f32> = (0..inn * out)
                .map(|i| ((i as f32 * seed).sin()) * 0.02)
                .collect();
            Linear::F32 {
                w: NT::from_vec_f32(v, (inn, out)).unwrap(),
                b: None,
            }
        };
        let g = |d: usize| NT::from_vec_f32(vec![1.0f32; d], (d,)).unwrap();
        let blk = Block {
            img_mod: lin(dim, 6 * dim, 0.1),
            txt_mod: lin(dim, 6 * dim, 0.2),
            to_q: lin(dim, dim, 0.3),
            to_k: lin(dim, dim, 0.4),
            to_v: lin(dim, dim, 0.5),
            add_q: lin(dim, dim, 0.6),
            add_k: lin(dim, dim, 0.7),
            add_v: lin(dim, dim, 0.8),
            norm_q: g(hd),
            norm_k: g(hd),
            norm_add_q: g(hd),
            norm_add_k: g(hd),
            to_out: lin(dim, dim, 0.9),
            to_add_out: lin(dim, dim, 1.0),
            img_mlp: (lin(dim, 4 * dim, 1.1), lin(4 * dim, dim, 1.2)),
            txt_mlp: (lin(dim, 4 * dim, 1.3), lin(4 * dim, dim, 1.4)),
            heads,
            head_dim: hd,
            eps: c.eps,
            device: crate::tensor::Device::Cpu,
        };
        let (si, st) = (24usize, 3usize);
        let img = NT::from_vec_f32(
            (0..si * dim)
                .map(|i| (i as f32 * 0.01).cos() * 0.1)
                .collect(),
            (si, dim),
        )
        .unwrap();
        let txt = NT::from_vec_f32(
            (0..st * dim)
                .map(|i| (i as f32 * 0.02).sin() * 0.1)
                .collect(),
            (st, dim),
        )
        .unwrap();
        let temb = NT::from_vec_f32(
            (0..dim).map(|i| (i as f32 * 0.03).sin() * 0.1).collect(),
            (1usize, dim),
        )
        .unwrap();
        let (oi, ot) = blk.forward(&img, &txt, &temb, &ic, &is, &tc, &ts).unwrap();
        assert_eq!(oi.shape().dims(), &[si, dim]);
        assert_eq!(ot.shape().dims(), &[st, dim]);
        assert!(oi.to_vec_f32().iter().all(|x| x.is_finite()));
        assert!(ot.to_vec_f32().iter().all(|x| x.is_finite()));
    }

    fn synth_block(dim: usize, heads: usize, hd: usize, eps: f64, s: f32) -> Block {
        let lin = |inn: usize, out: usize, seed: f32| {
            let v: Vec<f32> = (0..inn * out)
                .map(|i| ((i as f32 * seed).sin()) * 0.02)
                .collect();
            Linear::F32 {
                w: NT::from_vec_f32(v, (inn, out)).unwrap(),
                b: None,
            }
        };
        let g = |d: usize| NT::from_vec_f32(vec![1.0f32; d], (d,)).unwrap();
        Block {
            img_mod: lin(dim, 6 * dim, s),
            txt_mod: lin(dim, 6 * dim, s + 0.1),
            to_q: lin(dim, dim, s + 0.2),
            to_k: lin(dim, dim, s + 0.3),
            to_v: lin(dim, dim, s + 0.4),
            add_q: lin(dim, dim, s + 0.5),
            add_k: lin(dim, dim, s + 0.6),
            add_v: lin(dim, dim, s + 0.7),
            norm_q: g(hd),
            norm_k: g(hd),
            norm_add_q: g(hd),
            norm_add_k: g(hd),
            to_out: lin(dim, dim, s + 0.8),
            to_add_out: lin(dim, dim, s + 0.9),
            img_mlp: (lin(dim, 4 * dim, s + 1.0), lin(4 * dim, dim, s + 1.1)),
            txt_mlp: (lin(dim, 4 * dim, s + 1.2), lin(4 * dim, dim, s + 1.3)),
            heads,
            head_dim: hd,
            eps,
            device: crate::tensor::Device::Cpu,
        }
    }

    #[test]
    fn model_forward_shapes_and_finite() {
        let mut c = Config::default();
        c.num_attention_heads = 2; // dim = 256
        c.num_layers = 2;
        let (dim, hd, heads) = (c.dim(), c.attention_head_dim, c.num_attention_heads);
        let lin = |inn: usize, out: usize, seed: f32| {
            let v: Vec<f32> = (0..inn * out)
                .map(|i| ((i as f32 * seed).sin()) * 0.02)
                .collect();
            Linear::F32 {
                w: NT::from_vec_f32(v, (inn, out)).unwrap(),
                b: None,
            }
        };
        let rope = QwenEmbedRope::new(c.theta, c.axes_dims_rope, true).unwrap();
        let m = Model {
            img_in: lin(c.in_channels, dim, 0.1),
            txt_norm: NT::from_vec_f32(vec![1f32; c.joint_attention_dim], (c.joint_attention_dim,))
                .unwrap(),
            txt_in: lin(c.joint_attention_dim, dim, 0.2),
            time1: lin(256, dim, 0.3),
            time2: lin(dim, dim, 0.4),
            blocks: vec![
                synth_block(dim, heads, hd, c.eps, 0.5),
                synth_block(dim, heads, hd, c.eps, 2.0),
            ],
            norm_out: lin(dim, 2 * dim, 0.6),
            proj_out: lin(dim, c.out_channels * c.patch_size * c.patch_size, 0.7),
            cfg: c.clone(),
            rope,
            input_device: crate::tensor::Device::Cpu,
        };
        let (f, h, w, st) = (1usize, 4usize, 6usize, 3usize);
        let img = NT::from_vec_f32(
            vec![0.05f32; f * h * w * c.in_channels],
            (f * h * w, c.in_channels),
        )
        .unwrap();
        let txt = NT::from_vec_f32(
            vec![0.05f32; st * c.joint_attention_dim],
            (st, c.joint_attention_dim),
        )
        .unwrap();
        let out = m.forward(&img, &txt, 0.7, f, h, w).unwrap();
        assert_eq!(
            out.shape().dims(),
            &[f * h * w, c.out_channels * c.patch_size * c.patch_size]
        );
        assert!(out.to_vec_f32().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn block_load_f32_name_mapping() {
        use std::collections::HashMap;
        let (heads, hd) = (2usize, 128usize);
        let dim = heads * hd; // 256
        let mut m: HashMap<String, NT> = HashMap::new();
        let mut put = |name: &str, out: usize, inn: usize| {
            let w: Vec<f32> = (0..out * inn)
                .map(|i| (i as f32 * 0.001).sin() * 0.02)
                .collect();
            m.insert(
                format!("transformer_blocks.0.{name}.weight"),
                NT::from_vec_f32(w, (out, inn)).unwrap(),
            );
            m.insert(
                format!("transformer_blocks.0.{name}.bias"),
                NT::from_vec_f32(vec![0f32; out], (out,)).unwrap(),
            );
        };
        for n in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_out.0",
            "attn.to_add_out",
        ] {
            put(n, dim, dim);
        }
        put("img_mod.1", 6 * dim, dim);
        put("txt_mod.1", 6 * dim, dim);
        put("img_mlp.net.0.proj", 4 * dim, dim);
        put("img_mlp.net.2", dim, 4 * dim);
        put("txt_mlp.net.0.proj", 4 * dim, dim);
        put("txt_mlp.net.2", dim, 4 * dim);
        for n in [
            "attn.norm_q",
            "attn.norm_k",
            "attn.norm_added_q",
            "attn.norm_added_k",
        ] {
            m.insert(
                format!("transformer_blocks.0.{n}.weight"),
                NT::from_vec_f32(vec![1f32; hd], (hd,)).unwrap(),
            );
        }
        // loads without a missing-tensor error -> every reference resolves to a real GGUF name
        let blk = Block::load_f32(&m, 0, heads, hd, 1e-6).unwrap();

        let c = Config::default();
        let rope = QwenEmbedRope::new(c.theta, c.axes_dims_rope, true).unwrap();
        let ((ic, is), (tc, ts)) = rope.forward(1, 4, 6, 3).unwrap();
        let img = NT::from_vec_f32(vec![0.05f32; 24 * dim], (24usize, dim)).unwrap();
        let txt = NT::from_vec_f32(vec![0.05f32; 3 * dim], (3usize, dim)).unwrap();
        let temb = NT::from_vec_f32(vec![0.05f32; dim], (1usize, dim)).unwrap();
        let (oi, ot) = blk.forward(&img, &txt, &temb, &ic, &is, &tc, &ts).unwrap();
        assert_eq!(oi.shape().dims(), &[24, dim]);
        assert_eq!(ot.shape().dims(), &[3, dim]);
        assert!(oi.to_vec_f32().iter().all(|x| x.is_finite()));
    }
}
