//! Native Boogu-Image DiT (BooguTransformer2DModel): a FLUX/OmniGen2 + Lumina2
//! dual-stream MMDiT, following the OmniGen2 and Lumina2 architectures the checkpoint
//! implements - the block names below are theirs. Reuses the native
//! image-DiT primitives (quantized linear, tiled SDPA) and the FLUX VAE.
//!
//! Pipeline: Qwen3-VL text encode -> flow-match Euler DiT (CFG) -> FLUX VAE decode.
//! Weights are fp8_scaled safetensors (big matmuls fp8 e4m3 x per-tensor
//! `weight_scale`, embedders bf16). The matmul weights are re-block-quantized to Q8_0 and kept
//! ~1 byte resident on-device (via `fp8_scaled::load_qvarbuilder` -> QVarBuilder/QKernelMatMul), so the
//! DiT fits a single GPU; norms/embedders stay dense F32. This replaces the earlier eager fp8->F32
//! decode (which produced a ~38 GB dense model that fit no 16 GB GPU).
//!
//! This file builds up incrementally: config + weight loading + the Lumina/OmniGen2
//! sub-blocks (norm-zero, continuous-norm, SwiGLU FFN, GQA attention). The block
//! stacks (double/single/refiner), 3-axis RoPE, and the top-level forward + sampling
//! pipeline are wired on top of these units.

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::quantized::{GgmlDType, QKernelMatMul, QVarBuilder};
use crate::tensor::{Device, DeviceLocation, Error, Result, Tensor as NT};
use std::collections::HashMap;

/// Flatten a `HeteroPlan` into one `Device` per block (the plan owns placement - NEVER hardcode a
/// device). `DeviceKind::Cuda(i)` resolves via the probed `cuda_devices` map; CPU/OpenCL -> CPU.
fn plan_block_devices(plan: &HeteroPlan, cuda_devices: &HashMap<usize, Device>) -> Vec<Device> {
    let mut out = Vec::with_capacity(plan.total_layers);
    for seg in &plan.segments {
        let dev = match seg.kind {
            DeviceKind::Cuda(i) => cuda_devices.get(&i).cloned().unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };
        for _ in seg.layer_start..seg.layer_end {
            out.push(dev.clone());
        }
    }
    out
}

/// Boogu-Image DiT configuration (Turbo v0.1; derived from the safetensors shapes).
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden_size: usize,              // 3360
    pub num_heads: usize,                // 28
    pub num_kv_heads: usize,             // 7 (GQA)
    pub head_dim: usize,                 // 120
    pub num_double_stream_layers: usize, // 8
    pub num_single_stream_layers: usize, // 32
    pub num_refiner_layers: usize,       // 2, for each of context and noise
    pub ffn_inner: usize,                // SwiGLU inner (multiple_of-rounded), ~13568
    pub multiple_of: usize,              // 256
    pub patch_size: usize,               // 2
    pub in_channels: usize,              // 16 (FLUX VAE latent)
    pub text_dim: usize,                 // 4096 (Qwen3-VL hidden)
    pub time_embed_dim: usize,           // min(hidden, 1024) = 1024
    pub axes_dim: [usize; 3],            // 3-axis rope split (sum = head_dim = 120)
    pub theta: f32,                      // rope theta
    pub norm_eps: f64,                   // 1e-5
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden_size: 3360,
            num_heads: 28,
            num_kv_heads: 7,
            head_dim: 120,
            num_double_stream_layers: 8,
            num_single_stream_layers: 32,
            num_refiner_layers: 2,
            ffn_inner: 13568,
            multiple_of: 256,
            patch_size: 2,
            in_channels: 16,
            text_dim: 4096,
            time_embed_dim: 1024,
            // 3-axis (context, height, width); sum must equal head_dim (120).
            axes_dim: [40, 40, 40],
            theta: 10000.0,
            norm_eps: 1e-5,
        }
    }
}

// -- weight loading (fp8_scaled / bf16 -> F32 native) ----------------------------

/// Load every tensor of the DiT safetensors into an F32 name->tensor map on `dev`,
/// folding each fp8 matmul weight by its companion `<name>.weight_scale` (F32 scalar).
/// The safetensors loader already decodes fp8 e4m3/e5m2 and bf16 to F32.
pub fn load_state_dict(path: &str, dev: &Device) -> Result<HashMap<String, NT>> {
    // SAFETY: the file is mmapped for the lifetime of `st`; tensors are copied out
    // by `load_to` before `st` drops.
    let st = unsafe { crate::tensor::safetensors_io::SafeTensorsLoader::multi(&[path])? };
    let names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    let name_set: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
    let scale_suffix = ".weight_scale";
    let mut out: HashMap<String, NT> = HashMap::with_capacity(names.len());
    for name in &names {
        if name.ends_with(scale_suffix) {
            continue; // folded into its weight below
        }
        let mut t = st.load_to(name, crate::tensor::DType::F32, dev)?;
        // fp8_scaled: multiply the weight by its per-tensor F32 scale if present.
        let scale_name = format!("{name}_scale");
        if name.ends_with(".weight") && name_set.contains(scale_name.as_str()) {
            let s = st.load_to(&scale_name, crate::tensor::DType::F32, dev)?;
            if let Some(&scalar) = s.to_vec_f32().first() {
                t = t.affine(scalar, 0.0)?;
            }
        }
        out.insert(name.clone(), t);
    }
    Ok(out)
}

/// A quantized linear: the projection weight is kept ~1 byte on-device (Q8_0 QKernelMatMul) and
/// dequantized on the fly at matmul time; the optional bias `[out]` stays dense. `forward`:
/// `x @ w + b`.
enum QLinear {
    Quant { qm: QKernelMatMul, b: Option<NT> },
}

impl QLinear {
    /// Load `<prefix>.weight` as a Q8_0 QKernelMatMul (stored `[out, in]`) + optional dense `<prefix>.bias`.
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str) -> Result<Self> {
        let qm = vb.qmatmul_auto(&format!("{prefix}.weight"), dev)?;
        let b = if vb.contains(&format!("{prefix}.bias")) {
            Some(vb.get_f32_auto_on(&format!("{prefix}.bias"), dev)?)
        } else {
            None
        };
        Ok(QLinear::Quant { qm, b })
    }

    fn forward(&self, x: &NT) -> Result<NT> {
        let QLinear::Quant { qm, b } = self;
        let dims = x.shape().dims().to_vec();
        let k = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((rows, k))?;
        // Q8_0 weight stays resident on-device; dequant happens on the fly. Boogu's AdaLN
        // modulation reaches fp16-overflow activation magnitudes (like the Qwen-Image DiT), so
        // wide activations must accumulate in F32, not the fp16-accumulating tiled MMQ: on GPU,
        // decode-width (rows<=5) routes through MMVQ (int32 accumulation, correct + fast); wider
        // rows dequant the weight on-device and run one cuBLAS F32 GEMM. On CPU, an exact dense
        // F32 matmul.
        let y = if x2.device().is_cuda() {
            if rows <= 5 {
                qm.forward(&x2)?
            } else {
                qm.forward_dequant_gpu(&x2)?
            }
        } else {
            qm.forward_dequant_f32(&x2)?
        };
        let out = y.shape().dims()[1];
        let y = match b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut od = dims;
        *od.last_mut().unwrap() = out;
        y.reshape(od)
    }
}

// -- elementwise primitives ------------------------------------------------------

fn silu(x: &NT) -> Result<NT> {
    // x * sigmoid(x)
    let neg = x.affine(-1.0, 0.0)?;
    let sig = neg.exp()?.affine(1.0, 1.0)?.recip()?; // 1/(1+exp(-x))
    x.mul(&sig)
}

/// RMSNorm over the last dim with a learnable gamma `[dim]` (eps inside the sqrt).
fn rmsnorm(x: &NT, gamma: &NT, eps: f64) -> Result<NT> {
    // Fused GPU rms_norm kernel. The manual sqr+mean_keepdim path routes through reduce_dim, which
    // round-trips to CPU (D2H+H2D) per call - thousands of times across the block stack, leaving the
    // GPU idle between kernels. The fused kernel keeps the whole norm on-device.
    x.rms_norm(gamma, eps as f32)
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

// -- Lumina2 / OmniGen2 sub-blocks (exact ports) ---------------------------------

/// LuminaRMSNormZero: `emb = linear(silu(temb))`, chunked into
/// `[scale_msa, gate_msa, scale_mlp, gate_mlp]`; returns
/// `(rmsnorm(x)*(1+scale_msa), gate_msa, scale_mlp, gate_mlp)`.
pub struct LuminaRMSNormZero {
    linear: QLinear, // [time_dim, 4*hidden]
    norm_gamma: NT,  // [hidden]
    eps: f64,
}

impl LuminaRMSNormZero {
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str, eps: f64) -> Result<Self> {
        Ok(LuminaRMSNormZero {
            linear: QLinear::load(vb, dev, &format!("{prefix}.linear"))?,
            norm_gamma: vb.get_f32_auto_on(&format!("{prefix}.norm.weight"), dev)?,
            eps,
        })
    }

    /// `x [seq, hidden]`, `temb [1, time_dim]`. Returns `(x_mod [seq,hidden],
    /// gate_msa [1,hidden], scale_mlp [1,hidden], gate_mlp [1,hidden])`.
    fn forward(&self, x: &NT, temb: &NT) -> Result<(NT, NT, NT, NT)> {
        let dim = *self.norm_gamma.shape().dims().last().unwrap();
        let emb = self.linear.forward(&silu(temb)?)?; // [1, 4*hidden]
        let scale_msa = emb.narrow(1, 0, dim)?;
        let gate_msa = emb.narrow(1, dim, dim)?;
        let scale_mlp = emb.narrow(1, 2 * dim, dim)?;
        let gate_mlp = emb.narrow(1, 3 * dim, dim)?;
        // rmsnorm(x) * (1 + scale_msa), broadcasting scale over the sequence.
        let xn = rmsnorm(x, &self.norm_gamma, self.eps)?;
        let x_mod = xn.broadcast_mul(&scale_msa.affine(1.0, 1.0)?)?;
        Ok((x_mod, gate_msa, scale_mlp, gate_mlp))
    }
}

/// LuminaLayerNormContinuous: `emb = linear_1(silu(cond))`;
/// `x = layernorm(x)*(1+emb)`; then optional `linear_2`.
pub struct LuminaLayerNormContinuous {
    linear_1: QLinear,         // [cond_dim, hidden]
    linear_2: Option<QLinear>, // [hidden, out]
    eps: f64,
}

impl LuminaLayerNormContinuous {
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str, eps: f64, has_out: bool) -> Result<Self> {
        Ok(LuminaLayerNormContinuous {
            linear_1: QLinear::load(vb, dev, &format!("{prefix}.linear_1"))?,
            linear_2: if has_out {
                Some(QLinear::load(vb, dev, &format!("{prefix}.linear_2"))?)
            } else {
                None
            },
            eps,
        })
    }

    /// `x [seq, hidden]`, `cond [1, cond_dim]`.
    fn forward(&self, x: &NT, cond: &NT) -> Result<NT> {
        let emb = self.linear_1.forward(&silu(cond)?)?; // [1, hidden]
        let xn = layernorm_noaffine(x, self.eps)?;
        let x = xn.broadcast_mul(&emb.affine(1.0, 1.0)?)?;
        match &self.linear_2 {
            Some(l2) => l2.forward(&x),
            None => Ok(x),
        }
    }
}

/// LuminaFeedForward (SwiGLU): `linear_2(silu(linear_1(x)) * linear_3(x))`.
pub struct FeedForward {
    linear_1: QLinear, // gate
    linear_2: QLinear, // down
    linear_3: QLinear, // up
}

impl FeedForward {
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str) -> Result<Self> {
        Ok(FeedForward {
            linear_1: QLinear::load(vb, dev, &format!("{prefix}.linear_1"))?,
            linear_2: QLinear::load(vb, dev, &format!("{prefix}.linear_2"))?,
            linear_3: QLinear::load(vb, dev, &format!("{prefix}.linear_3"))?,
        })
    }

    fn forward(&self, x: &NT) -> Result<NT> {
        let h1 = self.linear_1.forward(x)?;
        let h2 = self.linear_3.forward(x)?;
        self.linear_2.forward(&silu(&h1)?.mul(&h2)?)
    }
}

/// GQA attention with per-head-dim QK RMSNorm and rope. `encoder_hidden_states`
/// supplies K/V (equals `hidden_states` for self-attention). Batch-1, seq-major.
pub struct Attention {
    to_q: QLinear,   // [dim, heads*head_dim]
    to_k: QLinear,   // [dim, kv_heads*head_dim]
    to_v: QLinear,   // [dim, kv_heads*head_dim]
    norm_q: NT,      // [head_dim]
    norm_k: NT,      // [head_dim]
    to_out: QLinear, // [heads*head_dim, dim]
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    /// `attn_prefix` locates to_q/k/v/norm_q/norm_k/to_out.0.
    fn load(vb: &QVarBuilder, dev: &Device, attn_prefix: &str, cfg: &Config) -> Result<Self> {
        Ok(Attention {
            to_q: QLinear::load(vb, dev, &format!("{attn_prefix}.to_q"))?,
            to_k: QLinear::load(vb, dev, &format!("{attn_prefix}.to_k"))?,
            to_v: QLinear::load(vb, dev, &format!("{attn_prefix}.to_v"))?,
            norm_q: vb.get_f32_auto_on(&format!("{attn_prefix}.norm_q.weight"), dev)?,
            norm_k: vb.get_f32_auto_on(&format!("{attn_prefix}.norm_k.weight"), dev)?,
            to_out: QLinear::load(vb, dev, &format!("{attn_prefix}.to_out.0"))?,
            heads: cfg.num_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.norm_eps,
        })
    }

    /// `hidden [sq, dim]`, `enc [sk, dim]`, rope `(cos,sin)` applied to q,k over the
    /// KEPT sequence (`None` skips rope). Returns `[sq, dim]`.
    fn forward(&self, hidden: &NT, enc: &NT, rope: Option<(&NT, &NT)>) -> Result<NT> {
        let sq = hidden.shape().dims()[0];
        let sk = enc.shape().dims()[0];
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let mut q = self.to_q.forward(hidden)?.reshape((sq, h, hd))?;
        let mut k = self.to_k.forward(enc)?.reshape((sk, kvh, hd))?;
        let v = self.to_v.forward(enc)?.reshape((sk, kvh, hd))?;
        q = rmsnorm(&q, &self.norm_q, self.eps)?;
        k = rmsnorm(&k, &self.norm_k, self.eps)?;
        if let Some((cos, sin)) = rope {
            q = crate::inference::model::qwen_image::dit::apply_rope(&q, cos, sin)?;
            k = crate::inference::model::qwen_image::dit::apply_rope(&k, cos, sin)?;
        }
        // GQA: repeat K/V heads to `h` (n_rep = h/kvh) before per-head SDPA.
        let n_rep = h / kvh;
        let k = repeat_kv_heads(&k, n_rep)?; // [sk, h, hd]
        let v = repeat_kv_heads(&v, n_rep)?;
        // [h, s, hd]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0 / (hd as f32).sqrt();
        // Deliberately the EXACT F32 entry point. The tensor-core variant of this same
        // function is worth 1.96x on the video DiT, and measured here it is a REGRESSION:
        // a 1024-square render went 17 s -> 57 s. This engine runs many short attention
        // tiles rather than one long sequence, and the casts cost more than the narrower
        // GEMMs save. Do not switch it without an A/B on this family.
        let out = crate::inference::model::acestep::ops::sdpa_tiled(
            &q, &k, &v, None, false, scale, 1.0, 512,
        )?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((sq, h * hd))?;
        self.to_out.forward(&out)
    }
}

/// Repeat each KV head `n_rep` times along the head axis: `[s, kvh, hd] -> [s, kvh*n_rep, hd]`,
/// grouped so head `j` maps to kv head `j / n_rep` (GQA layout).
fn repeat_kv_heads(x: &NT, n_rep: usize) -> Result<NT> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let d = x.shape().dims().to_vec();
    let (s, kvh, hd) = (d[0], d[1], d[2]);
    // [s, kvh, 1, hd] -> broadcast to [s, kvh, n_rep, hd] -> [s, kvh*n_rep, hd]
    let x = x.reshape((s, kvh, 1, hd))?;
    let rep = NT::cat(&vec![&x; n_rep], 2)?; // [s, kvh, n_rep, hd]
    rep.reshape((s, kvh * n_rep, hd))
}

// -- small shared helpers --------------------------------------------------------

/// tanh-gated residual: `h + tanh(gate) * sub`, with `gate [1,dim]` broadcast over
/// the sequence of `sub [seq,dim]`.
fn add_gated(h: &NT, gate: &NT, sub: &NT) -> Result<NT> {
    let g = gate.tanh()?;
    h.add(&sub.broadcast_mul(&g)?)
}

/// `(1 + scale) * x` with `scale [1,dim]` broadcast over `x [seq,dim]`.
fn scale_1p(x: &NT, scale: &NT) -> Result<NT> {
    x.broadcast_mul(&scale.affine(1.0, 1.0)?)
}

/// Sinusoidal timestep features `[1, 256]` (diffusers Timesteps: flip_sin_to_cos=True,
/// downscale_freq_shift=0): `emb[j] = cos(t*f_j)` (first half) `|| sin(t*f_j)` (second),
/// `f_j = exp(-ln(10000)*j/half)`. `t` is pre-scaled by the caller (timestep_scale).
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

// -- OmniGen2 single-stream / refiner block --------------------------------------

/// OmniGen2TransformerBlock. `modulation=true` (single-stream + noise/ref refiners):
/// adaLN-zero via `norm1` -> attn (tanh-gated) -> SwiGLU MLP (scale + tanh-gated).
/// `modulation=false` (context refiner): plain pre-RMSNorm residual block.
pub struct OmniGen2Block {
    /// Device this block lives on (assigned by the plan).
    device: Device,
    modulation: bool,
    norm1_zero: Option<LuminaRMSNormZero>, // modulation=true
    norm1_rms: Option<NT>,                 // modulation=false (gamma)
    attn: Attention,
    norm2: NT,     // RMSNorm gamma
    ffn_norm1: NT, // RMSNorm gamma
    ffn_norm2: NT, // RMSNorm gamma
    ff: FeedForward,
    eps: f64,
}

impl OmniGen2Block {
    fn load(
        vb: &QVarBuilder,
        dev: &Device,
        prefix: &str,
        cfg: &Config,
        modulation: bool,
    ) -> Result<Self> {
        let eps = cfg.norm_eps;
        let (norm1_zero, norm1_rms) = if modulation {
            (
                Some(LuminaRMSNormZero::load(
                    vb,
                    dev,
                    &format!("{prefix}.norm1"),
                    eps,
                )?),
                None,
            )
        } else {
            (
                None,
                Some(vb.get_f32_auto_on(&format!("{prefix}.norm1.weight"), dev)?),
            )
        };
        Ok(OmniGen2Block {
            device: dev.clone(),
            modulation,
            norm1_zero,
            norm1_rms,
            attn: Attention::load(vb, dev, &format!("{prefix}.attn"), cfg)?,
            norm2: vb.get_f32_auto_on(&format!("{prefix}.norm2.weight"), dev)?,
            ffn_norm1: vb.get_f32_auto_on(&format!("{prefix}.ffn_norm1.weight"), dev)?,
            ffn_norm2: vb.get_f32_auto_on(&format!("{prefix}.ffn_norm2.weight"), dev)?,
            ff: FeedForward::load(vb, dev, &format!("{prefix}.feed_forward"))?,
            eps,
        })
    }

    /// `h [seq,dim]`, self-attention rope `(cos,sin)`, `temb [1,time_dim]` (unused when
    /// `modulation=false`). Returns `[seq,dim]`.
    fn forward(&self, h: &NT, rope: (&NT, &NT), temb: &NT) -> Result<NT> {
        if self.modulation {
            let nz = self.norm1_zero.as_ref().unwrap();
            let (nh, gate_msa, scale_mlp, gate_mlp) = nz.forward(h, temb)?;
            let a = self.attn.forward(&nh, &nh, Some(rope))?;
            let h = add_gated(h, &gate_msa, &rmsnorm(&a, &self.norm2, self.eps)?)?;
            let ff_in = scale_1p(&rmsnorm(&h, &self.ffn_norm1, self.eps)?, &scale_mlp)?;
            let mlp = self.ff.forward(&ff_in)?;
            add_gated(&h, &gate_mlp, &rmsnorm(&mlp, &self.ffn_norm2, self.eps)?)
        } else {
            let gamma = self.norm1_rms.as_ref().unwrap();
            let nh = rmsnorm(h, gamma, self.eps)?;
            let a = self.attn.forward(&nh, &nh, Some(rope))?;
            let h = h.add(&rmsnorm(&a, &self.norm2, self.eps)?)?;
            let mlp = self.ff.forward(&rmsnorm(&h, &self.ffn_norm1, self.eps)?)?;
            h.add(&rmsnorm(&mlp, &self.ffn_norm2, self.eps)?)
        }
    }
}

// -- Boogu joint attention (dual-stream) -----------------------------------------

/// Joint attention over `[instruct ; img]` with separate per-stream q/k/v and output
/// projections, shared QK-RMSNorm, GQA + rope. Returns the recombined
/// `[L_instruct+L_img, dim]` after the per-stream out-projections and the final `to_out`.
pub struct JointAttention {
    img_to_q: QLinear,
    img_to_k: QLinear,
    img_to_v: QLinear,
    instruct_to_q: QLinear,
    instruct_to_k: QLinear,
    instruct_to_v: QLinear,
    img_out: QLinear,
    instruct_out: QLinear,
    to_out: QLinear, // to_out.0
    norm_q: NT,      // [head_dim]
    norm_k: NT,      // [head_dim]
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl JointAttention {
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str, cfg: &Config) -> Result<Self> {
        let p = format!("{prefix}.processor");
        Ok(JointAttention {
            img_to_q: QLinear::load(vb, dev, &format!("{p}.img_to_q"))?,
            img_to_k: QLinear::load(vb, dev, &format!("{p}.img_to_k"))?,
            img_to_v: QLinear::load(vb, dev, &format!("{p}.img_to_v"))?,
            instruct_to_q: QLinear::load(vb, dev, &format!("{p}.instruct_to_q"))?,
            instruct_to_k: QLinear::load(vb, dev, &format!("{p}.instruct_to_k"))?,
            instruct_to_v: QLinear::load(vb, dev, &format!("{p}.instruct_to_v"))?,
            img_out: QLinear::load(vb, dev, &format!("{p}.img_out"))?,
            instruct_out: QLinear::load(vb, dev, &format!("{p}.instruct_out"))?,
            to_out: QLinear::load(vb, dev, &format!("{prefix}.to_out.0"))?,
            norm_q: vb.get_f32_auto_on(&format!("{prefix}.norm_q.weight"), dev)?,
            norm_k: vb.get_f32_auto_on(&format!("{prefix}.norm_k.weight"), dev)?,
            heads: cfg.num_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.norm_eps,
        })
    }

    /// `img [L_img,dim]`, `instruct [L_instruct,dim]`, joint rope `(cos,sin)` over the
    /// concatenated `[instruct ; img]` sequence. Returns `[L_instruct+L_img, dim]`.
    fn forward(&self, img: &NT, instruct: &NT, rope: (&NT, &NT)) -> Result<NT> {
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let l_inst = instruct.shape().dims()[0];
        // Concatenate instruction first, then image (matches the reference processor order).
        let q = NT::cat(
            &[
                &self.instruct_to_q.forward(instruct)?,
                &self.img_to_q.forward(img)?,
            ],
            0,
        )?;
        let k = NT::cat(
            &[
                &self.instruct_to_k.forward(instruct)?,
                &self.img_to_k.forward(img)?,
            ],
            0,
        )?;
        let v = NT::cat(
            &[
                &self.instruct_to_v.forward(instruct)?,
                &self.img_to_v.forward(img)?,
            ],
            0,
        )?;
        let s = q.shape().dims()[0];
        let mut q = q.reshape((s, h, hd))?;
        let mut k = k.reshape((s, kvh, hd))?;
        let v = v.reshape((s, kvh, hd))?;
        q = rmsnorm(&q, &self.norm_q, self.eps)?;
        k = rmsnorm(&k, &self.norm_k, self.eps)?;
        let (cos, sin) = rope;
        q = crate::inference::model::qwen_image::dit::apply_rope(&q, cos, sin)?;
        k = crate::inference::model::qwen_image::dit::apply_rope(&k, cos, sin)?;
        let n_rep = h / kvh;
        let k = repeat_kv_heads(&k, n_rep)?;
        let v = repeat_kv_heads(&v, n_rep)?;
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0 / (hd as f32).sqrt();
        // Deliberately the EXACT F32 entry point. The tensor-core variant of this same
        // function is worth 1.96x on the video DiT, and measured here it is a REGRESSION:
        // a 1024-square render went 17 s -> 57 s. This engine runs many short attention
        // tiles rather than one long sequence, and the casts cost more than the narrower
        // GEMMs save. Do not switch it without an A/B on this family.
        let out = crate::inference::model::acestep::ops::sdpa_tiled(
            &q, &k, &v, None, false, scale, 1.0, 512,
        )?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((s, h * hd))?;
        // Split back to instruction/image, apply per-stream output projections, recombine.
        let inst = self.instruct_out.forward(&out.narrow(0, 0, l_inst)?)?;
        let imgp = self.img_out.forward(&out.narrow(0, l_inst, s - l_inst)?)?;
        let joined = NT::cat(&[&inst, &imgp], 0)?;
        self.to_out.forward(&joined)
    }
}

// -- Boogu double-stream block ---------------------------------------------------

/// BooguDoubleStreamBlock: joint attention over `[instruct ; img]` + image self-attention,
/// each stream with its own modulation and SwiGLU MLP. All attention residuals are
/// tanh-gated through a dedicated post-RMSNorm.
pub struct BooguDoubleStreamBlock {
    /// Device this block lives on (assigned by the plan).
    device: Device,
    img_norm1: LuminaRMSNormZero,
    img_norm2: LuminaRMSNormZero,
    img_norm3: LuminaRMSNormZero,
    instruct_norm1: LuminaRMSNormZero,
    instruct_norm2: LuminaRMSNormZero,
    img_instruct_attn: JointAttention,
    img_self_attn: Attention,
    img_feed_forward: FeedForward,
    instruct_feed_forward: FeedForward,
    img_attn_norm: NT,
    img_self_attn_norm: NT,
    img_ffn_norm1: NT,
    img_ffn_norm2: NT,
    instruct_attn_norm: NT,
    instruct_ffn_norm1: NT,
    instruct_ffn_norm2: NT,
    eps: f64,
}

impl BooguDoubleStreamBlock {
    fn load(vb: &QVarBuilder, dev: &Device, prefix: &str, cfg: &Config) -> Result<Self> {
        let eps = cfg.norm_eps;
        Ok(BooguDoubleStreamBlock {
            device: dev.clone(),
            img_norm1: LuminaRMSNormZero::load(vb, dev, &format!("{prefix}.img_norm1"), eps)?,
            img_norm2: LuminaRMSNormZero::load(vb, dev, &format!("{prefix}.img_norm2"), eps)?,
            img_norm3: LuminaRMSNormZero::load(vb, dev, &format!("{prefix}.img_norm3"), eps)?,
            instruct_norm1: LuminaRMSNormZero::load(
                vb,
                dev,
                &format!("{prefix}.instruct_norm1"),
                eps,
            )?,
            instruct_norm2: LuminaRMSNormZero::load(
                vb,
                dev,
                &format!("{prefix}.instruct_norm2"),
                eps,
            )?,
            img_instruct_attn: JointAttention::load(
                vb,
                dev,
                &format!("{prefix}.img_instruct_attn"),
                cfg,
            )?,
            img_self_attn: Attention::load(vb, dev, &format!("{prefix}.img_self_attn"), cfg)?,
            img_feed_forward: FeedForward::load(vb, dev, &format!("{prefix}.img_feed_forward"))?,
            instruct_feed_forward: FeedForward::load(
                vb,
                dev,
                &format!("{prefix}.instruct_feed_forward"),
            )?,
            img_attn_norm: vb.get_f32_auto_on(&format!("{prefix}.img_attn_norm.weight"), dev)?,
            img_self_attn_norm: vb
                .get_f32_auto_on(&format!("{prefix}.img_self_attn_norm.weight"), dev)?,
            img_ffn_norm1: vb.get_f32_auto_on(&format!("{prefix}.img_ffn_norm1.weight"), dev)?,
            img_ffn_norm2: vb.get_f32_auto_on(&format!("{prefix}.img_ffn_norm2.weight"), dev)?,
            instruct_attn_norm: vb
                .get_f32_auto_on(&format!("{prefix}.instruct_attn_norm.weight"), dev)?,
            instruct_ffn_norm1: vb
                .get_f32_auto_on(&format!("{prefix}.instruct_ffn_norm1.weight"), dev)?,
            instruct_ffn_norm2: vb
                .get_f32_auto_on(&format!("{prefix}.instruct_ffn_norm2.weight"), dev)?,
            eps,
        })
    }

    /// `img [L_img,dim]`, `instruct [L_instruct,dim]`, joint rope (over `[instruct;img]`),
    /// img rope (over the image tokens only), `temb [1,time_dim]`.
    /// Returns `(img, instruct)`.
    fn forward(
        &self,
        img: &NT,
        instruct: &NT,
        joint_rope: (&NT, &NT),
        img_rope: (&NT, &NT),
        temb: &NT,
    ) -> Result<(NT, NT)> {
        let l_inst = instruct.shape().dims()[0];
        let (img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp) =
            self.img_norm1.forward(img, temb)?;
        let (img_n2, img_shift_mlp, _, _) = self.img_norm2.forward(img, temb)?;
        let (img_n3, img_gate_self, _, _) = self.img_norm3.forward(img, temb)?;
        let (inst_n1, inst_gate_msa, inst_scale_mlp, inst_gate_mlp) =
            self.instruct_norm1.forward(instruct, temb)?;
        let (inst_n2, inst_shift_mlp, _, _) = self.instruct_norm2.forward(instruct, temb)?;

        let joint = self
            .img_instruct_attn
            .forward(&img_n1, &inst_n1, joint_rope)?;
        let inst_attn = joint.narrow(0, 0, l_inst)?;
        let img_attn = joint.narrow(0, l_inst, joint.shape().dims()[0] - l_inst)?;

        let img_self = self
            .img_self_attn
            .forward(&img_n3, &img_n3, Some(img_rope))?;

        // Image stream.
        let mut img = add_gated(
            img,
            &img_gate_msa,
            &rmsnorm(&img_attn, &self.img_attn_norm, self.eps)?,
        )?;
        img = add_gated(
            &img,
            &img_gate_self,
            &rmsnorm(&img_self, &self.img_self_attn_norm, self.eps)?,
        )?;
        let img_mlp_in = scale_1p(&img_n2, &img_scale_mlp)?.broadcast_add(&img_shift_mlp)?;
        let img_mlp =
            self.img_feed_forward
                .forward(&rmsnorm(&img_mlp_in, &self.img_ffn_norm1, self.eps)?)?;
        img = add_gated(
            &img,
            &img_gate_mlp,
            &rmsnorm(&img_mlp, &self.img_ffn_norm2, self.eps)?,
        )?;

        // Instruction stream.
        let mut inst = add_gated(
            instruct,
            &inst_gate_msa,
            &rmsnorm(&inst_attn, &self.instruct_attn_norm, self.eps)?,
        )?;
        let inst_mlp_in = scale_1p(&inst_n2, &inst_scale_mlp)?.broadcast_add(&inst_shift_mlp)?;
        let inst_mlp = self.instruct_feed_forward.forward(&rmsnorm(
            &inst_mlp_in,
            &self.instruct_ffn_norm1,
            self.eps,
        )?)?;
        inst = add_gated(
            &inst,
            &inst_gate_mlp,
            &rmsnorm(&inst_mlp, &self.instruct_ffn_norm2, self.eps)?,
        )?;

        Ok((img, inst))
    }
}

// -- top-level model -------------------------------------------------------------

/// The Boogu-Image DiT (`BooguTransformer2DModel`). Batch-1, sequence-major forward for
/// the text-to-image path (no reference images). Matmul weights are Q8_0 on-device; norms and
/// embedders are dense F32.
pub struct BooguTransformer2DModel {
    cfg: Config,
    /// The DiT's I/O device: the fastest planned device (block 0). The input latent + embedders +
    /// refiners live here, and the forward brings its output back here before norm_out. The
    /// double/single stream blocks may sit on other devices (per the plan).
    input_device: Device,
    // time + caption embedders.
    time1: QLinear,       // timestep_embedder.linear_1 [256->1024]
    time2: QLinear,       // timestep_embedder.linear_2 [1024->1024]
    caption_norm: NT,     // caption_embedder.0 RMSNorm gamma [text_dim]
    caption_lin: QLinear, // caption_embedder.1 [text_dim->hidden]
    // patch embedders.
    x_embedder: QLinear, // [patch^2*in_ch -> hidden]
    // NOT loaded: `ref_image_patch_embedder`, `image_index_embedding` and the
    // `ref_image_refiner` stack. They are the checkpoint's instruction-edit path, which
    // nothing here drives - `model_capabilities` advertises this family as txt2img only.
    // They were being read onto the card anyway, `num_refiner_layers` blocks of them,
    // and then never touched. Restore all three together when the edit path is wired.
    // block stacks.
    context_refiner: Vec<OmniGen2Block>, // modulation=false
    noise_refiner: Vec<OmniGen2Block>,   // modulation=true
    double_stream_layers: Vec<BooguDoubleStreamBlock>,
    single_stream_layers: Vec<OmniGen2Block>,
    norm_out: LuminaLayerNormContinuous,
}

impl BooguTransformer2DModel {
    /// Latent channel count (FLUX VAE = 16).
    pub fn in_channels(&self) -> usize {
        self.cfg.in_channels
    }
    /// DiT patch size (2).
    pub fn patch_size(&self) -> usize {
        self.cfg.patch_size
    }
    /// Device the DiT expects its input latent on (and where its output lands).
    pub fn input_device(&self) -> &Device {
        &self.input_device
    }

    /// Whether the stream blocks ended up on more than one device.
    ///
    /// This is what tells the request path the DiT is running DEMOTED - split across
    /// cards, or spilled to the host - because VRAM was tight at load. That placement
    /// costs a transfer at every block boundary on every step and outlives the pressure
    /// that caused it, so the engine needs to be able to see it and re-plan.
    pub fn is_split(&self) -> bool {
        let mut devs = self
            .double_stream_layers
            .iter()
            .map(|b| b.device.location())
            .chain(
                self.single_stream_layers
                    .iter()
                    .map(|b| b.device.location()),
            );
        let Some(first) = devs.next() else {
            return false;
        };
        devs.any(|d| d != first) || first != self.input_device.location()
    }
    /// Device the DiT output lands on (== input device: the forward moves the hidden state back
    /// here for norm_out + depatchify).
    pub fn output_device(&self) -> &Device {
        &self.input_device
    }

    /// Load the fp8_scaled DiT, placing the double/single stream blocks across devices per the
    /// `plan` (adaptive multi-GPU+CPU, fastest-first, spill - NO hardcoded device). The QVarBuilder
    /// is parsed once to CPU (its Q8_0 blobs are device-independent); each block's weights go on the
    /// device the plan assigns. Embedders + refiners + norm_out sit on the input (block-0) device;
    /// they are small and this keeps the multi-stream plumbing simple.
    pub fn load(
        path: &str,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
    ) -> Result<Self> {
        Self::load_cancellable(path, cuda_devices, plan, None)
    }

    /// [`Self::load`] with cooperative cancellation: checked per tensor during the fp8 decode
    /// and per block during assembly, so an abandoned load stops in seconds.
    pub fn load_cancellable(
        path: &str,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Self> {
        let cfg = Config::default();
        // SAFETY: `load_qvarbuilder` copies every tensor out of the mmap before returning.
        let vb = unsafe {
            crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                &[path],
                GgmlDType::Q8_0,
                &Device::Cpu,
                cancel,
            )
        }?;
        // The plan distributes the main block stack: double (0..n_double) then single (n_double..).
        let n_double = cfg.num_double_stream_layers;
        let n_single = cfg.num_single_stream_layers;
        let block_dev = plan_block_devices(plan, cuda_devices);
        if block_dev.len() != n_double + n_single {
            return Err(Error(format!(
                "boogu DiT: plan has {} blocks, expected {} (double {} + single {})",
                block_dev.len(),
                n_double + n_single,
                n_double,
                n_single
            )));
        }
        let input_device = block_dev.first().cloned().unwrap_or(Device::Cpu);
        let idev = &input_device;
        // Embedders + refiners on the input device.
        let mk = |prefix: &str, n: usize, modulation: bool| -> Result<Vec<OmniGen2Block>> {
            (0..n)
                .map(|i| OmniGen2Block::load(&vb, idev, &format!("{prefix}.{i}"), &cfg, modulation))
                .collect()
        };
        // Distributed stacks: each block on its planned device.
        let double_stream_layers = (0..n_double)
            .map(|i| {
                if let Some(c) = cancel {
                    c.bail()?;
                }
                BooguDoubleStreamBlock::load(
                    &vb,
                    &block_dev[i],
                    &format!("double_stream_layers.{i}"),
                    &cfg,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let single_stream_layers = (0..n_single)
            .map(|i| {
                if let Some(c) = cancel {
                    c.bail()?;
                }
                OmniGen2Block::load(
                    &vb,
                    &block_dev[n_double + i],
                    &format!("single_stream_layers.{i}"),
                    &cfg,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BooguTransformer2DModel {
            time1: QLinear::load(&vb, idev, "time_caption_embed.timestep_embedder.linear_1")?,
            time2: QLinear::load(&vb, idev, "time_caption_embed.timestep_embedder.linear_2")?,
            caption_norm: vb
                .get_f32_auto_on("time_caption_embed.caption_embedder.0.weight", idev)?,
            caption_lin: QLinear::load(&vb, idev, "time_caption_embed.caption_embedder.1")?,
            x_embedder: QLinear::load(&vb, idev, "x_embedder")?,
            context_refiner: mk("context_refiner", cfg.num_refiner_layers, false)?,
            noise_refiner: mk("noise_refiner", cfg.num_refiner_layers, true)?,
            double_stream_layers,
            single_stream_layers,
            // norm_out LayerNorm uses eps 1e-6 in the reference (not norm_eps).
            norm_out: LuminaLayerNormContinuous::load(&vb, idev, "norm_out", 1e-6, true)?,
            cfg,
            input_device: input_device.clone(),
        })
    }

    /// Build the 3-axis rope (cos,sin) for a `[instruct ; img]` joint sequence. Axis 0 is
    /// the caption/context position, axes 1/2 are the image row/col (patch-token) grid.
    /// Returns `(joint, cap, img)`, each `(cos [seq,head_dim/2], sin [seq,head_dim/2])`.
    fn build_rope(
        &self,
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
    ) -> Result<((NT, NT), (NT, NT), (NT, NT))> {
        let img_len = h_tokens * w_tokens;
        let seq = cap_len + img_len;
        let axes = self.cfg.axes_dim;
        let halves = [axes[0] / 2, axes[1] / 2, axes[2] / 2];
        let half_total: usize = halves.iter().sum();
        let theta = self.cfg.theta;
        // Per-axis integer positions over the joint sequence.
        let mut p0 = vec![0f32; seq];
        let mut p1 = vec![0f32; seq];
        let mut p2 = vec![0f32; seq];
        for l in 0..cap_len {
            p0[l] = l as f32;
            p1[l] = l as f32;
            p2[l] = l as f32;
        }
        let mut idx = cap_len;
        for r in 0..h_tokens {
            for c in 0..w_tokens {
                p0[idx] = cap_len as f32; // pe_shift after the caption
                p1[idx] = r as f32;
                p2[idx] = c as f32;
                idx += 1;
            }
        }
        let axes_pos = [&p0, &p1, &p2];
        let mut cos = vec![0f32; seq * half_total];
        let mut sin = vec![0f32; seq * half_total];
        for p in 0..seq {
            let mut col = 0;
            for a in 0..3 {
                let dim = axes[a] as f32;
                for j in 0..halves[a] {
                    let f = theta.powf(-(2.0 * j as f32) / dim);
                    let ang = axes_pos[a][p] * f;
                    cos[p * half_total + col] = ang.cos();
                    sin[p * half_total + col] = ang.sin();
                    col += 1;
                }
            }
        }
        // Built on CPU; the forward moves each (joint/cap/img) rope to the device(s) that need it.
        let joint_cos = NT::from_vec_f32(cos, (seq, half_total))?;
        let joint_sin = NT::from_vec_f32(sin, (seq, half_total))?;
        let cap = (
            joint_cos.narrow(0, 0, cap_len)?,
            joint_sin.narrow(0, 0, cap_len)?,
        );
        let img = (
            joint_cos.narrow(0, cap_len, img_len)?,
            joint_sin.narrow(0, cap_len, img_len)?,
        );
        Ok(((joint_cos, joint_sin), cap, img))
    }

    /// Text-to-image forward. `x [in_channels,H,W]` noisy latent, scalar `timestep` in
    /// `[0,1]` (flow-match sigma), `context [text_len, text_dim]` encoder hidden states,
    /// `num_tokens` = effective caption length (equals `text_len` for the unpadded path).
    /// Returns the velocity latent `[in_channels,H,W]`.
    pub fn forward(&self, x: &NT, timestep: f32, context: &NT, num_tokens: usize) -> Result<NT> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let d = x.shape().dims().to_vec();
        let (c, hgt, wid) = (d[0], d[1], d[2]);
        // pad_to_patch_size: pad H/W up to a patch multiple (no-op for even latents).
        let h_pad = hgt.div_ceil(p) * p;
        let w_pad = wid.div_ceil(p) * p;
        let idev = &self.input_device;
        let xv = x.to_vec_f32();
        let flat = patchify(&xv, c, hgt, wid, h_pad, w_pad, p); // [img_len, p*p*c]
        let h_tokens = h_pad / p;
        let w_tokens = w_pad / p;
        let img_len = h_tokens * w_tokens;
        let feat = p * p * c;
        let flat = NT::from_vec_f32(flat, (img_len, feat))?.to_device(idev)?;

        // time + caption embeddings (input device). The DMD pipeline feeds the raw sigma as the
        // timestep; the Lumina2 combined embedder scales it internally (timestep_scale = 1000).
        let t_scaled = timestep * 1000.0; // timestep_scale = 1000
        let tf = timestep_features(t_scaled, 256)?.to_device(idev)?;
        let temb = self.time2.forward(&silu(&self.time1.forward(&tf)?)?)?; // [1, 1024] on idev
        let context_i = context.to_device(idev)?;
        let mut text =
            self.caption_lin
                .forward(&rmsnorm(&context_i, &self.caption_norm, cfg.norm_eps)?)?; // [text_len, hidden]
        let cap_len = text.shape().dims()[0];
        debug_assert_eq!(
            cap_len, num_tokens,
            "unpadded T2I path expects text_len == num_tokens"
        );
        let _ = num_tokens;

        // 3-axis rope (CPU); materialize temb + joint/img rope on each device the plan uses (the
        // input device for the refiners, plus each distinct double/single block device).
        let (joint_cpu, cap_cpu, img_cpu) = self.build_rope(cap_len, h_tokens, w_tokens)?;
        let cap_i = (cap_cpu.0.to_device(idev)?, cap_cpu.1.to_device(idev)?);
        let mut devs: Vec<Device> = vec![idev.clone()];
        for l in &self.double_stream_layers {
            devs.push(l.device.clone());
        }
        for l in &self.single_stream_layers {
            devs.push(l.device.clone());
        }
        // per device: (temb, joint_cos, joint_sin, img_cos, img_sin)
        let mut rope: HashMap<DeviceLocation, (NT, NT, NT, NT, NT)> = HashMap::new();
        for dev in &devs {
            let loc = dev.location();
            if !rope.contains_key(&loc) {
                rope.insert(
                    loc,
                    (
                        temb.to_device(dev)?,
                        joint_cpu.0.to_device(dev)?,
                        joint_cpu.1.to_device(dev)?,
                        img_cpu.0.to_device(dev)?,
                        img_cpu.1.to_device(dev)?,
                    ),
                );
            }
        }
        let (temb_i, _, _, img_i_cos, img_i_sin) = &rope[&idev.location()];

        // Context refiner over the caption tokens (input device).
        for layer in &self.context_refiner {
            text = layer.forward(&text, (&cap_i.0, &cap_i.1), temb_i)?;
        }
        // Patch-embed + noise-refine the image tokens (input device; T2I: no reference images).
        let mut img = self.x_embedder.forward(&flat)?; // [img_len, hidden]
        for layer in &self.noise_refiner {
            img = layer.forward(&img, (img_i_cos, img_i_sin), temb_i)?;
        }

        // Dual-stream stage: hop BOTH streams to each block's device when it changes.
        let mut cur = idev.clone();
        for layer in &self.double_stream_layers {
            if layer.device.location() != cur.location() {
                cur.synchronize()?;
                img = img.to_device(&layer.device)?;
                text = text.to_device(&layer.device)?;
                layer.device.synchronize()?;
                cur = layer.device.clone();
            }
            let (t, jc, js, ic, is) = &rope[&layer.device.location()];
            let (ni, nt) = layer.forward(&img, &text, (jc, js), (ic, is), t)?;
            img = ni;
            text = nt;
        }

        // Single-stream stage over the concatenated [text ; img] sequence; hop the single hidden.
        let mut hidden = NT::cat(&[&text, &img], 0)?;
        for layer in &self.single_stream_layers {
            if layer.device.location() != cur.location() {
                cur.synchronize()?;
                hidden = hidden.to_device(&layer.device)?;
                layer.device.synchronize()?;
                cur = layer.device.clone();
            }
            let (t, jc, js, _, _) = &rope[&layer.device.location()];
            hidden = layer.forward(&hidden, (jc, js), t)?;
        }

        // Bring the hidden state back to the input device for norm_out + depatchify.
        if cur.location() != idev.location() {
            cur.synchronize()?;
            hidden = hidden.to_device(idev)?;
            idev.synchronize()?;
        }
        let hidden = self.norm_out.forward(&hidden, temb_i)?; // [seq, feat]
        let img_out = hidden.narrow(0, cap_len, img_len)?;
        let ov = img_out.to_vec_f32();
        let out = depatchify(&ov, c, hgt, wid, h_pad, w_pad, p); // [c, hgt, wid]
                                                                 // The official transformer returns the model prediction unnegated; the DMD step consumes
                                                                 // it as `latents + (1 - sigma) * pred`.
        NT::from_vec_f32(out, (c, hgt, wid))?.to_device(idev)
    }
}

/// Patchify `x [C, H, W]` (padded to `H_pad/W_pad`) into `[(H_pad/p)*(W_pad/p), p*p*C]`
/// with the reference `b c (h p1) (w p2) -> b (h w) (p1 p2 c)` channel ordering
/// (`c` fastest). Padded positions read zeros.
#[allow(clippy::too_many_arguments)]
fn patchify(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    h_pad: usize,
    w_pad: usize,
    p: usize,
) -> Vec<f32> {
    let ht = h_pad / p;
    let wt = w_pad / p;
    let feat = p * p * c;
    let mut out = vec![0f32; ht * wt * feat];
    for hi in 0..ht {
        for wi in 0..wt {
            let t = hi * wt + wi;
            for p1 in 0..p {
                for p2 in 0..p {
                    let (yy, xx) = (hi * p + p1, wi * p + p2);
                    if yy >= h || xx >= w {
                        continue; // zero pad
                    }
                    for ch in 0..c {
                        let col = (p1 * p + p2) * c + ch;
                        out[t * feat + col] = x[ch * h * w + yy * w + xx];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of `patchify`: `[(H_pad/p)*(W_pad/p), p*p*C] -> [C, H, W]` (cropped from the
/// padded grid), the reference `b (h w) (p1 p2 c) -> b c (h p1) (w p2)`.
#[allow(clippy::too_many_arguments)]
fn depatchify(
    hidden: &[f32],
    c: usize,
    h: usize,
    w: usize,
    h_pad: usize,
    w_pad: usize,
    p: usize,
) -> Vec<f32> {
    let ht = h_pad / p;
    let wt = w_pad / p;
    let feat = p * p * c;
    let mut out = vec![0f32; c * h * w];
    for hi in 0..ht {
        for wi in 0..wt {
            let t = hi * wt + wi;
            for p1 in 0..p {
                for p2 in 0..p {
                    let (yy, xx) = (hi * p + p1, wi * p + p2);
                    if yy >= h || xx >= w {
                        continue;
                    }
                    for ch in 0..c {
                        let col = (p1 * p + p2) * c + ch;
                        out[ch * h * w + yy * w + xx] = hidden[t * feat + col];
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Loads the ~10GB fp8 checkpoint and runs one forward on dummy latents. Ignored by
    // default (heavy, needs the local weights). Run with:
    //   cargo test -p loken --profile fast native_boogu_dit -- --ignored --nocapture
    #[test]
    #[ignore]
    fn boogu_dit_one_forward() {
        let path = format!(
            "{}/boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
            crate::inference::cache::hf::models_dir()
        );
        // Plan across whatever CUDA devices are present (fastest-first, spill to CPU); the plan
        // decides placement - the test never hardcodes a device.
        let mut cuda_devices = HashMap::new();
        let cuda_list: Vec<(usize, u64)> =
            crate::inference::place::device_probe::probe_cuda_devices(0)
                .into_iter()
                .map(|(i, f, d)| {
                    cuda_devices.insert(i, d);
                    (i, f)
                })
                .collect();
        let cfg = Config::default();
        let n_blocks = cfg.num_double_stream_layers + cfg.num_single_stream_layers;
        let sz = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let plan = HeteroPlan::calculate(n_blocks, sz, &cuda_list, &[], 1.0);
        println!("boogu DiT smoke test, plan: {:?}", plan.segments);
        let model =
            BooguTransformer2DModel::load(&path, &cuda_devices, &plan).expect("load boogu DiT");
        let dev = model.input_device().clone();
        // 16-channel latent, 8x8 spatial -> 4x4 = 16 image tokens; 6 caption tokens.
        let (c, h, w) = (16usize, 8usize, 8usize);
        let x = NT::from_vec_f32(vec![0.01f32; c * h * w], (c, h, w))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let text_len = 6usize;
        let ctx = NT::from_vec_f32(vec![0.02f32; text_len * 4096], (text_len, 4096))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let out = model.forward(&x, 0.5, &ctx, text_len).expect("forward");
        assert_eq!(out.shape().dims(), &[c, h, w]);
        let v = out.to_vec_f32();
        assert!(
            v.iter().all(|z| z.is_finite()),
            "output has non-finite values"
        );
        println!("boogu DiT forward ok: out[0..4]={:?}", &v[..4]);
    }

    // Reference-diff harness: run the SAME fixed inputs as the official torch implementation
    // (scratchpad/ref_diff/run_reference.py) and dump raw f32 outputs for offline comparison.
    // Inputs/outputs live in the dir given by BOOGU_REF_DIFF_DIR. Run with:
    //   BOOGU_REF_DIFF_DIR=... cargo test -p loken --release boogu_dit_ref_diff -- --ignored --nocapture
    #[test]
    #[ignore]
    fn boogu_dit_ref_diff() {
        let dir = std::env::var("BOOGU_REF_DIFF_DIR").expect("set BOOGU_REF_DIFF_DIR");
        let read_f32 = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(format!("{dir}/{name}")).expect(name);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let path = format!(
            "{}/boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
            crate::inference::cache::hf::models_dir()
        );
        let mut cuda_devices = HashMap::new();
        let cuda_list: Vec<(usize, u64)> =
            crate::inference::place::device_probe::probe_cuda_devices(0)
                .into_iter()
                .map(|(i, f, d)| {
                    cuda_devices.insert(i, d);
                    (i, f)
                })
                .collect();
        let cfg = Config::default();
        let n_blocks = cfg.num_double_stream_layers + cfg.num_single_stream_layers;
        let sz = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let plan = HeteroPlan::calculate(n_blocks, sz, &cuda_list, &[], 1.0);
        let model =
            BooguTransformer2DModel::load(&path, &cuda_devices, &plan).expect("load boogu DiT");
        let dev = model.input_device().clone();
        let (c, h, w) = (16usize, 16usize, 16usize);
        let x = NT::from_vec_f32(read_f32("in_latent_16x16x16.f32"), (c, h, w))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let text_len = 32usize;
        let ctx = NT::from_vec_f32(read_f32("in_context_32x4096.f32"), (text_len, 4096))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        for sigma in [0.001f32, 0.5] {
            let out = model.forward(&x, sigma, &ctx, text_len).expect("forward");
            let v = out.to_vec_f32();
            let n = v.len() as f32;
            let mean = v.iter().sum::<f32>() / n;
            let std = (v.iter().map(|z| (z - mean) * (z - mean)).sum::<f32>() / n).sqrt();
            let tag = format!("{sigma}").replace('.', "p");
            let bytes: Vec<u8> = v.iter().flat_map(|z| z.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/rust_out_sigma{tag}.f32"), bytes).unwrap();
            println!("sigma={sigma}: mean={mean:.5} std={std:.5} -> rust_out_sigma{tag}.f32");
        }
    }

    // Stage-level variant of the ref-diff: replicates the forward while dumping every stage so
    // the first diverging module (vs the torch reference's forward hooks) can be identified.
    #[test]
    #[ignore]
    fn boogu_dit_stage_diff() {
        let dir = std::env::var("BOOGU_REF_DIFF_DIR").expect("set BOOGU_REF_DIFF_DIR");
        let read_f32 = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(format!("{dir}/{name}")).expect(name);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let dump = |name: &str, t: &NT| {
            let v = t.to_vec_f32();
            let bytes: Vec<u8> = v.iter().flat_map(|z| z.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/rust_stage_{name}.f32"), bytes).unwrap();
        };
        let path = format!(
            "{}/boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
            crate::inference::cache::hf::models_dir()
        );
        let mut cuda_devices = HashMap::new();
        let cuda_list: Vec<(usize, u64)> =
            crate::inference::place::device_probe::probe_cuda_devices(0)
                .into_iter()
                .map(|(i, f, d)| {
                    cuda_devices.insert(i, d);
                    (i, f)
                })
                .collect();
        let cfg0 = Config::default();
        let n_blocks = cfg0.num_double_stream_layers + cfg0.num_single_stream_layers;
        let sz = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let plan = HeteroPlan::calculate(n_blocks, sz, &cuda_list, &[], 1.0);
        let model = BooguTransformer2DModel::load(&path, &cuda_devices, &plan).expect("load");
        let idev = model.input_device().clone();
        let (c, hgt, wid) = (16usize, 16usize, 16usize);
        let xv = read_f32("in_latent_16x16x16.f32");
        let ctxv = read_f32("in_context_32x4096.f32");
        let cap_len = 32usize;

        for sigma in [0.001f32, 0.5] {
            let tag = format!("{sigma}").replace('.', "p");
            let cfg = &model.cfg;
            let p = cfg.patch_size;
            let (h_pad, w_pad) = (hgt, wid);
            let flat = patchify(&xv, c, hgt, wid, h_pad, w_pad, p);
            let h_tokens = h_pad / p;
            let w_tokens = w_pad / p;
            let img_len = h_tokens * w_tokens;
            let feat = p * p * c;
            let flat = NT::from_vec_f32(flat, (img_len, feat))
                .unwrap()
                .to_device(&idev)
                .unwrap();

            let t_scaled = sigma * 1000.0;
            let tf = timestep_features(t_scaled, 256)
                .unwrap()
                .to_device(&idev)
                .unwrap();
            let temb = model
                .time2
                .forward(&silu(&model.time1.forward(&tf).unwrap()).unwrap())
                .unwrap();
            dump(&format!("temb_sigma{tag}"), &temb);
            let context_i = NT::from_vec_f32(ctxv.clone(), (cap_len, 4096))
                .unwrap()
                .to_device(&idev)
                .unwrap();
            let mut text = model
                .caption_lin
                .forward(&rmsnorm(&context_i, &model.caption_norm, cfg.norm_eps).unwrap())
                .unwrap();
            dump(&format!("caption_embed_sigma{tag}"), &text);

            let (joint_cpu, cap_cpu, img_cpu) =
                model.build_rope(cap_len, h_tokens, w_tokens).unwrap();
            let cap_i = (
                cap_cpu.0.to_device(&idev).unwrap(),
                cap_cpu.1.to_device(&idev).unwrap(),
            );
            let img_i = (
                img_cpu.0.to_device(&idev).unwrap(),
                img_cpu.1.to_device(&idev).unwrap(),
            );
            let joint_i = (
                joint_cpu.0.to_device(&idev).unwrap(),
                joint_cpu.1.to_device(&idev).unwrap(),
            );

            for (i, layer) in model.context_refiner.iter().enumerate() {
                text = layer.forward(&text, (&cap_i.0, &cap_i.1), &temb).unwrap();
                dump(&format!("context_refiner.{i}_sigma{tag}"), &text);
            }
            let mut img = model.x_embedder.forward(&flat).unwrap();
            dump(&format!("x_embedder_sigma{tag}"), &img);
            for (i, layer) in model.noise_refiner.iter().enumerate() {
                img = layer.forward(&img, (&img_i.0, &img_i.1), &temb).unwrap();
                dump(&format!("noise_refiner.{i}_sigma{tag}"), &img);
            }
            for (i, layer) in model.double_stream_layers.iter().enumerate() {
                let (ni, nt) = layer
                    .forward(
                        &img,
                        &text,
                        (&joint_i.0, &joint_i.1),
                        (&img_i.0, &img_i.1),
                        &temb,
                    )
                    .unwrap();
                img = ni;
                text = nt;
                dump(&format!("double.{i}_sigma{tag}"), &img);
            }
            let mut hidden = NT::cat(&[&text, &img], 0).unwrap();
            for (i, layer) in model.single_stream_layers.iter().enumerate() {
                hidden = layer
                    .forward(&hidden, (&joint_i.0, &joint_i.1), &temb)
                    .unwrap();
                dump(&format!("single.{i}_sigma{tag}"), &hidden);
            }
            let hidden = model.norm_out.forward(&hidden, &temb).unwrap();
            dump(&format!("norm_out_sigma{tag}"), &hidden);
            println!("stage dump complete for sigma={sigma}");
        }
    }
}
