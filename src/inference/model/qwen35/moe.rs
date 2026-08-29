//! qwen35moe (Qwen3.5-MoE / Qwen3-Next family) - hybrid gated-DeltaNet linear
//! attention + full attention + softmax-MoE with a scalar-gated shared expert.
//!
//! TEXT path only (vision ViT deferred). Block layout:
//!   x += operator(attn_norm(x))            // operator = gated-deltanet | gated-attention
//!   x += moe(attn_post_norm(x))
//!
//! - **gated DeltaNet** (linear attn, qwen3-next): attn_qkv -> causal conv1d(k=4)+silu
//!   -> split q,k,v; q/k L2-normed; per-token recurrent gated delta rule
//!   `S = g.S + beta.k⊗(v - Sᵀk); o = Sᵀq`; z-gated RMSNorm; ssm_out.
//!   (recurrent form ported from vllm recurrent_gated_delta_rule - avoids the
//!   chunked solve_tri/cumsum path a per-op formulation lacks.)
//! - **gated attention** (every 4th block): attn_q = [Q | gate] joint; Q/K RMS-norm;
//!   partial mRoPE (text -> standard NEOX rope on rope_dim); GQA; `o.sigmoid(gate)`; wo.
//! - **MoE**: softmax router (top-k, normalized) + SwiGLU experts + a shared expert
//!   gated by `sigmoid(scalar_gate.x)`.

use crate::inference::model::qwen35::vision::Qwen35Vision;
use crate::tensor::layer::{Embedding, RmsNorm};
use crate::tensor::quantized::{gguf_file, QMatMul, QTensor};
use crate::tensor::{DType, Device, IndexOp, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Optional mmap of the source GGUF: CPU-placed passthrough tensors become
/// zero-copy file views instead of heap copies.
type Mm<'a> = Option<&'a Arc<memmap2::Mmap>>;

fn load_q8<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
    mm: Mm<'_>,
) -> Result<QTensor> {
    use crate::tensor::quantized::GgmlDType;
    let passthrough = c
        .tensor_infos
        .get(name)
        .map(|i| crate::inference::moe_cuda::gemm::expert_kernels_serve(i.ggml_dtype))
        .unwrap_or(false);
    if passthrough {
        if let (Some(m), true) = (mm, d.is_cpu()) {
            if let Some(qt) = crate::tensor::quant_view::gguf_mmap_view(c, m, name)? {
                return Ok(qt);
            }
        }
        return c.tensor(r, name, d);
    }
    let cpu = c.tensor(r, name, &Device::Cpu)?;
    let f = cpu.dequantize(&Device::Cpu)?;
    QTensor::quantize_onto(&f, GgmlDType::Q8_0, d)
}
fn ld_f32<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
) -> Result<Tensor> {
    c.tensor(r, name, d)?.dequantize(d)
}
/// Row-wise concat of same-dtype quantized weights sharing one input dim into a
/// single QMatMul (the DeltaNet input-projection fusion). Returns None when the
/// tensors mix quant dtypes / use a dtype outside the GPU-kernel set - the
/// caller falls back to four separate weights.
fn load_fused_proj<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    names: &[String],
    d: &Device,
) -> Result<Option<QMatMul>> {
    use crate::tensor::quantized::GgmlDType;
    let mut dt = None;
    let mut k = None;
    for n in names {
        let Some(info) = c.tensor_infos.get(n) else {
            return Ok(None);
        };
        if !crate::inference::moe_cuda::gemm::expert_kernels_serve(info.ggml_dtype) {
            return Ok(None);
        }
        if *dt.get_or_insert(info.ggml_dtype) != info.ggml_dtype {
            return Ok(None);
        }
        let dims = info.shape.dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        if *k.get_or_insert(dims[1]) != dims[1] {
            return Ok(None);
        }
    }
    let (dt, k) = (dt.unwrap(), k.unwrap());
    let mut bytes = Vec::new();
    let mut rows = 0usize;
    for n in names {
        let t = c.tensor(r, n, &Device::Cpu)?;
        rows += t.shape().dims()[0];
        bytes.extend_from_slice(&t.data()?);
    }
    let qt = crate::tensor::quantized::QTensor::from_ggml_bytes(dt, &bytes, vec![rows, k], d)?;
    Ok(Some(QMatMul::from_arc(Arc::new(qt))?))
}
/// L2 norm over last dim: x . rsqrt(sum(x²)+eps).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let s = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_div(&(s + eps)?.sqrt()?)
}
/// per-head RMS norm over last dim, weight [dim].
fn head_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let v = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(v + eps)?.sqrt()?)?.broadcast_mul(w)
}

/// Per-head RMSNorm over the last dim for attention Q/K. On CUDA uses the fused
/// `head_rmsnorm` kernel (1 launch vs ~6 tensor ops); tensor-op fallback on CPU.
/// `x` is [b, heads, seq, hd]; output same shape in dtype `st`.
fn head_rms_fused(
    x: &Tensor,
    w: &Tensor,
    eps: f64,
    b: usize,
    heads: usize,
    seq: usize,
    hd: usize,
    st: DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let xf = x.to_dtype(DType::F32)?.reshape((b * heads * seq, hd))?;
        let y = crate::inference::moe_cuda::head_rmsnorm(&xf, w, eps as f32)?;
        return y.reshape((b, heads, seq, hd))?.to_dtype(st);
    }
    head_rms(&x.to_dtype(DType::F32)?, w, eps)?.to_dtype(st)
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
    pub rms_eps: f64,
    pub context_length: usize,
    pub rope_freq_base: f32,
    // deltanet
    pub conv_kernel: usize, // ssm.conv_kernel = 4
    pub head_kv_dim: usize, // ssm.state_size = 128
    pub n_k_heads: usize,   // ssm.group_count = 16
    pub n_v_heads: usize,   // ssm.time_step_rank = 32
    pub d_inner: usize,     // ssm.inner_size = 4096
}

impl Qwen35Config {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        // GGUF namespaces every key under the model's OWN architecture tag. Three tags
        // share this block - "qwen35moe", the dense "qwen35", and "qwen3next" - so read
        // the one the file actually declares instead of assuming the mixture's. Naming
        // only one is why the dense variant failed on `missing block_count` while its
        // metadata was sitting there under a different prefix.
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string());
        let g = |k: &str| {
            arch.as_deref()
                .and_then(|a| ct.metadata.get(&format!("{a}.{k}")))
                .or_else(|| ct.metadata.get(&format!("qwen35moe.{k}")))
                .or_else(|| ct.metadata.get(&format!("qwen35.{k}")))
                .or_else(|| ct.metadata.get(&format!("qwen3next.{k}")))
        };
        let req = |k: &str| -> Result<usize> {
            g(k).and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| {
                    crate::tensor::Error::msg(format!(
                        "{}: missing {k}",
                        arch.as_deref().unwrap_or("qwen35moe")
                    ))
                })
        };
        let ou = |k: &str, d: usize| {
            g(k).and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let of = |k: &str, d: f32| g(k).and_then(|v| v.to_f32().ok()).unwrap_or(d);
        Ok(Self {
            n_layers: req("block_count")?,
            d_model: req("embedding_length")?,
            n_head: ou("attention.head_count", 16),
            head_dim: ou("attention.key_length", 256),
            rope_dim: ou("rope.dimension_count", 64),
            n_expert: ou("expert_count", 256),
            n_expert_used: ou("expert_used_count", 8),
            expert_ff: ou("expert_feed_forward_length", 512),
            shared_ff: ou("expert_shared_feed_forward_length", 512),
            rms_eps: of("attention.layer_norm_rms_epsilon", 1e-6) as f64,
            context_length: ou("context_length", 32768),
            rope_freq_base: of("rope.freq_base", 1e7),
            conv_kernel: ou("ssm.conv_kernel", 4),
            head_kv_dim: ou("ssm.state_size", 128),
            n_k_heads: ou("ssm.group_count", 16),
            n_v_heads: ou("ssm.time_step_rank", 32),
            d_inner: ou("ssm.inner_size", 4096),
        })
    }
}

fn build_rope(
    rotary_dim: usize,
    max_seq: usize,
    base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = crate::inference::model::rope::inverse_frequencies(rotary_dim, base);
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for p in 0..max_seq {
        for (i, &f) in inv.iter().enumerate() {
            let a = p as f32 * f;
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_seq, half), device)?,
        Tensor::from_vec(sin, (max_seq, half), device)?,
    ))
}

/// mRoPE-2D cos/sin from per-token 3D positions `(t,h,w)`. For qwen3-vl IMROPE
/// (`mrope_interleaved=true`, sections `[11,11,10]` over the 32 freq pairs of
/// rope_dim=64), freq pair `j` rotates by position component `[t,h,w][j%3]`  - 
/// the bounds in ggml's is_imrope branch collapse to exactly this j%3 rule for
/// these section sizes. For text tokens `t==h==w` this is bit-identical to
/// `build_rope` (which is why the text path already works). Returns
/// (cos, sin) shaped `[seq, rope_dim/2]`, ready for `Qwen35Attn::rope`.
fn build_mrope_cos_sin(
    pos: &[[i32; 3]],
    rotary_dim: usize,
    base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = crate::inference::model::rope::inverse_frequencies(rotary_dim, base);
    let seq = pos.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (p, thw) in pos.iter().enumerate() {
        for (j, &f) in inv.iter().enumerate() {
            let pc = thw[j % 3] as f32; // t/h/w selected by j%3 (IMROPE interleave)
            let a = pc * f;
            cos[p * half + j] = a.cos();
            sin[p * half + j] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, half), device)?,
        Tensor::from_vec(sin, (seq, half), device)?,
    ))
}

/// Per-token 3D mRoPE positions `(t,h,w)` for a sequence whose token ids are
/// `ids`, where image-placeholder tokens (`image_token`) appear in contiguous
/// runs each backed by a merged grid `(gh, gw)` (gh = grid_h/merge,
/// gw = grid_w/merge; run length must equal gh*gw). Text tokens get a running
/// scalar `p` (t=h=w=p). For image run starting at scalar base `P0`: token i
/// gets `t=P0, h=P0+i/gw, w=P0+i%gw`; after the run the scalar advances to
/// `P0 + gw` (ollama qwen3vl rule - by merged width). `grids` supplies the
/// (gh,gw) for each image run in order.
fn mrope_positions(ids: &[u32], image_token: u32, grids: &[(usize, usize)]) -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(ids.len());
    let mut p: i32 = 0;
    let mut gi = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token {
            let (gh, gw) = grids.get(gi).copied().unwrap_or((1, 1));
            let n = gh * gw;
            let p0 = p;
            for k in 0..n.min(ids.len() - i) {
                out.push([p0, p0 + (k / gw) as i32, p0 + (k % gw) as i32]);
            }
            p = p0 + gw as i32; // advance by merged width
            i += n;
            gi += 1;
        } else {
            out.push([p, p, p]);
            p += 1;
            i += 1;
        }
    }
    out
}

/// Gated attention (Qwen3-Next): wq = [Q|gate], Q/K norm, partial rope, GQA,
/// sigmoid output gate.
/// Attention q/k/v projections - same input, fused row-wise at load when the
/// GGUF dtypes allow (same trick as [`DeltaProj`]).
enum AttnProj {
    /// rows = [q.gate(n_head.2.hd) | k(n_kv.hd) | v(n_kv.hd)]
    Fused {
        w: QMatMul,
        q_rows: usize,
        kv_rows: usize,
    },
    Split {
        wq: QMatMul,
        wk: QMatMul,
        wv: QMatMul,
    },
}

pub struct Qwen35Attn {
    qkv: AttnProj, // wq -> [n_head * head_dim * 2]  (Q + gate)
    wo: QMatMul,
    q_norm: Tensor,
    k_norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_dim: usize,
    num_kv_groups: usize,
    rms_eps: f64,
    // Fixed-size O(1)/token cache (slice_set) - was ConcatKvCache, whose per-token
    // cat re-copied the whole KV (O(N²) over a generation) and fed the equally
    // O(N) k.transpose().contiguous() below. Grows in steps beyond the budget.
    kv_cache: crate::tensor::KvCache,
}

impl Qwen35Attn {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Qwen35Config,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let qkv_names = [
            format!("{p}.attn_q.weight"),
            format!("{p}.attn_k.weight"),
            format!("{p}.attn_v.weight"),
        ];
        let k_rows = c
            .tensor_infos
            .get(&qkv_names[1])
            .map(|i| i.shape.dims()[0])
            .unwrap_or(cfg.head_dim);
        let n_kv_head = (k_rows / cfg.head_dim.max(1)).max(1);
        let qkv = match load_fused_proj(c, r, &qkv_names, device)? {
            Some(w) => AttnProj::Fused {
                w,
                q_rows: c.tensor_infos[&qkv_names[0]].shape.dims()[0],
                kv_rows: k_rows,
            },
            None => AttnProj::Split {
                wq: QMatMul::from_qtensor(load_q8(c, r, &qkv_names[0], device, mm)?)?,
                wk: QMatMul::from_qtensor(load_q8(c, r, &qkv_names[1], device, mm)?)?,
                wv: QMatMul::from_qtensor(load_q8(c, r, &qkv_names[2], device, mm)?)?,
            },
        };
        Ok(Self {
            qkv,
            wo: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.attn_output.weight"),
                device,
                mm,
            )?)?,
            q_norm: ld_f32(c, r, &format!("{p}.attn_q_norm.weight"), device)?,
            k_norm: ld_f32(c, r, &format!("{p}.attn_k_norm.weight"), device)?,
            cos,
            sin,
            n_head: cfg.n_head,
            n_kv_head,
            head_dim: cfg.head_dim,
            rope_dim: cfg.rope_dim,
            num_kv_groups: cfg.n_head / n_kv_head.max(1),
            rms_eps: cfg.rms_eps,
            // the reference KvCache allocates the FULL initial capacity on the first
            // append (grow_by = initial) and grows in those steps. qwen3.5 is 22GB
            // on 2x16GB - the fast GPU is near-brim, so a big pre-alloc starves
            // prefill activations (OOM). Keep the initial small (grows on demand).
            kv_cache: crate::tensor::KvCache::new(2, cfg.context_length.min(2048).max(256)),
        })
    }
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }
    /// Deep-copied KV state for the prefix cache. See `Cache::deep_copy`.
    fn snapshot(&self) -> Result<LayerStateSnap> {
        Ok(LayerStateSnap::Attn(self.kv_cache.deep_copy()?))
    }
    fn restore(&mut self, s: &LayerStateSnap) -> Result<()> {
        if let LayerStateSnap::Attn(kv) = s {
            self.kv_cache = kv.deep_copy()?;
        }
        Ok(())
    }

    /// partial NEOX rope on first rope_dim dims.
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Fused partial-NEOX rope (one launch), bit-identical. Gated to decode
        // (seq==1, the text path) - the seq>1 image-prefill mRoPE path keeps the
        // tensor-op chain untouched. Falls back if not on the CUDA F16 fast path.
        if x.dim(2)? == 1 && cos.dtype() == DType::F16 {
            if let Some(o) = crate::inference::kernel::fused::neox_rope_f16(
                x,
                cos,
                sin,
                self.rope_dim.min(self.head_dim),
            )? {
                return Ok(o);
            }
        }
        if self.rope_dim >= self.head_dim {
            return crate::tensor::ops::rope(&x.contiguous()?, cos, sin);
        }
        let rot = x.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
        let pass = x.narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?;
        let rot = crate::tensor::ops::rope(&rot, cos, sin)?;
        Tensor::cat(&[&rot, &pass], D::Minus1)?.contiguous()
    }

    pub fn forward(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let seq = x.dim(1)?;
        let cos = self.cos.narrow(0, input_pos, seq)?;
        let sin = self.sin.narrow(0, input_pos, seq)?;
        self.forward_rope(x, &cos, &sin, input_pos)
    }

    /// Attention with externally-supplied rope cos/sin `[seq, rope_dim/2]`
    /// (used for image-prefill mRoPE-2D); `input_pos` is the KV offset for the
    /// causal mask. The scalar-position `forward` delegates here.
    pub fn forward_rope(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        input_pos: usize,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dev = x.device().clone();
        let st = x.dtype();
        // wq -> [b, seq, n_head, 2*head_dim] : per head [Q(head_dim) | gate(head_dim)]
        // q/k/v in ONE matmul when fused (slices are views at decode; the
        // consumers below are all tensor ops, which honour storage offsets).
        let (qg_flat, k_flat, v_flat) = match &self.qkv {
            AttnProj::Fused { w, q_rows, kv_rows } => {
                let full = w.forward(x)?;
                (
                    full.narrow(D::Minus1, 0, *q_rows)?.contiguous()?,
                    full.narrow(D::Minus1, *q_rows, *kv_rows)?.contiguous()?,
                    full.narrow(D::Minus1, *q_rows + *kv_rows, *kv_rows)?
                        .contiguous()?,
                )
            }
            AttnProj::Split { wq, wk, wv } => (wq.forward(x)?, wk.forward(x)?, wv.forward(x)?),
        };
        let qg = qg_flat.reshape((b, seq, self.n_head, 2 * self.head_dim))?;
        let q = qg.narrow(D::Minus1, 0, self.head_dim)?.contiguous()?; // [b,seq,nh,hd]
        let gate = qg
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .contiguous()?;
        let q = q.transpose(1, 2)?.contiguous()?; // [b,nh,seq,hd]
        let k = k_flat
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v_flat
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // Q/K per-head RMS-norm (fused kernel on CUDA)
        let q = head_rms_fused(
            &q,
            &self.q_norm,
            self.rms_eps,
            b,
            self.n_head,
            seq,
            self.head_dim,
            st,
        )?;
        let k = head_rms_fused(
            &k,
            &self.k_norm,
            self.rms_eps,
            b,
            self.n_kv_head,
            seq,
            self.head_dim,
            st,
        )?;
        let cos = cos.to_dtype(q.dtype())?;
        let sin = sin.to_dtype(q.dtype())?;
        let q = self.rope(&q, &cos, &sin)?;
        let k = self.rope(&k, &cos, &sin)?;
        let (k, v) = self.kv_cache.append(&k, &v)?; // [b, n_kv, kv, hd]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let kv = input_pos + seq;
        let (nkv, gr, hd) = (self.n_kv_head, self.num_kv_groups, self.head_dim);
        // Fused flash-decode (seq=1): one F16 kernel does scores+softmax+V on the
        // KvCache narrow - no cuBLAS, no [n_head,kv] scores in HBM, fewer launches.
        // qwen3.5 is LAUNCH-bound (~20-29% util) so the saved launches can outweigh
        // the slower wide-head (hd=256) one-warp-per-head math (which loses on the
        // GPU-bound nemotron). No attention sinks. Falls back to the matmul chain.
        // kv <= 2048 gate: the flash-decode kernel scans kv_len sequentially (one
        // warp/head); at hd=256 that loses to the kv-parallel cuBLAS gemv once the
        // context is large, so beyond ~2K kv fall back to the cuBLAS chain (keeps
        // the short/medium win without risking the long-prompt cell).
        // Split-K flash-decode now parallelises the kv scan (fused_kernels::flash_decode
        // tiers nsplit 32/64/128), so the fast path holds at long ctx instead of
        // falling back to the materialised-scores cuBLAS chain past 512.
        let flash = if seq == 1
            && dev.is_cuda()
            && q.dtype() == DType::F16
            && hd % 32 == 0
            && hd <= 256
            && kv <= 16384
        {
            crate::inference::kernel::fused::flash_decode(
                &q.reshape((b, self.n_head, hd))?,
                &k,
                &v,
                None,
                None,
                scale as f32,
                b,
                self.n_head,
                nkv,
                kv,
                hd,
            )?
        } else {
            None
        };
        let out = if let Some(o) = flash {
            o.reshape((b, seq, self.n_head * hd))?
        } else {
            // GQA WITHOUT repeat_kv: group q heads per kv-head (reshape) and batch-
            // matmul so K/V are read 1x not grx. Bit-exact (identical dot products).
            // Pass the KvCache narrow's transpose straight to matmul (NO
            // .contiguous()): the substrate maps it to a cuBLAS gemm with transB=T that
            // reads K contiguously. Same for V. (Proven on nemotron.)
            let kc = k.transpose(2, 3)?; // [b, n_kv, hd, kv] (strided)
            let qr = q.reshape((b, nkv, gr * seq, hd))?; // q head j*gr+g -> [.,j,g*seq+s,.]
                                                         // Keep scores in the matmul's own dtype; the widen and the scale
                                                         // fold into the masked softmax below, which walks the tensor anyway.
            let scores_h = qr.matmul(&kc)?.reshape((b, self.n_head, seq, kv))?;
            let w = if seq > 1 {
                let mut mask = vec![0f32; seq * kv];
                for i in 0..seq {
                    for j in (input_pos + i + 1)..kv {
                        mask[i * kv + j] = f32::NEG_INFINITY;
                    }
                }
                let m = Tensor::from_vec(mask, (1, 1, seq, kv), &dev)?;
                crate::tensor::ops::softmax_scaled_masked(&scores_h, scale, &m)?
            } else {
                let scores = (scores_h.to_dtype(DType::F32)? * scale)?;
                crate::tensor::ops::softmax_last_dim(&scores)?
            }
            .to_dtype(v.dtype())?;
            let wr = w.reshape((b, nkv, gr * seq, kv))?;
            wr.matmul(&v)?
                .reshape((b, self.n_head, seq, hd))?
                .transpose(1, 2)?
                .reshape((b, seq, self.n_head * hd))? // [b,seq,nh*hd]
        };
        // sigmoid output gate (gate flattened to [b,seq,nh*hd])
        let gate = gate.reshape((b, seq, self.n_head * self.head_dim))?;
        let g = crate::tensor::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(out.dtype())?;
        let out = (out * g)?;
        self.wo.forward(&out)
    }
}

/// DeltaNet input projections. All four read the SAME hidden state, so when the
/// GGUF stores them in one quant dtype they are concatenated row-wise at load
/// into a single weight -> ONE activation-quantize + ONE mmvq launch per layer
/// instead of four of each (qwen3.5 decode is CPU-launch-bound: ~1100
/// launches/token; this removes ~180 of them).
enum DeltaProj {
    /// rows = [qkv(conv_dim) | z(n_v.v_dim) | a(n_v) | b(n_v)]
    Fused(QMatMul),
    Split {
        in_qkv: QMatMul,
        z_proj: QMatMul,
        a_proj: QMatMul,
        b_proj: QMatMul,
    },
    /// Qwen3-Next-80B (unsloth) GGUF packs beta+alpha into one `ssm_ba` proj
    /// (rows = [b(n_v) | a(n_v)]); split at project() time.
    Ba {
        in_qkv: QMatMul,
        z_proj: QMatMul,
        ba_proj: QMatMul,
    },
}

/// Host-side DeltaNet state + weight caches for the fused CPU recurrence
/// (`forward_cpu_fused`): plain `Vec<f32>` so the per-token loop never touches
/// the tensor-op dispatch layer (the per-token tensor-op path is ~30 small ops/layer,
/// each a single-threaded dispatch - the dominant CPU decode cost).
#[derive(Clone)]
struct CpuDeltaState {
    conv: Vec<f32>,  // [conv_dim * (kernel-1)] rolling conv window
    ssm: Vec<f32>,   // [n_v * v_dim * k_dim]
    w: Vec<f32>,     // conv weights [conv_dim * kernel]
    a_log: Vec<f32>, // [n_v]
    dt: Vec<f32>,    // [n_v]
    norm: Vec<f32>,  // [v_dim]
}

/// Gated DeltaNet linear attention (recurrent form).
pub struct Qwen35DeltaNet {
    proj: DeltaProj,   // in_qkv[8192] + attn_gate[4096] + ssm_alpha/beta[n_v]
    conv_w: Tensor,    // [conv_dim, kernel] F32
    a_log: Tensor,     // ssm_a [n_v_heads] F32
    dt_bias: Tensor,   // ssm_dt [n_v_heads] F32
    norm_w: Tensor,    // ssm_norm [v_head_dim] F32
    out_proj: QMatMul, // ssm_out -> d_model
    head_k_dim: usize,
    head_v_dim: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    conv_dim: usize,
    conv_kernel: usize,
    rms_eps: f64,
    device: Device,
    conv_state: Option<Tensor>, // [b, conv_dim, kernel-1]
    ssm_state: Option<Tensor>,  // [b, n_v_heads, v_dim, k_dim]
    cpu_state: Option<CpuDeltaState>,
}

impl Qwen35DeltaNet {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Qwen35Config,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let head_k_dim = cfg.head_kv_dim;
        let head_v_dim = cfg.d_inner / cfg.n_v_heads.max(1);
        let conv_dim = head_k_dim * cfg.n_k_heads * 2 + head_v_dim * cfg.n_v_heads;
        let proj_names = [
            format!("{p}.attn_qkv.weight"),
            format!("{p}.attn_gate.weight"),
            format!("{p}.ssm_alpha.weight"),
            format!("{p}.ssm_beta.weight"),
        ];
        let has = |n: &str| c.tensor_infos.contains_key(n);
        let proj = if has(&proj_names[2]) {
            // Qwen3.5-MoE layout: separate ssm_alpha/ssm_beta (fused if possible).
            match load_fused_proj(c, r, &proj_names, device)? {
                Some(w) => DeltaProj::Fused(w),
                None => DeltaProj::Split {
                    in_qkv: QMatMul::from_qtensor(load_q8(c, r, &proj_names[0], device, mm)?)?,
                    z_proj: QMatMul::from_qtensor(load_q8(c, r, &proj_names[1], device, mm)?)?,
                    a_proj: QMatMul::from_qtensor(load_q8(c, r, &proj_names[2], device, mm)?)?,
                    b_proj: QMatMul::from_qtensor(load_q8(c, r, &proj_names[3], device, mm)?)?,
                },
            }
        } else {
            // Qwen3-Next-80B layout: combined ssm_ba (beta‖alpha).
            DeltaProj::Ba {
                in_qkv: QMatMul::from_qtensor(load_q8(c, r, &proj_names[0], device, mm)?)?,
                z_proj: QMatMul::from_qtensor(load_q8(c, r, &proj_names[1], device, mm)?)?,
                ba_proj: QMatMul::from_qtensor(load_q8(
                    c,
                    r,
                    &format!("{p}.ssm_ba.weight"),
                    device,
                    mm,
                )?)?,
            }
        };
        Ok(Self {
            proj,
            conv_w: ld_f32(c, r, &format!("{p}.ssm_conv1d.weight"), device)?,
            a_log: ld_f32(c, r, &format!("{p}.ssm_a"), device)?.flatten_all()?,
            // qwen3.5-moe: ssm_dt ; qwen3-next-80B: ssm_dt.bias
            dt_bias: ld_f32(c, r, &format!("{p}.ssm_dt"), device)
                .or_else(|_| ld_f32(c, r, &format!("{p}.ssm_dt.bias"), device))?
                .flatten_all()?,
            norm_w: ld_f32(c, r, &format!("{p}.ssm_norm.weight"), device)?,
            out_proj: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ssm_out.weight"),
                device,
                mm,
            )?)?,
            head_k_dim,
            head_v_dim,
            n_k_heads: cfg.n_k_heads,
            n_v_heads: cfg.n_v_heads,
            conv_dim,
            conv_kernel: cfg.conv_kernel,
            rms_eps: cfg.rms_eps,
            device: device.clone(),
            conv_state: None,
            ssm_state: None,
            cpu_state: None,
        })
    }
    /// Input projections: (qkv[..,conv_dim], z[..,n_v.v_dim], alpha[..,n_v],
    /// beta[..,n_v]) in the activation dtype. Fused = one matmul + slices
    /// (contiguous views at decode seq==1; copies only on the prefill path).
    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let zn = self.n_v_heads * self.head_v_dim;
        let nv = self.n_v_heads;
        match &self.proj {
            DeltaProj::Fused(w) => {
                let full = w.forward(x)?;
                // At decode (seq==1) the narrows are contiguous VIEWS (offset
                // only, zero copies) - the moe_cuda FFI consumers are
                // start_offset-aware. At prefill (seq>1) they are strided and
                // contiguous() materialises them for the downstream reshapes.
                let qkv = full.narrow(D::Minus1, 0, self.conv_dim)?.contiguous()?;
                let z = full.narrow(D::Minus1, self.conv_dim, zn)?.contiguous()?;
                let a = full
                    .narrow(D::Minus1, self.conv_dim + zn, nv)?
                    .contiguous()?;
                let b = full
                    .narrow(D::Minus1, self.conv_dim + zn + nv, nv)?
                    .contiguous()?;
                Ok((qkv, z, a, b))
            }
            DeltaProj::Split {
                in_qkv,
                z_proj,
                a_proj,
                b_proj,
            } => Ok((
                in_qkv.forward(x)?,
                z_proj.forward(x)?,
                a_proj.forward(x)?,
                b_proj.forward(x)?,
            )),
            DeltaProj::Ba {
                in_qkv,
                z_proj,
                ba_proj,
            } => {
                // ssm_ba rows = [b(beta, n_v) | a(alpha/dt, n_v)] -> same (a, b) order
                // the recurrence expects (alpha then beta).
                let ba = ba_proj.forward(x)?;
                let b = ba.narrow(D::Minus1, 0, nv)?.contiguous()?;
                let a = ba.narrow(D::Minus1, nv, nv)?.contiguous()?;
                Ok((in_qkv.forward(x)?, z_proj.forward(x)?, a, b))
            }
        }
    }

    fn reset_state(&mut self, b: usize) -> Result<()> {
        self.conv_state = Some(Tensor::zeros_on(
            (b, self.conv_dim, self.conv_kernel - 1),
            DType::F32,
            &self.device,
        )?);
        self.ssm_state = Some(Tensor::zeros_on(
            (b, self.n_v_heads, self.head_v_dim, self.head_k_dim),
            DType::F32,
            &self.device,
        )?);
        Ok(())
    }

    /// Deep-copied recurrent state (conv window + SSM matrix, + the host-side CPU
    /// recurrence state) for the prefix cache. `affine(1,0)` forces fresh
    /// storage so a later in-place decode step can't mutate the snapshot.
    fn snapshot(&self) -> Result<LayerStateSnap> {
        let dc = |o: &Option<Tensor>| -> Result<Option<Tensor>> {
            Ok(match o {
                Some(t) => Some(t.affine(1.0, 0.0)?),
                None => None,
            })
        };
        Ok(LayerStateSnap::Delta {
            conv: dc(&self.conv_state)?,
            ssm: dc(&self.ssm_state)?,
            cpu: self.cpu_state.clone(),
        })
    }
    fn restore(&mut self, s: &LayerStateSnap) -> Result<()> {
        if let LayerStateSnap::Delta { conv, ssm, cpu } = s {
            let dc = |o: &Option<Tensor>| -> Result<Option<Tensor>> {
                Ok(match o {
                    Some(t) => Some(t.affine(1.0, 0.0)?),
                    None => None,
                })
            };
            self.conv_state = dc(conv)?;
            self.ssm_state = dc(ssm)?;
            self.cpu_state = cpu.clone();
        }
        Ok(())
    }

    pub fn forward(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if input_pos == 0 || self.conv_state.is_none() {
            self.reset_state(b)?;
        }
        // Fused single-launch gated-DeltaNet recurrence (≈40 tensor ops/token ->
        // one CUDA kernel/layer; same approach ollama/llama.cpp use). Bit-exact
        // to the per-token reference on prefill and decode. Used on CUDA for the head dims
        // the kernel templates; the per-token loop is the CPU/fallback path.
        #[cfg(feature = "cuda")]
        if b == 1
            && x.device().is_cuda()
            && self.head_k_dim == self.head_v_dim
            && matches!(self.head_v_dim, 64 | 128 | 256)
        {
            return self.forward_fused(x, seq);
        }
        // CPU: fused host-side recurrence (no per-op the tensor-op dispatch).
        if b == 1 && !x.device().is_cuda() {
            if input_pos == 0 {
                self.cpu_state = None;
            }
            return self.forward_cpu_fused(x, seq);
        }
        self.forward_pertoken(x, b, seq)
    }

    /// Fused CPU path: ONE batched projection matmul, then a plain-Rust f32
    /// loop for conv + gates + gated-delta recurrence + z-gated RMSNorm
    /// (rayon across the n_v heads), then ONE out_proj matmul. Same math and
    /// op order as `forward_pertoken` (the reference), without its ~30
    /// single-threaded the tensor-op dispatches per token per layer.
    fn forward_cpu_fused(&mut self, x: &Tensor, seq: usize) -> Result<Tensor> {
        use rayon::prelude::*;
        let (kg, vg) = (self.n_k_heads, self.n_v_heads);
        let (kd, vd) = (self.head_k_dim, self.head_v_dim);
        let qk_sz = kd * kg;
        let (ck, c_dim) = (self.conv_kernel, self.conv_dim);
        let scale = 1.0f32 / (kd as f32).sqrt();
        let rms_eps = self.rms_eps as f32;

        if self.cpu_state.is_none() {
            self.cpu_state = Some(CpuDeltaState {
                conv: vec![0f32; c_dim * (ck - 1)],
                ssm: vec![0f32; vg * vd * kd],
                w: self
                    .conv_w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                a_log: self.a_log.to_vec1::<f32>()?,
                dt: self.dt_bias.to_vec1::<f32>()?,
                norm: self
                    .norm_w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            });
        }

        // One projection pass over the whole sequence.
        let (qkv_t, z_t, a_t, b_t) = self.project(x)?;
        let qkv = qkv_t
            .to_dtype(DType::F32)?
            .reshape((seq, c_dim))?
            .to_vec2::<f32>()?;
        let zv = z_t
            .to_dtype(DType::F32)?
            .reshape((seq, vg * vd))?
            .to_vec2::<f32>()?;
        let av = a_t
            .to_dtype(DType::F32)?
            .reshape((seq, vg))?
            .to_vec2::<f32>()?;
        let bv = b_t
            .to_dtype(DType::F32)?
            .reshape((seq, vg))?
            .to_vec2::<f32>()?;

        let st = self.cpu_state.as_mut().unwrap();
        let mut y = vec![0f32; seq * vg * vd];
        let mut conv_out = vec![0f32; c_dim];
        let mut qn = vec![0f32; kg * kd];
        let mut kn = vec![0f32; kg * kd];
        let silu = |v: f32| v / (1.0 + (-v).exp());
        for t in 0..seq {
            // causal depthwise conv1d (window = state ++ current) + silu,
            // then roll the state left by one.
            let xt = &qkv[t];
            for c in 0..c_dim {
                let w = &st.w[c * ck..(c + 1) * ck];
                let s = &st.conv[c * (ck - 1)..(c + 1) * (ck - 1)];
                let mut acc = w[ck - 1] * xt[c];
                for j in 0..ck - 1 {
                    acc += w[j] * s[j];
                }
                conv_out[c] = silu(acc);
            }
            for c in 0..c_dim {
                let s = &mut st.conv[c * (ck - 1)..(c + 1) * (ck - 1)];
                s.rotate_left(1);
                s[ck - 2] = xt[c];
            }
            // l2norm q,k per k-head (q also scaled by 1/sqrt(kd))
            for g in 0..kg {
                let q = &conv_out[g * kd..(g + 1) * kd];
                let k = &conv_out[qk_sz + g * kd..qk_sz + (g + 1) * kd];
                let qs = (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
                let ks = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
                for d in 0..kd {
                    qn[g * kd + d] = q[d] / qs * scale;
                    kn[g * kd + d] = k[d] / ks;
                }
            }
            let v_all = &conv_out[2 * qk_sz..2 * qk_sz + vg * vd];
            let (zr, ar, br) = (&zv[t], &av[t], &bv[t]);
            let yt = &mut y[t * vg * vd..(t + 1) * vg * vd];
            let (a_log, dt, norm) = (&st.a_log, &st.dt, &st.norm);
            let (qn_r, kn_r) = (&qn, &kn);
            // recurrence per v-head (tiled GQA: v-head h reads k-head h % kg
            // - matches the reference broadcast (b, rep, kg, .) reshape)
            st.ssm
                .par_chunks_mut(vd * kd)
                .zip(yt.par_chunks_mut(vd))
                .enumerate()
                .for_each(|(h, (s, yh))| {
                    let j = h % kg;
                    let (q, k) = (&qn_r[j * kd..(j + 1) * kd], &kn_r[j * kd..(j + 1) * kd]);
                    let v = &v_all[h * vd..(h + 1) * vd];
                    let sp = (1.0 + (ar[h] + dt[h]).exp()).ln();
                    let g = (a_log[h] * sp).exp();
                    let beta = 1.0 / (1.0 + (-br[h]).exp());
                    let mut sumsq = 0f32;
                    let mut ovec = vec![0f32; vd];
                    for d in 0..vd {
                        let row = &mut s[d * kd..(d + 1) * kd];
                        let mut kv = 0f32;
                        for c in 0..kd {
                            row[c] *= g;
                            kv += row[c] * k[c];
                        }
                        let delta = (v[d] - kv) * beta;
                        let mut od = 0f32;
                        for c in 0..kd {
                            row[c] += delta * k[c];
                            od += row[c] * q[c];
                        }
                        ovec[d] = od;
                        sumsq += od * od;
                    }
                    // z-gated RMSNorm over v_head_dim
                    let inv = 1.0 / (sumsq / vd as f32 + rms_eps).sqrt();
                    for d in 0..vd {
                        let on = ovec[d] * inv * norm[d];
                        let z = zr[h * vd + d];
                        yh[d] = on * (z / (1.0 + (-z).exp()));
                    }
                });
        }
        let y = Tensor::from_vec(y, (1, seq, vg * vd), &x.device())?.to_dtype(x.dtype())?;
        self.out_proj.forward(&y)
    }

    fn forward_pertoken(&mut self, x: &Tensor, b: usize, seq: usize) -> Result<Tensor> {
        let kg = self.n_k_heads;
        let vg = self.n_v_heads;
        let kd = self.head_k_dim;
        let vd = self.head_v_dim;
        let rep = vg / kg; // GQA repeat factor (q,k heads -> v heads)
        let qk_sz = kd * kg;
        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let xt = x.i((.., t, ..))?; // [b, d_model]
            let (qkv_t, z_t, a_t, b_t) = self.project(&xt)?;
            let qkv = qkv_t.to_dtype(DType::F32)?; // [b, conv_dim]
                                                   // causal depthwise conv over conv_dim channels (kernel) + silu
            let cs = self.conv_state.as_ref().unwrap();
            let win = Tensor::cat(&[cs, &qkv.unsqueeze(qkv.rank())?], D::Minus1)?; // [b, conv_dim, kernel]
            let mut acc = self
                .conv_w
                .i((.., 0))?
                .broadcast_mul(&win.i((.., .., 0))?)?;
            for kk in 1..self.conv_kernel {
                acc = (acc
                    + self
                        .conv_w
                        .i((.., kk))?
                        .broadcast_mul(&win.i((.., .., kk))?)?)?;
            }
            self.conv_state = Some(
                win.narrow(D::Minus1, 1, self.conv_kernel - 1)?
                    .contiguous()?,
            );
            let conv = crate::tensor::ops::silu(&acc)?; // [b, conv_dim]
                                                        // split q,k,v
            let q = conv.narrow(D::Minus1, 0, qk_sz)?.reshape((b, kg, kd))?;
            let k = conv.narrow(D::Minus1, qk_sz, qk_sz)?.reshape((b, kg, kd))?;
            let v = conv
                .narrow(D::Minus1, 2 * qk_sz, vd * vg)?
                .reshape((b, vg, vd))?;
            // L2-norm q,k per head; scale q by 1/sqrt(kd)
            let q = (l2norm(&q, 1e-6)? * (1.0 / (kd as f64).sqrt()))?;
            let k = l2norm(&k, 1e-6)?;
            // GQA: repeat q,k heads to vg. qwen35 has ssm.v_head_reordered=true ->
            // TILED repeat (v-head h ← k-head h%kg), NOT interleave (h/rep).
            // (ollama Repeat4D(.,numVHeads,.) tiles the k-head block.)
            let q = q
                .unsqueeze(1)?
                .broadcast_as((b, rep, kg, kd))?
                .reshape((b, vg, kd))?;
            let k = k
                .unsqueeze(1)?
                .broadcast_as((b, rep, kg, kd))?
                .reshape((b, vg, kd))?;
            // gates: g = -exp(a_log) * softplus(alpha + dt) ; decay = exp(g) ; beta = sigmoid(b_proj)
            let alpha = a_t.to_dtype(DType::F32)?; // [b, n_v]
                                                   // softplus(alpha + dt_bias). The fused launch replaces broadcast_add + exp +
                                                   // add + log, and - more than the three dispatches - it keeps the tensor ON THE
                                                   // DEVICE: `log` has no device path, so the chain below bounces to the host and
                                                   // everything computed after it stays there, once per SSM layer per step. Same
                                                   // kernel and same fallback nemotron-h already uses; bit-identical.
            let sp = match crate::inference::kernel::fused::softplus_bias(&alpha, &self.dt_bias)? {
                Some(y) => y,
                None => ((alpha.broadcast_add(&self.dt_bias)?.exp()? + 1.0)?).log()?,
            };
            // g = ssm_a . softplus(alpha+dt) ; decay = exp(g) in (0,1).
            // GGUF `ssm_a` is ALREADY -exp(A_log) (all-negative, like nemotron)  - 
            // multiply directly, do NOT apply -exp() again.
            let gdecay = self.a_log.broadcast_mul(&sp)?.exp()?; // [b, n_v]
            let beta = crate::tensor::ops::sigmoid(&b_t.to_dtype(DType::F32)?)?; // [b, n_v]
                                                                                 // recurrent gated delta rule
            let s = self.ssm_state.as_ref().unwrap(); // [b, vg, vd, kd]
            let decay = gdecay.reshape((b, vg, 1, 1))?.broadcast_as(s.shape())?;
            let s = (s * decay)?;
            // kv_mem[v] = sum_k S[v,k]*k[k]
            let k_b = k.reshape((b, vg, 1, kd))?;
            let kv_mem = (&s * k_b.broadcast_as(s.shape())?)?.sum(D::Minus1)?; // [b,vg,vd]
            let delta = ((&v - kv_mem)? * beta.reshape((b, vg, 1))?.broadcast_as((b, vg, vd))?)?; // [b,vg,vd]
                                                                                                  // S[v,k] += delta[v] * k[k]
            let outer = delta
                .reshape((b, vg, vd, 1))?
                .broadcast_mul(&k.reshape((b, vg, 1, kd))?)?;
            let s = (s + outer)?;
            // o[v] = sum_k S[v,k]*q[k]
            let q_b = q.reshape((b, vg, 1, kd))?;
            let o = (&s * q_b.broadcast_as(s.shape())?)?.sum(D::Minus1)?; // [b,vg,vd]
            self.ssm_state = Some(s);
            // z-gated RMSNorm over v_head_dim, then flatten. The norm is the ordinary one -
            // mean of squares over the last axis, epsilon under the root, weight after - so
            // it is `RmsNorm`; the z gate that follows is this operator's own.
            let on = RmsNorm::new(self.norm_w.clone(), self.rms_eps as f32).forward(&o)?; // [b,vg,vd]
            let z = z_t.to_dtype(DType::F32)?.reshape((b, vg, vd))?;
            let y = (on * crate::tensor::ops::silu(&z)?)?.reshape((b, vg * vd))?;
            let out = self.out_proj.forward(&y.to_dtype(x.dtype())?)?; // [b, d_model]
            outs.push(out.unsqueeze(1)?);
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)
    }

    /// CUDA fast path: batch every non-recurrent step over the sequence, then a
    /// SINGLE fused `gated_delta_net` kernel call for the recurrence (vs the
    /// ~40 ops/token in `forward`'s loop). Bit-equivalent math; only the FP
    /// reduction order of `kv`/`attn` differs (warp-reduce vs a tensor-level sum).
    #[cfg(feature = "cuda")]
    fn forward_fused(&mut self, x: &Tensor, seq: usize) -> Result<Tensor> {
        let b = 1;
        let (kg, vg) = (self.n_k_heads, self.n_v_heads);
        let (kd, vd) = (self.head_k_dim, self.head_v_dim);
        let rep = vg / kg;
        let qk_sz = kd * kg;
        // 0. ALL input projections in one matmul (DeltaProj::Fused: one
        // activation-quantize + one mmvq instead of four of each).
        let (qkv_p, z_p, a_p, b_p) = self.project(x)?;
        // 1. in_qkv (batched) + fused causal depthwise conv1d + silu (ONE kernel,
        // replacing ~16 tensor ops/layer; also drops the two transposes since the
        // kernel works directly on [seq, conv_dim]).
        // F16 in_qkv output fed straight to the F16-input conv kernel (no ->F32 cast).
        let qkv2_f16 = qkv_p.reshape((seq, self.conv_dim))?.contiguous()?; // [seq, C] F16
        let cs2 = self
            .conv_state
            .as_ref()
            .unwrap()
            .reshape((self.conv_dim, self.conv_kernel - 1))?
            .contiguous()?; // [C, K-1]
        let (conv2, new_cs) = match crate::inference::moe_cuda::fused_conv_silu_f16in(
            &qkv2_f16,
            &cs2,
            &self.conv_w,
            self.conv_kernel,
        )? {
            Some(out) => out,
            None => crate::inference::moe_cuda::fused_conv_silu(
                &qkv2_f16.to_dtype(DType::F32)?,
                &cs2,
                &self.conv_w,
                self.conv_kernel,
            )?,
        }; // [seq, C], [C, K-1]
        self.conv_state = Some(new_cs.reshape((1, self.conv_dim, self.conv_kernel - 1))?);
        let conv = conv2.reshape((b, seq, self.conv_dim))?; // [1, seq, conv_dim]
                                                            // 2. split q,k,v (.contiguous(): the conv narrows carry non-zero offsets
                                                            // for k/v at seq==1 and are non-contiguous at seq>1).
        let q = conv
            .narrow(D::Minus1, 0, qk_sz)?
            .reshape((seq, kg, kd))?
            .contiguous()?;
        let k = conv
            .narrow(D::Minus1, qk_sz, qk_sz)?
            .reshape((seq, kg, kd))?
            .contiguous()?;
        let vf = conv
            .narrow(D::Minus1, 2 * qk_sz, vd * vg)?
            .reshape((seq, vg, vd))?
            .contiguous()?;
        // 3+4. fused l2norm + GQA-tile (kg -> vg) for q,k -> [seq, vg, kd] (ONE kernel each)
        let qf = crate::inference::moe_cuda::l2norm_gqa(&q, kg, kd, rep, 1e-6)?;
        let kf = crate::inference::moe_cuda::l2norm_gqa(&k, kg, kd, rep, 1e-6)?;
        // 5. fused gating (ONE kernel): g_pre = a_log.softplus(alpha+dt) (PRE-exp;
        // recurrence kernel does expf); beta = sigmoid(b_proj).
        // F16 a_proj/b_proj outputs fed straight to the F16-input gate kernel (no ->F32 cast).
        let alpha_f16 = a_p.reshape((seq, vg))?; // [seq, vg] F16
        let beta_f16 = b_p.reshape((seq, vg))?;
        let (gf, bf) = match crate::inference::moe_cuda::deltanet_gate_f16in(
            &alpha_f16,
            &beta_f16,
            &self.a_log,
            &self.dt_bias,
        )? {
            Some(out) => out,
            None => crate::inference::moe_cuda::deltanet_gate(
                &alpha_f16.to_dtype(DType::F32)?,
                &beta_f16.to_dtype(DType::F32)?,
                &self.a_log,
                &self.dt_bias,
            )?,
        };
        // 6. ONE fused recurrence kernel (qf,kf,vf are [seq,vg,*] contiguous)
        let state = self
            .ssm_state
            .as_ref()
            .unwrap()
            .reshape((vg, vd, kd))?
            .contiguous()?;
        let scale = (1.0 / (kd as f64).sqrt()) as f32;
        let (o, new_state) =
            crate::inference::moe_cuda::gated_delta_net(&qf, &kf, &vf, &gf, &bf, &state, scale)?;
        self.ssm_state = Some(new_state.reshape((b, vg, vd, kd))?);
        // 7. fused z-gated RMSNorm (one kernel: rmsnorm(o).norm_w.silu(z)) + out_proj
        let o2 = o.reshape((seq * vg, vd))?; // [N, D] F32
                                             // F16 z_proj output fed to the F16-I/O zgate kernel -> y in F16 (no z->F32
                                             // / y->F16 casts; bit-identical). out_proj then takes F16 directly.
        let z_f16 = z_p.reshape((seq * vg, vd))?; // F16
        let y = match crate::inference::moe_cuda::zgate_rmsnorm_f16io(
            &o2,
            &z_f16,
            &self.norm_w,
            self.rms_eps as f32,
        )? {
            Some(y) => y.reshape((b, seq, vg * vd))?,
            None => {
                let z = z_f16.to_dtype(DType::F32)?;
                crate::inference::moe_cuda::zgate_rmsnorm(
                    &o2,
                    &z,
                    &self.norm_w,
                    self.rms_eps as f32,
                )?
                .reshape((b, seq, vg * vd))?
                .to_dtype(x.dtype())?
            }
        };
        self.out_proj.forward(&y)
    }
}

/// Shared-expert gate+up projections - same input, so fused row-wise into one
/// weight when the GGUF dtypes allow (same trick as [`DeltaProj`]).
enum ShexpGateUp {
    /// rows = [gate(n) | up(n)]
    Fused {
        w: QMatMul,
        n: usize,
    },
    Split {
        gate: QMatMul,
        up: QMatMul,
    },
}

/// MoE: softmax router (top-k, normalized) + SwiGLU experts + scalar-gated shared expert.
pub struct Qwen35Moe {
    /// router logits [n_expert] ‖ shared-expert scalar gate [1], one F32 gemv
    /// (rows = n_expert + 1) instead of two per layer.
    router: crate::tensor::layer::Linear,
    n_expert: usize,
    gate_exps: Arc<QTensor>,
    up_exps: Arc<QTensor>,
    down_exps: Arc<QTensor>,
    gateup_shexp: ShexpGateUp,
    down_shexp: QMatMul,
    n_expert_used: usize,
    dtype: DType,
}
impl Qwen35Moe {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        cfg: &Qwen35Config,
        dtype: DType,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let gate_w = ld_f32(c, r, &format!("{p}.ffn_gate_inp.weight"), device)?;
        let shg_w = ld_f32(c, r, &format!("{p}.ffn_gate_inp_shexp.weight"), device)?;
        let n_expert = gate_w.dims()[0];
        // The router runs as one F32 gemv (see `forward`). The two source weights
        // can carry different on-disk dtypes (this GGUF: ffn_gate_inp = F32,
        // ffn_gate_inp_shexp = F16); on CPU `dequantize` keeps F16 blocks in F16
        // while F32 blocks land in F32, so the concat would mix dtypes. Coerce
        // both to F32 (= what the GPU path already produces) before the cat.
        let gate_w = gate_w.to_dtype(DType::F32)?;
        let shg_w = shg_w.to_dtype(DType::F32)?;
        let router_w = Tensor::cat(&[&gate_w, &shg_w.reshape((1, shg_w.elem_count()))?], 0)?;
        let shexp_names = [
            format!("{p}.ffn_gate_shexp.weight"),
            format!("{p}.ffn_up_shexp.weight"),
        ];
        let gateup_shexp = match load_fused_proj(c, r, &shexp_names, device)? {
            Some(w) => {
                let n = c.tensor_infos[&shexp_names[0]].shape.dims()[0];
                ShexpGateUp::Fused { w, n }
            }
            None => ShexpGateUp::Split {
                gate: QMatMul::from_qtensor(load_q8(c, r, &shexp_names[0], device, mm)?)?,
                up: QMatMul::from_qtensor(load_q8(c, r, &shexp_names[1], device, mm)?)?,
            },
        };
        Ok(Self {
            router: crate::tensor::layer::Linear::new(router_w, None)?,
            n_expert,
            gate_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_gate_exps.weight"),
                device,
                mm,
            )?),
            up_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_up_exps.weight"),
                device,
                mm,
            )?),
            down_exps: Arc::new(load_q8(
                c,
                r,
                &format!("{p}.ffn_down_exps.weight"),
                device,
                mm,
            )?),
            gateup_shexp,
            down_shexp: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_down_shexp.weight"),
                device,
                mm,
            )?)?,
            n_expert_used: cfg.n_expert_used,
            dtype,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use crate::inference::moe_cuda as moe;
        let (b, seq, hidden) = x.dims3()?;
        let xs = x
            .reshape((x.elem_count() / hidden, hidden))?
            .to_dtype(DType::F32)?;
        let n_tokens = xs.dim(0)?;
        // On the CUDA binary run with `--cpu` the moe_cuda expert/router kernels
        // reject CPU tensors - branch to the moe_cpu twins (nemotron_h/lfm2_moe
        // pattern). Without this qwen3.5:35b CPU returned 0 tokens (500 error).
        let on_cuda = xs.device().is_cuda();
        // router logits + shared-expert scalar gate in one gemv ([n, n_expert+1]).
        let router_out = self.router.forward(&xs)?;
        let logits = router_out
            .narrow(D::Minus1, 0, self.n_expert)?
            .contiguous()?;
        let gate_logit = router_out
            .narrow(D::Minus1, self.n_expert, 1)?
            .contiguous()?; // [n,1] pre-sigmoid
                            // softmax router, top-k, normalized
        let (topk_w, topk_ids) = if on_cuda {
            moe::topk_softmax(&logits, self.n_expert_used, true)?
        } else {
            crate::inference::moe_cpu::topk_softmax(&logits, self.n_expert_used, true)?
        };
        let topk_flat = topk_ids.flatten_all()?;
        // Decode: one-warp argsort (bit-identical to sort_last_dim); tensor-op fallback for prefill.
        let (expert_ids, sorted_token_ids) =
            match crate::inference::moe_cuda::argsort_small_u32(&topk_flat)? {
                Some(out) => out,
                None => topk_flat.sort_last_dim(true)?,
            };
        // Decode (seq==1): ONE fused kernel does gate GEMM + up GEMM + silu.mul
        // (quantizes the input to q8_1 once). Prefill keeps the wmma 2-GEMM path.
        // CPU branches to the moe_cpu twins (see on_cuda above).
        let h = if seq == 1 {
            if on_cuda {
                moe::moe_gemm_gguf_gate_up_silu_mul(
                    &xs,
                    &self.gate_exps,
                    &self.up_exps,
                    &sorted_token_ids,
                    &expert_ids,
                    self.n_expert_used,
                )?
            } else {
                crate::inference::moe_cpu::moe_gemm_gguf_gate_up_silu_mul(
                    &xs,
                    &self.gate_exps,
                    &self.up_exps,
                    &sorted_token_ids,
                    &expert_ids,
                    self.n_expert_used,
                )?
            }
        } else {
            let (gate, up) = if on_cuda {
                (
                    crate::inference::moe_cuda::moe_gemm_gguf(
                        &xs,
                        &self.gate_exps,
                        &None,
                        &sorted_token_ids,
                        &expert_ids,
                        self.n_expert_used,
                        true,
                        self.dtype,
                    )?,
                    crate::inference::moe_cuda::moe_gemm_gguf(
                        &xs,
                        &self.up_exps,
                        &None,
                        &sorted_token_ids,
                        &expert_ids,
                        self.n_expert_used,
                        true,
                        self.dtype,
                    )?,
                )
            } else {
                (
                    crate::inference::moe_cpu::moe_gemm_gguf(
                        &xs,
                        &self.gate_exps,
                        &None,
                        &sorted_token_ids,
                        &expert_ids,
                        self.n_expert_used,
                        true,
                        self.dtype,
                    )?,
                    crate::inference::moe_cpu::moe_gemm_gguf(
                        &xs,
                        &self.up_exps,
                        &None,
                        &sorted_token_ids,
                        &expert_ids,
                        self.n_expert_used,
                        true,
                        self.dtype,
                    )?,
                )
            };
            (crate::tensor::ops::silu(&gate)? * up)?
        };
        let routed = if on_cuda {
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                None,
                None,
            )?
        } else {
            crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
                &h,
                &self.down_exps,
                &sorted_token_ids,
                &expert_ids,
                &topk_w,
                self.n_expert_used,
                n_tokens,
                None,
                None,
            )?
        };
        // shared expert: SwiGLU * sigmoid(scalar_gate). silu(g).u via one F16
        // launch (F32-internal -> bit-identical to silu(g.f32).u.f32 -> f16), saving
        // cast+silu+mul+cast dispatches/layer (qwen3.5 is CPU-dispatch-bound).
        // xt: reuse the layer input directly when it is already in the weight
        // dtype (F16->F32->F16 is an exact roundtrip) - saves a cast launch/layer.
        let xt = if x.dtype() == self.dtype && x.is_contiguous() {
            x.reshape((x.elem_count() / hidden, hidden))?
        } else {
            xs.to_dtype(self.dtype)?
        };
        // gate+up fused into one matmul; the halves are contiguous views at
        // decode (silu_mul_f16 is start_offset-aware).
        let (g16, u16) = match &self.gateup_shexp {
            ShexpGateUp::Fused { w, n } => {
                let gu = w.forward(&xt)?;
                (
                    gu.narrow(D::Minus1, 0, *n)?.contiguous()?,
                    gu.narrow(D::Minus1, *n, *n)?.contiguous()?,
                )
            }
            ShexpGateUp::Split { gate, up } => (gate.forward(&xt)?, up.forward(&xt)?),
        };
        let gu = match crate::inference::kernel::fused::silu_mul_f16(&g16, &u16)? {
            Some(y) => y,
            None => {
                let g = crate::tensor::ops::silu(&g16.to_dtype(DType::F32)?)?;
                (g * u16.to_dtype(DType::F32)?)?.to_dtype(self.dtype)?
            }
        };
        // Fused output epilogue: out = routed + f32(down).sigmoid(gate_logit), one
        // launch (sigmoid + f16->f32 cast + per-token broadcast-mul + add). down/gate
        // stay un-cast/un-activated; the kernel does it all in F32 -> bit-identical.
        let down = self.down_shexp.forward(&gu)?; // F16
        let out =
            match crate::inference::kernel::fused::fused_shexp_out(&routed, &down, &gate_logit)? {
                Some(o) => o,
                None => {
                    let sh = down
                        .to_dtype(DType::F32)?
                        .broadcast_mul(&crate::tensor::ops::sigmoid(&gate_logit)?)?;
                    (routed + sh)?
                }
            };
        out.reshape((b, seq, hidden))?.to_dtype(x.dtype())
    }
}

/// Dense SwiGLU feed-forward, for the members of this family that ship one
/// expert per layer instead of a routed set.
///
/// It is structurally the MoE block's SHARED expert - the same fused gate/up
/// projection, the same fused activation, the same down projection - with the
/// dense tensor names and no router, so both go through one implementation. The
/// dense variants cannot fall back to the generic transformer: this family's
/// block is a hybrid whose DeltaNet layers the generic path cannot run, and the
/// FFN is the only part that differs between the two.
pub struct Qwen35Dense {
    gateup: ShexpGateUp,
    down: QMatMul,
    dtype: DType,
}
impl Qwen35Dense {
    pub fn load<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        layer: usize,
        dtype: DType,
        device: &Device,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let p = format!("blk.{layer}");
        let names = [format!("{p}.ffn_gate.weight"), format!("{p}.ffn_up.weight")];
        let gateup = match load_fused_proj(c, r, &names, device)? {
            Some(w) => {
                let n = c.tensor_infos[&names[0]].shape.dims()[0];
                ShexpGateUp::Fused { w, n }
            }
            None => ShexpGateUp::Split {
                gate: QMatMul::from_qtensor(load_q8(c, r, &names[0], device, mm)?)?,
                up: QMatMul::from_qtensor(load_q8(c, r, &names[1], device, mm)?)?,
            },
        };
        Ok(Self {
            gateup,
            down: QMatMul::from_qtensor(load_q8(
                c,
                r,
                &format!("{p}.ffn_down.weight"),
                device,
                mm,
            )?)?,
            dtype,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let rows = x.elem_count() / hidden;
        // Reuse the input directly when it already carries the weight dtype: the
        // F16 roundtrip is exact, and this saves a cast launch per layer.
        let xt = if x.dtype() == self.dtype && x.is_contiguous() {
            x.reshape((rows, hidden))?
        } else {
            x.reshape((rows, hidden))?.to_dtype(self.dtype)?
        };
        let (g16, u16) = match &self.gateup {
            ShexpGateUp::Fused { w, n } => {
                let gu = w.forward(&xt)?;
                (
                    gu.narrow(D::Minus1, 0, *n)?.contiguous()?,
                    gu.narrow(D::Minus1, *n, *n)?.contiguous()?,
                )
            }
            ShexpGateUp::Split { gate, up } => (gate.forward(&xt)?, up.forward(&xt)?),
        };
        let gu = match crate::inference::kernel::fused::silu_mul_f16(&g16, &u16)? {
            Some(y) => y,
            None => {
                let g = crate::tensor::ops::silu(&g16.to_dtype(DType::F32)?)?;
                (g * u16.to_dtype(DType::F32)?)?.to_dtype(self.dtype)?
            }
        };
        self.down
            .forward(&gu)?
            .reshape((b, seq, hidden))?
            .to_dtype(x.dtype())
    }
}

/// The layer's feed-forward, routed or dense. Which one a file carries is read
/// from the file itself - the router tensor's presence - never from the tag.
enum Ffn {
    Moe(Qwen35Moe),
    Dense(Qwen35Dense),
}
impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Moe(m) => m.forward(x),
            Ffn::Dense(d) => d.forward(x),
        }
    }
}

enum Op {
    Delta(Qwen35DeltaNet),
    Attn(Qwen35Attn),
}

/// Per-layer deep-copied state captured at a prompt boundary for the recurrent
/// prefix cache. DeltaNet layers can't trim their recurrent state to a
/// prefix the way attention KV can, so the only safe reuse is to snapshot the
/// full state at the boundary and restore it on an exact-prefix match.
enum LayerStateSnap {
    Delta {
        conv: Option<Tensor>,
        ssm: Option<Tensor>,
        cpu: Option<CpuDeltaState>,
    },
    Attn(crate::tensor::KvCache),
}
/// One resident prefix snapshot (in-memory only, never persisted - privacy).
/// `logits` = the prefill's last-position logits, so an exact-prefix hit can seed
/// the FIRST decode token without any re-forward (state is already at prompt_len).
struct Qwen35PrefixCache {
    prompt: Vec<u32>,
    layers: Vec<LayerStateSnap>,
    logits: Tensor,
}
struct Layer {
    attn_norm: crate::tensor::layer::RmsNorm,
    post_norm: crate::tensor::layer::RmsNorm,
    op: Op,
    ffn: Ffn,
    device: Device,
    rms_eps: f64,
}

/// Pre-norm fusing cast-F32 + rms_norm + cast-F16 (3 launches) into one kernel.
/// qwen3.5 is launch-bound (nsys: 21% GPU util); cutting ~192 norm launches/token
/// (2 norms x ~48 layers). Falls back to the tensor-op cast path off-CUDA / non-F16.
fn norm_fwd(norm: &crate::tensor::layer::RmsNorm, x: &Tensor, eps: f64) -> Result<Tensor> {
    if x.dtype() == DType::F16 && x.device().is_cuda() {
        return crate::inference::kernel::fused::fused_rmsnorm_f16(x, norm.weight(), eps as f32);
    }
    let st = x.dtype();
    norm.forward(&x.to_dtype(DType::F32)?)?.to_dtype(st)
}

pub struct Qwen35MoeModel {
    embed: Embedding,
    embed_dev: Device,
    layers: Vec<Layer>,
    norm: crate::tensor::layer::RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    rope_dim: usize,
    rope_freq_base: f32,
    /// Vision ViT (present iff the GGUF carries `v.*` tensors). Single-device,
    /// pinned to `embed_dev` (GPU 0).
    vision: Option<Qwen35Vision>,
    image_token: u32,
    /// In-memory recurrent prefix snapshot (env-gated). Lets an identical
    /// later prompt skip re-prefill. Never persisted to disk (privacy).
    prefix_cache: Option<Qwen35PrefixCache>,
}

impl Qwen35MoeModel {
    /// Snapshot the full per-layer state (recurrent + KV) as of position
    /// `prompt.len()`, with the prefill's last-position `logits`, for later
    /// exact-prefix reuse. Call AFTER prefilling [0, prompt.len()).
    pub fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> Result<()> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(match &l.op {
                Op::Delta(d) => d.snapshot()?,
                Op::Attn(a) => a.snapshot()?,
            });
        }
        self.prefix_cache = Some(Qwen35PrefixCache {
            prompt: prompt.to_vec(),
            layers,
            logits: logits.affine(1.0, 0.0)?,
        });
        Ok(())
    }

    /// If a snapshot exists whose prompt EXACTLY equals `prompt`, restore every
    /// layer's state from it (deep-copied, so the snapshot stays reusable) and
    /// return the prefill logits to seed decode from position `prompt.len()`.
    /// Else None (caller re-prefills). Exact-match only: a recurrent state can't
    /// be truncated to a shorter prefix.
    pub fn try_restore_prefix(&mut self, prompt: &[u32]) -> Result<Option<Tensor>> {
        match &self.prefix_cache {
            Some(pc) if pc.prompt.as_slice() == prompt => {}
            _ => return Ok(None),
        }
        let pc = self.prefix_cache.take().unwrap();
        for (l, snap) in self.layers.iter_mut().zip(pc.layers.iter()) {
            match &mut l.op {
                Op::Delta(d) => d.restore(snap)?,
                Op::Attn(a) => a.restore(snap)?,
            }
        }
        let logits = pc.logits.affine(1.0, 0.0)?;
        self.prefix_cache = Some(pc);
        Ok(Some(logits))
    }
    pub fn from_gguf<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
        gpu_avail: &[u64],
        file_size: u64,
        reserve_bytes: u64,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let cfg = Qwen35Config::from_gguf(content)?;
        let n_dev = devices.len().max(1);
        // VRAM-weighted split: fill the fastest GPU first (vs the old equal-count
        // split that let the slow GPU gate sequential decode). Shared with nemotron.
        // Real per-layer byte sizes (DeltaNet vs MoE blocks differ hugely) so the
        // fast GPU packs to its true budget instead of being under-filled.
        let layer_sizes =
            crate::inference::model::nemotron_h::layer_byte_sizes(content, cfg.n_layers);
        let plan = crate::inference::model::nemotron_h::plan_layer_devices(
            cfg.n_layers,
            n_dev,
            file_size,
            gpu_avail,
            &layer_sizes,
            reserve_bytes,
        );
        let dev_for = |l: usize| -> &Device { &devices[plan[l].min(n_dev - 1)] };
        let max_seq = cfg.context_length.min(32768).max(8192);
        let mut rope = Vec::with_capacity(n_dev);
        for d in devices {
            rope.push(build_rope(cfg.rope_dim, max_seq, cfg.rope_freq_base, d)?);
        }

        let embed_dev = devices[0].clone();
        let embed_t = content
            .tensor(reader, "token_embd.weight", &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_dtype(DType::F16)?
            .to_device(&embed_dev)?;
        let embed = Embedding::new(embed_t);

        let has = |l: usize, s: &str| content.tensor_infos.contains_key(&format!("blk.{l}.{s}"));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let device = dev_for(i);
            let di = plan[i].min(n_dev - 1);
            let (cos, sin) = &rope[di];
            let attn_norm = crate::tensor::layer::RmsNorm::new(
                ld_f32(
                    content,
                    reader,
                    &format!("blk.{i}.attn_norm.weight"),
                    device,
                )?,
                cfg.rms_eps as f32,
            );
            let post_norm = crate::tensor::layer::RmsNorm::new(
                ld_f32(
                    content,
                    reader,
                    &format!("blk.{i}.post_attention_norm.weight"),
                    device,
                )?,
                cfg.rms_eps as f32,
            );
            let op = if has(i, "attn_q.weight") {
                Op::Attn(Qwen35Attn::load(
                    content,
                    reader,
                    i,
                    &cfg,
                    cos.clone(),
                    sin.clone(),
                    device,
                    mm,
                )?)
            } else {
                Op::Delta(Qwen35DeltaNet::load(content, reader, i, &cfg, device, mm)?)
            };
            // Routed or dense is a property of the FILE: the dense members of this
            // family ship no router tensor. Reading it here keeps one loader for
            // the whole family instead of a second port selected by tag.
            let ffn = if content
                .tensor_infos
                .contains_key(&format!("blk.{i}.ffn_gate_inp.weight"))
            {
                Ffn::Moe(Qwen35Moe::load(
                    content, reader, i, &cfg, dtype, device, mm,
                )?)
            } else {
                Ffn::Dense(Qwen35Dense::load(content, reader, i, dtype, device, mm)?)
            };
            if i == 0 || i + 1 == cfg.n_layers {
                tracing::info!(
                    "qwen35moe layer {i} op={} on {:?}",
                    if matches!(op, Op::Attn(_)) {
                        "attn"
                    } else {
                        "deltanet"
                    },
                    device.location()
                );
            }
            layers.push(Layer {
                attn_norm,
                post_norm,
                op,
                ffn,
                device: device.clone(),
                rms_eps: cfg.rms_eps,
            });
        }
        let last = dev_for(cfg.n_layers - 1);
        let norm = crate::tensor::layer::RmsNorm::new(
            ld_f32(content, reader, "output_norm.weight", last)?,
            cfg.rms_eps as f32,
        );
        let lm_head = match load_q8(content, reader, "output.weight", last, mm) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => {
                QMatMul::from_qtensor(load_q8(content, reader, "token_embd.weight", last, mm)?)?
            }
        };
        // Vision ViT - only if the GGUF carries it (multimodal qwen3.5). Pinned
        // to GPU 0 (embed_dev); it's small (~0.5 GB F16) vs the 23 GB text model.
        let vision = if content.tensor_infos.contains_key("v.patch_embed.weight") {
            match Qwen35Vision::from_gguf(content, reader, cfg.d_model, &embed_dev) {
                Ok(v) => {
                    tracing::info!("qwen35moe: vision ViT loaded ({} blocks)", cfg.n_layers);
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!("qwen35moe: vision load failed ({e}); text-only");
                    None
                }
            }
        } else {
            None
        };
        let image_token = content
            .metadata
            .get("qwen35moe.vision.image_token_id")
            .or_else(|| content.metadata.get("qwen35moe.image_token_id"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(248056);

        Ok(Self {
            embed,
            embed_dev,
            layers,
            norm,
            lm_head,
            dtype,
            rope_dim: cfg.rope_dim,
            rope_freq_base: cfg.rope_freq_base,
            vision,
            image_token,
            prefix_cache: None,
        })
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }
    pub fn image_token(&self) -> u32 {
        self.image_token
    }

    /// Full image prefill: encode the image through the ViT, splice its
    /// `[n_merged, 2048]` embeddings over the contiguous `image_token` run in
    /// the text-embedding stream, then run `forward_embeds` with mRoPE-2D
    /// positions. `pixel_values` `[n_patches, 1536]` + patch grid `(gh, gw)`
    /// come from `image_processor::preprocess_qwen35vl`. The prompt must contain
    /// exactly `n_merged = (gh/2)*(gw/2)` consecutive `image_token` placeholders.
    /// Returns `(last_token_logits, next_input_pos)`. For decode, call
    /// `forward(tok, next_input_pos + i)` - the continuing **logical** mRoPE
    /// position (NOT the sequence length; the image compresses positions). The
    /// decode causal mask is skipped (seq==1) so only the rope position matters;
    /// the KV-cache length is tracked separately by the cache.
    pub fn forward_with_image(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        gh: usize,
        gw: usize,
    ) -> Result<(Tensor, usize)> {
        let vit = self.vision.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg("qwen35moe: no vision ViT loaded".to_string())
        })?;
        let img = vit.forward(pixel_values, gh, gw)?.to_dtype(self.dtype)?; // [n_merged, 2048]
        let n_merged = img.dim(0)?;
        let (mh, mw) = (gh / 2, gw / 2); // merged grid (spatial_merge_size=2)

        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let start = ids
            .iter()
            .position(|&t| t == self.image_token)
            .ok_or_else(|| {
                crate::tensor::Error::msg("qwen35moe: image_token not found in prompt".to_string())
            })?;
        let n_ph = ids[start..]
            .iter()
            .take_while(|&&t| t == self.image_token)
            .count();
        if n_ph != n_merged {
            return Err(crate::tensor::Error::msg(format!(
                "qwen35moe: prompt has {n_ph} image placeholders but ViT produced {n_merged} (grid {mh}x{mw})")));
        }

        // text embeds, then splice the ViT rows over the placeholder run
        let embeds = self.embed_tokens(input_ids)?; // [1, seq, d]
        let d = embeds.dim(2)?;
        let img3 = img.to_device(&self.embed_dev)?.reshape((1, n_merged, d))?;
        let pre = embeds.narrow(1, 0, start)?;
        let post_start = start + n_ph;
        let post_len = embeds.dim(1)? - post_start;
        let spliced = if post_len > 0 {
            let post = embeds.narrow(1, post_start, post_len)?;
            Tensor::cat(&[&pre, &img3, &post], 1)?
        } else {
            Tensor::cat(&[&pre, &img3], 1)?
        };

        let pos = mrope_positions(&ids, self.image_token, &[(mh, mw)]);
        // Next decode position = beyond every prefill logical position.
        let next_pos = pos
            .iter()
            .flat_map(|p| p.iter().copied())
            .max()
            .unwrap_or(0) as usize
            + 1;
        let logits = self.forward_embeds(&spliced, &pos)?;
        Ok((logits, next_pos))
    }

    /// Embed token ids WITHOUT running the transformer - used by the engine to
    /// build the text-embedding stream so image embeds can be spliced in before
    /// `forward_embeds`. Returns `[1, seq, d_model]` on `embed_dev`.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let ids = input_ids.to_device(&self.embed_dev)?;
        let mut x = self.embed.forward(&ids)?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }
        Ok(x)
    }

    /// Prefill forward over pre-computed embeddings (text embeds with image
    /// embeds spliced in at the image-token rows) using mRoPE-2D positions
    /// `pos` (`[t,h,w]` per token). Resets all caches on the first chunk,
    /// returns last-token logits `[1, vocab]`. Attention layers rotate with the
    /// per-token mRoPE cos/sin; DeltaNet layers are position-free as usual.
    ///
    /// CHUNKED (>n_ctx overflow class, same remedy as the pixtral splice):
    /// running a long splice (image ≈ 1-4k merged tokens + question) as ONE
    /// monolithic forward materialises `heads x seq x seq x f32` attention
    /// scores -> VRAM OOM at long context. Chunks of `PREFILL_CHUNK_TOKENS`
    /// bound the transient: attention appends into the growable `KvCache`
    /// per chunk (mask offset via `input_pos`), DeltaNet carries its
    /// conv/recurrent state across chunks (resets only at offset 0). The last
    /// chunk's logits are identical to the monolithic result.
    pub fn forward_embeds(&mut self, embeds: &Tensor, pos: &[[i32; 3]]) -> Result<Tensor> {
        const CHUNK: usize = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS;
        let (_b, seq, _) = embeds.dims3()?;
        debug_assert_eq!(seq, pos.len(), "embeds seq must match position count");
        let mut off = 0usize;
        let mut last: Option<Tensor> = None;
        while off < seq {
            let n = CHUNK.min(seq - off);
            let chunk = embeds.narrow(1, off, n)?;
            last = Some(self.forward_embeds_chunk(&chunk, &pos[off..off + n], off)?);
            off += n;
        }
        last.ok_or_else(|| crate::tensor::Error::msg("qwen35: empty spliced prefill".to_string()))
    }

    /// One chunk of the spliced prefill: all layers over `embeds[.., off..off+n]`
    /// with the matching mRoPE position slice, KV/recurrent state continuing
    /// from the previous chunk (`input_pos = off`; caches reset at `off == 0`).
    fn forward_embeds_chunk(
        &mut self,
        embeds: &Tensor,
        pos: &[[i32; 3]],
        input_pos: usize,
    ) -> Result<Tensor> {
        let (_b, seq, _) = embeds.dims3()?;
        let mut x = embeds.to_device(&self.embed_dev)?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }
        let (rope_dim, base) = (self.rope_dim, self.rope_freq_base);
        for layer in self.layers.iter_mut() {
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            let _st = x.dtype();
            let residual = x.clone();
            let h = norm_fwd(&layer.attn_norm, &x, layer.rms_eps)?;
            let h = match &mut layer.op {
                // DeltaNet self-resets its conv/recurrent state when input_pos==0
                // and carries it across later chunks.
                Op::Delta(o) => o.forward(&h, input_pos)?,
                Op::Attn(o) => {
                    if input_pos == 0 {
                        o.reset();
                    }
                    let (cos, sin) = build_mrope_cos_sin(pos, rope_dim, base, &layer.device)?;
                    o.forward_rope(&h, &cos, &sin, input_pos)?
                }
            };
            x = (residual + h)?;
            let residual = x.clone();
            let h = norm_fwd(&layer.post_norm, &x, layer.rms_eps)?;
            let h = layer.ffn.forward(&h)?;
            x = (residual + h)?;
        }
        let x = self.norm.forward(&x.to_dtype(DType::F32)?)?;
        let x = x.i((.., seq - 1, ..))?.to_dtype(self.dtype)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }

    pub fn forward(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        let ids = input_ids.to_device(&self.embed_dev)?;
        let mut x = self.embed.forward(&ids)?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }
        for layer in self.layers.iter_mut() {
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            let _st = x.dtype();
            let residual = x.clone();
            let h = norm_fwd(&layer.attn_norm, &x, layer.rms_eps)?;
            let h = match &mut layer.op {
                Op::Delta(o) => o.forward(&h, input_pos)?,
                Op::Attn(o) => {
                    if input_pos == 0 {
                        o.reset();
                    }
                    o.forward(&h, input_pos)?
                }
            };
            x = (residual + h)?;
            let residual = x.clone();
            let h = norm_fwd(&layer.post_norm, &x, layer.rms_eps)?;
            let t1 = std::time::Instant::now();
            let h = layer.ffn.forward(&h)?;
            x = (residual + h)?;
        }
        let x = self.norm.forward(&x.to_dtype(DType::F32)?)?;
        let x = x.i((.., seq - 1, ..))?.to_dtype(self.dtype)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // mRoPE with t==h==w must be bit-identical to the 1D scalar build_rope  - 
    // this is the invariant that makes the (already-correct) text path a
    // regression guard for the IMROPE interleave rule used by vision.
    #[test]
    fn mrope_text_collapses_to_build_rope() {
        let dev = Device::Cpu;
        let (rd, base, n) = (64usize, 1e7f32, 8usize);
        let (c1, s1) = build_rope(rd, n, base, &dev).unwrap();
        let pos: Vec<[i32; 3]> = (0..n as i32).map(|p| [p, p, p]).collect();
        let (c2, s2) = build_mrope_cos_sin(&pos, rd, base, &dev).unwrap();
        let maxd = |a: Tensor, b: Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max_keepdim(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let dc = maxd(c1, c2);
        let ds = maxd(s1, s2);
        assert!(
            dc < 1e-5 && ds < 1e-5,
            "mrope(t=h=w) != build_rope: cosΔ={dc} sinΔ={ds}"
        );
    }

    // 3D position layout for a [text, text, imgx4 (2x2 merged grid), text]
    // sequence per the ollama qwen3vl rule.
    #[test]
    fn mrope_positions_image_block() {
        let img = 248056u32;
        let ids = vec![10u32, 20, img, img, img, img, 30];
        let pos = mrope_positions(&ids, img, &[(2, 2)]);
        assert_eq!(
            pos,
            vec![
                [0, 0, 0],
                [1, 1, 1], // text
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3], // image 2x2: t=P0, h=P0+i/w, w=P0+i%w
                [4, 4, 4], // text resumes at P0+merged_width(2)
            ]
        );
    }
}
