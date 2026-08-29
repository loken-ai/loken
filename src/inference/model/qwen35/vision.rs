//! qwen35moe (Qwen3-VL) vision encoder - 27-block ViT + patch-embed + learned
//! pos-embed (bilinear-interpolated) + 2D rope + 2x2 spatial-merge merger.
//!
//! Input: pixel_values [n_patches, patch_dim=1536] (from image_processor::
//! preprocess_qwen35vl, 2x2-merge-grouped order) + grid (grid_h, grid_w).
//! Output: image embeddings [n_merged, 2048] (text embedding dim) where
//! n_merged = (grid_h/2).(grid_w/2), spliced into the text stream at the
//! image_token positions.
//!
//! Ported from ollama's `model/models/qwen3vl/model_vision.go` (cos/sin
//! rotate-half 2D rope) + llama.cpp qwen3vl.cpp. patch_embed is effectively a
//! Linear(1536->1152) (conv stride=patch ⇒ per-patch independent). All weights
//! F16 (run F32). NOT YET VALIDATED end-to-end - needs ollama image-embedding
//! comparison once splice+API are wired (steps 4-5).

use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Result, Tensor, D};
use std::io::{Read, Seek};

fn ld<R: Read + Seek>(c: &gguf_file::Content, r: &mut R, name: &str, d: &Device) -> Result<Tensor> {
    c.tensor(r, name, d)?.dequantize(d)?.to_dtype(DType::F32)
}
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub n_layers: usize,      // 27
    pub hidden: usize,        // 1152
    pub n_head: usize,        // 16
    pub head_dim: usize,      // 72
    pub patch_size: usize,    // 16
    pub merge_size: usize,    // 2
    pub eps: f64,             // 1e-6
    pub rope_theta: f32,      // 10000
    pub grid_per_side: usize, // 48 (sqrt of num pos embeddings)
    pub out_dim: usize,       // 2048 (text hidden)
}
impl VisionConfig {
    pub fn from_gguf(ct: &gguf_file::Content, text_hidden: usize) -> Self {
        let g = |k: &str| ct.metadata.get(&format!("qwen35moe.vision.{k}"));
        let u = |k: &str, d: usize| {
            g(k).and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let f = |k: &str, d: f32| g(k).and_then(|v| v.to_f32().ok()).unwrap_or(d);
        let hidden = u("embedding_length", 1152);
        let n_head = u("attention.head_count", 16);
        // num_positional_embeddings not always present -> infer from pos_embed rows at load.
        VisionConfig {
            n_layers: u("block_count", 27),
            hidden,
            n_head,
            head_dim: hidden / n_head.max(1),
            patch_size: u("patch_size", 16),
            merge_size: u("spatial_merge_size", 2),
            eps: f("attention.layer_norm_epsilon", 1e-6) as f64,
            rope_theta: f("rope.freq_base", 10000.0),
            grid_per_side: 48,
            out_dim: text_hidden,
        }
    }
}

struct VBlock {
    q: Tensor,
    qb: Tensor,
    k: Tensor,
    kb: Tensor,
    v: Tensor,
    vb: Tensor,
    o: Tensor,
    ob: Tensor,
    n1w: Tensor,
    n1b: Tensor,
    n2w: Tensor,
    n2b: Tensor,
    fc1: Tensor,
    fc1b: Tensor,
    fc2: Tensor,
    fc2b: Tensor,
}

fn layernorm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)?
        .broadcast_mul(w)?
        .broadcast_add(b)
}
fn lin(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.broadcast_matmul(&w.t()?)?.broadcast_add(b)
}
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

pub struct Qwen35Vision {
    cfg: VisionConfig,
    patch_w: Tensor,   // [hidden, patch_dim]
    patch_b: Tensor,   // [hidden]
    pos_embed: Tensor, // [grid_per_side², hidden]
    blocks: Vec<VBlock>,
    merger_nw: Tensor,
    merger_nb: Tensor,
    merger_fc1: Tensor,
    merger_fc1b: Tensor,
    merger_fc2: Tensor,
    merger_fc2b: Tensor,
    device: Device,
}

impl Qwen35Vision {
    pub fn from_gguf<R: Read + Seek>(
        c: &gguf_file::Content,
        r: &mut R,
        text_hidden: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut cfg = VisionConfig::from_gguf(c, text_hidden);
        // patch_embed weight [16,16,2,3456] = 1152x1536 elems -> Linear(1536->1152).
        let pw = ld(c, r, "v.patch_embed.weight", device)?;
        let patch_dim = pw.elem_count() / cfg.hidden;
        let patch_w = pw.reshape((cfg.hidden, patch_dim))?;
        let patch_b = ld(c, r, "v.patch_embed.bias", device)?;
        let pos_embed = ld(c, r, "v.pos_embed.weight", device)?; // [n_pos, hidden]
        cfg.grid_per_side = (pos_embed.dim(0)? as f64).sqrt().round() as usize;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("v.blk.{i}");
            blocks.push(VBlock {
                q: ld(c, r, &format!("{p}.attn_q.weight"), device)?,
                qb: ld(c, r, &format!("{p}.attn_q.bias"), device)?,
                k: ld(c, r, &format!("{p}.attn_k.weight"), device)?,
                kb: ld(c, r, &format!("{p}.attn_k.bias"), device)?,
                v: ld(c, r, &format!("{p}.attn_v.weight"), device)?,
                vb: ld(c, r, &format!("{p}.attn_v.bias"), device)?,
                o: ld(c, r, &format!("{p}.attn_out.weight"), device)?,
                ob: ld(c, r, &format!("{p}.attn_out.bias"), device)?,
                n1w: ld(c, r, &format!("{p}.norm1.weight"), device)?,
                n1b: ld(c, r, &format!("{p}.norm1.bias"), device)?,
                n2w: ld(c, r, &format!("{p}.norm2.weight"), device)?,
                n2b: ld(c, r, &format!("{p}.norm2.bias"), device)?,
                fc1: ld(c, r, &format!("{p}.mlp.linear_fc1.weight"), device)?,
                fc1b: ld(c, r, &format!("{p}.mlp.linear_fc1.bias"), device)?,
                fc2: ld(c, r, &format!("{p}.mlp.linear_fc2.weight"), device)?,
                fc2b: ld(c, r, &format!("{p}.mlp.linear_fc2.bias"), device)?,
            });
        }
        Ok(Self {
            patch_w,
            patch_b,
            pos_embed,
            merger_nw: ld(c, r, "v.merger.norm.weight", device)?,
            merger_nb: ld(c, r, "v.merger.norm.bias", device)?,
            merger_fc1: ld(c, r, "v.merger.linear_fc1.weight", device)?,
            merger_fc1b: ld(c, r, "v.merger.linear_fc1.bias", device)?,
            merger_fc2: ld(c, r, "v.merger.linear_fc2.weight", device)?,
            merger_fc2b: ld(c, r, "v.merger.linear_fc2.bias", device)?,
            cfg,
            blocks,
            device: device.clone(),
        })
    }

    /// Bilinear-interpolate the learned pos_embed (grid_per_side²) to (gh,gw),
    /// in the 2x2-merge-grouped patch order (matching preprocess). Returns [n_patches, hidden].
    fn interp_pos_embed(&self, gh: usize, gw: usize) -> Result<Tensor> {
        let gps = self.cfg.grid_per_side;
        let ms = self.cfg.merge_size;
        let n = gh * gw;
        // gather 4 corner rows + bilinear weights for each patch, in merge-grouped order
        let mut idx: Vec<u32> = Vec::with_capacity(n * 4);
        let mut wts: Vec<f32> = Vec::with_capacity(n * 4);
        let step_h = (gps - 1) as f32 / (gh.max(2) - 1) as f32;
        let step_w = (gps - 1) as f32 / (gw.max(2) - 1) as f32;
        let mut h = 0;
        while h < gh {
            let mut w = 0;
            while w < gw {
                for mh in 0..ms {
                    for mw in 0..ms {
                        let (yh, xw) = (h + mh, w + mw);
                        let y = yh as f32 * step_h;
                        let x = xw as f32 * step_w;
                        let (fy, fx) = (y.floor() as usize, x.floor() as usize);
                        let cy = (fy + 1).min(gps - 1);
                        let cx = (fx + 1).min(gps - 1);
                        let (dy, dx) = (y - fy as f32, x - fx as f32);
                        idx.push((fy * gps + fx) as u32);
                        wts.push((1.0 - dy) * (1.0 - dx));
                        idx.push((fy * gps + cx) as u32);
                        wts.push((1.0 - dy) * dx);
                        idx.push((cy * gps + fx) as u32);
                        wts.push(dy * (1.0 - dx));
                        idx.push((cy * gps + cx) as u32);
                        wts.push(dy * dx);
                    }
                }
                w += ms;
            }
            h += ms;
        }
        let idx_t = Tensor::from_vec(idx, n * 4, &self.device)?;
        let rows = self.pos_embed.index_select(&idx_t, 0)?; // [n*4, hidden]
        let wts_t = Tensor::from_vec(wts, (n * 4, 1), &self.device)?;
        let weighted = rows
            .broadcast_mul(&wts_t)?
            .reshape((n, 4, self.cfg.hidden))?;
        weighted.sum(1) // [n, hidden]
    }

    /// 2D rope cos/sin for the patches in merge-grouped order. Returns (cos,sin) [n_patches, head_dim].
    fn rope_cos_sin(&self, gh: usize, gw: usize) -> Result<(Tensor, Tensor)> {
        let hd = self.cfg.head_dim;
        let half = hd / 2; // 36
        let quarter = half / 2; // 18
        let theta = self.cfg.rope_theta as f64;
        let ms = self.cfg.merge_size;
        let n = gh * gw;
        // per-patch (y,x) in merge-grouped order; freq[j] = pos / theta^(2j/half)
        let mut cos = vec![0f32; n * hd];
        let mut sin = vec![0f32; n * hd];
        let invf: Vec<f64> = (0..quarter)
            .map(|j| 1.0 / theta.powf((2 * j) as f64 / half as f64))
            .collect();
        let mut pi = 0usize;
        let mut h = 0;
        while h < gh {
            let mut w = 0;
            while w < gw {
                for mh in 0..ms {
                    for mw in 0..ms {
                        let (y, x) = ((h + mh) as f64, (w + mw) as f64);
                        // first `quarter` freqs use y, next `quarter` use x -> [y-freqs(18), x-freqs(18)] = half(36)
                        for j in 0..quarter {
                            let ay = (y * invf[j]) as f32;
                            let ax = (x * invf[j]) as f32;
                            // dims [j] and [j+half] share angle ay (rotate-half pairs i, i+half)
                            cos[pi * hd + j] = ay.cos();
                            sin[pi * hd + j] = ay.sin();
                            cos[pi * hd + half + j] = ay.cos();
                            sin[pi * hd + half + j] = ay.sin();
                            cos[pi * hd + quarter + j] = ax.cos();
                            sin[pi * hd + quarter + j] = ax.sin();
                            cos[pi * hd + half + quarter + j] = ax.cos();
                            sin[pi * hd + half + quarter + j] = ax.sin();
                        }
                        pi += 1;
                    }
                }
                w += ms;
            }
            h += ms;
        }
        Ok((
            Tensor::from_vec(cos, (n, hd), &self.device)?,
            Tensor::from_vec(sin, (n, hd), &self.device)?,
        ))
    }

    pub fn forward(&self, pixel_values: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let xs = pixel_values.to_dtype(DType::F32)?.to_device(&self.device)?; // [n_patches, 1536]
        let n = xs.dim(0)?;
        // patch embed (linear) + pos embed
        let mut h = lin(&xs, &self.patch_w, &self.patch_b)?; // [n, hidden]
        h = (h + self.interp_pos_embed(grid_h, grid_w)?)?;
        let (cos, sin) = self.rope_cos_sin(grid_h, grid_w)?; // [n, head_dim]
        let nh = self.cfg.n_head;
        let hd = self.cfg.head_dim;
        for blk in &self.blocks {
            let res = h.clone();
            let hn = layernorm(&h, &blk.n1w, &blk.n1b, self.cfg.eps)?;
            // qkv: [n, nh*hd] -> [nh, n, hd]
            let q = lin(&hn, &blk.q, &blk.qb)?
                .reshape((n, nh, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            let k = lin(&hn, &blk.k, &blk.kb)?
                .reshape((n, nh, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            let v = lin(&hn, &blk.v, &blk.vb)?
                .reshape((n, nh, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            // apply rope: cos/sin [n,hd] broadcast over heads
            let cb = cos.reshape((1, n, hd))?;
            let sb = sin.reshape((1, n, hd))?;
            let q = (q.broadcast_mul(&cb)? + rotate_half(&q)?.broadcast_mul(&sb)?)?;
            let k = (k.broadcast_mul(&cb)? + rotate_half(&k)?.broadcast_mul(&sb)?)?;
            // full attention [nh, n, n]
            let scale = 1.0 / (hd as f64).sqrt();
            let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
            let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
            let attn = probs.matmul(&v)?; // [nh, n, hd]
            let attn = attn.transpose(0, 1)?.reshape((n, nh * hd))?;
            let attn = lin(&attn, &blk.o, &blk.ob)?;
            h = (res + attn)?;
            let res = h.clone();
            let hn = layernorm(&h, &blk.n2w, &blk.n2b, self.cfg.eps)?;
            let ff = lin(&lin(&hn, &blk.fc1, &blk.fc1b)?.gelu()?, &blk.fc2, &blk.fc2b)?;
            h = (res + ff)?;
        }
        // merger: group 2x2 -> [n/4, hidden*4]; norm (over hidden, pre-merge); fc2(gelu(fc1))
        // patches are already in 2x2-merge-grouped order, so consecutive groups of 4 = one merged token.
        let hn = layernorm(&h, &self.merger_nw, &self.merger_nb, self.cfg.eps)?; // [n, hidden]
        let merged = hn.reshape((n / 4, self.cfg.hidden * 4))?; // [n_merged, 4608]
        let out = lin(
            &lin(&merged, &self.merger_fc1, &self.merger_fc1b)?.gelu()?,
            &self.merger_fc2,
            &self.merger_fc2b,
        )?;
        Ok(out) // [n_merged, out_dim=2048]
    }
}
