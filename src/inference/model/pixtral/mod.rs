//! Pixtral (image->text) - the Pixtral-ViT vision encoder + 2-layer GELU projector that maps patch
//! features into the Mistral-Nemo embedding space (5120).
//!
//! Loaded from the llama.cpp `mmproj-*-pixtral-12b-*.gguf` (`clip-vision`, projector_type "pixtral").
//! Tensors: `v.*` = the ViT, `mm.*` = the projector, `v.token_embd.img_break` = the [IMG_BREAK]
//! embedding (5120, inserted between image rows in the decoder stream). Vision config (GGUF meta):
//! dim 1024, 24 blocks, 16 heads (hd 64), ffn 4096, patch 16, image <=1024, SiLU, RoPE-2D, -> 5120.
//!
//! Pixtral-ViT (bias-free attention, RMSNorm, SwiGLU, 2D-RoPE θ=10000; ln_pre then blocks, NO final
//! norm): image [1,3,H,W] -> Conv2d(3->1024,k16,s16) -> [gh.gw,1024] -> ln_pre -> 24x[rms->attn(+rope)->res,
//! rms->SwiGLU->res] -> projector mm.1(1024->5120)->GELU->mm.2(5120->5120) -> [gh.gw, 5120].
//!
//! Pixtral RoPE-2D: freqs=1/θ^(i/32), i∈0..32; height uses freqs[0::2], width freqs[1::2]; patch
//! (r,c) -> rot = cat(r.freqs_h[16], c.freqs_w[16]) -> emb=cat(rot,rot)[64] -> cos/sin, rotate_half.
//!
//! Self-contained on the native substrate (mmproj-dequant->F32, explicit matmuls). The native GGUF
//! reader reverses ggml `ne` -> weights arrive [out,in], conv [out,in,kh,kw].

use crate::tensor::layer::{Conv2d, Conv2dConfig};
use crate::tensor::{Device, Result, Tensor};

const DIM: usize = 1024;
const HEADS: usize = 16;
const HD: usize = DIM / HEADS; // 64
const LAYERS: usize = 24;
const PATCH: usize = 16;
const PROJ: usize = 5120; // Mistral-Nemo hidden
const THETA: f32 = 10000.0;
const EPS: f32 = 1e-5;

fn err(m: String) -> crate::tensor::Error {
    crate::tensor::Error(m)
}

// CLIP normalization (clip.vision.image_mean/std from the mmproj).
const MEAN: [f32; 3] = [0.48145467, 0.4578275, 0.40821072];
const STD: [f32; 3] = [0.26862955, 0.2613026, 0.2757771];

/// Preprocess EXACTLY like HF `PixtralImageProcessor` (so the ViT sees identical pixel_values):
/// ratio = max(h,w)/longest_edge; if >1 floor-scale both; grid = CEIL(dim/16); resize the ORIGINAL
/// straight to (gh.16, gw.16) with bicubic (CatmullRom = PIL BICUBIC); CLIP-normalize.
/// Returns native `[1,3,H,W]` + the patch grid.
pub fn preprocess(
    img: &image::RgbImage,
    longest_edge: usize,
    dev: &Device,
) -> Result<(Tensor, usize, usize)> {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let ratio = (h0 as f32 / longest_edge as f32).max(w0 as f32 / longest_edge as f32);
    let (mut h, mut w) = (h0, w0);
    if ratio > 1.0 {
        h = (h0 as f32 / ratio).floor() as usize;
        w = (w0 as f32 / ratio).floor() as usize;
    }
    // A very thin/wide image floors a side to 0; keep at least one patch so
    // `(dim-1)` can't underflow (release: wraps to usize::MAX -> OOM abort).
    let (h, w) = (h.max(PATCH), w.max(PATCH));
    let (gh, gw) = ((h - 1) / PATCH + 1, (w - 1) / PATCH + 1);
    let (h, w) = (gh * PATCH, gw * PATCH);
    let rz = if w == w0 && h == h0 {
        img.clone()
    } else {
        image::imageops::resize(
            img,
            w as u32,
            h as u32,
            image::imageops::FilterType::CatmullRom,
        )
    };
    let mut v = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = rz.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                v[c * h * w + y * w + x] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    let t = Tensor::from_vec_f32(v, (1usize, 3usize, h, w))?.to_device(dev)?;
    Ok((t, gh, gw))
}

/// `x [n,in] . wᵀ` (`w` is `[out,in]`) -> `[n,out]`, optional bias.
fn lin(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.transpose(0, 1)?.contiguous()?)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

/// RMSNorm: x / sqrt(mean(x²)+eps) . w. Pixtral vision uses these genuinely-small trained weights
/// BARE (γ ≈ 0.02, some negative) - VERIFIED against the HF safetensors (ln_pre.weight mean 0.0215
/// range [-0.004,0.112] == the mmproj value, i.e. the mmproj stored the raw HF weight, no delta).
/// Confirmed bit-exact (cosine 1.0 per-layer) vs the real transformers PixtralVisionModel. The small
/// γ keeps the per-block signal small; the residual still grows to ~3 by the last block (attention/
/// FFN projections dominate), matching transformers exactly.
fn rmsnorm(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.rms_norm(w, EPS)
}

/// rotate_half: [x1, x2] (halves along last dim) -> [-x2, x1].
/// The GGUF/mmproj stores the ViT Q/K projections PERMUTED for interleaved (GPT-J) rope - the same
/// row-interleave llama.cpp applies at conversion. Our 2D-RoPE uses `rotate_half` (NeoX/split-half,
/// matching transformers `PixtralVisionModel`), so un-permute Q/K rows back to NeoX layout at load:
/// per head, neox[i]=int[2i], neox[HD/2+i]=int[2i+1]. Without this the ViT features are wrong
/// (cos≈0.03 vs the reference) and perception fails (dog->cow). V/O are rope-free - left untouched.
fn deinterleave_qk(w: Vec<f32>, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let (nh, half) = (out_dim / HD, HD / 2);
    let mut o = vec![0f32; w.len()];
    for h in 0..nh {
        let base = h * HD;
        for i in 0..half {
            let (s0, d0) = ((base + 2 * i) * in_dim, (base + i) * in_dim);
            let (s1, d1) = ((base + 2 * i + 1) * in_dim, (base + half + i) * in_dim);
            o[d0..d0 + in_dim].copy_from_slice(&w[s0..s0 + in_dim]);
            o[d1..d1 + in_dim].copy_from_slice(&w[s1..s1 + in_dim]);
        }
    }
    o
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    let last = d[d.len() - 1];
    let h = last / 2;
    let x1 = x.narrow(d.len() - 1, 0, h)?;
    let x2 = x.narrow(d.len() - 1, h, h)?;
    Tensor::cat(&[&x2.affine(-1.0, 0.0)?, &x1], d.len() - 1)
}

/// Pixtral 2D-RoPE cos/sin for a grid of `ghxgw` patches (row-major). Returns `([n,HD],[n,HD])`.
fn rope_cos_sin(gh: usize, gw: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = HD / 2; // 32
    let nf = half / 2; // 16
                       // freqs[i] = 1/θ^(i/half), i∈0..half ; h uses even indices, w uses odd.
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / THETA.powf(i as f32 / half as f32))
        .collect();
    let fh: Vec<f32> = (0..nf).map(|i| freqs[2 * i]).collect();
    let fw: Vec<f32> = (0..nf).map(|i| freqs[2 * i + 1]).collect();
    let n = gh * gw;
    let mut cos = vec![0f32; n * HD];
    let mut sin = vec![0f32; n * HD];
    for r in 0..gh {
        for c in 0..gw {
            let p = r * gw + c;
            for i in 0..nf {
                let ah = r as f32 * fh[i];
                let aw = c as f32 * fw[i];
                // rot = [r.fh (16), c.fw (16)] ; emb = cat(rot,rot) (64)
                for &(off, a) in &[(i, ah), (nf + i, aw)] {
                    cos[p * HD + off] = a.cos();
                    cos[p * HD + half + off] = a.cos();
                    sin[p * HD + off] = a.sin();
                    sin[p * HD + half + off] = a.sin();
                }
            }
        }
    }
    Ok((
        Tensor::from_vec_f32(cos, (n, HD))?.to_device(dev)?,
        Tensor::from_vec_f32(sin, (n, HD))?.to_device(dev)?,
    ))
}

struct Block {
    ln1_w: Tensor,
    ln2_w: Tensor,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
}

impl Block {
    /// x `[seq, DIM]`, full (non-causal) attention over all patches. cos/sin `[seq, HD]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let h = rmsnorm(x, &self.ln1_w)?;
        let q = lin(&h, &self.q_w, None)?.reshape((seq, HEADS, HD))?;
        let k = lin(&h, &self.k_w, None)?.reshape((seq, HEADS, HD))?;
        let v = lin(&h, &self.v_w, None)?.reshape((seq, HEADS, HD))?;
        // rope: cos/sin [seq,HD] -> [seq,1,HD]
        let cos3 = cos.reshape((seq, 1, HD))?;
        let sin3 = sin.reshape((seq, 1, HD))?;
        let q = q
            .broadcast_mul(&cos3)?
            .add(&rotate_half(&q)?.broadcast_mul(&sin3)?)?;
        let k = k
            .broadcast_mul(&cos3)?
            .add(&rotate_half(&k)?.broadcast_mul(&sin3)?)?;
        // [seq,H,HD] -> [H,seq,HD]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0f32 / (HD as f32).sqrt();
        // Query-tiled attention (never materializes [H,seq,seq]) - for Pixtral at native 1024px the
        // grid is up to 64x64=4096 patches, so the full scores tensor would be GBs. Non-causal.
        let o = crate::inference::model::acestep::ops::sdpa_tiled(
            &q, &k, &v, None, false, scale, 1.0, 512,
        )?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((seq, DIM))?;
        let attn = lin(&o, &self.o_w, None)?;
        let x = x.add(&attn)?;
        // SwiGLU FFN: down(silu(gate(x)) . up(x))
        let h = rmsnorm(&x, &self.ln2_w)?;
        let g = lin(&h, &self.gate_w, None)?.silu()?;
        let u = lin(&h, &self.up_w, None)?;
        let ff = lin(&g.mul(&u)?, &self.down_w, None)?;
        x.add(&ff)
    }
}

pub struct PixtralVision {
    patch: Conv2d,
    pre_ln_w: Tensor,
    blocks: Vec<Block>,
    mm1_w: Tensor,
    mm1_b: Tensor,
    mm2_w: Tensor,
    mm2_b: Tensor,
    img_break: Tensor, // [PROJ] - inserted between image rows in the decoder stream
    dev: Device,
}

impl PixtralVision {
    pub fn load_mmproj(path: &str, dev: &Device) -> Result<Self> {
        use crate::tensor::quantized::gguf_file;
        use crate::tensor::DType as CDType;
        use crate::tensor::Device as CDevice;
        let content = gguf_file::open_mapped(std::path::Path::new(path))
            .map_err(|e| err(format!("gguf {path}: {e}")))?;
        let dqd = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let t = content
                .tensor(name, &CDevice::Cpu)
                .and_then(|t| t.dequantize(&CDevice::Cpu))
                .and_then(|t| t.to_dtype(CDType::F32))
                .map_err(|e| err(format!("dq {name}: {e}")))?;
            let dims = t.dims().to_vec();
            let v = t
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| err(format!("vec {name}: {e}")))?;
            Ok((v, dims))
        };
        let t2 = |v: Vec<f32>, r: usize, c: usize| -> Result<Tensor> {
            Tensor::from_vec_f32(v, (r, c)).and_then(|t| t.to_device(dev))
        };
        let t1 = |v: Vec<f32>| -> Result<Tensor> {
            let n = v.len();
            Tensor::from_vec_f32(v, (n,)).and_then(|t| t.to_device(dev))
        };

        // patch conv: native reader gives [out,in,kh,kw] = [1024,3,16,16].
        let (pw, pd) = dqd("v.patch_embd.weight")?;
        let patchw = Tensor::from_vec_f32(pw, (pd[0], pd[1], pd[2], pd[3]))?.to_device(dev)?;
        let patch = Conv2d::new(
            patchw,
            None,
            Conv2dConfig {
                padding: 0,
                stride: PATCH,
                ..Default::default()
            },
        );

        let (pre, _) = dqd("v.pre_ln.weight")?;

        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("v.blk.{i}");
            let (qw, _) = dqd(&format!("{p}.attn_q.weight"))?;
            let (kw, _) = dqd(&format!("{p}.attn_k.weight"))?;
            // GGUF Q/K are interleaved-rope permuted -> un-permute to NeoX for our rotate_half rope.
            let qw = deinterleave_qk(qw, DIM, DIM);
            let kw = deinterleave_qk(kw, DIM, DIM);
            let (vw, _) = dqd(&format!("{p}.attn_v.weight"))?;
            let (ow, _) = dqd(&format!("{p}.attn_out.weight"))?;
            let (l1, _) = dqd(&format!("{p}.ln1.weight"))?;
            let (l2, _) = dqd(&format!("{p}.ln2.weight"))?;
            let (gw, gd) = dqd(&format!("{p}.ffn_gate.weight"))?; // [4096,1024]
            let (uw, ud) = dqd(&format!("{p}.ffn_up.weight"))?; // [4096,1024]
            let (dw, dd) = dqd(&format!("{p}.ffn_down.weight"))?; // [1024,4096]
            blocks.push(Block {
                ln1_w: t1(l1)?,
                ln2_w: t1(l2)?,
                q_w: t2(qw, DIM, DIM)?,
                k_w: t2(kw, DIM, DIM)?,
                v_w: t2(vw, DIM, DIM)?,
                o_w: t2(ow, DIM, DIM)?,
                gate_w: t2(gw, gd[0], gd[1])?,
                up_w: t2(uw, ud[0], ud[1])?,
                down_w: t2(dw, dd[0], dd[1])?,
            });
        }
        let (m1, m1d) = dqd("mm.1.weight")?; // [5120,1024]
        let (m1b, _) = dqd("mm.1.bias")?;
        let (m2, m2d) = dqd("mm.2.weight")?; // [5120,5120]
        let (m2b, _) = dqd("mm.2.bias")?;
        let (ib, _) = dqd("v.token_embd.img_break")?;
        Ok(Self {
            patch,
            pre_ln_w: t1(pre)?,
            blocks,
            mm1_w: t2(m1, m1d[0], m1d[1])?,
            mm1_b: t1(m1b)?,
            mm2_w: t2(m2, m2d[0], m2d[1])?,
            mm2_b: t1(m2b)?,
            img_break: t1(ib)?,
            dev: dev.clone(),
        })
    }

    /// `img [1,3,H,W]` (H,W multiples of 16, CLIP-normalized) -> patch embeds `[gh.gw, PROJ]` and the
    /// grid `(gh, gw)` (for img_break row insertion by the caller).
    pub fn forward(&self, img: &Tensor) -> Result<(Tensor, (usize, usize))> {
        let (_, e, g) = self.forward_hidden(img)?;
        Ok((e, g))
    }

    /// Same as `forward` but also returns the ViT hidden state BEFORE the projector (`[gh.gw, DIM]`),
    /// i.e. `(vit_out, embeds, grid)` - for A/B against the transformers `PixtralVisionModel`.
    pub fn forward_hidden(&self, img: &Tensor) -> Result<(Tensor, Tensor, (usize, usize))> {
        let img = img.to_device(&self.dev)?;
        let d = img.shape().dims();
        let (h, w) = (d[d.len() - 2], d[d.len() - 1]);
        let (gh, gw) = (h / PATCH, w / PATCH);
        let x = self.patch.forward(&img)?; // [1,1024,gh,gw]
        let x = x.reshape((DIM, gh * gw))?.transpose(0, 1)?.contiguous()?; // [gh.gw, 1024]
        let mut x = rmsnorm(&x, &self.pre_ln_w)?;
        let (cos, sin) = rope_cos_sin(gh, gw, &self.dev)?;
        for b in self.blocks.iter() {
            x = b.forward(&x, &cos, &sin)?;
        }
        let vit_out = x.clone(); // [gh.gw, DIM] pre-projector
                                 // projector: mm.1 -> GELU -> mm.2
        let h = lin(&x, &self.mm1_w, Some(&self.mm1_b))?;
        let h = h.gelu_erf()?;
        let x = lin(&h, &self.mm2_w, Some(&self.mm2_b))?; // [gh.gw, 5120]
        Ok((vit_out, x, (gh, gw)))
    }

    /// The [IMG_BREAK] embedding (`[PROJ]`) - inserted between image rows in the decoder stream.
    pub fn img_break(&self) -> &Tensor {
        &self.img_break
    }

    /// Engine entry: raw RGB image -> decoder-ready image embeds `[1, gh.gw + gh-1, 5120]` (compat,
    /// CPU) with the `[IMG_BREAK]` embedding interleaved between patch rows (one per row except the
    /// last - the caller closes the region with the `[IMG_END]` TOKEN in the text suffix). Native
    /// 1024 longest-edge (the model's image_size), HF-exact preprocessing.
    pub fn encode_image_with_breaks(&self, img: &image::RgbImage) -> Result<crate::tensor::Tensor> {
        let (px, gh, gw) = preprocess(img, 1024, &self.dev)?;
        let (patch_emb, _) = self.forward(&px)?;
        let pv = patch_emb.to_vec_f32(); // [gh.gw, 5120]
        let bv = self.img_break.to_vec_f32(); // [5120]
        let mut seq: Vec<f32> = Vec::with_capacity((gh * gw + gh) * PROJ);
        for r in 0..gh {
            seq.extend_from_slice(&pv[r * gw * PROJ..(r + 1) * gw * PROJ]);
            if r + 1 < gh {
                seq.extend_from_slice(&bv);
            }
        }
        let n_tok = gh * gw + (gh - 1);
        crate::tensor::Tensor::from_vec(
            seq,
            &[1usize, n_tok, PROJ][..],
            &crate::tensor::Device::Cpu,
        )
        .map_err(|e| err(format!("embeds: {e}")))
    }
}
