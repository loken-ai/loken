//! Qwen2.5-VL vision tower (the semantic-conditioning encoder for Qwen-Image-Edit). Ported from
//! transformers `modeling_qwen2_5_vl.py` against a byte-exact oracle (scripts/qwen25vl_vision_oracle.py).
//!
//! Input: pixel_values `[n_patches, 1176]` (the processor's (c,t,h,w) patch flattening) + grid `(t,h,w)`.
//! Output: merged image embeddings `[n_patches/4, 3584]` that fill the LLM's `<|image_pad|>` slots.
//!
//! Arch (7B): dim 1280, 32 blocks, 16 heads (hd 80), patch 14, temporal_patch 2, merge 2, ffn 3420
//! (SwiGLU), RMSNorm eps 1e-6, 2D-RoPE (theta 1e4). Window attention (window 112px = 4 merged-tokens)
//! in all blocks EXCEPT full-attention at {7,15,23,31}. Merger: RMSNorm->group2x2->5120->GELU->3584.
//! Weights: mmproj GGUF (`v.*`/`mm.*`), all dequantized to F32 (the tower is small).
use crate::tensor::Result;
use crate::tensor::{Device, Tensor};

const DIM: usize = 1280;
const DEPTH: usize = 32;
const HEADS: usize = 16;
const HD: usize = 80; // head_dim
const FFN: usize = 3420;
const MERGE: usize = 2;
const PATCH: usize = 14;
const WINDOW: usize = 112;
const OUT: usize = 3584;
const EPS: f32 = 1e-6;
const FULLATT: [usize; 4] = [7, 15, 23, 31];

struct Block {
    n1: Tensor,
    n2: Tensor, // RMSNorm gammas [1280]
    qkv_w: Tensor,
    qkv_b: Tensor, // [3840,1280], [3840]
    proj_w: Tensor,
    proj_b: Tensor, // [1280,1280], [1280]
    gate_w: Tensor,
    gate_b: Tensor, // [3420,1280]
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor, // [1280,3420]
}

pub struct Qwen25Vision {
    patch_w: Tensor, // [1280,1176]
    blocks: Vec<Block>,
    ln_q: Tensor, // merger RMSNorm [1280]
    mm0_w: Tensor,
    mm0_b: Tensor, // [5120,5120]
    mm2_w: Tensor,
    mm2_b: Tensor, // [3584,5120]
    dev: Device,
}

fn lin(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    // x [n,in] . wᵀ (w is [out,in]) -> [n,out]
    let y = x.matmul(&w.transpose(0, 1)?.contiguous()?)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

impl Qwen25Vision {
    pub fn load_mmproj(path: &str, dev: &Device) -> Result<Self> {
        use crate::tensor::quantized::gguf_file;
        use crate::tensor::DType as CDType;
        use crate::tensor::Device as CDevice;
        let content = gguf_file::open_mapped(std::path::Path::new(path))
            .map_err(|e| err(format!("gguf {path}: {e}")))?;
        let dqv = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
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
        let t2 = |v: Vec<f32>, d: (usize, usize), dev: &Device| {
            Tensor::from_vec_f32(v, d).and_then(|t| t.to_device(dev))
        };
        let t1 = |v: Vec<f32>, dev: &Device| {
            let n = v.len();
            Tensor::from_vec_f32(v, (n,)).and_then(|t| t.to_device(dev))
        };

        // patch_embed: two [1280,3,14,14] temporal slices -> W[1280,1176], j=((c*2+t)*14+h)*14+w.
        let (s0, _) = dqv("v.patch_embd.weight")?;
        let (s1, _) = dqv("v.patch_embd.weight.1")?;
        let (cin, ps) = (3usize, PATCH);
        let pin = cin * 2 * ps * ps;
        let at = |s: &[f32], o: usize, c: usize, h: usize, w: usize| {
            s[((o * cin + c) * ps + h) * ps + w]
        };
        let mut pw = vec![0f32; DIM * pin];
        for o in 0..DIM {
            for c in 0..cin {
                for t in 0..2 {
                    let s = if t == 0 { &s0 } else { &s1 };
                    for h in 0..ps {
                        for w in 0..ps {
                            pw[o * pin + ((c * 2 + t) * ps + h) * ps + w] = at(s, o, c, h, w);
                        }
                    }
                }
            }
        }
        let patch_w = t2(pw, (DIM, pin), dev)?;

        let mut blocks = Vec::with_capacity(DEPTH);
        for b in 0..DEPTH {
            let g = format!("v.blk.{b}");
            let (qw, _) = dqv(&format!("{g}.attn_q.weight"))?;
            let (kw, _) = dqv(&format!("{g}.attn_k.weight"))?;
            let (vw, _) = dqv(&format!("{g}.attn_v.weight"))?;
            let (qb, _) = dqv(&format!("{g}.attn_q.bias"))?;
            let (kb, _) = dqv(&format!("{g}.attn_k.bias"))?;
            let (vb, _) = dqv(&format!("{g}.attn_v.bias"))?;
            let mut qkvw = qw;
            qkvw.extend(kw);
            qkvw.extend(vw); // [3840,1280] row-concat
            let mut qkvb = qb;
            qkvb.extend(kb);
            qkvb.extend(vb); // [3840]
            let (pw2, _) = dqv(&format!("{g}.attn_out.weight"))?;
            let (pb, _) = dqv(&format!("{g}.attn_out.bias"))?;
            let (gw, _) = dqv(&format!("{g}.ffn_gate.weight"))?;
            let (gb, _) = dqv(&format!("{g}.ffn_gate.bias"))?;
            let (uw, _) = dqv(&format!("{g}.ffn_up.weight"))?;
            let (ub, _) = dqv(&format!("{g}.ffn_up.bias"))?;
            let (dw, _) = dqv(&format!("{g}.ffn_down.weight"))?;
            let (db, _) = dqv(&format!("{g}.ffn_down.bias"))?;
            let (n1, _) = dqv(&format!("{g}.ln1.weight"))?;
            let (n2, _) = dqv(&format!("{g}.ln2.weight"))?;
            blocks.push(Block {
                n1: t1(n1, dev)?,
                n2: t1(n2, dev)?,
                qkv_w: t2(qkvw, (3 * DIM, DIM), dev)?,
                qkv_b: t1(qkvb, dev)?,
                proj_w: t2(pw2, (DIM, DIM), dev)?,
                proj_b: t1(pb, dev)?,
                gate_w: t2(gw, (FFN, DIM), dev)?,
                gate_b: t1(gb, dev)?,
                up_w: t2(uw, (FFN, DIM), dev)?,
                up_b: t1(ub, dev)?,
                down_w: t2(dw, (DIM, FFN), dev)?,
                down_b: t1(db, dev)?,
            });
        }
        let (lq, _) = dqv("v.post_ln.weight")?;
        let (m0w, _) = dqv("mm.0.weight")?;
        let (m0b, _) = dqv("mm.0.bias")?;
        let (m2w, _) = dqv("mm.2.weight")?;
        let (m2b, _) = dqv("mm.2.bias")?;
        Ok(Qwen25Vision {
            patch_w,
            blocks,
            ln_q: t1(lq, dev)?,
            mm0_w: t2(m0w, (5120, 5120), dev)?,
            mm0_b: t1(m0b, dev)?,
            mm2_w: t2(m2w, (OUT, 5120), dev)?,
            mm2_b: t1(m2b, dev)?,
            dev: dev.clone(),
        })
    }

    /// Full forward: pixel_values [n,1176], grid (t,h,w) -> merged embeds [n/4, 3584].
    pub fn forward(&self, pixel_values: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (gt, gh, gw) = grid;
        let seq = gt * gh * gw;
        let dev = &self.dev;
        // patch_embed
        let mut hs = lin(pixel_values, &self.patch_w, None)?; // [seq,1280]

        // position_ids (block-major (h,w) per patch), rotary, window permute -> cos/sin [seq,80]
        let pos = position_ids(gt, gh, gw); // [(h,w); seq]
        let (window_index, cu_win) = window_index(gt, gh, gw);
        let cu_full = vec![0usize, seq];
        let (cos, sin) = rope_cos_sin(&pos, &window_index, dev)?; // [seq,80] each (window-permuted)

        // permute hidden by window_index in groups of merge² (=4)
        hs = permute_units(&hs, &window_index, dev)?;

        for (bi, blk) in self.blocks.iter().enumerate() {
            let cu = if FULLATT.contains(&bi) {
                &cu_full
            } else {
                &cu_win
            };
            // attn
            let normed = hs.rms_norm(&blk.n1, EPS)?;
            let attn = blk.attention(&normed, &cos, &sin, cu, dev)?;
            hs = hs.add(&attn)?;
            // mlp (SwiGLU)
            let n2 = hs.rms_norm(&blk.n2, EPS)?;
            let g = lin(&n2, &blk.gate_w, Some(&blk.gate_b))?.silu()?;
            let u = lin(&n2, &blk.up_w, Some(&blk.up_b))?;
            let m = lin(&g.mul(&u)?, &blk.down_w, Some(&blk.down_b))?;
            hs = hs.add(&m)?;
        }
        // merger: RMSNorm -> group 4 -> 5120 -> GELU -> 3584, then un-permute
        let normed = hs.rms_norm(&self.ln_q, EPS)?; // [seq,1280]
        let grouped = normed.reshape((seq / 4, 4 * DIM))?; // [seq/4,5120]
        let x = lin(&grouped, &self.mm0_w, Some(&self.mm0_b))?.gelu()?;
        let merged = lin(&x, &self.mm2_w, Some(&self.mm2_b))?; // [seq/4,3584]
                                                               // un-permute: reverse of window_index (on merged-token units)
        unpermute_merged(&merged, &window_index, dev)
    }
}

/// rotate_half over the last dim (HD): cat(-x[..,half:], x[..,:half]).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(2)?;
    let half = d / 2;
    let x1 = x.narrow(2, 0, half)?;
    let x2 = x.narrow(2, half, half)?.affine(-1.0, 0.0)?;
    Tensor::cat(&[&x2, &x1], 2)
}

impl Block {
    /// Variable-length attention over segments defined by `cu` (cumulative boundaries), 16 heads.
    fn attention(
        &self,
        normed: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu: &[usize],
        dev: &Device,
    ) -> Result<Tensor> {
        let seq = normed.dim(0)?;
        let qkv = lin(normed, &self.qkv_w, Some(&self.qkv_b))?; // [seq,3840]
        let q = qkv.narrow(1, 0, DIM)?.reshape((seq, HEADS, HD))?;
        let k = qkv.narrow(1, DIM, DIM)?.reshape((seq, HEADS, HD))?;
        let v = qkv
            .narrow(1, 2 * DIM, DIM)?
            .reshape((seq, HEADS, HD))?
            .contiguous()?;
        // rope: q*cos + rotate_half(q)*sin ; cos/sin [seq,80] -> [seq,1,80]
        let cos3 = cos.reshape((seq, 1, HD))?;
        let sin3 = sin.reshape((seq, 1, HD))?;
        let q = q
            .broadcast_mul(&cos3)?
            .add(&rotate_half(&q)?.broadcast_mul(&sin3)?)?
            .contiguous()?;
        let k = k
            .broadcast_mul(&cos3)?
            .add(&rotate_half(&k)?.broadcast_mul(&sin3)?)?
            .contiguous()?;
        let scale = 1.0f32 / (HD as f32).sqrt();
        let mut out = vec![0f32; seq * HEADS * HD];
        for w in cu.windows(2) {
            let (s, e) = (w[0], w[1]);
            let l = e - s;
            if l == 0 {
                continue;
            }
            // [l,16,80] -> [16,l,80]
            let qs = q.narrow(0, s, l)?.transpose(0, 1)?.contiguous()?;
            let ks = k.narrow(0, s, l)?.transpose(0, 1)?.contiguous()?;
            let vs = v.narrow(0, s, l)?.transpose(0, 1)?.contiguous()?;
            // shared sdpa (native_acestep_ops): qs @ ksᵀ * scale -> softmax -> @ vs,
            // same math as the previous inline scores/probs chain.
            let o = crate::inference::model::acestep::ops::sdpa(
                &qs, &ks, &vs, None, false, scale, 1.0,
            )?; // [16,l,80]
            let o = o.transpose(0, 1)?.contiguous()?; // [l,16,80]
            let ov = o.to_vec_f32();
            for i in 0..(l * HEADS * HD) {
                out[s * HEADS * HD + i] = ov[i];
            }
        }
        let attn = Tensor::from_vec_f32(out, (seq, HEADS * HD))?.to_device(dev)?;
        lin(&attn, &self.proj_w, Some(&self.proj_b))
    }
}

// -- construction helpers (validated against oracle dumps) -------------------------------------

/// Block-major (h,w) position ids: meshgrid(h,w) reshaped (h/2,2,w/2,2).transpose(1,2).flatten.
fn position_ids(gt: usize, gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let (mh, mw) = (gh / MERGE, gw / MERGE);
    let mut out = Vec::with_capacity(gt * gh * gw);
    for _t in 0..gt {
        for bh in 0..mh {
            for bw in 0..mw {
                for ih in 0..MERGE {
                    for iw in 0..MERGE {
                        out.push((bh * MERGE + ih, bw * MERGE + iw));
                    }
                }
            }
        }
    }
    out
}

/// Window reorder indices (merged-token units) + cumulative window boundaries (patch units).
fn window_index(gt: usize, gh: usize, gw: usize) -> (Vec<usize>, Vec<usize>) {
    let vm = WINDOW / MERGE / PATCH; // 4
    let unit = MERGE * MERGE; // 4
    let (lh, lw) = (gh / MERGE, gw / MERGE); // llm grid
    let pad_h = (vm - lh % vm) % vm;
    let pad_w = (vm - lw % vm) % vm;
    let (nh, nw) = ((lh + pad_h) / vm, (lw + pad_w) / vm);
    let mut window_index = Vec::new();
    let mut cu = vec![0usize];
    let mut base = 0usize;
    for _t in 0..gt {
        // index[i,j] = i*lw + j for i<lh,j<lw else -100 (padded)
        for wh in 0..nh {
            for ww in 0..nw {
                let mut count = 0usize;
                for ih in 0..vm {
                    for iw in 0..vm {
                        let (gi, gj) = (wh * vm + ih, ww * vm + iw);
                        if gi < lh && gj < lw {
                            window_index.push(base + gi * lw + gj);
                            count += 1;
                        }
                    }
                }
                let last = *cu.last().unwrap();
                cu.push(last + count * unit);
            }
        }
        base += lh * lw;
    }
    // unique_consecutive on cu
    let mut cuu = Vec::with_capacity(cu.len());
    for c in cu {
        if cuu.last() != Some(&c) {
            cuu.push(c);
        }
    }
    (window_index, cuu)
}

/// cos/sin [seq,80]: rotary(pos)=[h.invf(20), w.invf(20)] -> window-permute (units of 4) -> cat(rot,rot).
fn rope_cos_sin(
    pos: &[(usize, usize)],
    window_index: &[usize],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = HD / 2; // 40
    let nf = half / 2; // 20 freqs
    let inv: Vec<f32> = (0..nf)
        .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / half as f32))
        .collect();
    let seq = pos.len();
    // rotary[p] = [h*inv(20), w*inv(20)] (40)
    let mut rot = vec![0f32; seq * half];
    for (p, &(h, w)) in pos.iter().enumerate() {
        for i in 0..nf {
            rot[p * half + i] = h as f32 * inv[i];
            rot[p * half + nf + i] = w as f32 * inv[i];
        }
    }
    // window-permute rot in units of 4 merged patches
    let permuted = permute_units_vec(&rot, half, window_index);
    // emb = cat(rot,rot) -> [seq,80]; cos/sin
    let mut cos = vec![0f32; seq * HD];
    let mut sin = vec![0f32; seq * HD];
    for p in 0..seq {
        for i in 0..half {
            let a = permuted[p * half + i];
            cos[p * HD + i] = a.cos();
            cos[p * HD + half + i] = a.cos();
            sin[p * HD + i] = a.sin();
            sin[p * HD + half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec_f32(cos, (seq, HD))?.to_device(dev)?,
        Tensor::from_vec_f32(sin, (seq, HD))?.to_device(dev)?,
    ))
}

/// Reorder rows of a flat [seq,width] by window_index, in groups of 4 (spatial_merge_unit).
fn permute_units_vec(x: &[f32], width: usize, window_index: &[usize]) -> Vec<f32> {
    // x viewed as [seq/4, 4, width]; output[k] = x[window_index[k]] (each a 4xwidth block)
    let unit = MERGE * MERGE;
    let seq = x.len() / width;
    let nu = seq / unit;
    let mut out = vec![0f32; x.len()];
    for (k, &src) in window_index.iter().enumerate() {
        // block src -> block k
        for r in 0..unit {
            for c in 0..width {
                out[(k * unit + r) * width + c] = x[(src * unit + r) * width + c];
            }
        }
    }
    let _ = nu;
    out
}

fn permute_units(x: &Tensor, window_index: &[usize], dev: &Device) -> Result<Tensor> {
    let width = x.dim(1)?;
    let v = x.to_vec_f32();
    let out = permute_units_vec(&v, width, window_index);
    Tensor::from_vec_f32(out, (x.dim(0)?, width))?.to_device(dev)
}

/// Un-permute merged tokens [nu,3584] by reverse of window_index (merged-token units).
fn unpermute_merged(merged: &Tensor, window_index: &[usize], dev: &Device) -> Result<Tensor> {
    let width = merged.dim(1)?;
    let nu = merged.dim(0)?;
    let v = merged.to_vec_f32();
    let mut out = vec![0f32; v.len()];
    // merged row k corresponds to original merged-token window_index[k]
    for (k, &orig) in window_index.iter().enumerate() {
        for c in 0..width {
            out[orig * width + c] = v[k * width + c];
        }
    }
    let _ = nu;
    Tensor::from_vec_f32(out, (merged.dim(0)?, width))?.to_device(dev)
}

fn err(s: String) -> crate::tensor::Error {
    crate::tensor::Error(s)
}

// CLIP normalization constants (mmproj clip.vision.image_mean/std)
const IMEAN: [f32; 3] = [0.48145467, 0.4578275, 0.40821072];
const ISTD: [f32; 3] = [0.26862955, 0.2613026, 0.2757771];

/// smart_resize (factor 28, min 56², max 1280.28²): round to 28-multiples under the pixel cap.
pub fn smart_resize(h: usize, w: usize) -> (usize, usize) {
    let factor = (PATCH * MERGE) as f64; // 28
    let (minp, maxp) = (56.0 * 56.0, 1280.0 * 28.0 * 28.0);
    let r = |x: f64| (x / factor).round() * factor;
    let (mut hb, mut wb) = (r(h as f64).max(factor), r(w as f64).max(factor));
    let (hf, wf) = (h as f64, w as f64);
    if hb * wb > maxp {
        let beta = (hf * wf / maxp).sqrt();
        hb = ((hf / beta / factor).floor() * factor).max(factor);
        wb = ((wf / beta / factor).floor() * factor).max(factor);
    } else if hb * wb < minp {
        let beta = (minp / (hf * wf)).sqrt();
        hb = (hf * beta / factor).ceil() * factor;
        wb = (wf * beta / factor).ceil() * factor;
    }
    (hb as usize, wb as usize)
}

/// Preprocess an RGB image -> (pixel_values [n,1176], grid (1,gh,gw)) matching the Qwen2.5-VL processor:
/// smart_resize -> bicubic -> rescale/normalize -> block-major (c,t,h,w) patch flatten (t duplicated).
pub fn preprocess(img: &image::RgbImage, dev: &Device) -> Result<(Tensor, (usize, usize, usize))> {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let (rh, rw) = smart_resize(h0, w0);
    let resized = image::imageops::resize(
        img,
        rw as u32,
        rh as u32,
        image::imageops::FilterType::CatmullRom,
    );
    // normalized [C,H,W]
    let mut norm = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                norm[c * rh * rw + y * rw + x] = (p[c] as f32 / 255.0 - IMEAN[c]) / ISTD[c];
            }
        }
    }
    let (gh, gw) = (rh / PATCH, rw / PATCH);
    let (mh, mw) = (gh / MERGE, gw / MERGE);
    let n = gh * gw;
    let feat = 3 * 2 * PATCH * PATCH; // 1176 (c,t,h,w), t duplicated
    let mut pv = vec![0f32; n * feat];
    let mut row = 0usize;
    for bh in 0..mh {
        for bw in 0..mw {
            for ih in 0..MERGE {
                for iw in 0..MERGE {
                    let (ph_base, pw_base) = ((bh * MERGE + ih) * PATCH, (bw * MERGE + iw) * PATCH);
                    let mut j = 0usize;
                    for c in 0..3 {
                        for _t in 0..2 {
                            for hh in 0..PATCH {
                                for ww in 0..PATCH {
                                    pv[row * feat + j] =
                                        norm[c * rh * rw + (ph_base + hh) * rw + (pw_base + ww)];
                                    j += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    let t = Tensor::from_vec_f32(pv, (n, feat))?.to_device(dev)?;
    Ok((t, (1, gh, gw)))
}
