//! Kyutai `tts-1.6b-en_fr` - the Depformer (component 6 of the port).
//!
//! A tiny per-step autoregressive generator that unrolls the 32 audio codebooks of
//! ONE main step as a length-32 causal sequence. It is a 4-layer streaming transformer
//! (dim 1024, 16 heads hd 64, causal, NO rope, NO cross-attn) with **weights_per_step**:
//! each of the 32 positions selects one of 11 weight slots via a fixed schedule, applied
//! to the attention in/out projections and the gating FFN.
//!
//! Position `cb`'s input = `depformer_in[slot](transformer_out) + emb(token[cb])`, where
//!   - cb == 0: `token` is the TEXT token, embedded by `depformer_text_emb`
//!     (demux_second_stream + out1/out2 projecting 128 -> 1024);
//!   - cb  > 0: `token` is the previous audio codebook, embedded by `depformer_emb[cb-1]`
//!     (lookup 128 -> low_rank 1024).
//! Output per codebook: `logits[cb] = linears[cb](dep_output[cb])` (depformer_norms are
//! Identity). Each head produces `AUDIO_CARD` logits.

use crate::inference::model::acestep::ops::sdpa;
use crate::tensor::ops::causal_mask;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

const MAIN_DIM: usize = 2048; // main transformer hidden feeding depformer_in
const DIM: usize = 1024;
const N_HEAD: usize = 16;
const HD: usize = DIM / N_HEAD; // 64
const HIDDEN: usize = 2048; // gating hidden (linear_in -> 2*HIDDEN, linear_out ← HIDDEN)
const N_LAYERS: usize = 4;
const DEP_Q: usize = 32; // audio codebooks generated per step
const N_SLOTS: usize = 11;
const EMB_DIM: usize = 128; // codebook / text embedding table dim (pre low-rank)
const AUDIO_CARD: usize = 2048; // logits per codebook
const AUDIO_EMB_ROWS: usize = 2049;
const TEXT_EMB_ROWS: usize = 8001;
const TEXT_CARD: usize = 8001; // demux modulus
const RMS_EPS: f32 = 1e-8;

/// weights_per_step schedule: position `cb` -> weight slot.
const SCHEDULE: [usize; DEP_Q] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    10, 10,
];

struct Layer {
    norm1: Tensor,          // rms alpha [DIM]
    norm2: Tensor,          // rms alpha [DIM]
    in_projs: Vec<Tensor>,  // 11 x [3*DIM, DIM] fused QKV
    out_projs: Vec<Tensor>, // 11 x [DIM, DIM]
    g_in: Vec<Tensor>,      // 11 x [2*HIDDEN, DIM]
    g_out: Vec<Tensor>,     // 11 x [DIM, HIDDEN]
}
impl Layer {
    fn load(vb: &VarBuilder, i: usize) -> Result<Self> {
        let p = vb.pp(format!("layers.{i}"));
        let a = p.pp("self_attn");
        let g = p.pp("gating");
        // The per-step attention weights are stored CONCATENATED along dim 0
        // (in_proj_weight [N_SLOTS*3*DIM, DIM], out_proj.weight [N_SLOTS*DIM, DIM]);
        // slice per slot. Gating keeps one module per slot.
        let in_proj = a.get((N_SLOTS * 3 * DIM, DIM), "in_proj_weight")?;
        let out_proj = a.get((N_SLOTS * DIM, DIM), "out_proj.weight")?;
        let mut in_projs = Vec::with_capacity(N_SLOTS);
        let mut out_projs = Vec::with_capacity(N_SLOTS);
        let mut g_in = Vec::with_capacity(N_SLOTS);
        let mut g_out = Vec::with_capacity(N_SLOTS);
        for s in 0..N_SLOTS {
            in_projs.push(in_proj.narrow(0, s * 3 * DIM, 3 * DIM)?.contiguous()?);
            out_projs.push(out_proj.narrow(0, s * DIM, DIM)?.contiguous()?);
            g_in.push(g.get((2 * HIDDEN, DIM), &format!("{s}.linear_in.weight"))?);
            g_out.push(g.get((DIM, HIDDEN), &format!("{s}.linear_out.weight"))?);
        }
        Ok(Self {
            norm1: p.get((1, 1, DIM), "norm1.alpha")?.reshape(DIM)?,
            norm2: p.get((1, 1, DIM), "norm2.alpha")?.reshape(DIM)?,
            in_projs,
            out_projs,
            g_in,
            g_out,
        })
    }

    // [S, dim] -> [1, H, S, hd]
    fn heads(t: &Tensor, s: usize) -> Result<Tensor> {
        t.reshape((s, N_HEAD, HD))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    /// `mask`: the `[s, s]` additive causal mask - built ONCE per
    /// depformer forward by the caller instead of being rebuilt (host loop +
    /// upload) inside every one of the N_LAYERS layer calls.
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let s = x.shape().dims2()?.0;
        let scale = 1.0 / (HD as f32).sqrt();
        // -- self-attention (causal, per-step QKV/out weights, no rope) --
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = per_step_proj(&h, &self.in_projs)?; // [S, 3*DIM]
        let q = Self::heads(&qkv.narrow(1, 0, DIM)?, s)?;
        let k = Self::heads(&qkv.narrow(1, DIM, DIM)?, s)?;
        let v = Self::heads(&qkv.narrow(1, 2 * DIM, DIM)?, s)?;
        let att = sdpa(&q, &k, &v, Some(mask), false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, DIM))?;
        let att = per_step_proj(&att, &self.out_projs)?;
        let x = x.add(&att)?;
        // -- SiLU-GLU gating FFN (per-step weights) --
        let h = x.rms_norm(&self.norm2, RMS_EPS)?;
        let hin = per_step_proj(&h, &self.g_in)?; // [S, 2*HIDDEN]
        let gate = hin
            .narrow(1, 0, HIDDEN)?
            .silu()?
            .mul(&hin.narrow(1, HIDDEN, HIDDEN)?)?;
        let gout = per_step_proj(&gate, &self.g_out)?;
        x.add(&gout)
    }

    /// Streaming single-codebook step: `x: [1, DIM]` for position `cb`, using the slot's
    /// weights and a KV cache accumulated across the 32 codebooks of one main step. No
    /// rope, no cross-attn, no mask (the single query attends all cached earlier positions).
    fn forward_step(
        &self,
        x: &Tensor,
        lc: &mut DepLayerCache,
        cb: usize,
        _dev: &Device,
    ) -> Result<Tensor> {
        let slot = SCHEDULE[cb];
        let scale = 1.0 / (HD as f32).sqrt();
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = h.matmul_t(&self.in_projs[slot])?; // [1, 3*DIM]
        let q = Self::heads(&qkv.narrow(1, 0, DIM)?, 1)?;
        let k = Self::heads(&qkv.narrow(1, DIM, DIM)?, 1)?;
        let v = Self::heads(&qkv.narrow(1, 2 * DIM, DIM)?, 1)?;
        let kc = match lc.k.take() {
            None => k,
            Some(p) => Tensor::cat(&[&p, &k], 2)?,
        };
        let vc = match lc.v.take() {
            None => v,
            Some(p) => Tensor::cat(&[&p, &v], 2)?,
        };
        let att = sdpa(&q, &kc, &vc, None, false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, DIM))?
            .matmul_t(&self.out_projs[slot])?;
        lc.k = Some(kc);
        lc.v = Some(vc);
        let x = x.add(&att)?;
        let h = x
            .rms_norm(&self.norm2, RMS_EPS)?
            .matmul_t(&self.g_in[slot])?; // [1, 2*HIDDEN]
        let gate = h
            .narrow(1, 0, HIDDEN)?
            .silu()?
            .mul(&h.narrow(1, HIDDEN, HIDDEN)?)?;
        x.add(&gate.matmul_t(&self.g_out[slot])?)
    }
}

/// Per-layer streaming state for the depformer (self-attn K/V, reset each main step).
struct DepLayerCache {
    k: Option<Tensor>, // [1, H, cached, hd]
    v: Option<Tensor>,
}

/// Streaming KV cache for one main step's 32-codebook depformer unroll.
pub struct DepCache {
    layers: Vec<DepLayerCache>,
}

/// Apply per-position weights (one weight per `SCHEDULE[pos]`) to `x: [S, in]`,
/// grouping the contiguous runs that share a slot into a single GEMM each.
fn per_step_proj(x: &Tensor, weights: &[Tensor]) -> Result<Tensor> {
    let s = x.shape().dims2()?.0;
    let mut blocks: Vec<Tensor> = Vec::new();
    let mut i = 0;
    while i < s {
        let slot = SCHEDULE[i];
        let mut j = i;
        while j < s && SCHEDULE[j] == slot {
            j += 1;
        }
        blocks.push(x.narrow(0, i, j - i)?.matmul_t(&weights[slot])?);
        i = j;
    }
    let refs: Vec<&Tensor> = blocks.iter().collect();
    Tensor::cat(&refs, 0)
}

pub struct KyutaiDepformer {
    dep_in: Vec<Tensor>,    // 11 x [DIM, MAIN_DIM]
    text_emb: Tensor,       // [8001, 128]
    text_out1: Tensor,      // [DIM, 128]
    text_out2: Tensor,      // [DIM, 128]
    audio_emb: Vec<Tensor>, // 31 x [2049, 128]
    audio_lr: Vec<Tensor>,  // 31 x [DIM, 128] low-rank projection
    layers: Vec<Layer>,
    linears: Vec<Tensor>, // 32 x [AUDIO_CARD, DIM]
    device: Device,
}
impl KyutaiDepformer {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mut dep_in = Vec::with_capacity(N_SLOTS);
        for s in 0..N_SLOTS {
            dep_in.push(vb.get((DIM, MAIN_DIM), &format!("depformer_in.{s}.weight"))?);
        }
        let mut audio_emb = Vec::with_capacity(DEP_Q - 1);
        let mut audio_lr = Vec::with_capacity(DEP_Q - 1);
        for i in 0..DEP_Q - 1 {
            audio_emb.push(vb.get(
                (AUDIO_EMB_ROWS, EMB_DIM),
                &format!("depformer_emb.{i}.weight"),
            )?);
            audio_lr.push(vb.get(
                (DIM, EMB_DIM),
                &format!("depformer_emb.{i}.low_rank.weight"),
            )?);
        }
        let dt = vb.pp("depformer");
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(Layer::load(&dt, i)?);
        }
        let mut linears = Vec::with_capacity(DEP_Q);
        for cb in 0..DEP_Q {
            linears.push(vb.get((AUDIO_CARD, DIM), &format!("linears.{cb}.weight"))?);
        }
        Ok(Self {
            dep_in,
            text_emb: vb.get((TEXT_EMB_ROWS, EMB_DIM), "depformer_text_emb.weight")?,
            text_out1: vb.get((DIM, EMB_DIM), "depformer_text_emb.out1.weight")?,
            text_out2: vb.get((DIM, EMB_DIM), "depformer_text_emb.out2.weight")?,
            audio_emb,
            audio_lr,
            layers,
            linears,
            device,
        })
    }

    /// Demuxed text embedding (cb==0 input token). `token: [1]` u32 -> `[1, DIM]`.
    fn text_embed(&self, token: u32) -> Result<Tensor> {
        let card = TEXT_CARD as u32;
        let left = token % card;
        let left_t = Tensor::from_vec_u32(vec![left], vec![1])?.to_device(&self.device)?;
        let mut y = self
            .text_emb
            .index_select(&left_t, 0)?
            .matmul_t(&self.text_out1)?;
        if token / card >= 1 {
            let right = token / card - 1;
            let right_t = Tensor::from_vec_u32(vec![right], vec![1])?.to_device(&self.device)?;
            y = y.add(
                &self
                    .text_emb
                    .index_select(&right_t, 0)?
                    .matmul_t(&self.text_out2)?,
            )?;
        }
        Ok(y)
    }

    /// Embed one depformer input position `cb` (0..DEP_Q-1) given its input token.
    /// `tout: [1, MAIN_DIM]`. cb0 token = text (demux emb); cb>0 = prev audio codebook.
    fn embed_pos(&self, tout: &Tensor, cb: usize, token: u32) -> Result<Tensor> {
        let slot = SCHEDULE[cb];
        let proj = tout.matmul_t(&self.dep_in[slot])?; // [1, DIM]
        let tok_emb = if cb == 0 {
            self.text_embed(token)?
        } else {
            let id = Tensor::from_vec_u32(vec![token], vec![1])?.to_device(&self.device)?;
            self.audio_emb[cb - 1]
                .index_select(&id, 0)?
                .matmul_t(&self.audio_lr[cb - 1])?
        };
        proj.add(&tok_emb)
    }

    /// Autoregressive generation head: given `transformer_out` and the input tokens for
    /// positions `0..L-1` (`input_tokens[0]`=text, `input_tokens[j>0]`=sampled codebook
    /// j-1), run the depformer over that length-L prefix and return the logits
    /// `[AUDIO_CARD]` for the LAST position (codebook L-1). Re-runs the prefix each call
    /// (O(L²) over 32 positions - cheap) instead of a streaming KV cache.
    pub fn forward_prefix(
        &self,
        transformer_out: &Tensor,
        input_tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let l = input_tokens.len();
        assert!((1..=DEP_Q).contains(&l));
        let tout = transformer_out.reshape((1, MAIN_DIM))?;
        let mut rows: Vec<Tensor> = Vec::with_capacity(l);
        for (cb, &tok) in input_tokens.iter().enumerate() {
            rows.push(self.embed_pos(&tout, cb, tok)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let mut x = Tensor::cat(&refs, 0)?; // [L, DIM]
        let mask = causal_mask(l, f32::NEG_INFINITY, &self.device)?;
        for lyr in &self.layers {
            x = lyr.forward(&x, &mask)?;
        }
        x.narrow(0, l - 1, 1)?
            .matmul_t(&self.linears[l - 1])?
            .flatten_all()?
            .to_vec1_f32()
    }

    /// A fresh KV cache for one main step (the 32-codebook unroll is independent per step).
    pub fn new_cache(&self) -> DepCache {
        DepCache {
            layers: (0..self.layers.len())
                .map(|_| DepLayerCache { k: None, v: None })
                .collect(),
        }
    }

    /// Streaming autoregressive step: logits `[AUDIO_CARD]` for codebook `cb`, given its
    /// input token (`text_token` for cb 0, the previous sampled codebook otherwise) and the
    /// running `cache`. O(1) in prior codebooks vs `forward_prefix`'s O(cb) re-run.
    pub fn forward_step(
        &self,
        transformer_out: &Tensor,
        cb: usize,
        input_token: u32,
        cache: &mut DepCache,
    ) -> Result<Vec<f32>> {
        let tout = transformer_out.reshape((1, MAIN_DIM))?;
        let mut h = self.embed_pos(&tout, cb, input_token)?; // [1, DIM]
        for (l, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = l.forward_step(&h, lc, cb, &self.device)?;
        }
        h.matmul_t(&self.linears[cb])?.flatten_all()?.to_vec1_f32()
    }

    /// Build the length-32 depformer input sequence `[DEP_Q, DIM]`.
    fn build_input(&self, transformer_out: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let tout = transformer_out.reshape((1, MAIN_DIM))?;
        let mut rows: Vec<Tensor> = Vec::with_capacity(DEP_Q);
        for cb in 0..DEP_Q {
            let slot = SCHEDULE[cb];
            let proj = tout.matmul_t(&self.dep_in[slot])?; // [1, DIM]
            let tok_emb = if cb == 0 {
                self.text_embed(tokens[0])?
            } else {
                let id =
                    Tensor::from_vec_u32(vec![tokens[cb]], vec![1])?.to_device(&self.device)?;
                self.audio_emb[cb - 1]
                    .index_select(&id, 0)?
                    .matmul_t(&self.audio_lr[cb - 1])?
            };
            rows.push(proj.add(&tok_emb)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Run one main step: unroll the 32 codebooks and return per-codebook logits
    /// `[DEP_Q, AUDIO_CARD]`. `transformer_out: [MAIN_DIM]` is the LM post-out_norm
    /// hidden for this step; `tokens: [32]` are the input tokens (token[0] = text,
    /// token[cb>0] = previous audio codebook).
    pub fn forward(&self, transformer_out: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let (_, _, logits) = self.forward_debug(transformer_out, tokens)?;
        Ok(logits)
    }

    /// Same as `forward` but also returns the built input `[DEP_Q, DIM]` and the
    /// post-transformer output `[DEP_Q, DIM]` for parity isolation.
    pub fn forward_debug(
        &self,
        transformer_out: &Tensor,
        tokens: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (input, _, out, logits) = self.forward_layers(transformer_out, tokens)?;
        Ok((input, out, logits))
    }

    /// Returns `(input, per-layer outputs, final output, logits)` for parity isolation.
    pub fn forward_layers(
        &self,
        transformer_out: &Tensor,
        tokens: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>, Tensor, Tensor)> {
        assert_eq!(tokens.len(), DEP_Q);
        let input = self.build_input(transformer_out, tokens)?;
        let mut x = input.clone();
        let mut per_layer = Vec::with_capacity(N_LAYERS);
        let mask = causal_mask(DEP_Q, f32::NEG_INFINITY, &self.device)?;
        for l in &self.layers {
            x = l.forward(&x, &mask)?;
            per_layer.push(x.clone());
        }
        let mut logits: Vec<Tensor> = Vec::with_capacity(DEP_Q);
        for cb in 0..DEP_Q {
            let row = x.narrow(0, cb, 1)?; // [1, DIM]
            logits.push(row.matmul_t(&self.linears[cb])?); // [1, AUDIO_CARD]
        }
        let refs: Vec<&Tensor> = logits.iter().collect();
        Ok((input, per_layer, x, Tensor::cat(&refs, 0)?))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl KyutaiDepformer {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.layers.len())
    }
}
