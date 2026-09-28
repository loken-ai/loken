//! Kyutai `tts-1.6b-en_fr` - the Depformer (component 6 of the port).
//!
//! A small per-step autoregressive generator that unrolls the audio codebooks of ONE main
//! step as a causal sequence, one position per codebook. It is a streaming transformer
//! (causal, NO rope, NO cross-attn) with **weights_per_step**: each position selects one
//! weight slot through the schedule the checkpoint's `config.json` states, applied to the
//! attention in/out projections and the gating FFN.
//!
//! Position `cb`'s input = `depformer_in[slot](transformer_out) + emb(token[cb])`, where
//!   - cb == 0: `token` is the TEXT token, embedded by `depformer_text_emb`
//!     (demux_second_stream + out1/out2 projecting the low rank to the model width);
//!   - cb  > 0: `token` is the previous audio codebook, embedded by `depformer_emb[cb-1]`
//!     (lookup at the low rank, then projected to the model width).
//! Output per codebook: `logits[cb] = linears[cb](dep_output[cb])` (depformer_norms are
//! Identity). Each head produces one logit per codebook entry.

use crate::inference::model::kyutai::config::KyutaiConfig;
use crate::inference::model::sdpa::sdpa;
use crate::tensor::ops::causal_mask;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

const RMS_EPS: f32 = 1e-8;

/// The depformer's shape, read from the checkpoint's `config.json`.
#[derive(Debug, Clone)]
struct DepGeometry {
    /// Width of the main transformer's output that feeds `depformer_in`.
    main_dim: usize,
    dim: usize,
    heads: usize,
    layers: usize,
    /// Width of one half of the SiLU-GLU gate.
    hidden: usize,
    /// Weight slot of each codebook position; its length is the codebooks per step.
    schedule: Vec<usize>,
    /// Width of the embedding tables before their projection to `dim`.
    low_rank: usize,
    /// Entries of one codebook, and so logits per head.
    audio_card: usize,
    /// Rows of the text embedding, which is also the demux modulus.
    text_rows: usize,
}

impl DepGeometry {
    fn from_config(cfg: &KyutaiConfig) -> Result<Self> {
        let schedule = cfg.usizes(&["depformer_weights_per_step_schedule"])?;
        let codebooks = cfg.usize(&["dep_q"])?;
        if schedule.len() != codebooks {
            return Err(crate::tensor::Error(format!(
                "kyutai config.json: a schedule of {} slots for {codebooks} codebooks",
                schedule.len()
            )));
        }
        Ok(Self {
            main_dim: cfg.usize(&["dim"])?,
            dim: cfg.usize(&["depformer_dim"])?,
            heads: cfg.usize(&["depformer_num_heads"])?,
            layers: cfg.usize(&["depformer_num_layers"])?,
            hidden: KyutaiConfig::gate_width(cfg.usize(&["depformer_dim_feedforward"])?),
            schedule,
            low_rank: cfg.usize(&["depformer_low_rank_embeddings"])?,
            audio_card: cfg.usize(&["card"])?,
            text_rows: cfg.usize(&["text_card"])? + 1,
        })
    }

    /// Weight slots per layer: one past the highest the schedule names.
    fn slots(&self) -> usize {
        self.schedule.iter().max().map_or(0, |m| m + 1)
    }

    /// Codebooks unrolled per main step.
    fn codebooks(&self) -> usize {
        self.schedule.len()
    }
}

struct Layer {
    norm1: Tensor,          // rms alpha [dim]
    norm2: Tensor,          // rms alpha [dim]
    in_projs: Vec<Tensor>,  // slots x [3*dim, dim] fused QKV
    out_projs: Vec<Tensor>, // slots x [dim, dim]
    g_in: Vec<Tensor>,      // slots x [2*hidden, dim]
    g_out: Vec<Tensor>,     // slots x [dim, hidden]
    dim: usize,
    heads: usize,
    hidden: usize,
}
impl Layer {
    fn load(vb: &VarBuilder, i: usize, g: &DepGeometry) -> Result<Self> {
        let (dim, hidden, slots) = (g.dim, g.hidden, g.slots());
        let p = vb.pp(format!("layers.{i}"));
        let a = p.pp("self_attn");
        let gt = p.pp("gating");
        // The per-step attention weights are stored CONCATENATED along dim 0
        // (in_proj_weight [slots*3*dim, dim], out_proj.weight [slots*dim, dim]);
        // slice per slot. Gating keeps one module per slot.
        let in_proj = a.get((slots * 3 * dim, dim), "in_proj_weight")?;
        let out_proj = a.get((slots * dim, dim), "out_proj.weight")?;
        let mut in_projs = Vec::with_capacity(slots);
        let mut out_projs = Vec::with_capacity(slots);
        let mut g_in = Vec::with_capacity(slots);
        let mut g_out = Vec::with_capacity(slots);
        for s in 0..slots {
            in_projs.push(in_proj.narrow(0, s * 3 * dim, 3 * dim)?.contiguous()?);
            out_projs.push(out_proj.narrow(0, s * dim, dim)?.contiguous()?);
            g_in.push(gt.get((2 * hidden, dim), &format!("{s}.linear_in.weight"))?);
            g_out.push(gt.get((dim, hidden), &format!("{s}.linear_out.weight"))?);
        }
        Ok(Self {
            norm1: p.get((1, 1, dim), "norm1.alpha")?.reshape(dim)?,
            norm2: p.get((1, 1, dim), "norm2.alpha")?.reshape(dim)?,
            in_projs,
            out_projs,
            g_in,
            g_out,
            dim,
            heads: g.heads,
            hidden,
        })
    }
    fn head_dim(&self) -> usize {
        self.dim / self.heads.max(1)
    }

    // [S, dim] -> [1, H, S, hd]
    fn split_heads(&self, t: &Tensor, s: usize) -> Result<Tensor> {
        t.reshape((s, self.heads, self.head_dim()))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    /// `mask`: the `[s, s]` additive causal mask - built ONCE per
    /// depformer forward by the caller instead of being rebuilt (host loop +
    /// upload) inside every layer call.
    fn forward(&self, x: &Tensor, mask: &Tensor, schedule: &[usize]) -> Result<Tensor> {
        let (dim, hidden) = (self.dim, self.hidden);
        let s = x.shape().dims2()?.0;
        let scale = 1.0 / (self.head_dim() as f32).sqrt();
        // -- self-attention (causal, per-step QKV/out weights, no rope) --
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = per_step_proj(&h, &self.in_projs, schedule)?; // [S, 3*dim]
        let q = self.split_heads(&qkv.narrow(1, 0, dim)?, s)?;
        let k = self.split_heads(&qkv.narrow(1, dim, dim)?, s)?;
        let v = self.split_heads(&qkv.narrow(1, 2 * dim, dim)?, s)?;
        let att = sdpa(&q, &k, &v, Some(mask), false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, dim))?;
        let att = per_step_proj(&att, &self.out_projs, schedule)?;
        let x = x.add(&att)?;
        // -- SiLU-GLU gating FFN (per-step weights) --
        let h = x.rms_norm(&self.norm2, RMS_EPS)?;
        let hin = per_step_proj(&h, &self.g_in, schedule)?; // [S, 2*hidden]
        let gate = hin
            .narrow(1, 0, hidden)?
            .silu()?
            .mul(&hin.narrow(1, hidden, hidden)?)?;
        let gout = per_step_proj(&gate, &self.g_out, schedule)?;
        x.add(&gout)
    }

    /// Streaming single-codebook step: `x: [1, dim]` for one position, using weight `slot`
    /// and a KV cache accumulated across the codebooks of one main step. No rope, no
    /// cross-attn, no mask (the single query attends all cached earlier positions).
    fn forward_step(&self, x: &Tensor, lc: &mut DepLayerCache, slot: usize) -> Result<Tensor> {
        let (dim, hidden) = (self.dim, self.hidden);
        let scale = 1.0 / (self.head_dim() as f32).sqrt();
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = h.matmul_t(&self.in_projs[slot])?; // [1, 3*dim]
        let q = self.split_heads(&qkv.narrow(1, 0, dim)?, 1)?;
        let k = self.split_heads(&qkv.narrow(1, dim, dim)?, 1)?;
        let v = self.split_heads(&qkv.narrow(1, 2 * dim, dim)?, 1)?;
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
            .reshape((1, dim))?
            .matmul_t(&self.out_projs[slot])?;
        lc.k = Some(kc);
        lc.v = Some(vc);
        let x = x.add(&att)?;
        let h = x
            .rms_norm(&self.norm2, RMS_EPS)?
            .matmul_t(&self.g_in[slot])?; // [1, 2*hidden]
        let gate = h
            .narrow(1, 0, hidden)?
            .silu()?
            .mul(&h.narrow(1, hidden, hidden)?)?;
        x.add(&gate.matmul_t(&self.g_out[slot])?)
    }
}

/// Per-layer streaming state for the depformer (self-attn K/V, reset each main step).
struct DepLayerCache {
    k: Option<Tensor>, // [1, H, cached, hd]
    v: Option<Tensor>,
}

/// Streaming KV cache for one main step's depformer unroll.
pub struct DepCache {
    layers: Vec<DepLayerCache>,
}

/// Apply per-position weights (one weight per `schedule[pos]`) to `x: [S, in]`,
/// grouping the contiguous runs that share a slot into a single GEMM each.
fn per_step_proj(x: &Tensor, weights: &[Tensor], schedule: &[usize]) -> Result<Tensor> {
    let s = x.shape().dims2()?.0;
    let mut blocks: Vec<Tensor> = Vec::new();
    let mut i = 0;
    while i < s {
        let slot = schedule[i];
        let mut j = i;
        while j < s && schedule[j] == slot {
            j += 1;
        }
        blocks.push(x.narrow(0, i, j - i)?.matmul_t(&weights[slot])?);
        i = j;
    }
    let refs: Vec<&Tensor> = blocks.iter().collect();
    Tensor::cat(&refs, 0)
}

pub struct KyutaiDepformer {
    dep_in: Vec<Tensor>,    // slots x [dim, main_dim]
    text_emb: Tensor,       // [text_rows, low_rank]
    text_out1: Tensor,      // [dim, low_rank]
    text_out2: Tensor,      // [dim, low_rank]
    audio_emb: Vec<Tensor>, // (codebooks-1) x [audio_card+1, low_rank]
    audio_lr: Vec<Tensor>,  // (codebooks-1) x [dim, low_rank] low-rank projection
    layers: Vec<Layer>,
    linears: Vec<Tensor>, // codebooks x [audio_card, dim]
    g: DepGeometry,
    device: Device,
}
impl KyutaiDepformer {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let g = DepGeometry::from_config(&KyutaiConfig::read(path)?)?;
        let (dim, low_rank, codebooks) = (g.dim, g.low_rank, g.codebooks());
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mut dep_in = Vec::with_capacity(g.slots());
        for s in 0..g.slots() {
            dep_in.push(vb.get((dim, g.main_dim), &format!("depformer_in.{s}.weight"))?);
        }
        let mut audio_emb = Vec::with_capacity(codebooks - 1);
        let mut audio_lr = Vec::with_capacity(codebooks - 1);
        for i in 0..codebooks - 1 {
            audio_emb.push(vb.get(
                (g.audio_card + 1, low_rank),
                &format!("depformer_emb.{i}.weight"),
            )?);
            audio_lr.push(vb.get(
                (dim, low_rank),
                &format!("depformer_emb.{i}.low_rank.weight"),
            )?);
        }
        let dt = vb.pp("depformer");
        let mut layers = Vec::with_capacity(g.layers);
        for i in 0..g.layers {
            layers.push(Layer::load(&dt, i, &g)?);
        }
        let mut linears = Vec::with_capacity(codebooks);
        for cb in 0..codebooks {
            linears.push(vb.get((g.audio_card, dim), &format!("linears.{cb}.weight"))?);
        }
        Ok(Self {
            dep_in,
            text_emb: vb.get((g.text_rows, low_rank), "depformer_text_emb.weight")?,
            text_out1: vb.get((dim, low_rank), "depformer_text_emb.out1.weight")?,
            text_out2: vb.get((dim, low_rank), "depformer_text_emb.out2.weight")?,
            audio_emb,
            audio_lr,
            layers,
            linears,
            g,
            device,
        })
    }

    /// Demuxed text embedding (cb==0 input token). `token: [1]` u32 -> `[1, dim]`.
    fn text_embed(&self, token: u32) -> Result<Tensor> {
        let card = self.g.text_rows as u32;
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

    /// Embed one depformer input position `cb` given its input token.
    /// `tout: [1, main_dim]`. cb0 token = text (demux emb); cb>0 = prev audio codebook.
    fn embed_pos(&self, tout: &Tensor, cb: usize, token: u32) -> Result<Tensor> {
        let slot = self.g.schedule[cb];
        let proj = tout.matmul_t(&self.dep_in[slot])?; // [1, dim]
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
    /// `[audio_card]` for the LAST position (codebook L-1). Re-runs the prefix each call
    /// (O(L²) over the codebooks - cheap) instead of a streaming KV cache.
    pub fn forward_prefix(
        &self,
        transformer_out: &Tensor,
        input_tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let l = input_tokens.len();
        assert!((1..=self.g.codebooks()).contains(&l));
        let tout = transformer_out.reshape((1, self.g.main_dim))?;
        let mut rows: Vec<Tensor> = Vec::with_capacity(l);
        for (cb, &tok) in input_tokens.iter().enumerate() {
            rows.push(self.embed_pos(&tout, cb, tok)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let mut x = Tensor::cat(&refs, 0)?; // [L, dim]
        let mask = causal_mask(l, f32::NEG_INFINITY, &self.device)?;
        for lyr in &self.layers {
            x = lyr.forward(&x, &mask, &self.g.schedule)?;
        }
        x.narrow(0, l - 1, 1)?
            .matmul_t(&self.linears[l - 1])?
            .flatten_all()?
            .to_vec1_f32()
    }

    /// A fresh KV cache for one main step (the codebook unroll is independent per step).
    pub fn new_cache(&self) -> DepCache {
        DepCache {
            layers: (0..self.layers.len())
                .map(|_| DepLayerCache { k: None, v: None })
                .collect(),
        }
    }

    /// Streaming autoregressive step: logits `[audio_card]` for codebook `cb`, given its
    /// input token (`text_token` for cb 0, the previous sampled codebook otherwise) and the
    /// running `cache`. O(1) in prior codebooks vs `forward_prefix`'s O(cb) re-run.
    pub fn forward_step(
        &self,
        transformer_out: &Tensor,
        cb: usize,
        input_token: u32,
        cache: &mut DepCache,
    ) -> Result<Vec<f32>> {
        let tout = transformer_out.reshape((1, self.g.main_dim))?;
        let mut h = self.embed_pos(&tout, cb, input_token)?; // [1, dim]
        let slot = self.g.schedule[cb];
        for (l, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = l.forward_step(&h, lc, slot)?;
        }
        h.matmul_t(&self.linears[cb])?.flatten_all()?.to_vec1_f32()
    }

    /// Build the depformer input sequence `[codebooks, dim]`.
    fn build_input(&self, transformer_out: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let tout = transformer_out.reshape((1, self.g.main_dim))?;
        let rows = (0..self.g.codebooks())
            .map(|cb| self.embed_pos(&tout, cb, tokens[cb]))
            .collect::<Result<Vec<Tensor>>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Run one main step: unroll the codebooks and return per-codebook logits
    /// `[codebooks, audio_card]`. `transformer_out: [main_dim]` is the LM post-out_norm
    /// hidden for this step; `tokens: [codebooks]` are the input tokens (token[0] = text,
    /// token[cb>0] = previous audio codebook).
    pub fn forward(&self, transformer_out: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let (_, _, logits) = self.forward_debug(transformer_out, tokens)?;
        Ok(logits)
    }

    /// Same as `forward` but also returns the built input `[codebooks, dim]` and the
    /// post-transformer output `[codebooks, dim]` for parity isolation.
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
        let codebooks = self.g.codebooks();
        assert_eq!(tokens.len(), codebooks);
        let input = self.build_input(transformer_out, tokens)?;
        let mut x = input.clone();
        let mut per_layer = Vec::with_capacity(self.layers.len());
        let mask = causal_mask(codebooks, f32::NEG_INFINITY, &self.device)?;
        for l in &self.layers {
            x = l.forward(&x, &mask, &self.g.schedule)?;
            per_layer.push(x.clone());
        }
        let mut logits: Vec<Tensor> = Vec::with_capacity(codebooks);
        for cb in 0..codebooks {
            let row = x.narrow(0, cb, 1)?; // [1, dim]
            logits.push(row.matmul_t(&self.linears[cb])?); // [1, audio_card]
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
