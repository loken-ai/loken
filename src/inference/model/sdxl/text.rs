//! SDXL text conditioning: the two CLIP towers, combined the way the reference does.
//!
//! The settings below were READ OFF a working reference rather than guessed, because each
//! of them silently changes the image rather than failing:
//!
//! - both towers are read at their PENULTIMATE hidden state (`layer_idx = -2`) and
//!   WITHOUT the final layer norm (`layer_norm_hidden_state = False`);
//! - the context is `cat([clip_l, clip_g], dim = -1)` -> 768 + 1280 = 2048;
//! - the pooled vector is bigG's ALONE, and it is PROJECTED through
//!   `text_projection` (`return_projected_pooled = True`);
//! - the towers pad differently: CLIP-L pads with the end token, bigG pads with 0.
//!
//! CLIP-L needs no new code: SDXL single-file checkpoints store it under the
//! HuggingFace names our `native_clip` already reads. bigG is the same
//! architecture (1280 wide, 32 layers, 20 heads, GELU) but ships in the open_clip
//! layout, whose attention keeps ONE fused `in_proj` instead of three projections -
//! so its tower lives here, and the fusion is kept (one GEMM instead of three).

use crate::inference::model::clip::text::{Config as ClipConfig, Transformer as ClipTransformer};
use crate::tensor::layer::{layer_norm, LayerNorm, Linear};
use crate::tensor::ops::causal_mask;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// Sequence length both towers are trained at.
pub const CONTEXT_TOKENS: usize = 77;
/// CLIP-L width, bigG width, and the concatenated context width.
pub const CLIP_L_DIM: usize = 768;
pub const CLIP_G_DIM: usize = 1280;
pub const CONTEXT_DIM: usize = CLIP_L_DIM + CLIP_G_DIM;

/// The token id both towers use for end-of-text; CLIP-L also pads with it.
pub const EOS_TOKEN: u32 = 49407;
pub const BOS_TOKEN: u32 = 49406;

/// Where each tower lives inside an SDXL single-file checkpoint.
const L_PREFIX: &str = "conditioner.embedders.0.transformer.text_model";
const G_PREFIX: &str = "conditioner.embedders.1.model";

/// bigG's published config.
const G_LAYERS: usize = 32;
const G_HEADS: usize = 20;
const G_MLP: usize = 5120;

/// One open_clip residual attention block: `ln_1 -> attn -> +x -> ln_2 -> mlp -> +x`.
///
/// The fused `in_proj` is kept fused: q, k and v come out of a single GEMM and are
/// split afterwards, which is both what the checkpoint stores and cheaper than three
/// separate projections.
struct GBlock {
    ln1: LayerNorm,
    ln2: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    fc: Linear,
    proj: Linear,
}

impl GBlock {
    fn new(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(CLIP_G_DIM, 1e-5, &vb.pp("ln_1"))?,
            ln2: layer_norm(CLIP_G_DIM, 1e-5, &vb.pp("ln_2"))?,
            in_proj: fused_linear(3 * CLIP_G_DIM, CLIP_G_DIM, &vb.pp("attn"), "in_proj")?,
            out_proj: named_linear(CLIP_G_DIM, CLIP_G_DIM, &vb.pp("attn").pp("out_proj"))?,
            fc: named_linear(G_MLP, CLIP_G_DIM, &vb.pp("mlp").pp("c_fc"))?,
            proj: named_linear(CLIP_G_DIM, G_MLP, &vb.pp("mlp").pp("c_proj"))?,
        })
    }

    /// `xs`: `[b, seq, dim]`, `mask`: additive causal mask `[1, 1, seq, seq]`.
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.shape().dims3()?;
        let hd = CLIP_G_DIM / G_HEADS;

        let h = self.ln1.forward(xs)?;
        let qkv = self.in_proj.forward(&h)?;
        // [b, seq, 3*dim] -> three [b, heads, seq, head_dim]
        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(2, i * CLIP_G_DIM, CLIP_G_DIM)?
                .reshape(vec![b, seq, G_HEADS, hd])?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = scores.softmax_last_dim()?.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape(vec![b, seq, CLIP_G_DIM])?;
        let xs = xs.add(&self.out_proj.forward(&attn)?)?;

        let h = self.ln2.forward(&xs)?;
        // open_clip's bigG text tower uses plain GELU (not the quick approximation).
        let h = self.proj.forward(&self.fc.forward(&h)?.gelu()?)?;
        xs.add(&h)
    }
}

/// The bigG text tower: token + positional embeddings, 32 residual blocks, and the
/// output projection used for the pooled vector.
pub struct ClipG {
    token_embedding: Tensor,
    positional_embedding: Tensor,
    blocks: Vec<GBlock>,
    ln_final: LayerNorm,
    text_projection: Tensor,
    device: Device,
}

impl ClipG {
    fn new(vb: &VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(G_LAYERS);
        let vb_blocks = vb.pp("transformer").pp("resblocks");
        for i in 0..G_LAYERS {
            blocks.push(GBlock::new(&vb_blocks.pp(i.to_string()))?);
        }
        Ok(Self {
            token_embedding: vb
                .pp("token_embedding")
                .get((49408, CLIP_G_DIM), "weight")?,
            positional_embedding: vb.get((CONTEXT_TOKENS, CLIP_G_DIM), "positional_embedding")?,
            blocks,
            ln_final: layer_norm(CLIP_G_DIM, 1e-5, &vb.pp("ln_final"))?,
            // open_clip stores this as a bare `[dim, dim]` tensor, applied on the right.
            text_projection: vb.get((CLIP_G_DIM, CLIP_G_DIM), "text_projection")?,
            device: vb.device().clone(),
        })
    }

    /// Penultimate hidden state `[1, seq, 1280]` (no final norm) and the PROJECTED
    /// pooled vector `[1, 1280]` taken at the EOS position after the final norm.
    pub fn forward(&self, ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let seq = ids.len();
        let idt = Tensor::from_vec_u32(ids.to_vec(), vec![seq])?.to_device(&self.device)?;
        let emb = self
            .token_embedding
            .index_select(&idt, 0)?
            .reshape(vec![1, seq, CLIP_G_DIM])?;
        let pos = self
            .positional_embedding
            .narrow(0, 0, seq)?
            .reshape(vec![1, seq, CLIP_G_DIM])?;
        let mut xs = emb.add(&pos)?;

        // Built in F32; it is ADDED to the scores, so it has to arrive in whatever dtype
        // this tower is resident in.
        let mask = causal_mask(seq, f32::MIN, &self.device)?
            .reshape(vec![1, 1, seq, seq])?
            .to_dtype(xs.dtype())?;
        // Stop one block early: that hidden state IS the conditioning.
        let penultimate_at = self.blocks.len() - 1;
        let mut penultimate = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            if i == penultimate_at {
                penultimate = Some(xs.clone());
            }
            xs = blk.forward(&xs, &mask)?;
        }
        let hidden = penultimate.unwrap_or_else(|| xs.clone());

        // Pooled: EOS row of the FULL (final-normed) output, then projected.
        let out = self.ln_final.forward(&xs)?;
        let eos = ids
            .iter()
            .position(|t| *t == EOS_TOKEN)
            .unwrap_or(seq.saturating_sub(1));
        let row = out.narrow(1, eos, 1)?.reshape(vec![1, CLIP_G_DIM])?;
        let pooled = row.matmul(&self.text_projection)?;
        Ok((hidden, pooled))
    }
}

/// `{prefix}.weight` `[out, in]` + `{prefix}.bias`.
fn named_linear(out_dim: usize, in_dim: usize, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = vb.get(out_dim, "bias")?;
    Linear::new(w, Some(b))
}

/// open_clip's fused attention input projection: `{name}_weight` / `{name}_bias`
/// rather than a `{name}.weight` sub-path.
fn fused_linear(out_dim: usize, in_dim: usize, vb: &VarBuilder, name: &str) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), &format!("{name}_weight"))?;
    let b = vb.get(out_dim, &format!("{name}_bias"))?;
    Linear::new(w, Some(b))
}

/// Both towers, resident.
pub struct SdxlTextEncoders {
    l: ClipTransformer,
    g: ClipG,
    device: Device,
}

impl SdxlTextEncoders {
    pub fn load(checkpoint: &str, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[checkpoint], dtype, device) }?;
        // CLIP-L is stored under the HuggingFace names `native_clip` already reads.
        let l_cfg = ClipConfig {
            vocab_size: 49408,
            embed_dim: CLIP_L_DIM,
            activation: crate::tensor::ops::Activation::QuickGelu,
            intermediate_size: 3072,
            max_position_embeddings: CONTEXT_TOKENS,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: CLIP_L_DIM,
        };
        Ok(Self {
            l: ClipTransformer::new(vb.pp(L_PREFIX), &l_cfg)?,
            g: ClipG::new(&vb.pp(G_PREFIX))?,
            device: device.clone(),
        })
    }

    /// Pad a token sequence to the context length the way each tower expects.
    ///
    /// CLIP-L pads with the END token and bigG pads with 0 - the reference sets
    /// `pad_with_end` per tower, and padding bigG with 49407 shifts its embedding.
    pub fn pad_tokens(tokens: &[u32], for_g: bool) -> Vec<u32> {
        let pad = if for_g { 0 } else { EOS_TOKEN };
        let mut v = Vec::with_capacity(CONTEXT_TOKENS);
        v.push(BOS_TOKEN);
        for t in tokens.iter().take(CONTEXT_TOKENS - 2) {
            v.push(*t);
        }
        v.push(EOS_TOKEN);
        while v.len() < CONTEXT_TOKENS {
            v.push(pad);
        }
        v
    }

    /// Encode ALREADY-TOKENIZED text (BPE ids without BOS/EOS) into the UNet's
    /// context `[1, 77, 2048]` and bigG's projected pooled vector `[1, 1280]`.
    pub fn encode(&self, tokens: &[u32]) -> Result<(Tensor, Tensor)> {
        let ids_l = Self::pad_tokens(tokens, false);
        let ids_g = Self::pad_tokens(tokens, true);
        let hl = self
            .l
            .hidden_state(&ids_l, 1, CONTEXT_TOKENS, 1)?
            .to_device(&self.device)?;
        let (hg, pooled) = self.g.forward(&ids_g)?;
        // cat on the FEATURE axis: L first, then G (the reference's order).
        let ctx = Tensor::cat(&[&hl, &hg], 2)?;
        Ok((ctx, pooled))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    #[test]
    fn the_context_width_is_the_sum_of_both_towers() {
        assert_eq!(CONTEXT_DIM, 2048);
    }

    /// The towers pad differently; getting bigG's pad wrong is silent.
    #[test]
    fn padding_differs_per_tower_and_always_brackets_with_bos_eos() {
        let toks = [100u32, 200, 300];
        let l = SdxlTextEncoders::pad_tokens(&toks, false);
        let g = SdxlTextEncoders::pad_tokens(&toks, true);
        assert_eq!(l.len(), CONTEXT_TOKENS);
        assert_eq!(g.len(), CONTEXT_TOKENS);
        assert_eq!(l[0], BOS_TOKEN);
        assert_eq!(g[0], BOS_TOKEN);
        assert_eq!(l[4], EOS_TOKEN, "the EOS goes right after the prompt");
        assert_eq!(g[4], EOS_TOKEN);
        assert_eq!(
            *l.last().unwrap(),
            EOS_TOKEN,
            "CLIP-L pads with the end token"
        );
        assert_eq!(*g.last().unwrap(), 0, "bigG pads with zero");
    }

    /// An over-long prompt must still produce exactly one context window.
    #[test]
    fn a_long_prompt_is_truncated_to_the_context_window() {
        let toks: Vec<u32> = (0..500).collect();
        let v = SdxlTextEncoders::pad_tokens(&toks, false);
        assert_eq!(v.len(), CONTEXT_TOKENS);
        assert_eq!(v[0], BOS_TOKEN);
        assert_eq!(*v.last().unwrap(), EOS_TOKEN);
    }

    /// Gate: load BOTH towers out of a real checkpoint and check the conditioning
    /// they produce. A missed name or a wrong width fails the load or the shape;
    /// a degenerate (all-equal) embedding would pass a shape-only check.
    #[test]
    #[ignore = "needs an SDXL checkpoint under the configured models dir"]
    fn both_towers_load_and_produce_a_2048_wide_context() {
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let ckpt = std::fs::read_dir(dir.join("raymnants"))
            .expect("raymnants dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .expect("an SDXL .safetensors");
        let dev = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .next()
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
        let enc = SdxlTextEncoders::load(ckpt.to_str().unwrap(), &dev, DType::F32)
            .expect("both towers load");

        // Arbitrary ids stand in for a tokenizer here: this gate is about the
        // NETWORK plumbing, not the BPE.
        let (ctx, pooled) = enc.encode(&[320u32, 1125, 539]).expect("encode");
        assert_eq!(
            ctx.dims(),
            &[1, CONTEXT_TOKENS, CONTEXT_DIM],
            "context shape"
        );
        assert_eq!(pooled.dims(), &[1, CLIP_G_DIM], "pooled shape");
        let c = ctx.to_device(&Device::Cpu).unwrap().to_vec_f32();
        let p = pooled.to_device(&Device::Cpu).unwrap().to_vec_f32();
        assert!(
            c.iter().all(|x| x.is_finite()),
            "context has non-finite values"
        );
        assert!(
            p.iter().all(|x| x.is_finite()),
            "pooled has non-finite values"
        );
        // Both halves must carry signal: a tower that failed to load its weights
        // would produce a constant slab.
        let half = CONTEXT_DIM / 2;
        let spread = |off: usize, n: usize| {
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for t in 0..CONTEXT_TOKENS {
                for d in 0..n {
                    let v = c[t * CONTEXT_DIM + off + d];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            hi - lo
        };
        let (sl, sg) = (spread(0, CLIP_L_DIM), spread(CLIP_L_DIM, CLIP_G_DIM));
        println!(
            "SDXL text: context {:?}, L spread {sl:.3}, G spread {sg:.3}",
            ctx.dims()
        );
        assert!(sl > 0.1, "the CLIP-L half is flat ({sl})");
        assert!(sg > 0.1, "the bigG half is flat ({sg})");
        let _ = half;
    }
}
