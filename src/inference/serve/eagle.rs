//! EAGLE-1 speculative-decoding draft head - native-substrate port.
//!
//! Decode on a single request is memory-bandwidth-bound: every output token
//! re-reads all the target weights. Speculative decoding breaks that wall by
//! drafting K tokens cheaply and *verifying* them in ONE target forward pass
//! (same bandwidth as one token, but commits up to K+1). EAGLE-1 (Li et al.
//! 2024) is the strongest draft-free flavor for our setup: the draft is a tiny
//! trained head that runs at the FEATURE level and reuses the target's own
//! embedding table + LM head - so it shares the exact tokenizer (no
//! draft-model/tokenizer-match wall) and reaches 70-90% acceptance at ~0.3% of
//! the target's parameter cost. Originally built in four phases (measured
//! 1.71-1.82x held-out on deepcoder); removed when the substrate was written
//! here, and restored on the native `crate::tensor` substrate. The
//! generic_transformer integration hooks (`capture_feature`, `eagle_embed`,
//! pre-norm->logits, `forward_from_hidden`) survived that migration.
//!
//! Head architecture (one qwen2-style decoder layer):
//!   x       = fc( concat[ feature_t , embed(token_{t+1}) ] )      // 2h -> h
//!   x       = x + Attn( RMSNorm(x) )      (GQA + RoPE, causal over the draft)
//!   f_{t+1} = x + SwiGLU( RMSNorm(x) )    // the predicted next feature
//! then the caller maps f_{t+1} through the TARGET lm_head to sample token_{t+2}.
//!
//! `feature_t` is the target's last-layer hidden state (pre-LM-head). This
//! module owns ONLY the small trained head; the embedding and lm_head are the
//! target's and are applied by the caller (the draft/verify loop).
//!
//! INFERENCE-ONLY on the native substrate: training (the EAGLE feature+token
//! self-distillation loss) needs autograd, which the native substrate does not
//! provide - so training moves
//! out-of-tree (a Python head-trainer on captured (feature, next-token) pairs ->
//! safetensors), loaded here via the substrate's `VarBuilder` for inference.

use crate::tensor::layer::{Linear, RmsNorm};
use crate::tensor::ops;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor, D};

/// Geometry of one EAGLE decoder layer - mirrors the target's qwen2 attention
/// (so the predicted feature lands in the target's representation space).
#[derive(Debug, Clone)]
pub struct EagleConfig {
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    /// qwen2 uses non-interleaved RoPE (`rope`); llama/mistral use `rope_i`.
    pub use_rope_i: bool,
    /// qwen2 attention carries q/k/v biases; most llama-likes do not.
    pub qkv_bias: bool,
}

impl EagleConfig {
    /// Config for a deepseek-r1 / deepcoder (Qwen2-distill) target - the first
    /// EAGLE target (headline TP model; verified geometry in tp_model.rs).
    pub fn qwen2(
        hidden: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        intermediate: usize,
    ) -> Self {
        Self {
            hidden,
            n_head,
            n_kv_head,
            head_dim,
            intermediate,
            rms_eps: 1e-5,
            rope_base: 1e6,
            use_rope_i: false,
            qkv_bias: true,
        }
    }
}

const MAX_DRAFT_POS: usize = 4096;

/// Precompute RoPE cos/sin tables [MAX_DRAFT_POS, head_dim/2] (mirrors tp_model).
fn rope_tables(hd: usize, base: f32, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = hd / 2;
    let theta: Vec<f32> = crate::inference::model::rope::inverse_frequencies(hd, base);
    let theta = Tensor::new(theta.as_slice(), dev)?;
    let idx = Tensor::arange(0f32, MAX_DRAFT_POS as f32)?
        .to_device(dev)?
        .to_dtype(DType::F32)?
        .reshape((MAX_DRAFT_POS, 1))?;
    let ang = idx.matmul(&theta.reshape((1, half))?)?;
    Ok((ang.cos()?, ang.sin()?))
}

/// Per-draft KV cache for the head's single attention layer. One per in-flight
/// draft (the head itself is stateless/shared). Holds [n_kv, T, head_dim].
#[derive(Default)]
pub struct EagleCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl EagleCache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
    /// Number of tokens currently cached.
    pub fn len(&self) -> usize {
        self.k.as_ref().map(|t| t.dim(1).unwrap_or(0)).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `rope` / `rope_i` have the same signature; pick at build time.
type RopeFn = fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>;

pub struct EagleHead {
    fc: Linear, // [hidden] <- concat[feature, embed] ([2*hidden])
    attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ffn_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    cos: Tensor,
    sin: Tensor,
    cfg: EagleConfig,
}

impl EagleHead {
    /// Build the head from a `VarBuilder` (mmaped safetensors of a trained head).
    /// Weight names: fc, attn_norm, q_proj, k_proj, v_proj, o_proj, ffn_norm,
    /// gate_proj, up_proj, down_proj (qwen2 layout, `<name>.weight`/`.bias`).
    /// cos/sin are recomputed (not stored weights).
    pub fn new(cfg: EagleConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        let lin = |inp: usize, out: usize, name: &str, bias: bool| -> Result<Linear> {
            let p = vb.pp(name);
            let w = p.get((out, inp), "weight")?;
            let b = if bias {
                Some(p.get(out, "bias")?)
            } else {
                None
            };
            Linear::new(w, b)
        };
        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(
                vb.pp(name).get(h, "weight")?,
                cfg.rms_eps as f32,
            ))
        };
        let (cos, sin) = rope_tables(cfg.head_dim, cfg.rope_base, &dev)?;
        Ok(Self {
            fc: lin(2 * h, h, "fc", false)?,
            attn_norm: norm("attn_norm")?,
            q_proj: lin(h, qd, "q_proj", cfg.qkv_bias)?,
            k_proj: lin(h, kvd, "k_proj", cfg.qkv_bias)?,
            v_proj: lin(h, kvd, "v_proj", cfg.qkv_bias)?,
            o_proj: lin(qd, h, "o_proj", false)?,
            ffn_norm: norm("ffn_norm")?,
            gate_proj: lin(h, cfg.intermediate, "gate_proj", false)?,
            up_proj: lin(h, cfg.intermediate, "up_proj", false)?,
            down_proj: lin(cfg.intermediate, h, "down_proj", false)?,
            cos: cos.to_dtype(dtype)?,
            sin: sin.to_dtype(dtype)?,
            cfg,
        })
    }

    fn rope_fn(&self) -> RopeFn {
        if self.cfg.use_rope_i {
            ops::rope_i
        } else {
            ops::rope
        }
    }

    /// One EAGLE step over a draft sequence of length `seq` starting at absolute
    /// position `pos` (causal). Inputs are [seq, hidden]; returns the predicted
    /// next-feature sequence [seq, hidden] (caller applies the target lm_head).
    /// No KV cache - full causal forward (used for parity vs `forward_step`).
    pub fn forward(&self, feature: &Tensor, embed_next: &Tensor, pos: usize) -> Result<Tensor> {
        let (seq, h) = feature.dims2()?;
        debug_assert_eq!(h, self.cfg.hidden);
        let (nh, nkv, hd) = (self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim);

        let x = Tensor::cat(&[feature, embed_next], D::Minus1)?; // [seq, 2h]
        let x = self.fc.forward(&x)?; // [seq, h]

        // -- attention sublayer --
        let residual = x.clone();
        let hn = self.attn_norm.forward(&x.contiguous()?)?;
        let q = self
            .q_proj
            .forward(&hn)?
            .reshape((seq, nh, hd))?
            .transpose(0, 1)?; // [nh, seq, hd]
        let k = self
            .k_proj
            .forward(&hn)?
            .reshape((seq, nkv, hd))?
            .transpose(0, 1)?; // [nkv, seq, hd]
        let v = self
            .v_proj
            .forward(&hn)?
            .reshape((seq, nkv, hd))?
            .transpose(0, 1)?; // [nkv, seq, hd]

        let crow = self.cos.narrow(0, pos, seq)?; // [seq, hd/2]
        let srow = self.sin.narrow(0, pos, seq)?;
        let rope_fn = self.rope_fn();
        let rope = |t: &Tensor| -> Result<Tensor> {
            Ok(rope_fn(&t.unsqueeze(0)?.contiguous()?, &crow, &srow)?.squeeze(0)?)
        };
        let q = rope(&q)?; // [nh, seq, hd]
        let k = rope(&k)?;

        let rep = nh / nkv;
        let k = crate::tensor::ops::repeat_kv_unbatched(&k, rep)?; // [nh, seq, hd]
        let v = crate::tensor::ops::repeat_kv_unbatched(&v, rep)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.contiguous()?.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh, seq, seq]
        let mask = ops::causal_mask(seq, f32::NEG_INFINITY, &feature.device())?
            .to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&mask)?;
        let probs = ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v.contiguous()?)?; // [nh, seq, hd]
        let ctx = ctx.transpose(0, 1)?.reshape((seq, nh * hd))?; // [seq, nh*hd]
        let x = (residual + self.o_proj.forward(&ctx)?)?;

        // -- SwiGLU FFN sublayer --
        let residual = x.clone();
        let hn = self.ffn_norm.forward(&x.contiguous()?)?;
        let gate = self.gate_proj.forward(&hn)?.silu()?;
        let up = self.up_proj.forward(&hn)?;
        let ffn = self.down_proj.forward(&(gate * up)?)?;
        let x = (residual + ffn)?;
        Ok(x) // predicted features [seq, h]
    }

    /// Incremental single-token draft step with a KV cache - the per-step
    /// primitive of the autoregressive draft loop. `feature`/`embed_next` are
    /// [1, hidden]; appends to `cache` and attends over all cached positions
    /// (causal by construction). Returns the predicted feature [1, hidden].
    pub fn forward_step(
        &self,
        cache: &mut EagleCache,
        feature: &Tensor,
        embed_next: &Tensor,
        pos: usize,
    ) -> Result<Tensor> {
        let (nh, nkv, hd) = (self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim);

        let x = Tensor::cat(&[feature, embed_next], D::Minus1)?; // [1, 2h]
        let x = self.fc.forward(&x)?; // [1, h]

        let residual = x.clone();
        let hn = self.attn_norm.forward(&x.contiguous()?)?;
        let q = self
            .q_proj
            .forward(&hn)?
            .reshape((1, nh, hd))?
            .transpose(0, 1)?; // [nh, 1, hd]
        let k = self
            .k_proj
            .forward(&hn)?
            .reshape((1, nkv, hd))?
            .transpose(0, 1)?; // [nkv, 1, hd]
        let v = self
            .v_proj
            .forward(&hn)?
            .reshape((1, nkv, hd))?
            .transpose(0, 1)?; // [nkv, 1, hd]

        let crow = self.cos.narrow(0, pos, 1)?; // [1, hd/2]
        let srow = self.sin.narrow(0, pos, 1)?;
        let rope_fn = self.rope_fn();
        let rope = |t: &Tensor| -> Result<Tensor> {
            Ok(rope_fn(&t.unsqueeze(0)?.contiguous()?, &crow, &srow)?.squeeze(0)?)
        };
        let q = rope(&q)?; // [nh, 1, hd]
        let k = rope(&k)?; // [nkv, 1, hd]

        // Append the new K/V to the cache -> [nkv, T, hd].
        let k_all = match &cache.k {
            None => k.clone(),
            Some(c) => Tensor::cat(&[c, &k], 1)?,
        };
        let v_all = match &cache.v {
            None => v.clone(),
            Some(c) => Tensor::cat(&[c, &v], 1)?,
        };
        cache.k = Some(k_all.clone());
        cache.v = Some(v_all.clone());

        let rep = nh / nkv;
        let k_all = crate::tensor::ops::repeat_kv_unbatched(&k_all, rep)?; // [nh, T, hd]
        let v_all = crate::tensor::ops::repeat_kv_unbatched(&v_all, rep)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q
            .contiguous()?
            .matmul(&k_all.transpose(1, 2)?.contiguous()?)?
            * scale)?; // [nh, 1, T]
        let probs = ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_all.contiguous()?)?; // [nh, 1, hd]
        let ctx = ctx.transpose(0, 1)?.reshape((1, nh * hd))?; // [1, nh*hd]
        let x = (residual + self.o_proj.forward(&ctx)?)?;

        let residual = x.clone();
        let hn = self.ffn_norm.forward(&x.contiguous()?)?;
        let gate = self.gate_proj.forward(&hn)?.silu()?;
        let up = self.up_proj.forward(&hn)?;
        let ffn = self.down_proj.forward(&(gate * up)?)?;
        let x = (residual + ffn)?;
        Ok(x) // [1, h]
    }

    /// Greedy chain draft of `k` tokens from the target's feature at the last
    /// committed token (position `pos`). The first draft is free - the target's
    /// own greedy next token `sample(lm_head(f_pos))` - and each subsequent draft
    /// is predicted by the head. `embed` and `sample` are the TARGET's embedding
    /// lookup and (lm_head->argmax), supplied by the verify loop so the head shares
    /// the target's exact tokenizer. Returns `k` drafted token ids.
    pub fn draft_chain(
        &self,
        init_feature: &Tensor, // [1, hidden] target feature at `pos`
        pos: usize,
        k: usize,
        embed: impl Fn(u32) -> Result<Tensor>, // token id -> [1, hidden]
        sample: impl Fn(&Tensor) -> Result<u32>, // [1, hidden] feature -> token id
    ) -> Result<Vec<u32>> {
        let mut cache = EagleCache::new();
        let mut feature = init_feature.clone(); // f_pos (from the target)
        let mut token = sample(&feature)?; // tok_{pos+1} = lm_head(f_pos) - free
        let mut drafts = Vec::with_capacity(k);
        drafts.push(token);
        for i in 1..k {
            let e = embed(token)?; // embed(tok_{pos+i})
            let pred = self.forward_step(&mut cache, &feature, &e, pos + i)?; // f_{pos+i}
            token = sample(&pred)?; // tok_{pos+i+1}
            drafts.push(token);
            feature = pred;
        }
        Ok(drafts)
    }
}

/// Drives the EAGLE speculative-decode loop over a target, parameterized by the
/// target's ops as closures (so the loop is unit-testable without a live model).
/// Output is provably the target's greedy decode (`verify_accept` guarantees it);
/// the head only changes how MANY tokens commit per target forward (the speedup).
pub struct EagleDecoder {
    head: EagleHead,
    k: usize,
}

impl EagleDecoder {
    pub fn new(head: EagleHead, k: usize) -> Self {
        Self { head, k }
    }
    pub fn k(&self) -> usize {
        self.k
    }

    /// One decode cycle from `(feature, pos)`. Closures:
    ///  - `embed(tok) -> [1,h]`        : target token embedding
    ///  - `sample(feat) -> tok`        : target lm_head -> argmax (drives the draft)
    ///  - `verify(drafts) -> (greedy[K+1], feats[K+1])` : ONE target forward over the
    ///    draft sequence.
    /// Returns `(committed, next_feature, next_pos)`; `committed` is `n_accepted+1`
    /// guaranteed-correct target tokens.
    pub fn step(
        &self,
        feature: &Tensor,
        pos: usize,
        embed: impl Fn(u32) -> Result<Tensor>,
        sample: impl Fn(&Tensor) -> Result<u32>,
        verify: impl Fn(&[u32]) -> Result<(Vec<u32>, Vec<Tensor>)>,
    ) -> Result<(Vec<u32>, Tensor, usize)> {
        let drafts = self
            .head
            .draft_chain(feature, pos, self.k, &embed, &sample)?;
        let (greedy, feats) = verify(&drafts)?; // [K+1], [K+1]
        let (committed, n) = verify_accept(&drafts, &greedy);
        let next_feature = if n >= 1 {
            feats[n - 1].clone()
        } else {
            feature.clone()
        };
        let next_pos = pos + committed.len();
        Ok((committed, next_feature, next_pos))
    }
}

/// EAGLE-1 greedy verification (the speculative-decode accept step).
///
/// `target_next[i]` is the target's greedy argmax having seen the committed token
/// plus `drafts[0..i]`. Accept the longest prefix where `drafts[i] ==
/// target_next[i]`, then append the target's own token at the first divergence
/// (`target_next[n]`), always correct. Makes spec-decode output GREEDY-IDENTICAL
/// to plain decode. Returns `(committed, n_accepted)` with `n_accepted + 1` tokens.
pub fn verify_accept(drafts: &[u32], target_next: &[u32]) -> (Vec<u32>, usize) {
    let k = drafts.len();
    debug_assert!(target_next.len() >= k + 1, "need K+1 target greedy tokens");
    let mut n = 0;
    while n < k && drafts[n] == target_next[n] {
        n += 1;
    }
    let mut committed: Vec<u32> = drafts[..n].to_vec();
    committed.push(target_next[n]); // correction / bonus token - always valid
    (committed, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build a random-initialized head for tests (training is out-of-tree, so the
    /// old VarMap path is gone): synthesize each weight tensor and load via a
    /// `from_tensors` VarBuilder - the same constructor the serving path uses.
    fn random_head(cfg: EagleConfig, dev: &Device) -> EagleHead {
        let h = cfg.hidden;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, rows: usize, cols: usize| {
            m.insert(
                name.to_string(),
                Tensor::randn(0f32, 0.02, (rows, cols), dev).unwrap(),
            );
        };
        put("fc.weight", h, 2 * h);
        put("q_proj.weight", qd, h);
        put("k_proj.weight", kvd, h);
        put("v_proj.weight", kvd, h);
        put("o_proj.weight", h, qd);
        put("gate_proj.weight", cfg.intermediate, h);
        put("up_proj.weight", cfg.intermediate, h);
        put("down_proj.weight", h, cfg.intermediate);
        if cfg.qkv_bias {
            m.insert(
                "q_proj.bias".into(),
                Tensor::randn(0f32, 0.02, (qd,), dev).unwrap(),
            );
            m.insert(
                "k_proj.bias".into(),
                Tensor::randn(0f32, 0.02, (kvd,), dev).unwrap(),
            );
            m.insert(
                "v_proj.bias".into(),
                Tensor::randn(0f32, 0.02, (kvd,), dev).unwrap(),
            );
        }
        // RMSNorm weights ~1.0.
        m.insert(
            "attn_norm.weight".into(),
            Tensor::ones((h,), DType::F32, dev).unwrap(),
        );
        m.insert(
            "ffn_norm.weight".into(),
            Tensor::ones((h,), DType::F32, dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        EagleHead::new(cfg, vb).unwrap()
    }

    #[test]
    fn eagle_head_forward_shapes() {
        let dev = Device::Cpu;
        let cfg = EagleConfig::qwen2(5120, 40, 8, 128, 27648);
        let head = random_head(cfg, &dev);
        let seq = 5;
        let feature = Tensor::randn(0f32, 1.0, (seq, 5120), &dev).unwrap();
        let embed = Tensor::randn(0f32, 1.0, (seq, 5120), &dev).unwrap();
        let out = head.forward(&feature, &embed, 0).unwrap();
        assert_eq!(out.dims(), &[seq, 5120]);
        let m = out
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max_keepdim(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite(), "EAGLE forward produced non-finite output");
    }

    #[test]
    fn eagle_step_matches_full() {
        let dev = Device::Cpu;
        let cfg = EagleConfig::qwen2(256, 8, 2, 32, 512);
        let head = random_head(cfg, &dev);
        let seq = 6;
        let features = Tensor::randn(0f32, 1.0, (seq, 256), &dev).unwrap();
        let embeds = Tensor::randn(0f32, 1.0, (seq, 256), &dev).unwrap();

        let full = head.forward(&features, &embeds, 0).unwrap(); // [seq, 256]

        let mut cache = EagleCache::new();
        let mut rows: Vec<Tensor> = Vec::new();
        for t in 0..seq {
            let f = features.narrow(0, t, 1).unwrap(); // [1, 256]
            let e = embeds.narrow(0, t, 1).unwrap();
            rows.push(head.forward_step(&mut cache, &f, &e, t).unwrap());
        }
        let inc = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap(); // [seq, 256]

        let diff = (full - inc)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max_keepdim(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-3, "incremental vs full max abs diff = {diff}");
        assert_eq!(cache.len(), seq);
    }

    #[test]
    fn eagle_draft_chain_runs() {
        let dev = Device::Cpu;
        let (h, vocab) = (256usize, 100usize);
        let cfg = EagleConfig::qwen2(h, 8, 2, 32, 512);
        let head = random_head(cfg, &dev);

        let embed_table = Tensor::randn(0f32, 0.02, (vocab, h), &dev).unwrap();
        let lm_head_w = Tensor::randn(0f32, 0.02, (vocab, h), &dev).unwrap(); // [vocab, h]
        let embed = |t: u32| -> Result<Tensor> { embed_table.narrow(0, t as usize, 1) };
        let sample = |f: &Tensor| -> Result<u32> {
            let logits = f.matmul_t(&lm_head_w)?; // [1, vocab]
            Ok(logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0])
        };

        let f_pos = Tensor::randn(0f32, 1.0, (1, h), &dev).unwrap();
        let k = 5;
        let drafts = head.draft_chain(&f_pos, 10, k, embed, sample).unwrap();
        assert_eq!(drafts.len(), k);
        assert!(
            drafts.iter().all(|&t| (t as usize) < vocab),
            "draft token out of vocab"
        );
    }

    #[test]
    fn eagle_verify_accept() {
        let (c, n) = verify_accept(&[5, 7, 9, 2], &[5, 7, 3, 1, 4]);
        assert_eq!(n, 2);
        assert_eq!(c, vec![5, 7, 3]); // 2 accepted drafts + target correction (3)

        let (c, n) = verify_accept(&[5, 7], &[5, 7, 8]);
        assert_eq!(n, 2);
        assert_eq!(c, vec![5, 7, 8]);

        let (c, n) = verify_accept(&[5], &[3, 9]);
        assert_eq!(n, 0);
        assert_eq!(c, vec![3]);
    }

    #[test]
    fn eagle_decoder_greedy_equivalent() {
        // The WHOLE EAGLE loop must reproduce the target's greedy decode EXACTLY,
        // regardless of the (random/untrained) head - verify_accept guarantees it.
        let dev = Device::Cpu;
        let (h, vocab) = (64usize, 50u32);
        let cfg = EagleConfig::qwen2(h, 4, 2, 16, 128);
        let head = random_head(cfg, &dev);
        let k = 4;
        let dec = EagleDecoder::new(head, k);

        let greedy_next =
            |prefix: &[u32]| -> u32 { (prefix.last().copied().unwrap_or(0) + 7) % vocab };

        let n_out = 20usize;
        let mut truth = Vec::new();
        {
            let mut p = vec![0u32];
            for _ in 0..n_out {
                let t = greedy_next(&p);
                truth.push(t);
                p.push(t);
            }
        }

        let embed = |_t: u32| -> Result<Tensor> { Tensor::zeros_on((1, h), DType::F32, &dev) };
        let sample = |_f: &Tensor| -> Result<u32> { Ok(0u32) };

        let mut out: Vec<u32> = Vec::new();
        let mut prefix = vec![0u32];
        let mut feature = Tensor::zeros_on((1, h), DType::F32, &dev).unwrap();
        let mut pos = 0usize;
        while out.len() < n_out {
            let prefix_now = prefix.clone();
            let verify = |drafts: &[u32]| -> Result<(Vec<u32>, Vec<Tensor>)> {
                let mut greedy = Vec::with_capacity(drafts.len() + 1);
                let mut feats = Vec::with_capacity(drafts.len() + 1);
                let mut ctx = prefix_now.clone();
                greedy.push(greedy_next(&ctx));
                feats.push(Tensor::zeros_on((1, h), DType::F32, &dev)?);
                for &d in drafts {
                    ctx.push(d);
                    greedy.push(greedy_next(&ctx));
                    feats.push(Tensor::zeros_on((1, h), DType::F32, &dev)?);
                }
                Ok((greedy, feats))
            };
            let (committed, next_feature, next_pos) =
                dec.step(&feature, pos, embed, sample, verify).unwrap();
            assert!(
                !committed.is_empty(),
                "a cycle must commit at least the bonus token"
            );
            out.extend(&committed);
            prefix.extend(&committed);
            feature = next_feature;
            pos = next_pos;
        }
        out.truncate(n_out);
        assert_eq!(
            out, truth,
            "EAGLE loop output must equal the target greedy decode"
        );
    }
}
