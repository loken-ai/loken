//! Module containing quantized MixFormer model implementation.
//!
//! MixFormer is the phi-2 architecture: attention and feed-forward run in PARALLEL on the
//! same input rather than in sequence, so one layer norm feeds both and their outputs are
//! summed. It is NOT a mixture-of-experts, whatever this comment used to say - there is no
//! router, no gate and no top-k anywhere in the file, and someone looking for expert
//! routing was being sent to the wrong place.
//! This implementation provides quantization for reduced memory usage.
//!
//! Key features:
//! - Parallel attention and feed-forward computation
//! - Rotary positional embeddings
//! - Optional key-value caching
//! - Support for 8-bit quantization
//!

use crate::tensor::layer::qlinear::{
    q_layer_norm as layer_norm, qlinear as linear, QLinear as Linear, QMlp,
};
use crate::tensor::ops::Activation;
pub use crate::tensor::quantized::QVarBuilder as VarBuilder;
use crate::tensor::{DType, Device, IndexOp, Module, Result, Tensor, D};

/// Phi/MixFormer configuration, declared here with public
/// fields so the whole model lives in this crate.
///
/// Every field here is read to build the model. The published `config.json` carries three more
/// - a positional limit that nothing consults, a weight-tying flag this family never sets, and
/// a vocabulary padding multiple that nothing pads to - and a field no code reads is a claim
/// about the model that no code has to keep true.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    /// Width, depth, and the width the feed-forward widens to. `n_inner` absent means four
    /// times the width, which is what every published size uses.
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_inner: Option<usize>,
    pub n_head: usize,
    /// How much of each head the rotary turns. The rest passes through unturned - this family
    /// rotates a prefix rather than the whole head.
    pub rotary_dim: usize,
    pub activation_function: Activation,
    pub layer_norm_epsilon: f64,
}

impl Config {
    /// The two published sizes differ in their width and their depth, and in nothing else.
    ///
    /// The rotary dimension is not a third difference: it is the head width capped at
    /// thirty-two, and both sizes have thirty-two heads, so both land on thirty-two - writing
    /// it out per size invited them to drift.
    fn of_size(n_embd: usize, n_layer: usize) -> Self {
        const HEADS: usize = 32;
        Self {
            vocab_size: 51200,
            n_embd,
            n_layer,
            n_inner: None,
            n_head: HEADS,
            rotary_dim: usize::min(32, n_embd / HEADS),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn v1_5() -> Self {
        Self::of_size(2048, 24)
    }

    pub fn v2() -> Self {
        Self::of_size(2560, 32)
    }
}

const MAX_SEQ_LEN: usize = 4096;

/// The token embedding, under the name this checkpoint gives it.
///
/// It is the shared layer: a wrapper here held one field and forwarded to it, which is a type
/// whose whole content is where to look for the weights.
fn embedding(cfg: &Config, vb: VarBuilder) -> Result<crate::tensor::layer::Embedding> {
    crate::tensor::layer::qlinear::q_embedding(cfg.vocab_size, cfg.n_embd, &vb.pp("wte"))
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, dev: &Device) -> Result<Self> {
        let (cos, sin) =
            crate::inference::model::rope::precomput_freqs_cis_yarn(dim, 10000.0, None, max_seq_len, dev)?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        qkv: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, seqlen, three, _, headdim) = qkv.dims5()?;
        if three != 3 {
            crate::tensor::bail!("unexpected shape for qkv {:?}", qkv.shape())
        }
        let (_rotary_seqlen, rotary_dim_half) = self.cos.dims2()?;
        let rotary_dim = rotary_dim_half * 2;

        // The checkpoint fuses the three projections along axis 2 and stores them in the order
        // query, key, value. Selecting a position there drops the axis, leaving each operand as
        // `(batch, seq, heads, headdim)`.
        let operand = |which: usize| qkv.i((.., .., which));
        let (q, k, v) = (operand(0)?, operand(1)?, operand(2)?);

        // The tables span the longest context; a step reads the window its positions fall in,
        // carried to the operands' dtype so the rotation runs at a single precision.
        let window = |table: &Tensor| -> Result<Tensor> {
            let w = table.narrow(0, seqlen_offset, seqlen)?;
            if w.dtype() == q.dtype() {
                Ok(w)
            } else {
                w.to_dtype(q.dtype())
            }
        };
        let (c, s) = (window(&self.cos)?, window(&self.sin)?);

        // This family rotates only the first `rotary_dim` of each head and carries the rest
        // through untouched, so the rotation is applied to a slice and the tail is put back.
        // The shared op covers every device and dtype - a fused kernel where there is one, a
        // host pass where there is not - which is why there is no second chain here rotating
        // the pairs by hand. It wants the head axis ahead of the position axis, so each operand
        // is transposed in and back out.
        let turn = |x: &Tensor| -> Result<Tensor> {
            let xt = x.transpose(1, 2)?.contiguous()?;
            if rotary_dim == headdim {
                return crate::tensor::ops::rope(&xt, &c, &s)?.transpose(1, 2);
            }
            let turned = crate::tensor::ops::rope(
                &xt.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
                &c,
                &s,
            )?;
            let carried = xt.narrow(D::Minus1, rotary_dim, headdim - rotary_dim)?;
            Tensor::cat(&[&turned, &carried], D::Minus1)?.transpose(1, 2)
        };
        Ok((turn(&q)?, turn(&k)?, v))
    }
}

#[cfg(test)]
mod rotary_tests {
    use super::*;
    use crate::tensor::Device;

    /// The rotation reaches the head's first `rotary_dim` values and no further.
    ///
    /// This family rotates part of each head and carries the rest through; there used to be a
    /// second implementation of that to compare against, which proved only that the pair
    /// agreed. Held against the definition instead: pair `j` of the rotated part turns by the
    /// table's angle, and every value past `rotary_dim` comes back as it went in.
    #[test]
    fn a_partial_rotary_turns_the_head_it_covers_and_leaves_the_rest() {
        let (seq, heads, headdim, rotary_dim) = (3usize, 2usize, 8usize, 4usize);
        let rope = RotaryEmbedding::new(rotary_dim, 16, &Device::Cpu).unwrap();
        let n = seq * 3 * heads * headdim;
        let src: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.07 - 0.6).collect();
        let qkv = Tensor::from_vec(src.clone(), (1, seq, 3, heads, headdim), &Device::Cpu).unwrap();

        let (q, _k, v) = rope.apply_rotary_emb_qkv(&qkv, 0).unwrap();
        let got = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // v is handed back untouched, which is what says the split found the right operand.
        let vv = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let at = |which: usize, s: usize, h: usize, d: usize| {
            src[((s * 3 + which) * heads + h) * headdim + d]
        };
        let half = rotary_dim / 2;
        for s in 0..seq {
            for h in 0..heads {
                for d in 0..headdim {
                    let out = got[(s * heads + h) * headdim + d];
                    let want = if d >= rotary_dim {
                        at(0, s, h, d)
                    } else {
                        let j = d % half;
                        let w = 1f32 / 10_000f32.powf((2 * j) as f32 / rotary_dim as f32);
                        let (c, sn) = ((s as f32 * w).cos(), (s as f32 * w).sin());
                        let (lo, hi) = (at(0, s, h, j), at(0, s, h, j + half));
                        if d < half {
                            lo * c - hi * sn
                        } else {
                            hi * c + lo * sn
                        }
                    };
                    assert!(
                        (out - want).abs() < 1e-5,
                        "seq {s} head {h} channel {d}: {out} is not {want}"
                    );
                    assert_eq!(vv[(s * heads + h) * headdim + d], at(2, s, h, d));
                }
            }
        }
    }
}

/// The feed-forward, with landing places for the two intermediates a decode step produces.
///
/// Widen, activate, narrow is the shared [`QMlp`], and nothing about it is particular to this
/// family. What is particular is the pair of buffers: a captured decode graph records the
/// addresses its kernels read and write, so an intermediate allocated afresh each token would
/// have the replay reading somewhere the current step never wrote. The widened vector and the
/// activated vector are therefore written into buffers made once, at the width the widening
/// projection declares, and only on the path that is captured.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    net: QMlp,
    fc1_buf: Option<Tensor>,
    act_buf: Option<Tensor>,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_inner = cfg.n_inner.unwrap_or(4 * cfg.n_embd);
        Ok(Self {
            net: QMlp::new(
                linear(cfg.n_embd, n_inner, &vb.pp("fc1"))?,
                cfg.activation_function,
                linear(n_inner, cfg.n_embd, &vb.pp("fc2"))?,
            ),
            fc1_buf: None,
            act_buf: None,
        })
    }

    /// The same three steps, with the two intermediates parked in the stable buffers.
    ///
    /// Only a single-token step on a card qualifies: prefill is never captured, so it takes the
    /// plain path and pays nothing for buffers it would not reuse.
    fn forward_decode(&mut self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let use_buffers = xs.device().is_cuda() && dims.len() == 3 && dims[0] == 1 && dims[1] == 1;
        if !use_buffers {
            return self.net.forward(xs);
        }
        let widened = self.net.fc1().forward(xs)?;
        if self.fc1_buf.is_none() {
            let shape = (1, 1, self.net.hidden());
            self.fc1_buf = Some(Tensor::zeros_on(shape, widened.dtype(), &xs.device())?);
            self.act_buf = Some(Tensor::zeros_on(shape, widened.dtype(), &xs.device())?);
        }
        let fc1_buf = self.fc1_buf.as_ref().unwrap();
        fc1_buf.slice_set(&widened, 1, 0)?;
        let act_buf = self.act_buf.as_ref().unwrap();
        let activated = fc1_buf.apply(&self.net.act())?;
        act_buf.slice_set(&activated, 1, 0)?;
        act_buf.apply(self.net.fc2())
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.net.forward(xs)
    }
}

#[derive(Debug)]
struct CausalLMHead {
    ln: crate::tensor::layer::LayerNorm,
    linear: Linear,
}

impl CausalLMHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, &vb.pp("ln"))?,
            linear: linear(cfg.n_embd, cfg.vocab_size, &vb.pp("linear"))?,
        })
    }

    /// One position's hidden state turned into a score per vocabulary entry.
    ///
    /// The scores leave in F32 whatever the weights are held in: sampling compares them against
    /// each other, and a half-precision tail collapses distinctions that a temperature exists
    /// to act on.
    fn logits(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.ln.forward(xs)?;
        self.linear.forward(&normed)?.to_dtype(DType::F32)
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MHA {
    wqkv: Linear,
    out_proj: Linear,
    rotary_emb: RotaryEmbedding,
    // In-place KV cache: pre-allocated buffers + position cursor.
    // Avoids the O(N²) Tensor::cat per-decode-token re-alloc that the
    // old `kv_cache: Option<(Tensor, Tensor)>` introduced. Lazily
    // sized on first append; max bound is MAX_SEQ_LEN.
    kv_buf_k: Option<Tensor>,
    kv_buf_v: Option<Tensor>,
    kv_pos: usize,
    // Stable buffers for graph-safe decode.
    // qkv_buf: output of wqkv.forward([1,1,3*n_embd])
    // attn_out_buf: output of attn matmul before out_proj ([1,1,n_embd])
    qkv_buf: Option<Tensor>,
    attn_out_buf: Option<Tensor>,
    n_embd: usize,
    head_dim: usize,
    n_head: usize,
    softmax_scale: f64,
}

impl MHA {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.n_embd / cfg.n_head;
        Ok(Self {
            // One projection produces all three operands, which is how the checkpoint stores
            // them, so the width it emits is three times the model's.
            wqkv: linear(cfg.n_embd, 3 * cfg.n_embd, &vb.pp("Wqkv"))?,
            out_proj: linear(cfg.n_embd, cfg.n_embd, &vb.pp("out_proj"))?,
            rotary_emb: RotaryEmbedding::new(cfg.rotary_dim, MAX_SEQ_LEN, vb.device())?,
            kv_buf_k: None,
            kv_buf_v: None,
            kv_pos: 0,
            qkv_buf: None,
            attn_out_buf: None,
            n_embd: cfg.n_embd,
            head_dim,
            n_head: cfg.n_head,
            // The scores are divided by the square root of the head width, handed over as the
            // multiplier the attention takes.
            softmax_scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let _traced = tracing::trace_span!("mha").entered();
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let use_decode_bufs = xs.device().is_cuda() && b_size == 1 && seq_len == 1;
        let qkv_raw = self.wqkv.forward(xs)?;
        // Route wqkv output through stable qkv_buf for graph-safe decode.
        let qkv_routed = if use_decode_bufs {
            let qkv_dtype = qkv_raw.dtype();
            if self.qkv_buf.is_none() {
                self.qkv_buf = Some(Tensor::zeros_on(
                    (1, 1, 3 * self.n_embd),
                    qkv_dtype,
                    &xs.device(),
                )?);
            }
            let buf = self.qkv_buf.as_ref().unwrap();
            buf.slice_set(&qkv_raw, 1, 0)?;
            buf.clone()
        } else {
            qkv_raw
        };
        let qkv = qkv_routed.reshape((b_size, seq_len, 3, self.n_head, self.head_dim))?;
        let seqlen_offset = self.kv_pos;
        let (q, k_new, v_new) = self.rotary_emb.apply_rotary_emb_qkv(&qkv, seqlen_offset)?;

        // In-place KV append. Lazily allocate the max-seq buffer on first
        // call (saves O(N²) Tensor::cat per decode step).
        let n_kv_heads = k_new.dim(2)?;
        if self.kv_buf_k.is_none() {
            // Allocate to MAX_SEQ_LEN on the same device/dtype as k_new.
            let shape = (b_size, MAX_SEQ_LEN, n_kv_heads, self.head_dim);
            self.kv_buf_k = Some(Tensor::zeros_on(shape, k_new.dtype(), &k_new.device())?);
            self.kv_buf_v = Some(Tensor::zeros_on(shape, v_new.dtype(), &v_new.device())?);
            self.kv_pos = 0;
        }
        let buf_k = self.kv_buf_k.as_ref().unwrap();
        let buf_v = self.kv_buf_v.as_ref().unwrap();
        // Reset cache when the caller restarts a fresh prompt
        // (RotaryEmb's seqlen_offset==0 signals start).
        if seqlen_offset == 0 {
            self.kv_pos = 0;
        }
        // Write new tokens at [kv_pos..kv_pos+seq_len, :, :].
        buf_k.slice_set(&k_new, 1, self.kv_pos)?;
        buf_v.slice_set(&v_new, 1, self.kv_pos)?;
        self.kv_pos += seq_len;
        let k = buf_k.narrow(1, 0, self.kv_pos)?;
        let v = buf_v.narrow(1, 0, self.kv_pos)?;
        // Each head attends on its own, so the attention wants batch and head collapsed into a
        // single leading axis: `(batch, positions, heads, headdim)` becomes `(batch * heads,
        // positions, headdim)`. The query carries this step's positions, the key and the value
        // every position cached so far.
        let heads_first = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.flatten_to(1) };
        let (q, k, v) = (heads_first(&q)?, heads_first(&k)?, heads_first(&v)?);
        // The mask this used to build for itself is the causal one, and asking for it by the
        // two lengths means the prefill triangle and the decode identity are one call: at one
        // query against everything cached, nothing is masked.
        let attn_output = crate::inference::model::acestep::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            true,
            self.softmax_scale as f32,
            1.0,
        )?;
        // b*h,t,d
        let attn_output = attn_output
            .reshape((b_size, self.n_head, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        // Route attn_output through stable attn_out_buf for graph-safe
        // decode.
        let attn_routed = if use_decode_bufs {
            let aout_dtype = attn_output.dtype();
            if self.attn_out_buf.is_none() {
                self.attn_out_buf = Some(Tensor::zeros_on(
                    (1, 1, self.n_embd),
                    aout_dtype,
                    &xs.device(),
                )?);
            }
            let buf = self.attn_out_buf.as_ref().unwrap();
            buf.slice_set(&attn_output, 1, 0)?;
            buf.clone()
        } else {
            attn_output
        };
        attn_routed.apply(&self.out_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_buf_k = None;
        self.kv_buf_v = None;
        self.kv_pos = 0;
    }
}

#[derive(Debug)]
struct ParallelBlock {
    ln: crate::tensor::layer::LayerNorm,
    mixer: MHA,
    mlp: MLP,
}

impl ParallelBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, &vb.pp("ln"))?,
            mixer: MHA::new(cfg, vb.pp("mixer"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
        })
    }

    /// One layer: attention and feed-forward over the SAME normalised input, both added to it.
    ///
    /// This is what makes the family parallel rather than stacked - the feed-forward does not
    /// read what the attention produced, so one norm serves both and the two sums commute into
    /// the residual together.
    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let _traced = tracing::trace_span!("block").entered();
        let normed = xs.apply(&self.ln)?;
        let attended = self.mixer.forward(&normed)?;
        // The feed-forward takes the decode path, which parks its intermediates in buffers a
        // captured graph can replay; it falls back to the plain three steps for prefill.
        let expanded = self.mlp.forward_decode(&normed)?;
        attended + expanded + xs
    }

    fn clear_kv_cache(&mut self) {
        self.mixer.clear_kv_cache()
    }
}

#[derive(Debug)]
pub struct MixFormerSequentialForCausalLM {
    embedding: crate::tensor::layer::Embedding,
    blocks: Vec<ParallelBlock>,
    head: CausalLMHead,
    // Stable output buffer for graph-safe decode:
    // single-token forward writes logits to this buffer via slice_set
    // and returns a view of it. Engine sees the same device pointer
    // across decode tokens - required for graph capture's replay to
    // read valid logits.
    logits_buf: Option<Tensor>,
    /// Stable input ID buffer: single-token decode
    /// writes the new token into this buffer; the captured graph reads
    /// from the same device pointer on replay. Engine overwrites this
    /// buffer per-token via `set_input_id()` before launching the graph.
    input_buf: Option<Tensor>,
    vocab_size: usize,
}

impl MixFormerSequentialForCausalLM {
    /// The nested naming: everything under `transformer`, with the head beside it.
    pub fn new_v2(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner = vb.pp("transformer");
        Self::build(cfg, inner.pp("embd"), inner.pp("h"), 0, vb.pp("lm_head"))
    }

    /// The numbered naming: every part is a layer, the embedding first and the head last.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let layers = vb.pp("layers");
        let head = layers.pp(cfg.n_layer + 1);
        Self::build(cfg, layers.pp(0), layers.clone(), 1, head)
    }

    /// The same model either way: the two namings differ in where the parts are and in nothing
    /// else, so `first_block` is all that is left of the difference by the time it gets here.
    fn build(
        cfg: &Config,
        embedding_at: VarBuilder,
        blocks_at: VarBuilder,
        first_block: usize,
        head: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            embedding: embedding(cfg, embedding_at)?,
            blocks: (0..cfg.n_layer)
                .map(|i| ParallelBlock::new(cfg, blocks_at.pp(first_block + i)))
                .collect::<Result<_>>()?,
            head: CausalLMHead::new(cfg, head)?,
            logits_buf: None,
            input_buf: None,
            vocab_size: cfg.vocab_size,
        })
    }

    /// Borrow the stable logits buffer (None before first decode call).
    /// Used by the engine's CUDA graph capture path to read replayed
    /// logits from the same device pointer the captured graph wrote.
    pub fn logits_buf(&self) -> Option<&Tensor> {
        self.logits_buf.as_ref()
    }

    /// Borrow the stable input ID buffer. Used by engine to write the
    /// per-token next_token into the same device pointer the captured
    /// graph reads on replay.
    pub fn input_buf(&self) -> Option<&Tensor> {
        self.input_buf.as_ref()
    }

    /// Initialize the stable input buffer (must match shape (1,1) u32 on
    /// model device). Called lazily by the engine on first decode call.
    pub fn ensure_input_buf(&mut self, device: &crate::tensor::Device) -> Result<()> {
        if self.input_buf.is_none() {
            self.input_buf = Some(Tensor::zeros_on((1, 1), DType::U32, device)?);
        }
        Ok(())
    }

    /// Write `token` into the stable input buffer (overwrites the existing
    /// device pointer's contents). Captured graph reads from this buffer
    /// during replay, so the engine can change the input per-token
    /// without invalidating the graph.
    pub fn set_input_id(&self, token: u32) -> Result<()> {
        if let Some(buf) = self.input_buf.as_ref() {
            let tmp = Tensor::from_vec(vec![token], (1, 1), &buf.device())?;
            buf.slice_set(&tmp, 0, 0)?;
        }
        Ok(())
    }

    /// The stack of blocks, then the head over the last position only.
    ///
    /// Every position goes through the blocks because attention reads them all, but only the
    /// last one is being asked for a next token - so the widest matmul in the model, the one
    /// against the whole vocabulary, runs on a single row whatever the prompt's length.
    ///
    /// Both entry points end here: a plain prompt and a prompt with an image differ only in
    /// what they hand over as the embedded sequence.
    fn decode(&mut self, embedded: Tensor) -> Result<Tensor> {
        let last = embedded.dim(1)? - 1;
        let hidden = self
            .blocks
            .iter_mut()
            .try_fold(embedded, |xs, block| block.forward(&xs))?;
        self.head.logits(&hidden.narrow(1, last, 1)?)?.squeeze(1)
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let _traced = tracing::trace_span!("mixformer").entered();
        let (batch, _) = xs.dims2()?;
        let raw_logits = self.decode(xs.apply(&self.embedding)?)?;
        // A captured decode graph writes its logits at the address it was captured with, so a
        // single-sequence step on a card lands them in a buffer that outlives the step and
        // hands back a view of it; anything wider is not captured and keeps its own tensor.
        if batch == 1 && raw_logits.device().is_cuda() {
            let dtype = raw_logits.dtype();
            if self.logits_buf.is_none() {
                self.logits_buf = Some(Tensor::zeros_on(
                    (1, self.vocab_size),
                    dtype,
                    &raw_logits.device(),
                )?);
            }
            let buf = self.logits_buf.as_ref().unwrap();
            buf.slice_set(&raw_logits, 0, 0)?;
            Ok(buf.clone())
        } else {
            Ok(raw_logits)
        }
    }

    /// A prompt with a picture in it.
    ///
    /// The picture's embeddings are already in the decoder's width - the vision tower's
    /// projection put them there - so they are simply another stretch of the sequence. Their
    /// place in it is not free: this model was trained with the opening token first, the
    /// picture next and the text after it, and a sequence assembled in any other order asks
    /// the decoder about positions it never saw a picture at.
    pub fn forward_with_img(
        &mut self,
        bos_token: &Tensor,
        xs: &Tensor,
        img_embeds: &Tensor,
    ) -> Result<Tensor> {
        let _traced = tracing::trace_span!("mixformer").entered();
        let text = xs.apply(&self.embedding)?;
        let opening = bos_token.apply(&self.embedding)?;
        self.decode(Tensor::cat(&[&opening, img_embeds, &text], 1)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.blocks.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}
