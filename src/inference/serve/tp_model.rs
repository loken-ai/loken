//! Tensor-parallel decoder (any number of ranks) for dense GQA+SwiGLU models: qwen2 (deepseek-r1 32B, with
//! QKV bias) and the no-bias llama/mistral/mistral3 24B family (devstral-small-2,
//! mistral-small3.2, magistral-small). QKV bias is read only when present.
//!
//! The GenericHetero pipeline runs ONE GPU per decode token (the others idle) - that's why
//! vLLM (tensor-parallel) beats us on 2-GPU models (deepseek-r1 26 vs 37 tok/s). This module
//! runs EVERY GPU on every layer: q/k/v/gate/up are column-parallel (each rank owns a head /
//! intermediate slice), o/down are row-parallel (each rank's partial is summed by a cross-GPU
//! all-reduce). Built entirely on the validated `tp_decode` primitives (TpColumn/TpRow/
//! all_reduce_sum) - see `bin/tp_{weight_slice,ffn,attn,layer,gqa}_test`.
//!
//! Scope: decode-first; `tp_support::tensor_parallel_support` decides from the file's
//! tensors and metadata which models this decoder runs. Loader slices each
//! layer's weights across the ranks; forward keeps the hidden state replicated and reduces
//! once per attention + once per FFN. Per-rank KV cache holds that rank's kv-head subset.

use crate::tensor::quantized::{gguf_file, QMatMul, QTensor};
use crate::tensor::{Device, IndexOp, Tensor};
use anyhow::{bail, Result};
use std::io::Cursor;
use std::sync::Arc;

use crate::inference::model_facts::rope_pairing;
use crate::inference::serve::tp_decode::{
    all_reduce_sum_nccl, all_reduce_sum_nccl_f16, make_comms, TpColumn, TpColumnFused, TpRow,
};

/// Cross-GPU all-reduce precision for the tensor-parallel decode reduction.
///
/// f16 halves the PCIe payload of the all-reduce and is fully correct - bit-identical greedy
/// output and deterministic across runs on deepseek-r1 TP=2.
///
/// For DECODE the message is 1 token ([1,hidden]) and the AR overlaps compute on its own NCCL
/// stream -> halving an already-hidden ~53µs reduction is NEUTRAL (~41 tok/s either way).
/// BUT for PREFILL the message is [seq, hidden] (~43 MB at 2.5K x hidden 5120) and is ON the
/// critical path over the PHB/PCIe host bounce (no P2P) - halving it nearly DOUBLES prefill:
/// devstral 24B prefill 705 -> 1359 tok/s (+93%, measured 2026-06-20), narrowing the TTFT +
/// energy gap to ollama's single-GPU prefill (1793). Prefill win, decode neutral, greedy
/// bit-identical - so f16 is not a setting, it is the answer, and the environment switch that
/// selected it while the question was open is gone.
/// (One-shot peer-read AR is impossible here - see `tp_decode::all_reduce_one_shot_unavailable`.)
fn ar_f16_enabled() -> bool {
    true
}

fn meta_u32(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    ct.metadata.get(key).and_then(|v| v.to_u32().ok())
}
fn meta_f32(ct: &gguf_file::Content, key: &str) -> Option<f32> {
    ct.metadata.get(key).and_then(|v| v.to_f32().ok())
}

pub struct TpConfig {
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub rope_base: f32,
    pub eps: f64,
    pub vocab: usize,
    /// Adjacent-pair (interleaved) RoPE when true, half-split (NeoX) otherwise, from the
    /// architecture facts table.
    pub use_rope_i: bool,
}

impl TpConfig {
    fn from_gguf(ct: &gguf_file::Content, arch: &str) -> Result<Self> {
        let g = |k: &str| meta_u32(ct, &format!("{arch}.{k}"));
        let n_head = g("attention.head_count").unwrap_or(0) as usize;
        let hidden = g("embedding_length").unwrap_or(0) as usize;
        let head_dim = g("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| if n_head > 0 { hidden / n_head } else { 0 });
        let vocab = ct
            .tensor_infos
            .get("token_embd.weight")
            .map(|t| t.shape.dims()[0])
            .unwrap_or(0);
        Ok(Self {
            n_layers: g("block_count").unwrap_or(0) as usize,
            n_head,
            n_kv_head: g("attention.head_count_kv").unwrap_or(n_head as u32) as usize,
            head_dim,
            hidden,
            rope_base: meta_f32(ct, &format!("{arch}.rope.freq_base")).unwrap_or(10000.0),
            eps: meta_f32(ct, &format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5)
                as f64,
            vocab,
            use_rope_i: rope_pairing(arch).map_err(anyhow::Error::msg)?,
        })
    }
}

/// One transformer layer split across the ranks.
struct TpLayer {
    attn_norm: Vec<Tensor>, // rmsnorm weight, replicated on every rank
    ffn_norm: Vec<Tensor>,
    qk: TpColumnFused, // fused q|k column-parallel (+bias), one matmul/rank (both Q4_K)
    v: TpColumn,       // separate (v is Q6_K in Q4_K_M mixed quant - can't share the mvq)
    o: TpRow,          // row-parallel
    gate_up: TpColumnFused, // fused gate|up column-parallel, one matmul/rank (both Q4_K)
    down: TpRow,
    // per-rank KV cache: [nkv_rank, total_seq, hd] on each rank's device
    kv: Vec<Option<(Tensor, Tensor)>>,
}

pub struct TpQwen2 {
    cfg: TpConfig,
    cards: Vec<Device>,
    embed: crate::tensor::layer::Embedding, // the file's [vocab, hidden] blocks, decoded per lookup
    layers: Vec<TpLayer>,
    out_norm: Tensor, // on cards[0]
    lm_head: QMatMul, // on cards[0]
    // RoPE tables (interleaved), per device, [max_seq, hd/2]
    cos: Vec<Tensor>,
    sin: Vec<Tensor>,
    // NCCL comms (rank r = cards[r]) for the overlapping all-reduce.
    comms: Vec<cudarc::nccl::safe::Comm>,
    // Run the cross-GPU all-reduce in f16 (half the PCIe payload) vs f32.
    ar_f16: bool,
}

// RoPE table length. Sized so a full num_ctx=8192 request (prompt + generation)
// stays in-table with headroom. The table is tiny (MAX_SEQ x hd/2 x f32 ≈ a few
// MB/device), so a generous cap costs negligible VRAM while removing the silent
// position ceiling that capped TP at 8192 total positions.
const MAX_SEQ: usize = 16384;

fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // Fused single-kernel rmsnorm (vs the 6-op manual version) - cuts ~5 launches per norm,
    // x2 norms x2 ranks xn_layers per token. Needs contiguous input.
    Ok(crate::tensor::ops::rms_norm(
        &x.contiguous()?,
        w,
        eps as f32,
    )?)
}

fn rope_tables(hd: usize, base: f32, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = hd / 2;
    let theta: Vec<f32> = crate::inference::model::rope::inverse_frequencies(hd, base);
    let theta = Tensor::new(theta.as_slice(), dev)?;
    let idx = Tensor::arange_u32(0u32, MAX_SEQ as u32)?
        .to_device(dev)?
        .to_dtype(crate::tensor::DType::F32)?
        .reshape((MAX_SEQ, 1))?;
    let ang = idx.matmul(&theta.reshape((1, half))?)?;
    Ok((ang.cos()?, ang.sin()?))
}

impl TpQwen2 {
    /// Per-device layer distribution for the /api/ps topology view. Under
    /// tensor parallelism EVERY layer runs on BOTH GPUs (column/row-parallel
    /// weight shards + all-reduce), so both devices carry the full layer
    /// range. Returns `(total_layers, [(device_type, device_id, start, end)])`
    /// with inclusive ranges - same shape as GenericHetero's accessor.
    pub fn device_layer_distribution(&self) -> (usize, Vec<(String, usize, u32, u32)>) {
        let n = self.cfg.n_layers;
        let last = (n.saturating_sub(1)) as u32;
        let gid = |d: &Device| match d.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => 0,
        };
        // "CUDA (TP)" still contains "CUDA" so the GUI colours it as a GPU row,
        // while the suffix signals the layers are shared, not partitioned.
        (
            n,
            self.cards
                .iter()
                .map(|d| ("CUDA (TP)".to_string(), gid(d), 0, last))
                .collect(),
        )
    }

    /// The decoder over `cards`, one rank each, rank `r` taking `ffn_shares[r]` parts of the
    /// feed-forward rows: the placement decided both, and the shares it chose fit.
    pub fn from_gguf(
        mmap_bytes: &[u8],
        mmap_owner: &std::sync::Arc<memmap2::Mmap>,
        arch: &str,
        cards: &[Device],
        ffn_shares: &[usize],
    ) -> Result<Self> {
        let ranks = cards.len();
        if ranks < 2 || ffn_shares.len() != ranks {
            bail!(
                "the tensor-parallel decoder needs 2 or more cards with a share each, {} cards and {} shares given",
                ranks,
                ffn_shares.len()
            );
        }
        let cards: Vec<Device> = cards.to_vec();
        // Disable the old substrate's per-op CUDA event tracking on both ranks. It records an event after
        // every op (~1.18M cuEventRecord/run in the nsys profile = 7% of CUDA API time) for
        // cross-stream ordering - but within a device ops are already stream-ordered, and the
        // ONLY cross-GPU dependency (the all-reduce) is handled by NCCL's own stream sync, not
        // the old substrate's to_device. So the events are pure overhead here.
        for d in &cards {
            if let Ok(cd) = d.as_cuda_device() {
                unsafe {
                    cd.disable_event_tracking();
                }
            }
        }
        let mut cur = Cursor::new(mmap_bytes);
        let ct = gguf_file::Content::read_mapped(&mut cur, mmap_owner.clone())?;
        let cfg = TpConfig::from_gguf(&ct, arch)?;
        if cfg.n_head == 0
            || cfg.n_kv_head == 0
            || cfg.n_head % ranks != 0
            || cfg.n_kv_head % ranks != 0
        {
            bail!(
                "TP={ranks} needs n_head/n_kv_head divisible by {ranks}, got {}/{}",
                cfg.n_head,
                cfg.n_kv_head
            );
        }
        let cpu = crate::tensor::Device::Cpu;
        let rd = |ct: &gguf_file::Content,
                  _cur: &mut Cursor<&[u8]>,
                  name: &str|
         -> Result<QTensor> { Ok(ct.tensor_mapped(mmap_bytes, name, &cpu)?) };
        // embeddings: kept in the file's format, the rows of a prompt decoded per lookup
        let embed = crate::tensor::layer::Embedding::from_stored(Arc::new(rd(
            &ct,
            &mut cur,
            "token_embd.weight",
        )?))?;
        // layers
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blk.{i}");
            let norm = |ct: &gguf_file::Content,
                        cur: &mut Cursor<&[u8]>,
                        n: &str|
             -> Result<Vec<Tensor>> {
                // CONVERSION: a norm scale is a small vector held F32 on each card.
                let w = rd(ct, cur, n)?.dequantize(&cpu)?;
                cards.iter().map(|d| Ok(w.to_device(d)?)).collect()
            };
            // QKV bias is present on qwen2 but absent on llama/mistral/mistral3 - read it
            // only when the GGUF actually carries the tensor, else None (no-bias dense arch).
            let bias = |ct: &gguf_file::Content,
                        cur: &mut Cursor<&[u8]>,
                        n: &str|
             -> Result<Option<Tensor>> {
                if ct.tensor_infos.contains_key(n) {
                    // CONVERSION: a bias is a small vector held F32.
                    Ok(Some(rd(ct, cur, n)?.dequantize(&cpu)?))
                } else {
                    Ok(None)
                }
            };
            let attn_norm = norm(&ct, &mut cur, &format!("{p}.attn_norm.weight"))?;
            let wq = rd(&ct, &mut cur, &format!("{p}.attn_q.weight"))?;
            let bq = bias(&ct, &mut cur, &format!("{p}.attn_q.bias"))?;
            let wk = rd(&ct, &mut cur, &format!("{p}.attn_k.weight"))?;
            let bk = bias(&ct, &mut cur, &format!("{p}.attn_k.bias"))?;
            let wv = rd(&ct, &mut cur, &format!("{p}.attn_v.weight"))?;
            let bv = bias(&ct, &mut cur, &format!("{p}.attn_v.bias"))?;
            let wo = rd(&ct, &mut cur, &format!("{p}.attn_output.weight"))?;
            let ffn_norm = norm(&ct, &mut cur, &format!("{p}.ffn_norm.weight"))?;
            let wg = rd(&ct, &mut cur, &format!("{p}.ffn_gate.weight"))?;
            let wu = rd(&ct, &mut cur, &format!("{p}.ffn_up.weight"))?;
            let wd = rd(&ct, &mut cur, &format!("{p}.ffn_down.weight"))?;
            layers.push(TpLayer {
                attn_norm,
                ffn_norm,
                qk: TpColumnFused::from_qtensors(
                    &[(&wq, bq.as_ref()), (&wk, bk.as_ref())],
                    cfg.head_dim,
                    &cards,
                )?,
                v: TpColumn::from_qtensor_bias(&wv, bv.as_ref(), &cards)?,
                o: TpRow::from_qtensor(&wo, &cards)?,
                // Asymmetric FFN split: the faster ranks take the larger shares of the
                // intermediate rows so all finish together at the FFN all-reduce. gate/up
                // rows and down cols split at the SAME, QK_K-aligned points.
                gate_up: TpColumnFused::from_qtensors_shares(
                    &[(&wg, None), (&wu, None)],
                    ffn_shares,
                    crate::tensor::quant_cpu::QK_K,
                    &cards,
                )?,
                down: TpRow::from_qtensor_shares(&wd, ffn_shares, &cards)?,
                kv: vec![None; ranks],
            });
        }
        // (verified: deepseek-r1:32b qwen2 - layers=64 n_head=40 n_kv=8 hd=128 hidden=5120
        //  rope_base=1e6 eps=1e-5 vocab=152064; attn_q/output=Q4K, ffn_down=Q6K - all QK_K=256
        //  so the row-parallel slice_qtensor_cols block math is exact.)
        let out_norm = rd(&ct, &mut cur, "output_norm.weight")?
            // CONVERSION: a norm scale is a small vector held F32.
            .dequantize(&cpu)?
            .to_device(&cards[0])?;
        let lm_qt = match ct.tensor(&mut cur, "output.weight", &cards[0]) {
            Ok(qt) => qt,
            Err(_) => ct.tensor(&mut Cursor::new(mmap_bytes), "token_embd.weight", &cards[0])?, // tied
        };
        let lm_head = QMatMul::from_arc(Arc::new(lm_qt))?;
        let tables = cards
            .iter()
            .map(|d| rope_tables(cfg.head_dim, cfg.rope_base, d))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin): (Vec<Tensor>, Vec<Tensor>) = tables.into_iter().unzip();
        let comms = make_comms(&cards)?;
        let ar_f16 = ar_f16_enabled();
        Ok(Self {
            cfg,
            cards,
            embed,
            layers,
            out_norm,
            lm_head,
            cos,
            sin,
            comms,
            ar_f16,
        })
    }

    pub fn reset_kv(&mut self) {
        for l in &mut self.layers {
            l.kv.iter_mut().for_each(|c| *c = None);
        }
    }

    /// Trim every per-rank KV cache to `n` valid positions (seq is dim 1 of the
    /// [nkv_r, total_seq, hd] F16 cache). Enables the global prompt-cache prefix
    /// reuse for TP models (#32): the kept prefix is the EXACT F16 K/V a full
    /// prefill of [0,n) would have produced, so re-prefilling only the divergent
    /// suffix is greedy-identical. `n==0` drops the cache; `n>=cur` is a no-op
    /// (strict extension). Infallible - a narrow error leaves the cache intact
    /// (falls back to re-prefill, never corrupts).
    pub fn trim_kv(&mut self, n: usize) {
        let trim_one = |c: &mut Option<(Tensor, Tensor)>| {
            if n == 0 {
                *c = None;
                return;
            }
            if let Some((k, v)) = c.as_ref() {
                let cur = k.dim(1).unwrap_or(0);
                if n < cur {
                    if let (Ok(kt), Ok(vt)) = (k.narrow(1, 0, n), v.narrow(1, 0, n)) {
                        *c = Some((kt, vt))
                    }
                }
            }
        };
        for l in &mut self.layers {
            l.kv.iter_mut().for_each(trim_one);
        }
    }

    /// One rank's GQA attention with KV append. q:[seq, nh_r*hd], k/v:[seq, nkv_r*hd].
    fn rank_attn(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        nh_r: usize,
        nkv_r: usize,
        hd: usize,
        pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        use_rope_i: bool,
    ) -> Result<Tensor> {
        let seq = q.dim(0)?;
        let scale = (hd as f64).powf(-0.5);
        let heads = |t: &Tensor, n: usize| -> crate::tensor::Result<Tensor> {
            t.reshape((seq, n, hd))?.transpose(0, 1)?.contiguous()
        };
        let crow = cos.narrow(0, pos, seq)?;
        let srow = sin.narrow(0, pos, seq)?;
        let rope_fn = if use_rope_i {
            crate::tensor::ops::rope_i
        } else {
            crate::tensor::ops::rope
        };
        let rope = |t: &Tensor, n: usize| -> crate::tensor::Result<Tensor> {
            let h = heads(t, n)?.unsqueeze(0)?;
            rope_fn(&h.contiguous()?, &crow, &srow)?.squeeze(0)
        };
        let q = rope(q, nh_r)?; // [nh_r, seq, hd]
        let k_new = rope(k, nkv_r)?; // [nkv_r, seq, hd]
        let v_new = heads(v, nkv_r)?;
        // Store the per-rank KV cache in F16 (half the VRAM of F32). deepseek-r1:32b
        // fills both 16 GB cards to ~90% with weights; an F32 KV at long context
        // (64 layers x [nkv_r, ctx, hd] x 2 for K/V) is ~2.1 GB/GPU at 8192 and OOMs.
        // F16 halves it to ~1 GB/GPU. The cache is upcast back to F32 only for the
        // attention matmul below - a transient bounded by the current sequence, not
        // the steady footprint. Greedy output is unchanged (the upcast is exact for
        // values that came from F16; the only loss is in the appended new tokens,
        // which is below the sampler's resolution and matches the KV-quant precedent
        // already shipped for the GenericHetero path).
        let k_new16 = k_new.to_dtype(crate::tensor::DType::F16)?;
        let v_new16 = v_new.to_dtype(crate::tensor::DType::F16)?;
        // append to cache along seq dim (cache holds F16)
        let (k_full16, v_full16) = match cache.take() {
            Some((kc, vc)) => (
                Tensor::cat(&[&kc, &k_new16], 1)?,
                Tensor::cat(&[&vc, &v_new16], 1)?,
            ),
            None => (k_new16.clone(), v_new16.clone()),
        };
        *cache = Some((k_full16.clone(), v_full16.clone()));
        // PREFILL (seq>1) as one flash kernel over the F16 cache: query i sits at position
        // pos + i = i + total - seq, the causal offset the kernel derives.
        if seq > 1 {
            if let Some(y) = crate::inference::kernel::fused::flash_prefill_gqa(
                &q.unsqueeze(0)?,
                &k_full16.unsqueeze(0)?,
                &v_full16.unsqueeze(0)?,
                scale as f32,
                0,
            )? {
                return Ok(y.reshape((seq, nh_r * hd))?.to_dtype(q.dtype())?);
            }
        }
        // Upcast to F32 for the score/output matmuls (transient, freed after this layer).
        let k_full = k_full16.to_dtype(crate::tensor::DType::F32)?;
        let v_full = v_full16.to_dtype(crate::tensor::DType::F32)?;
        let total = k_full.dim(1)?;
        let rep = nh_r / nkv_r;
        if seq == 1 {
            // DECODE GQA without expanding K/V to nh_r heads: group the query by kv-head
            // ([nh_r,1,hd] -> [nkv_r, rep, hd]) and batch-matmul against K/V kept at [nkv_r,...].
            // Avoids the two expand+reshape copies (copy2d) per rank - those copies + their
            // layout HtoD were a big chunk of the per-token CUDA API overhead. No causal mask
            // needed (the single query sees all `total` keys).
            let qg = q.reshape((nkv_r, rep, hd))?; // q is [nh_r, 1, hd] = [nkv_r*rep, 1, hd]
            let scores = (qg.matmul(&k_full.transpose(1, 2)?.contiguous()?)? * scale)?; // [nkv_r, rep, total]
            let out = crate::tensor::ops::softmax_last_dim(&scores)?.matmul(&v_full)?; // [nkv_r, rep, hd]
            return Ok(out.reshape((1, nh_r * hd))?);
        }
        // PREFILL (seq>1): expand path + causal mask.
        let k = k_full
            .unsqueeze(1)?
            .expand((nkv_r, rep, total, hd))?
            .reshape((nh_r, total, hd))?;
        let v = v_full
            .unsqueeze(1)?
            .expand((nkv_r, rep, total, hd))?
            .reshape((nh_r, total, hd))?;
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh_r, seq, total]
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..total).map(move |j| {
                    if j <= pos + i {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let scores = scores.broadcast_add(&Tensor::from_vec(mask, (seq, total), &q.device())?)?;
        let out = crate::tensor::ops::softmax_last_dim(&scores)?.matmul(&v)?; // [nh_r, seq, hd]
        Ok(out
            .transpose(0, 1)?
            .reshape((seq, nh_r * hd))?
            .contiguous()?)
    }

    /// Cross-GPU all-reduce(sum) - f16 transport (default) or f32 (legacy/A-B).
    /// Free-standing (takes comms+flag, not `&self`) so it composes with the `iter_mut()`
    /// borrow of `self.layers` in the forward loop.
    fn all_reduce(
        f16: bool,
        comms: &[cudarc::nccl::safe::Comm],
        ps: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if f16 {
            all_reduce_sum_nccl_f16(ps, comms)
        } else {
            all_reduce_sum_nccl(ps, comms)
        }
    }

    /// Forward over `tokens` (length seq) starting at absolute position `pos`.
    /// Returns logits for the LAST token: [vocab].
    pub fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor> {
        // A fresh sequence always starts at pos 0 -> reset the per-layer KV caches so a new
        // request doesn't attend to the previous one's tokens. (The engine drives TP with a
        // fresh prefill at pos 0 per request; TP doesn't do session-reuse/speculative.)
        if pos == 0 {
            self.reset_kv();
        }
        let seq = tokens.len();
        let (nh, nkv, hd) = (self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim);
        let ranks = self.cards.len();
        let (nh_r, nkv_r) = (nh / ranks, nkv / ranks);
        let eps = self.cfg.eps;
        // embed -> [seq, hidden] on rank 0, replicated to the others
        let ids = Tensor::new(tokens, &Device::Cpu)?;
        let emb = self.embed.forward(&ids)?; // [seq, hidden] CPU
        let first = emb
            .to_device(&self.cards[0])?
            .to_dtype(crate::tensor::DType::F32)?;
        let mut xs: Vec<Tensor> = Vec::with_capacity(ranks);
        for card in &self.cards[1..] {
            xs.push(first.to_device(card)?);
        }
        xs.insert(0, first);

        let ar_f16 = self.ar_f16;
        let comms = &self.comms;
        for l in self.layers.iter_mut() {
            // -- attention --
            let hs = xs
                .iter()
                .zip(&l.attn_norm)
                .map(|(x, w)| rmsnorm(x, w, eps))
                .collect::<Result<Vec<_>>>()?;
            // Fused QK: one matmul/rank, then split [q|k] (decode seq=1 narrows are zero-copy);
            // v stays separate (Q6_K). q/k/v all share each rank's input.
            let qk = l.qk.forward(&hs)?;
            let vs = l.v.forward(&hs)?;
            let ri = self.cfg.use_rope_i;
            let mut attn = Vec::with_capacity(ranks);
            for r in 0..ranks {
                let parts = l.qk.split(&qk[r], r)?;
                attn.push(Self::rank_attn(
                    &parts[0],
                    &parts[1],
                    &vs[r],
                    &mut l.kv[r],
                    nh_r,
                    nkv_r,
                    hd,
                    pos,
                    &self.cos[r],
                    &self.sin[r],
                    ri,
                )?);
            }
            let partial = l.o.forward(&attn)?;
            let reduced = Self::all_reduce(ar_f16, comms, &partial)?;
            xs = xs
                .iter()
                .zip(&reduced)
                .map(|(x, o)| Ok((x + o)?))
                .collect::<Result<Vec<_>>>()?;
            // -- FFN --
            let hs = xs
                .iter()
                .zip(&l.ffn_norm)
                .map(|(x, w)| rmsnorm(x, w, eps))
                .collect::<Result<Vec<_>>>()?;
            // Fused gate|up: one matmul/rank, then split + fused SiLU.mul.
            let gu = l.gate_up.forward(&hs)?;
            let mut acts = Vec::with_capacity(ranks);
            for r in 0..ranks {
                let parts = l.gate_up.split(&gu[r], r)?;
                acts.push(crate::inference::kernel::fused::fused_silu_mul(
                    &parts[0], &parts[1],
                )?);
            }
            let partial = l.down.forward(&acts)?;
            let reduced = Self::all_reduce(ar_f16, comms, &partial)?;
            xs = xs
                .iter()
                .zip(&reduced)
                .map(|(x, f)| Ok((x + f)?))
                .collect::<Result<Vec<_>>>()?;
        }
        // final norm + lm_head on g0, last token only
        let last = xs[0].i(seq - 1)?.unsqueeze(0)?; // [1, hidden]
        let h = rmsnorm(&last, &self.out_norm, eps)?;
        let logits = self.lm_head.forward(&h)?; // [1, vocab]
        Ok(logits.squeeze(0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::DType;

    const HIDDEN: usize = 512;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const FFN: usize = 1024;
    const VOCAB: usize = 256;
    const LAYERS: usize = 2;
    const EPS: f64 = 1e-5;

    /// A small qwen2-shaped file with random weights: the quantised matrices as the decoder
    /// slices them, the rest F32. Returns the path and the dequantised weights by tensor name.
    fn toy_file(
        dir: &std::path::Path,
    ) -> (
        std::path::PathBuf,
        std::collections::HashMap<String, Tensor>,
    ) {
        let cpu = Device::Cpu;
        let mut entries = Vec::new();
        let mut dense = std::collections::HashMap::new();
        let mut put = |name: &str, dims: &[usize], quant: bool, scale: f32| {
            let w = Tensor::randn(0f32, scale, dims.to_vec(), &cpu).unwrap();
            let (dtype, data, held) = if quant {
                let q = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
                let held = q.dequantize(&cpu).unwrap();
                (GgmlDType::Q4K, q.data().unwrap().to_vec(), held)
            } else {
                let flat = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let bytes = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
                (GgmlDType::F32, bytes, w)
            };
            entries.push(GgufEntry {
                name: name.to_string(),
                dims: dims.to_vec(),
                dtype,
                data,
            });
            dense.insert(name.to_string(), held);
        };
        put("token_embd.weight", &[VOCAB, HIDDEN], false, 1.0);
        put("output_norm.weight", &[HIDDEN], false, 0.1);
        put("output.weight", &[VOCAB, HIDDEN], true, 0.05);
        for i in 0..LAYERS {
            let p = format!("blk.{i}");
            put(&format!("{p}.attn_norm.weight"), &[HIDDEN], false, 0.1);
            put(
                &format!("{p}.attn_q.weight"),
                &[HEADS * HEAD_DIM, HIDDEN],
                true,
                0.05,
            );
            put(&format!("{p}.attn_q.bias"), &[HEADS * HEAD_DIM], false, 0.1);
            put(
                &format!("{p}.attn_k.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                true,
                0.05,
            );
            put(
                &format!("{p}.attn_k.bias"),
                &[KV_HEADS * HEAD_DIM],
                false,
                0.1,
            );
            put(
                &format!("{p}.attn_v.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                true,
                0.05,
            );
            put(
                &format!("{p}.attn_v.bias"),
                &[KV_HEADS * HEAD_DIM],
                false,
                0.1,
            );
            put(
                &format!("{p}.attn_output.weight"),
                &[HIDDEN, HEADS * HEAD_DIM],
                true,
                0.05,
            );
            put(&format!("{p}.ffn_norm.weight"), &[HIDDEN], false, 0.1);
            put(&format!("{p}.ffn_gate.weight"), &[FFN, HIDDEN], true, 0.05);
            put(&format!("{p}.ffn_up.weight"), &[FFN, HIDDEN], true, 0.05);
            put(&format!("{p}.ffn_down.weight"), &[HIDDEN, FFN], true, 0.05);
        }
        let u32v = |v: usize| Value::U32(v as u32);
        let meta = vec![
            (
                "general.architecture".to_string(),
                Value::String("qwen2".into()),
            ),
            ("qwen2.block_count".to_string(), u32v(LAYERS)),
            ("qwen2.embedding_length".to_string(), u32v(HIDDEN)),
            ("qwen2.attention.head_count".to_string(), u32v(HEADS)),
            ("qwen2.attention.head_count_kv".to_string(), u32v(KV_HEADS)),
            ("qwen2.attention.key_length".to_string(), u32v(HEAD_DIM)),
            ("qwen2.feed_forward_length".to_string(), u32v(FFN)),
            ("qwen2.rope.freq_base".to_string(), Value::F32(10000.0)),
            (
                "qwen2.attention.layer_norm_rms_epsilon".to_string(),
                Value::F32(EPS as f32),
            ),
        ];
        let path = dir.join("toy-qwen2.gguf");
        write_gguf_with_metadata(&path, &meta, &entries).unwrap();
        (path, dense)
    }

    /// The same model on one card with dense weights, the last token's logits.
    fn reference(
        w: &std::collections::HashMap<String, Tensor>,
        tokens: &[u32],
        dev: &Device,
    ) -> Vec<f32> {
        let seq = tokens.len();
        let get = |n: &str| w[n].to_device(dev).unwrap();
        let lin = |x: &Tensor, n: &str| {
            x.matmul(&get(n).t().unwrap().contiguous().unwrap())
                .unwrap()
        };
        let ids = Tensor::new(tokens, &Device::Cpu).unwrap();
        let mut x = get("token_embd.weight")
            .index_select(&ids.to_device(dev).unwrap(), 0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let (cos, sin) = rope_tables(HEAD_DIM, 10000.0, dev).unwrap();
        let heads = |t: &Tensor, n: usize| {
            t.reshape((seq, n, HEAD_DIM))
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let rope = |t: &Tensor, n: usize| {
            let h = heads(t, n).unsqueeze(0).unwrap();
            crate::tensor::ops::rope(
                &h.contiguous().unwrap(),
                &cos.narrow(0, 0, seq).unwrap(),
                &sin.narrow(0, 0, seq).unwrap(),
            )
            .unwrap()
            .squeeze(0)
            .unwrap()
        };
        for i in 0..LAYERS {
            let p = format!("blk.{i}");
            let h = rmsnorm(&x, &get(&format!("{p}.attn_norm.weight")), EPS).unwrap();
            let q = lin(&h, &format!("{p}.attn_q.weight"))
                .broadcast_add(&get(&format!("{p}.attn_q.bias")))
                .unwrap();
            let k = lin(&h, &format!("{p}.attn_k.weight"))
                .broadcast_add(&get(&format!("{p}.attn_k.bias")))
                .unwrap();
            let v = lin(&h, &format!("{p}.attn_v.weight"))
                .broadcast_add(&get(&format!("{p}.attn_v.bias")))
                .unwrap();
            let (q, k, v) = (rope(&q, HEADS), rope(&k, KV_HEADS), heads(&v, KV_HEADS));
            let rep = HEADS / KV_HEADS;
            let expand = |t: &Tensor| {
                t.unsqueeze(1)
                    .unwrap()
                    .expand((KV_HEADS, rep, seq, HEAD_DIM))
                    .unwrap()
                    .reshape((HEADS, seq, HEAD_DIM))
                    .unwrap()
            };
            let scores = (q
                .matmul(&expand(&k).transpose(1, 2).unwrap().contiguous().unwrap())
                .unwrap()
                * (HEAD_DIM as f64).powf(-0.5))
            .unwrap();
            let mask: Vec<f32> = (0..seq)
                .flat_map(|a| (0..seq).map(move |b| if b <= a { 0.0 } else { f32::NEG_INFINITY }))
                .collect();
            let scores = scores
                .broadcast_add(&Tensor::from_vec(mask, (seq, seq), dev).unwrap())
                .unwrap();
            let att = crate::tensor::ops::softmax_last_dim(&scores)
                .unwrap()
                .matmul(&expand(&v))
                .unwrap();
            let att = att
                .transpose(0, 1)
                .unwrap()
                .reshape((seq, HEADS * HEAD_DIM))
                .unwrap()
                .contiguous()
                .unwrap();
            x = (&x + lin(&att, &format!("{p}.attn_output.weight"))).unwrap();
            let h = rmsnorm(&x, &get(&format!("{p}.ffn_norm.weight")), EPS).unwrap();
            let g = lin(&h, &format!("{p}.ffn_gate.weight"));
            let u = lin(&h, &format!("{p}.ffn_up.weight"));
            let act = (crate::tensor::ops::silu(&g).unwrap() * u).unwrap();
            x = (&x + lin(&act, &format!("{p}.ffn_down.weight"))).unwrap();
        }
        let last = x.i(seq - 1).unwrap().unsqueeze(0).unwrap();
        let h = rmsnorm(&last, &get("output_norm.weight"), EPS).unwrap();
        lin(&h, "output.weight")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    /// The decoder over two cards, with an even and with an uneven feed-forward share, gives the
    /// logits of the same model on one card.
    #[test]
    #[ignore = "needs two CUDA cards and NCCL"]
    fn the_ranks_reproduce_the_single_card_logits() {
        let (Ok(c0), Ok(c1)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("loken-tp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (path, weights) = toy_file(&dir);
        let file = std::fs::File::open(&path).unwrap();
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let tokens = [3u32, 17, 99, 5, 42];
        let want = reference(&weights, &tokens, &c0);
        for shares in [vec![1usize, 1], vec![3, 1]] {
            let mut tp =
                TpQwen2::from_gguf(&map, &map, "qwen2", &[c0.clone(), c1.clone()], &shares)
                    .unwrap();
            let got = tp
                .forward(&tokens, 0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let c = cosine(&got, &want);
            assert!(c > 0.99, "shares {shares:?}: cosine {c}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
