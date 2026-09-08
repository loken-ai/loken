//! Tensor-parallel (TP=2) decoder for dense GQA+SwiGLU models: qwen2 (deepseek-r1 32B, with
//! QKV bias) and the no-bias llama/mistral/mistral3 24B family (devstral-small-2,
//! mistral-small3.2, magistral-small). QKV bias is read only when present.
//!
//! The GenericHetero pipeline runs ONE GPU per decode token (the other idles) - that's why
//! vLLM (tensor-parallel) beats us on 2-GPU models (deepseek-r1 26 vs 37 tok/s). This module
//! runs BOTH GPUs on every layer: q/k/v/gate/up are column-parallel (each rank owns a head /
//! intermediate slice), o/down are row-parallel (each rank's partial is summed by a cross-GPU
//! all-reduce). Built entirely on the validated `tp_decode` primitives (TpColumn/TpRow/
//! all_reduce_sum) - see `bin/tp_{weight_slice,ffn,attn,layer,gqa}_test`.
//!
//! Scope: decode-first, qwen2 only (GQA + QKV bias + interleaved RoPE). Loader slices each
//! layer's weights across the 2 GPUs; forward keeps the hidden state replicated and reduces
//! once per attention + once per FFN. Per-rank KV cache holds that rank's kv-head subset.

use crate::tensor::quantized::{gguf_file, QMatMul, QTensor};
use crate::tensor::{Device, IndexOp, Tensor};
use anyhow::{bail, Result};
use std::io::Cursor;
use std::sync::Arc;

use crate::inference::serve::tp_decode::{
    all_reduce_sum_nccl, all_reduce_sum_nccl_f16, make_comms, TpColumn, TpColumnFused, TpRow,
};

/// Cross-GPU all-reduce precision for the TP=2 decode reduction.
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

/// How much of the feed-forward rows the fast card takes, as `(num, den)`.
///
/// Bandwidth says 2/3: the fast card gets through roughly twice the matmul per second, so
/// giving it two thirds makes both finish together and removes its idle wait at the
/// all-reduce. That is right whenever the resulting shard FITS.
///
/// It did not fit once: a 28.7 GB model asked 19.1 GB of a card holding 16.6, the load
/// climbed to 14.0 GB on one card while the other sat at 8.6, and the upload hit
/// CUDA_ERROR_OUT_OF_MEMORY. The cascade then re-planned off the GPU entirely and the model
/// ran at 2.5 tok/s instead of the ~25 it fits for. A ratio derived from speed alone is a
/// ratio that assumes room, so the capacity has a veto here: fall back to an even split,
/// and if even halves do not fit, say so rather than OOM inside the loader.
fn ffn_split_frac(weights: u64, d0: &Device, d1: &Device) -> Result<(usize, usize)> {
    let free = |d: &Device| -> u64 {
        crate::tensor::cuda_ext::mem_get_info(d)
            .map(|(f, _)| f as u64)
            .unwrap_or(0)
    };
    let (f0, f1) = (free(d0), free(d1));
    if f0 == 0 || f1 == 0 {
        return Ok((2, 3)); // no reading available: keep the measured default
    }
    // The rows this fraction governs dominate a dense transformer, so charging the whole
    // file against them is the conservative reading - it can only refuse a split that
    // would have just fitted, never accept one that will not.
    for (num, den) in [(2usize, 3usize), (1, 2)] {
        let want0 = weights.saturating_mul(num as u64) / den as u64;
        let want1 = weights.saturating_sub(want0);
        if want0 <= f0 && want1 <= f1 {
            if (num, den) != (2, 3) {
                tracing::info!(
                    "TP split {num}/{den} instead of 2/3: {:.1} GB against {:.1} + {:.1} GB free",
                    weights as f64 / 1e9,
                    f0 as f64 / 1e9,
                    f1 as f64 / 1e9
                );
            }
            return Ok((num, den));
        }
    }
    bail!(
        "TP=2 cannot hold {:.1} GB across {:.1} + {:.1} GB of free VRAM",
        weights as f64 / 1e9,
        f0 as f64 / 1e9,
        f1 as f64 / 1e9
    )
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
    /// Interleaved (llama/mistral) vs non-interleaved/NeoX (qwen2 and most others) RoPE.
    /// Mirrors GenericHetero's `use_rope_i: matches!(arch, "llama"|"mistral"|"mistral3")`.
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
            use_rope_i: matches!(arch, "llama" | "mistral" | "mistral3"),
        })
    }
}

/// One transformer layer split across the 2 GPUs.
struct TpLayer {
    attn_norm0: Tensor, // rmsnorm weight, replicated on g0 / g1
    attn_norm1: Tensor,
    ffn_norm0: Tensor,
    ffn_norm1: Tensor,
    qk: TpColumnFused, // fused q|k column-parallel (+bias), one matmul/rank (both Q4_K)
    v: TpColumn,       // separate (v is Q6_K in Q4_K_M mixed quant - can't share the mvq)
    o: TpRow,          // row-parallel
    gate_up: TpColumnFused, // fused gate|up column-parallel, one matmul/rank (both Q4_K)
    down: TpRow,
    // per-rank KV cache: [nkv_rank, total_seq, hd] on g0 / g1
    kv0: Option<(Tensor, Tensor)>,
    kv1: Option<(Tensor, Tensor)>,
}

pub struct TpQwen2 {
    cfg: TpConfig,
    g0: Device,
    g1: Device,
    embed: Tensor, // dequantized [vocab, hidden] on CPU
    layers: Vec<TpLayer>,
    out_norm: Tensor, // on g0
    lm_head: QMatMul, // on g0
    // RoPE tables (interleaved), per device, [max_seq, hd/2]
    cos0: Tensor,
    sin0: Tensor,
    cos1: Tensor,
    sin1: Tensor,
    // NCCL comms (rank0=g0, rank1=g1) for the overlapping all-reduce.
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
            vec![
                ("CUDA (TP)".to_string(), gid(&self.g0), 0, last),
                ("CUDA (TP)".to_string(), gid(&self.g1), 0, last),
            ],
        )
    }

    pub fn from_gguf(
        mmap_bytes: &[u8],
        mmap_owner: &std::sync::Arc<memmap2::Mmap>,
        arch: &str,
        g0: Device,
        g1: Device,
    ) -> Result<Self> {
        // Disable the old substrate's per-op CUDA event tracking on both ranks. It records an event after
        // every op (~1.18M cuEventRecord/run in the nsys profile = 7% of CUDA API time) for
        // cross-stream ordering - but within a device ops are already stream-ordered, and the
        // ONLY cross-GPU dependency (the all-reduce) is handled by NCCL's own stream sync, not
        // the old substrate's to_device. So the events are pure overhead here.
        for d in [&g0, &g1] {
            if let Ok(cd) = d.as_cuda_device() {
                unsafe {
                    cd.disable_event_tracking();
                }
            }
        }
        let mut cur = Cursor::new(mmap_bytes);
        let ct = gguf_file::Content::read_mapped(&mut cur, mmap_owner.clone())?;
        let cfg = TpConfig::from_gguf(&ct, arch)?;
        if cfg.n_head == 0 || cfg.n_kv_head == 0 || cfg.n_head % 2 != 0 || cfg.n_kv_head % 2 != 0 {
            bail!(
                "TP=2 needs even n_head/n_kv_head, got {}/{}",
                cfg.n_head,
                cfg.n_kv_head
            );
        }
        // Decided before a single byte is uploaded: a shard that does not fit has to be
        // refused here, where the caller can still fall back cleanly, not discovered
        // halfway through the upload as an OOM.
        let (ffn_num, ffn_den) = ffn_split_frac(mmap_bytes.len() as u64, &g0, &g1)?;
        let cpu = crate::tensor::Device::Cpu;
        let rd = |ct: &gguf_file::Content,
                  _cur: &mut Cursor<&[u8]>,
                  name: &str|
         -> Result<QTensor> { Ok(ct.tensor_mapped(mmap_bytes, name, &cpu)?) };
        // embeddings (dequantized on CPU, index_select per token)
        let embed = rd(&ct, &mut cur, "token_embd.weight")?.dequantize(&cpu)?;
        // layers
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blk.{i}");
            let norm = |ct: &gguf_file::Content,
                        cur: &mut Cursor<&[u8]>,
                        n: &str|
             -> Result<(Tensor, Tensor)> {
                let w = rd(ct, cur, n)?.dequantize(&cpu)?;
                Ok((w.to_device(&g0)?, w.to_device(&g1)?))
            };
            // QKV bias is present on qwen2 but absent on llama/mistral/mistral3 - read it
            // only when the GGUF actually carries the tensor, else None (no-bias dense arch).
            let bias = |ct: &gguf_file::Content,
                        cur: &mut Cursor<&[u8]>,
                        n: &str|
             -> Result<Option<Tensor>> {
                if ct.tensor_infos.contains_key(n) {
                    Ok(Some(rd(ct, cur, n)?.dequantize(&cpu)?))
                } else {
                    Ok(None)
                }
            };
            let (attn_norm0, attn_norm1) = norm(&ct, &mut cur, &format!("{p}.attn_norm.weight"))?;
            let wq = rd(&ct, &mut cur, &format!("{p}.attn_q.weight"))?;
            let bq = bias(&ct, &mut cur, &format!("{p}.attn_q.bias"))?;
            let wk = rd(&ct, &mut cur, &format!("{p}.attn_k.weight"))?;
            let bk = bias(&ct, &mut cur, &format!("{p}.attn_k.bias"))?;
            let wv = rd(&ct, &mut cur, &format!("{p}.attn_v.weight"))?;
            let bv = bias(&ct, &mut cur, &format!("{p}.attn_v.bias"))?;
            let wo = rd(&ct, &mut cur, &format!("{p}.attn_output.weight"))?;
            let (ffn_norm0, ffn_norm1) = norm(&ct, &mut cur, &format!("{p}.ffn_norm.weight"))?;
            let wg = rd(&ct, &mut cur, &format!("{p}.ffn_gate.weight"))?;
            let wu = rd(&ct, &mut cur, &format!("{p}.ffn_up.weight"))?;
            let wd = rd(&ct, &mut cur, &format!("{p}.ffn_down.weight"))?;
            layers.push(TpLayer {
                attn_norm0,
                attn_norm1,
                ffn_norm0,
                ffn_norm1,
                qk: TpColumnFused::from_qtensors(
                    &[(&wq, bq.as_ref()), (&wk, bk.as_ref())],
                    &g0,
                    &g1,
                )?,
                v: TpColumn::from_qtensor_bias(&wv, bv.as_ref(), &g0, &g1)?,
                o: TpRow::from_qtensor(&wo, &g0, &g1)?,
                // Asymmetric FFN split: GPU0 (5070 Ti) is ~2x the matmul throughput of GPU1
                // (5060 Ti), so give it ~2/3 of the intermediate rows -> both finish together,
                // killing GPU0's idle-wait at the FFN all-reduce (nsys-measured). gate/up rows
                // and down cols split at the SAME point (asym_split(27648,2,3)=18432, QK_K-aligned).
                gate_up: TpColumnFused::from_qtensors_frac(
                    &[(&wg, None), (&wu, None)],
                    ffn_num,
                    ffn_den,
                    &g0,
                    &g1,
                )?,
                down: TpRow::from_qtensor_frac(&wd, ffn_num, ffn_den, &g0, &g1)?,
                kv0: None,
                kv1: None,
            });
        }
        // (verified: deepseek-r1:32b qwen2 - layers=64 n_head=40 n_kv=8 hd=128 hidden=5120
        //  rope_base=1e6 eps=1e-5 vocab=152064; attn_q/output=Q4K, ffn_down=Q6K - all QK_K=256
        //  so the row-parallel slice_qtensor_cols block math is exact.)
        let out_norm = rd(&ct, &mut cur, "output_norm.weight")?
            .dequantize(&cpu)?
            .to_device(&g0)?;
        let lm_qt = match ct.tensor(&mut cur, "output.weight", &g0) {
            Ok(qt) => qt,
            Err(_) => ct.tensor(&mut Cursor::new(mmap_bytes), "token_embd.weight", &g0)?, // tied
        };
        let lm_head = QMatMul::from_arc(Arc::new(lm_qt))?;
        let (cos0, sin0) = rope_tables(cfg.head_dim, cfg.rope_base, &g0)?;
        let (cos1, sin1) = rope_tables(cfg.head_dim, cfg.rope_base, &g1)?;
        let comms = make_comms(&g0, &g1)?;
        let ar_f16 = ar_f16_enabled();
        Ok(Self {
            cfg,
            g0,
            g1,
            embed,
            layers,
            out_norm,
            lm_head,
            cos0,
            sin0,
            cos1,
            sin1,
            comms,
            ar_f16,
        })
    }

    pub fn reset_kv(&mut self) {
        for l in &mut self.layers {
            l.kv0 = None;
            l.kv1 = None;
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
            trim_one(&mut l.kv0);
            trim_one(&mut l.kv1);
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
        p0: &Tensor,
        p1: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if f16 {
            all_reduce_sum_nccl_f16(p0, p1, comms)
        } else {
            all_reduce_sum_nccl(p0, p1, comms)
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
        let (nh2, nkv2) = (nh / 2, nkv / 2);
        let eps = self.cfg.eps;
        // embed -> [seq, hidden] on g0, replicate to g1
        let ids = Tensor::new(tokens, &Device::Cpu)?;
        let emb = self.embed.index_select(&ids, 0)?; // [seq, hidden] CPU
        let mut x0 = emb
            .to_device(&self.g0)?
            .to_dtype(crate::tensor::DType::F32)?;
        let mut x1 = x0.to_device(&self.g1)?;

        let ar_f16 = self.ar_f16;
        let comms = &self.comms;
        for l in self.layers.iter_mut() {
            // -- attention --
            let h0 = rmsnorm(&x0, &l.attn_norm0, eps)?;
            let h1 = rmsnorm(&x1, &l.attn_norm1, eps)?;
            // Fused QK: one matmul/rank, then split [q|k] (decode seq=1 narrows are zero-copy);
            // v stays separate (Q6_K). q/k/v all share input h0/h1.
            let (qk0, qk1) = l.qk.forward(&h0, &h1)?;
            let s0 = l.qk.split(&qk0, 0)?;
            let s1 = l.qk.split(&qk1, 1)?;
            let (v0, v1) = l.v.forward(&h0, &h1)?;
            let ri = self.cfg.use_rope_i;
            let a0 = Self::rank_attn(
                &s0[0], &s0[1], &v0, &mut l.kv0, nh2, nkv2, hd, pos, &self.cos0, &self.sin0, ri,
            )?;
            let a1 = Self::rank_attn(
                &s1[0], &s1[1], &v1, &mut l.kv1, nh2, nkv2, hd, pos, &self.cos1, &self.sin1, ri,
            )?;
            let (p0, p1) = l.o.forward(&a0, &a1)?;
            let (o0, o1) = Self::all_reduce(ar_f16, comms, &p0, &p1)?;
            x0 = (&x0 + &o0)?;
            x1 = (&x1 + &o1)?;
            // -- FFN --
            let h0 = rmsnorm(&x0, &l.ffn_norm0, eps)?;
            let h1 = rmsnorm(&x1, &l.ffn_norm1, eps)?;
            // Fused gate|up: one matmul/rank, then split + fused SiLU.mul.
            let (gu0, gu1) = l.gate_up.forward(&h0, &h1)?;
            let p0 = l.gate_up.split(&gu0, 0)?;
            let p1 = l.gate_up.split(&gu1, 1)?;
            let m0 = crate::inference::kernel::fused::fused_silu_mul(&p0[0], &p0[1])?;
            let m1 = crate::inference::kernel::fused::fused_silu_mul(&p1[0], &p1[1])?;
            let (dp0, dp1) = l.down.forward(&m0, &m1)?;
            let (f0, f1) = Self::all_reduce(ar_f16, comms, &dp0, &dp1)?;
            x0 = (&x0 + &f0)?;
            x1 = (&x1 + &f1)?;
        }
        // final norm + lm_head on g0, last token only
        let last = x0.i(seq - 1)?.unsqueeze(0)?; // [1, hidden]
        let h = rmsnorm(&last, &self.out_norm, eps)?;
        let logits = self.lm_head.forward(&h)?; // [1, vocab]
        Ok(logits.squeeze(0)?)
    }
}
