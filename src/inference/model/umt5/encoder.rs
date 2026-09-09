//! Wan 2.1 video (#?? S2): full-Rust umT5-XXL text encoder (prompt -> context `[S, 4096]`)
//! for the Wan DiT cross-attention. Built on `crate::tensor`, with no external tensor library and no Python.
//!
//! umT5 is the same T5 v1.1 family as our FLAN-T5 encoder (gated-gelu FFN, scale-only
//! RMSNorm, no-bias qkv, NO 1/√d attention scaling, relative-position bucket bias) with two
//! differences handled here: (1) `shared_pos=False` - each of the 24 encoder blocks owns its
//! OWN relative-position embedding table, so the additive position bias is rebuilt per layer
//! from that block's table (NOT the shared block-0 table FLAN-T5 uses); (2) larger geometry
//! (dim 4096 / dim_ffn 10240 / 64 heads x 64 / 24 layers) with bf16 weights from the Wan
//! `.pth`, loaded and computed in f32 on CPU for the sanity test.
//!
//! The Wan checkpoint uses `wan/modules/t5.py` module names (`token_embedding`,
//! `blocks.{i}.{attn.{q,k,v,o}, norm1, ffn.{gate.0,fc1,fc2}, norm2, pos_embedding}`, `norm`),
//! NOT HF `encoder.block...` names - see the key-dump test for the locked mapping.

use crate::tensor::layer::{Embedding, Linear, RmsNorm};
use crate::tensor::pth::read_pt;
use crate::tensor::{Device, Error, Result, Tensor};
use std::collections::HashMap;

// umT5-XXL geometry (confirmed against the Wan checkpoint).
const N_LAYERS: usize = 24;
const N_HEADS: usize = 64;
const D_KV: usize = 64;
const NUM_BUCKETS: usize = 32;
const MAX_DISTANCE: usize = 128;
const EPS: f32 = 1e-6;

struct Umt5Block {
    norm1: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm2: RmsNorm,
    /// gated-gelu FFN: `fc2( gelu(gate(x)) * fc1(x) )`.
    gate: Linear,
    fc1: Linear,
    fc2: Linear,
    /// This block's OWN relative-position table `[NUM_BUCKETS, N_HEADS]` (host, row-major).
    rel_bias: Vec<f32>,
}

pub struct Umt5Encoder {
    embed: Embedding,
    blocks: Vec<Umt5Block>,
    final_norm: RmsNorm,
    device: Device,
}

/// T5 bidirectional relative-position bucket (port of `_relative_position_bucket`).
/// `rel = key_pos - query_pos`.
fn rel_bucket(rel: i64) -> usize {
    let mut ret = 0usize;
    let nb = NUM_BUCKETS / 2; // bidirectional -> half the buckets carry the sign
    if rel > 0 {
        ret += nb;
    }
    let n = rel.unsigned_abs() as usize;
    let max_exact = nb / 2;
    if n < max_exact {
        ret + n
    } else {
        let large = max_exact
            + ((n as f64 / max_exact as f64).ln() / (MAX_DISTANCE as f64 / max_exact as f64).ln()
                * (nb - max_exact) as f64) as usize;
        ret + large.min(nb - 1)
    }
}

impl Umt5Encoder {
    /// Load the umT5-XXL encoder from the Wan `.pth` (bf16 on disk -> f32 on `device`).
    pub fn from_pth(path: &str, device: Device) -> Result<Self> {
        Self::from_pth_dtype(path, device, crate::tensor::DType::F32)
    }

    /// Load the encoder in an explicit compute `dtype`. F32 is the reference (CPU); a
    /// 2-byte dtype HALVES the resident footprint, which is what lets the XXL encoder fit a
    /// card at all. The per-layer relative-position tables stay host-f32 (`rel_bias`);
    /// `encode` casts the additive bias it builds to the active tensors' dtype, so the math
    /// stays consistent.
    pub fn from_pth_dtype(path: &str, device: Device, dtype: crate::tensor::DType) -> Result<Self> {
        let _prof = std::env::var("UMT5_PROF").is_ok();
        let _t0 = std::time::Instant::now();
        let raw: HashMap<String, Tensor> = read_pt(path)?;
        if _prof {
            eprintln!(
                "[umt5-prof] read_pt (parse+parallel decode): {:.1}s",
                _t0.elapsed().as_secs_f64()
            );
        }
        let _t1 = std::time::Instant::now();
        let get = |name: &str| -> Result<Tensor> {
            raw.get(name)
                .ok_or_else(|| Error(format!("umt5: missing tensor `{name}`")))?
                .to_dtype(dtype)?
                .to_device(&device)
        };
        let lin =
            |name: &str| -> Result<Linear> { Linear::new(get(&format!("{name}.weight"))?, None) };
        let rms = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(get(&format!("{name}.weight"))?, EPS))
        };

        let embed = Embedding::new(get("token_embedding.weight")?);
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let b = format!("blocks.{i}");
            // Per-layer relative-position table (shared_pos=False).
            let rel = get(&format!("{b}.pos_embedding.embedding.weight"))?; // [NUM_BUCKETS, N_HEADS]
            blocks.push(Umt5Block {
                norm1: rms(&format!("{b}.norm1"))?,
                q: lin(&format!("{b}.attn.q"))?,
                k: lin(&format!("{b}.attn.k"))?,
                v: lin(&format!("{b}.attn.v"))?,
                o: lin(&format!("{b}.attn.o"))?,
                norm2: rms(&format!("{b}.norm2"))?,
                gate: lin(&format!("{b}.ffn.gate.0"))?,
                fc1: lin(&format!("{b}.ffn.fc1"))?,
                fc2: lin(&format!("{b}.ffn.fc2"))?,
                rel_bias: rel.to_vec_f32(),
            });
        }
        let final_norm = rms("norm")?;
        if _prof {
            eprintln!(
                "[umt5-prof] to_dtype({dtype:?})+to_device+assembly: {:.1}s",
                _t1.elapsed().as_secs_f64()
            );
        }
        Ok(Self {
            embed,
            blocks,
            final_norm,
            device,
        })
    }

    /// Device the encoder weights currently live on.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Copy of the encoder on `dev` converted to `dtype`. Used by the OOM fallback to turn the
    /// cached CPU f16 staging into the F32 the CPU forward requires - one in-RAM conversion
    /// (seconds) instead of re-parsing the 11 GB checkpoint (~2 minutes).
    pub fn to_dtype_on(&self, dev: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        let blocks = self
            .blocks
            .iter()
            .map(|b| -> Result<Umt5Block> {
                Ok(Umt5Block {
                    norm1: b.norm1.to_dtype_on(dev, dtype)?,
                    q: b.q.to_dtype_on(dev, dtype)?,
                    k: b.k.to_dtype_on(dev, dtype)?,
                    v: b.v.to_dtype_on(dev, dtype)?,
                    o: b.o.to_dtype_on(dev, dtype)?,
                    norm2: b.norm2.to_dtype_on(dev, dtype)?,
                    gate: b.gate.to_dtype_on(dev, dtype)?,
                    fc1: b.fc1.to_dtype_on(dev, dtype)?,
                    fc2: b.fc2.to_dtype_on(dev, dtype)?,
                    rel_bias: b.rel_bias.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed: self.embed.to_dtype_on(dev, dtype)?,
            blocks,
            final_norm: self.final_norm.to_dtype_on(dev, dtype)?,
            device: dev.clone(),
        })
    }

    /// Move every weight tensor onto `dev` (CPU<->CUDA). Mirrors the Wan-VAE per-component
    /// `to_device`: the encoder is loaded cheaply on CPU, then relocated to the planner's
    /// chosen card (or kept on CPU as the OOM fallback). The per-layer relative-position
    /// tables stay host-resident (`rel_bias`).
    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        let blocks = self
            .blocks
            .iter()
            .map(|b| -> Result<Umt5Block> {
                Ok(Umt5Block {
                    norm1: b.norm1.to_device(dev)?,
                    q: b.q.to_device(dev)?,
                    k: b.k.to_device(dev)?,
                    v: b.v.to_device(dev)?,
                    o: b.o.to_device(dev)?,
                    norm2: b.norm2.to_device(dev)?,
                    gate: b.gate.to_device(dev)?,
                    fc1: b.fc1.to_device(dev)?,
                    fc2: b.fc2.to_device(dev)?,
                    rel_bias: b.rel_bias.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed: self.embed.to_device(dev)?,
            blocks,
            final_norm: self.final_norm.to_device(dev)?,
            device: dev.clone(),
        })
    }

    /// Encode token ids -> `[S, D_MODEL]` (F32, on the encoder's device).
    pub fn encode(&self, ids: &[u32]) -> Result<Tensor> {
        let s = ids.len();
        if s == 0 {
            return Err(Error("umt5: empty token sequence".into()));
        }
        let ids_t = Tensor::from_vec_u32(ids.to_vec(), (s,))?.to_device(&self.device)?;
        let mut x = self.embed.forward(&ids_t)?; // [S, D_MODEL] (no embedding scaling in T5)
        let dt = x.dtype(); // F32 (CPU reference) or F16 (GPU); bias must match it to add.

        // Bucket id for every (query i, key j) pair - geometry only, reused per layer.
        let mut buckets = vec![0usize; s * s];
        for i in 0..s {
            for j in 0..s {
                buckets[i * s + j] = rel_bucket(j as i64 - i as i64);
            }
        }

        for blk in &self.blocks {
            // Per-layer additive relative-position bias [N_HEADS, S, S] from this block's table.
            let mut bias = vec![0f32; N_HEADS * s * s];
            for i in 0..s {
                for j in 0..s {
                    let b = buckets[i * s + j];
                    for h in 0..N_HEADS {
                        bias[h * s * s + i * s + j] = blk.rel_bias[b * N_HEADS + h];
                    }
                }
            }
            let bias = Tensor::from_vec_f32(bias, (N_HEADS, s, s))?
                .to_device(&self.device)?
                .to_dtype(dt)?;

            let h = blk.norm1.forward(&x)?;
            x = x.add(&self.attn(blk, &h, &bias, s)?)?;
            let h2 = blk.norm2.forward(&x)?;
            // Wan t5.py FFN: fc2( gelu(gate(x)) * fc1(x) ); its GELU is the tanh approximation.
            let ff = blk
                .fc2
                .forward(&blk.gate.forward(&h2)?.gelu()?.mul(&blk.fc1.forward(&h2)?)?)?;
            x = x.add(&ff)?;
        }
        self.final_norm.forward(&x)
    }

    /// T5 self-attention: NO 1/√d scaling (folded into the trained weights); add the per-layer
    /// relative-position bias to the scores before softmax.
    fn attn(&self, blk: &Umt5Block, h: &Tensor, bias: &Tensor, s: usize) -> Result<Tensor> {
        let inner = N_HEADS * D_KV;
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((s, N_HEADS, D_KV))?.transpose(0, 1)?.contiguous() // [H,S,D_KV]
        };
        let q = heads(blk.q.forward(h)?)?;
        let k = heads(blk.k.forward(h)?)?;
        let v = heads(blk.v.forward(h)?)?;
        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?; // [H,S,S]
        let probs = scores.add(bias)?.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?; // [H,S,D_KV]
        let ctx = ctx.transpose(0, 1)?.contiguous()?.reshape((s, inner))?;
        blk.o.forward(&ctx)
    }
}

/// What ONE `encode` holds on its device for a prompt of `seq` tokens at `dtype`, on top of
/// the weights.
///
/// Read off `encode`/`attn` above rather than chosen: each block builds an additive
/// relative-position bias `[N_HEADS, seq, seq]`, uploads it as f32 and casts it to the
/// compute dtype (both copies are live across the cast), and the attention then carries the
/// scores, the biased scores and the probabilities in that same shape. The bias is rebuilt
/// per block instead of accumulated, so the peak is ONE block's and does not grow with their
/// number - it grows with the SQUARE of the prompt, and T5 attention is unusually wide
/// (every head keeps a full `seq x seq` map). A placement that charges the weights alone
/// therefore approves a card with no room left to run on, and the failure lands mid-encode.
pub fn umt5_encode_peak_bytes(seq: usize, dtype: crate::tensor::DType) -> u64 {
    /// `[N_HEADS, seq, seq]` buffers alive at once in `attn`: the scores, the biased
    /// scores, the probabilities.
    const LIVE_SCORE_MAPS: u64 = 3;
    /// `[seq, d_model]` buffers alive at once in a block: the residual, its normalised
    /// copy, q, k, v and the attention output.
    const LIVE_HIDDEN_MAPS: u64 = 6;
    let elem = dtype.size_in_bytes() as u64;
    let f32_elem = std::mem::size_of::<f32>() as u64;
    let d_model = (N_HEADS * D_KV) as u64;
    let score_map = (N_HEADS as u64) * (seq as u64) * (seq as u64);
    // The bias is staged as f32 and cast, so it costs one of each.
    let bias = score_map.saturating_mul(f32_elem + elem);
    let scores = score_map.saturating_mul(elem * LIVE_SCORE_MAPS);
    let hidden = (seq as u64).saturating_mul(d_model * elem * LIVE_HIDDEN_MAPS);
    bias.saturating_add(scores).saturating_add(hidden)
}

/// Resolve the umT5-XXL encoder `.pth` inside the Wan2.1 snapshot (via the S1 resolver).
pub fn umt5_pth() -> std::path::PathBuf {
    crate::inference::model::wan::vae::wan_file("models_t5_umt5-xxl-enc-bf16.pth")
}

/// Resolve the umT5-XXL `tokenizer.json` inside the Wan2.1 snapshot.
pub fn umt5_tokenizer_path() -> std::path::PathBuf {
    crate::inference::model::wan::vae::wan_file("google/umt5-xxl/tokenizer.json")
}

/// Tokenize a prompt with the umT5 tokenizer.json and append the T5 eos (`</s>` = id 1).
pub fn umt5_tokenize(text: &str) -> Result<Vec<u32>> {
    let path = umt5_tokenizer_path();
    let tk = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| Error(format!("umt5 tokenizer ({path:?}): {e}")))?;
    let enc = tk
        .encode(text, false)
        .map_err(|e| Error(format!("umt5 encode: {e}")))?;
    let mut ids = enc.get_ids().to_vec();
    if ids.last() != Some(&1) {
        ids.push(1);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    /// Regression gate for the all-white-video bug: umT5-XXL activations
    /// OVERFLOW f16 (T5 has no attention scaling), so an all-F16 GPU forward
    /// returns 100% NaN -> NaN latents -> saturated white frames. The GPU
    /// path must run BF16 (f32's exponent, same footprint). Asserts zero NaN
    /// and closeness to the F32 CPU reference.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "loads the 11 GB umT5 on GPU"]
    fn umt5_gpu_bf16_matches_cpu_f32() {
        use super::*;
        use crate::tensor::DType;
        let path = umt5_pth();
        let path = path.to_str().unwrap();
        let ids = umt5_tokenize("a red car driving on a coastal road").unwrap();
        let cd = crate::tensor::cuda::CudaDevice::new(0).unwrap();
        let dev = crate::tensor::Device::Cuda(cd);
        let encb = Umt5Encoder::from_pth_dtype(path, dev, DType::BF16).unwrap();
        let outb = encb
            .encode(&ids)
            .unwrap()
            .to_device(&crate::tensor::Device::Cpu)
            .unwrap()
            .to_vec_f32();
        assert_eq!(
            outb.iter().filter(|v| v.is_nan()).count(),
            0,
            "GPU BF16 encode has NaNs"
        );
        drop(encb);
        let encc = Umt5Encoder::from_pth(path, crate::tensor::Device::Cpu).unwrap();
        let refv = encc.encode(&ids).unwrap().to_vec_f32();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in outb.iter().zip(&refv) {
            num += ((*a - *b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        eprintln!("umT5 GPU BF16 vs CPU F32 rel_rms = {rel:.4}");
        assert!(rel < 0.05, "GPU BF16 encode drifted: rel_rms {rel:.4}");
    }

    use super::*;

    /// Dump the real `.pth` key scheme + shapes/dtypes to lock the module mapping.
    #[test]
    #[ignore = "needs Wan2.1 umT5-XXL .pth (config hf models dir)"]
    fn dump_keys() {
        let path = umt5_pth();
        println!("umt5 pth: {path:?}");
        let raw = read_pt(path.to_str().unwrap()).unwrap();
        let mut keys: Vec<&String> = raw.keys().collect();
        keys.sort();
        println!("total tensors: {}", keys.len());
        for k in &keys {
            // Print block 0 + the non-block (top-level) keys in full; summarize the rest.
            if k.starts_with("blocks.0.") || !k.starts_with("blocks.") {
                let t = &raw[*k];
                println!("  {k}  {:?} {:?}", t.dims(), t.dtype());
            }
        }
    }

    /// Sanity: tokenize -> encode -> assert shape [S,4096], all finite, print mean/std.
    #[test]
    #[ignore = "needs Wan2.1 umT5-XXL .pth + tokenizer (config hf models dir)"]
    fn sanity_encode() {
        let ids = umt5_tokenize("a cat playing piano").unwrap();
        println!("tokens ({}): {ids:?}", ids.len());
        let enc = Umt5Encoder::from_pth(umt5_pth().to_str().unwrap(), Device::Cpu).unwrap();
        let out = enc.encode(&ids).unwrap();
        let d = out.dims();
        assert_eq!(d.len(), 2);
        assert_eq!(d[0], ids.len());
        assert_eq!(d[1], 4096);
        let v = out.to_vec_f32();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        println!(
            "umt5 encode ok: {} tokens x {} dim, mean={:.4} std={:.4}",
            d[0],
            d[1],
            mean,
            var.sqrt()
        );
    }
}

impl Umt5Encoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.blocks.len())
    }
}
