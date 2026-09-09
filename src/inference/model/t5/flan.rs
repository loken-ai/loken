//! T5 text encoder family - FLAN-T5-large (v1.1 / gated-gelu, EzAudio) and t5-base
//! (v1.0 / ReLU, Stable Audio Open). Full Rust on the native substrate (no external tensor library,
//! no extra crates, no Python).
//!
//! Encoder-only: token embedding -> PRE-norm blocks (self-attention with a SHARED
//! relative-position bias and NO 1/√d scaling, then the FFN - gated-GELU for v1.1,
//! plain ReLU for v1.0) -> final RMSNorm -> `[S, d_model]`. T5 LayerNorm is scale-only
//! RMSNorm (no bias, no mean-subtraction), so it maps directly onto our `RmsNorm`.
//! Weights load from `model.safetensors` (we use only the `encoder.*` + `shared`
//! tensors); geometry comes from `config.json` and the FFN variant is auto-detected
//! from the tensor names. One prompt at a time (no padding / no batch).

use crate::tensor::layer::{Embedding, Linear, RmsNorm};
use crate::tensor::safetensors_io::SafeTensorsLoader;
use crate::tensor::{Device, Error, Result, Tensor};

const D_KV: usize = 64;
const NUM_BUCKETS: usize = 32;
const MAX_DISTANCE: usize = 128;
const EPS: f32 = 1e-6;

struct T5Block {
    ln_attn: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln_ff: RmsNorm,
    /// `wi1 = Some` -> v1.1 gated-GELU (`wo(gelu(wi0 x) * wi1 x)`);
    /// `wi1 = None` -> v1.0 ReLU (`wo(relu(wi0 x))`).
    wi0: Linear,
    wi1: Option<Linear>,
    wo: Linear,
}

pub struct T5Encoder {
    embed: Embedding,
    blocks: Vec<T5Block>,
    final_ln: RmsNorm,
    n_heads: usize,
    /// `relative_attention_bias.weight` [NUM_BUCKETS, n_heads] kept on host (tiny); the
    /// per-sequence additive bias is built from it at encode time and shared by all blocks.
    rel_bias: Vec<f32>,
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

/// Read `num_layers` / `num_heads` from the model dir's `config.json`.
fn read_geometry(dir: &str) -> Result<(usize, usize)> {
    let raw = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| Error(format!("t5 config.json: {e}")))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| Error(format!("t5 config.json: {e}")))?;
    let n = |k: &str| -> Result<usize> {
        cfg[k]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| Error(format!("t5 config.json: missing {k}")))
    };
    Ok((n("num_layers")?, n("num_heads")?))
}

impl T5Encoder {
    /// Load an encoder from a T5 `model.safetensors` directory (any size/variant).
    pub fn from_dir(dir: &str) -> Result<Self> {
        let (n_layers, n_heads) = read_geometry(dir)?;
        let path = format!("{dir}/model.safetensors");
        let sz = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(3_000_000_000);
        // The fastest card that holds it, else the host - the fleet's own answer for a
        // component loaded once, rather than a helper belonging to another family.
        let device = crate::inference::place::plan::place_whole(sz, sz / 4);
        let st = unsafe { SafeTensorsLoader::multi(&[&path]) }?;
        let get = |name: &str| -> Result<Tensor> { st.load(name)?.to_device(&device) };
        let lin =
            |name: &str| -> Result<Linear> { Linear::new(get(&format!("{name}.weight"))?, None) };
        let rms = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(get(&format!("{name}.weight"))?, EPS))
        };

        let embed = Embedding::new(get("shared.weight")?);
        // v1.1 checkpoints ship gated `wi_0`/`wi_1`; v1.0 a single `wi` (ReLU).
        let gated = st
            .load("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
            .is_ok();
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let a = format!("encoder.block.{i}.layer.0");
            let f = format!("encoder.block.{i}.layer.1");
            let (wi0, wi1) = if gated {
                (
                    lin(&format!("{f}.DenseReluDense.wi_0"))?,
                    Some(lin(&format!("{f}.DenseReluDense.wi_1"))?),
                )
            } else {
                (lin(&format!("{f}.DenseReluDense.wi"))?, None)
            };
            blocks.push(T5Block {
                ln_attn: rms(&format!("{a}.layer_norm"))?,
                q: lin(&format!("{a}.SelfAttention.q"))?,
                k: lin(&format!("{a}.SelfAttention.k"))?,
                v: lin(&format!("{a}.SelfAttention.v"))?,
                o: lin(&format!("{a}.SelfAttention.o"))?,
                ln_ff: rms(&format!("{f}.layer_norm"))?,
                wi0,
                wi1,
                wo: lin(&format!("{f}.DenseReluDense.wo"))?,
            });
        }
        let final_ln = rms("encoder.final_layer_norm")?;
        // Shared relative-position bias lives only in block 0 (T5 convention).
        let rel = get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let rel_bias = rel.to_vec_f32(); // [NUM_BUCKETS, n_heads] row-major
        Ok(Self {
            embed,
            blocks,
            final_ln,
            n_heads,
            rel_bias,
            device,
        })
    }

    /// Encode token ids -> `[S, D_MODEL]` (on the encoder's device, F32).
    pub fn encode(&self, ids: &[u32]) -> Result<Tensor> {
        let s = ids.len();
        if s == 0 {
            return Err(Error("t5: empty token sequence".into()));
        }
        let ids_t = Tensor::from_vec_u32(ids.to_vec(), (s,))?.to_device(&self.device)?;
        let mut x = self.embed.forward(&ids_t)?; // [S, D_MODEL] (no embedding scaling in T5)

        // Additive relative-position bias [n_heads, S, S], shared across blocks.
        let nh = self.n_heads;
        let mut bias = vec![0f32; nh * s * s];
        for i in 0..s {
            for j in 0..s {
                let b = rel_bucket(j as i64 - i as i64);
                for h in 0..nh {
                    bias[h * s * s + i * s + j] = self.rel_bias[b * nh + h];
                }
            }
        }
        let bias = Tensor::from_vec_f32(bias, (nh, s, s))?.to_device(&self.device)?;

        for blk in &self.blocks {
            let h = blk.ln_attn.forward(&x)?;
            x = x.add(&self.attn(blk, &h, &bias, s)?)?;
            let h2 = blk.ln_ff.forward(&x)?;
            let inner = match &blk.wi1 {
                Some(wi1) => blk.wi0.forward(&h2)?.gelu()?.mul(&wi1.forward(&h2)?)?,
                None => blk.wi0.forward(&h2)?.relu()?,
            };
            x = x.add(&blk.wo.forward(&inner)?)?;
        }
        self.final_ln.forward(&x)
    }

    /// T5 self-attention: NO 1/√d scaling (folded into the trained weights); add the shared
    /// relative-position bias to the scores before softmax.
    fn attn(&self, blk: &T5Block, h: &Tensor, bias: &Tensor, s: usize) -> Result<Tensor> {
        let nh = self.n_heads;
        let inner = nh * D_KV;
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((s, nh, D_KV))?.transpose(0, 1)?.contiguous() // [H,S,D_KV]
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

/// The repository FLAN-T5-large is published in.
const T5_REPO: &str = "models--google--flan-t5-large";

/// The FLAN-T5-large snapshot holding the weights, or the `main` path so a caller's
/// error names where they were expected.
pub fn flan_t5_dir() -> String {
    crate::inference::cache::hf::snapshot_with(T5_REPO, "model.safetensors")
        .unwrap_or_else(|| crate::inference::cache::hf::snapshots(T5_REPO).join("main"))
        .to_string_lossy()
        .into_owned()
}

/// Tokenize a prompt with flan-t5's tokenizer.json and append the T5 eos (`</s>` = id 1).
pub fn t5_tokenize(dir: &str, text: &str) -> Result<Vec<u32>> {
    let tk = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| Error(format!("t5 tokenizer: {e}")))?;
    let enc = tk
        .encode(text, false)
        .map_err(|e| Error(format!("t5 encode: {e}")))?;
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
    use super::*;
    #[test]
    #[ignore = "needs flan-t5-large model.safetensors (config hf models dir)"]
    fn sanity_encode() {
        let dir = flan_t5_dir();
        let dir = dir.as_str();
        let ids = t5_tokenize(dir, "dog barking in the rain").unwrap();
        let enc = T5Encoder::from_dir(dir).unwrap();
        let out = enc.encode(&ids).unwrap();
        let d = out.dims();
        assert_eq!(d.len(), 2);
        assert_eq!(d[1], 1024);
        assert_eq!(d[0], ids.len());
        let v = out.to_vec_f32();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        println!(
            "T5 encode ok: {} tokens x {} dim, mean={:.4} std={:.4}",
            d[0],
            d[1],
            mean,
            var.sqrt()
        );
    }
}

impl T5Encoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.blocks.len())
    }
}
