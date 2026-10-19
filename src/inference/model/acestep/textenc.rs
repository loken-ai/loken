//! ACE-Step 1.5 - text encoder ( M1): Qwen3-Embedding-0.6B (separate GGUF,
//! arch `acestep-text-enc`), at the width, heads and depth its tensors have; θ1e6,
//! **causal**, vocab embed lookup. Produces `text_hidden [S,H]` that the condition
//! encoder's text_projector maps to the DiT width.
//! Reuses the Qwen3 pre-norm encoder layer (DetokLayer) validated by the FSQ
//! detok / cond encoder; only the causal mask + vocab embed differ.

use crate::inference::model::acestep::fsq::{
    detok_load_t, load_component, load_stack, rope_tables, DetokLayer, StackGeometry,
};
use crate::inference::model::qwen3::encoder::load_matrix;
use crate::tensor::{Device, Tensor};

pub struct TextEncoder {
    pub embed_tokens: Tensor, // [V, H], half precision
    pub layers: Vec<DetokLayer>,
    pub norm: Tensor, // [H]
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub device: Device,
}

impl TextEncoder {
    /// Load Qwen3-Embedding-0.6B-Q8_0.gguf (`embed_tokens` / `layers.{i}` / `norm`).
    pub fn from_gguf(path: &str) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let c = gguf_file::read_mapped_file(&f)?;
        // The checkpoint holds nothing but the encoder, so every tensor counts.
        load_component("text encoder", &c, &[], |device| {
            let g = StackGeometry::read(&c, "")?;
            Ok(TextEncoder {
                layers: load_stack(&c, &mut f, "", g.layers, device)?,
                embed_tokens: load_matrix(&c, &mut f, "embed_tokens.weight", device)?,
                norm: detok_load_t(&c, &mut f, "norm.weight", device)?,
                hidden: g.hidden,
                n_head: g.n_head,
                n_kv: g.n_kv,
                head_dim: g.head_dim,
                rope_theta: 1e6,
                device: device.clone(),
            })
        })
    }

    fn rms_eps(&self) -> f32 {
        1e-6
    }

    fn qk_roped(
        &self,
        lin: &crate::inference::model::qwen3::encoder::Linear,
        qknorm: &Tensor,
        x: &Tensor,
        n_heads: usize,
        s: usize,
    ) -> crate::tensor::Result<Tensor> {
        let d = self.head_dim;
        let proj = lin.forward(x)?;
        let q = proj
            .reshape((s, n_heads, d))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let q = q.rms_norm(qknorm, self.rms_eps())?;
        let (cosv, sinv) = rope_tables(s, d, self.rope_theta);
        let cos = Tensor::from_vec_f32(cosv, (s, d / 2))?.to_device(&self.device)?;
        let sin = Tensor::from_vec_f32(sinv, (s, d / 2))?.to_device(&self.device)?;
        q.rope(&cos, &sin) // [1,heads,S,D]
    }

    /// Causal GQA self-attention: `x [S,H]` -> `[S,H]` (key `ki` visible iff `ki<=qi`).
    fn self_attn(&self, l: &DetokLayer, x: &Tensor, s: usize) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::sdpa::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.qk_roped(&l.q, &l.q_norm, x, nh, s)?; // [1,nh,s,d]
        let k = self.qk_roped(&l.k, &l.k_norm, x, nkv, s)?; // [1,nkv,s,d]
        let v =
            l.v.forward(x)?
                .reshape((s, nkv, d))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?; // [1,nkv,s,d]
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?); // [1,nh,s,d]
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = sdpa(&q, &k, &v, None, true, scale, 1.0)?; // causal mask
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.o.forward(&ao) // [s,h]
    }

    fn enc_layer(&self, l: &DetokLayer, x: &Tensor, s: usize) -> crate::tensor::Result<Tensor> {
        let eps = self.rms_eps();
        let norm = x.rms_norm(&l.input_ln, eps)?;
        let x = x.add(&self.self_attn(l, &norm, s)?)?;
        let norm2 = x.rms_norm(&l.post_ln, eps)?;
        let gate = l.gate.forward(&norm2)?;
        let up = l.up.forward(&norm2)?;
        let ff = gate.silu()?.mul(&up)?;
        x.add(&l.down.forward(&ff)?)
    }

    /// Vocab embed lookup only (no encoder layers) -> `[S,H]` row-major - the
    /// cond encoder's `lyric_embed` (oracle qwen3_embed_lookup).
    pub fn embed_lookup(&self, token_ids: &[u32]) -> crate::tensor::Result<Vec<f32>> {
        let h = self.hidden;
        let mut out = vec![0f32; token_ids.len() * h];
        for (i, &id) in token_ids.iter().enumerate() {
            let row: Vec<f32> = self
                .embed_tokens
                .narrow(0, id as usize, 1)?
                .flatten_all()?
                .to_vec1_f32()?;
            out[i * h..(i + 1) * h].copy_from_slice(&row);
        }
        Ok(out)
    }

    /// Encode token ids -> `text_hidden [S,H]` (row-major). Vocab embed lookup ->
    /// causal Qwen3 layers -> final norm.
    pub fn forward(&self, token_ids: &[u32]) -> crate::tensor::Result<Vec<f32>> {
        let s = token_ids.len();
        // vocab embed lookup (gather rows) - embed_tokens is on device, so is `hid`.
        let rows: Vec<Tensor> = token_ids
            .iter()
            .map(|&id| self.embed_tokens.narrow(0, id as usize, 1))
            .collect::<crate::tensor::Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        let mut hid = Tensor::cat(&refs, 0)?.to_dtype(crate::tensor::DType::F32)?; // [s,h] on device
        for l in &self.layers {
            hid = self.enc_layer(l, &hid, s)?;
        }
        hid.rms_norm(&self.norm, self.rms_eps())?
            .flatten_all()?
            .to_vec1_f32()
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    fn load_dump(path: &str) -> (Vec<f32>, Vec<usize>) {
        let b = std::fs::read(path).unwrap();
        let nd = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut shape = Vec::with_capacity(nd);
        for i in 0..nd {
            shape.push(i32::from_le_bytes(b[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize);
        }
        let off = 4 + nd * 4;
        let data = b[off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (data, shape)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    #[ignore = "needs Qwen3-Embedding GGUF (config.test HF hub) + /tmp/acedump_detok dumps - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_text_encoder_vs_oracle() {
        let gguf = crate::inference::cache::hf::file(
            "models--Serveurperso--ACE-Step-1.5-GGUF",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
        );
        let m = TextEncoder::from_gguf(gguf.to_str().unwrap()).unwrap();
        let (ids_f, _) = load_dump("/tmp/acedump_detok/text_ids.bin");
        let ids: Vec<u32> = ids_f.iter().map(|&x| x as u32).collect();
        let (th, ths) = load_dump("/tmp/acedump_detok/text_hidden.bin"); // [S,1024]
        let out = m.forward(&ids).unwrap();
        assert_eq!(ths, vec![ids.len(), 1024], "shape");
        let c = cosine(&out, &th);
        println!("text encoder cosine={c:.6} (S={})", ids.len());
        assert!(c > 0.999, "text encoder cosine {c} too low");
    }
}

impl TextEncoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        let layers: Vec<_> = self.layers.iter().collect();
        let mut runs =
            crate::inference::model::qwen3::encoder::stack_placement(&self.device, &layers);
        if let Some(first) = runs.first_mut() {
            first.bytes +=
                crate::inference::model::qwen3::encoder::tensor_bytes(&self.embed_tokens);
        }
        runs
    }
}
