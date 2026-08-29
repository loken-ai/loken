//! ACE-Step 1.5 - text encoder ( M1): Qwen3-Embedding-0.6B (separate GGUF,
//! arch `acestep-text-enc`). 28L, H=1024, 16 q / 8 kv heads, head_dim 128,
//! ffn 3072, θ1e6, **causal**, vocab embed lookup. Produces `text_hidden [S,1024]`
//! that the condition encoder's text_projector maps to 2048 (see native_acestep_cond).
//! Reuses the Qwen3 pre-norm encoder layer (DetokLayer) validated by the FSQ
//! detok / cond encoder; only the causal mask + vocab embed differ.

use crate::inference::model::qwen3::encoder::{
    lin as detok_lin, load_t as detok_load_t, rope_tables, Layer as DetokLayer,
};
use crate::tensor::{Device, Tensor};

pub struct TextEncoder {
    pub embed_tokens: Tensor,    // [V, 1024]
    pub layers: Vec<DetokLayer>, // 28
    pub norm: Tensor,            // [1024]
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
        // See `gguf_resident_bytes`. This checkpoint holds nothing but the encoder, so
        // every tensor counts - but each is expanded to f32 on the way to the card, so
        // the quantised file under-stated what goes resident by four.
        let sz = crate::inference::model::acestep::ops::gguf_resident_bytes(&c, &[]);
        // The fastest card that holds it, else the host - the fleet's own answer for a
        // component loaded once. The reserve is a share of the weights because the
        // activations of a fixed-shape encoder scale with its width, not with a request.
        let device = crate::inference::place::plan::place_whole(sz, sz / 4);
        let mut layers = Vec::with_capacity(28);
        for l in 0..28usize {
            let p = format!("layers.{l}");
            layers.push(DetokLayer {
                input_ln: detok_load_t(
                    &c,
                    &mut f,
                    &format!("{p}.input_layernorm.weight"),
                    &device,
                )?,
                post_ln: detok_load_t(
                    &c,
                    &mut f,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &device,
                )?,
                q: detok_lin(&c, &mut f, &format!("{p}.self_attn.q_proj"), false, &device)?,
                k: detok_lin(&c, &mut f, &format!("{p}.self_attn.k_proj"), false, &device)?,
                v: detok_lin(&c, &mut f, &format!("{p}.self_attn.v_proj"), false, &device)?,
                o: detok_lin(&c, &mut f, &format!("{p}.self_attn.o_proj"), false, &device)?,
                q_norm: detok_load_t(&c, &mut f, &format!("{p}.self_attn.q_norm.weight"), &device)?,
                k_norm: detok_load_t(&c, &mut f, &format!("{p}.self_attn.k_norm.weight"), &device)?,
                gate: detok_lin(&c, &mut f, &format!("{p}.mlp.gate_proj"), false, &device)?,
                up: detok_lin(&c, &mut f, &format!("{p}.mlp.up_proj"), false, &device)?,
                down: detok_lin(&c, &mut f, &format!("{p}.mlp.down_proj"), false, &device)?,
            });
        }
        Ok(TextEncoder {
            embed_tokens: detok_load_t(&c, &mut f, "embed_tokens.weight", &device)?,
            layers,
            norm: detok_load_t(&c, &mut f, "norm.weight", &device)?,
            hidden: 1024,
            n_head: 16,
            n_kv: 8,
            head_dim: 128,
            rope_theta: 1e6,
            device,
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
        let proj = x.matmul_t(lin.weight()?)?;
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
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.qk_roped(&l.q, &l.q_norm, x, nh, s)?; // [1,nh,s,d]
        let k = self.qk_roped(&l.k, &l.k_norm, x, nkv, s)?; // [1,nkv,s,d]
        let v = x
            .matmul_t(l.v.weight()?)?
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
        let gate = norm2.matmul_t(l.gate.weight()?)?;
        let up = norm2.matmul_t(l.up.weight()?)?;
        let ff = gate.silu()?.mul(&up)?;
        x.add(&ff.matmul_t(l.down.weight()?)?)
    }

    /// Vocab embed lookup only (no encoder layers) -> `[S,1024]` row-major - the
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

    /// Encode token ids -> `text_hidden [S,1024]` (row-major). Vocab embed lookup ->
    /// 28 causal Qwen3 layers -> final norm.
    pub fn forward(&self, token_ids: &[u32]) -> crate::tensor::Result<Vec<f32>> {
        let s = token_ids.len();
        // vocab embed lookup (gather rows) - embed_tokens is on device, so is `hid`.
        let rows: Vec<Tensor> = token_ids
            .iter()
            .map(|&id| self.embed_tokens.narrow(0, id as usize, 1))
            .collect::<crate::tensor::Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        let mut hid = Tensor::cat(&refs, 0)?; // [s,h] on device
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
