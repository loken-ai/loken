//! ACE-Step 1.5 - condition encoder ( M1). Builds the DiT cross-attn source
//! `enc_hidden [S_total, 2048]` packed as `cat(lyric, timbre[0:1], text_proj)`
//! (oracle `cond-enc.h`):
//!   - lyric_embed [S_lyric,1024] -> Linear(1024->2048)+bias -> 8x Qwen3 encoder
//!     (alternating SWA-128/full) -> norm -> [S_lyric,2048]
//!   - text_hidden [S_text,1024] -> text_projector Linear(1024->2048) no bias -> [S_text,2048]
//!   - timbre_feats [S_ref,64] -> Linear(64->2048)+bias -> (XL: CLS) -> 4x encoder ->
//!     norm -> frame[0] -> [1,2048]
//! Weights live in the DiT GGUF (`encoder.*`). Reuses the Qwen3 pre-norm encoder
//! layer (`DetokLayer`) validated by the FSQ de-tokenizer; only the alternating
//! sliding-window mask (even layers SWA-128, odd full) differs from the detok's
//! full-attention layers.

use crate::inference::model::acestep::fsq::{
    detok_lin, detok_load_t, rope_tables, DetokLayer, DetokLinear,
};
use crate::tensor::{Device, Tensor};

/// The ACE-Step condition encoder (lyric + timbre encoders + text projector).
pub struct CondModel {
    pub lyric_embed: DetokLinear,      // Linear(1024->2048)+bias
    pub lyric_layers: Vec<DetokLayer>, // 8
    pub lyric_norm: Tensor,
    pub timbre_embed: DetokLinear,      // Linear(64->2048)+bias
    pub timbre_layers: Vec<DetokLayer>, // 4
    pub timbre_norm: Tensor,
    pub timbre_cls: Option<Tensor>, // [H] learned CLS (XL only)
    pub text_proj: DetokLinear,     // Linear(1024->2048) no bias
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub device: Device,
}

impl CondModel {
    /// Load the condition encoder from acestep-v15-turbo-Q8_0.gguf (`encoder.*`).
    /// Geometry mirrors the DiT/detok (H2048/16h/8kv/hd128/θ1e6); lyric 8L, timbre 4L.
    pub fn from_gguf(path: &str) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let c = gguf_file::read_mapped_file(&f)?;
        // Placed on what THIS component occupies, read from the checkpoint's own tensor
        // directory. The typed 4 GB that used to stand here was a figure nobody measured
        // deciding a real placement, and the file size beside it was the whole shared
        // GGUF - denoiser included - for an encoder that is a fraction of it.
        let device = {
            let sz = crate::inference::model::acestep::ops::gguf_resident_bytes(&c, &["encoder."]);
            crate::inference::place::plan::place_whole(
                sz,
                crate::inference::place::runtime_demand::load_runtime_floor(sz),
            )
        };
        let load_layers = |c: &gguf_file::Content,
                           f: &mut std::fs::File,
                           prefix: &str,
                           n: usize,
                           device: &Device|
         -> crate::tensor::Result<Vec<DetokLayer>> {
            let mut v = Vec::with_capacity(n);
            for l in 0..n {
                let p = format!("{prefix}.layers.{l}");
                v.push(DetokLayer {
                    input_ln: detok_load_t(c, f, &format!("{p}.input_layernorm.weight"), device)?,
                    post_ln: detok_load_t(
                        c,
                        f,
                        &format!("{p}.post_attention_layernorm.weight"),
                        device,
                    )?,
                    q: detok_lin(c, f, &format!("{p}.self_attn.q_proj"), false, device)?,
                    k: detok_lin(c, f, &format!("{p}.self_attn.k_proj"), false, device)?,
                    v: detok_lin(c, f, &format!("{p}.self_attn.v_proj"), false, device)?,
                    o: detok_lin(c, f, &format!("{p}.self_attn.o_proj"), false, device)?,
                    q_norm: detok_load_t(c, f, &format!("{p}.self_attn.q_norm.weight"), device)?,
                    k_norm: detok_load_t(c, f, &format!("{p}.self_attn.k_norm.weight"), device)?,
                    gate: detok_lin(c, f, &format!("{p}.mlp.gate_proj"), false, device)?,
                    up: detok_lin(c, f, &format!("{p}.mlp.up_proj"), false, device)?,
                    down: detok_lin(c, f, &format!("{p}.mlp.down_proj"), false, device)?,
                });
            }
            Ok(v)
        };
        let lyric_layers = load_layers(&c, &mut f, "encoder.lyric_encoder", 8, &device)?;
        let timbre_layers = load_layers(&c, &mut f, "encoder.timbre_encoder", 4, &device)?;
        // CLS token is used only by XL models (gated on the `acestep.encoder_hidden_size`
        // metadata, NOT tensor presence - the special_token tensor exists but is unused
        // on 2B models). Oracle cond-enc.h: use_timbre_cls = encoder_hidden_size > 0.
        let use_cls = c
            .metadata
            .get("acestep.encoder_hidden_size")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0)
            > 0;
        let timbre_cls = if use_cls {
            detok_load_t(&c, &mut f, "encoder.timbre_encoder.special_token", &device).ok()
        } else {
            None
        };
        Ok(CondModel {
            lyric_embed: detok_lin(
                &c,
                &mut f,
                "encoder.lyric_encoder.embed_tokens",
                true,
                &device,
            )?,
            lyric_layers,
            lyric_norm: detok_load_t(&c, &mut f, "encoder.lyric_encoder.norm.weight", &device)?,
            timbre_embed: detok_lin(
                &c,
                &mut f,
                "encoder.timbre_encoder.embed_tokens",
                true,
                &device,
            )?,
            timbre_layers,
            timbre_norm: detok_load_t(&c, &mut f, "encoder.timbre_encoder.norm.weight", &device)?,
            timbre_cls,
            text_proj: detok_lin(&c, &mut f, "encoder.text_projector", false, &device)?,
            hidden: 2048,
            n_head: 16,
            n_kv: 8,
            head_dim: 128,
            rope_theta: 1e6,
            sliding_window: 128,
            device,
        })
    }

    fn rms_eps(&self) -> f32 {
        1e-6
    }

    /// proj -> heads -> qk-norm -> NEOX RoPE -> `[1,heads,S,D]` on-device (validated layout).
    fn qk_roped(
        &self,
        lin: &DetokLinear,
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

    /// Bidirectional GQA self-attention with a sliding-window mask (`win`; pass
    /// `usize::MAX` for full). `x [S,H]` -> `[S,H]` post o_proj.
    fn self_attn(
        &self,
        l: &DetokLayer,
        x: &Tensor,
        s: usize,
        win: usize,
    ) -> crate::tensor::Result<Tensor> {
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
        // Bidirectional sliding-window mask (|i-j|>win = -inf); None for full attention.
        let mask = if win < usize::MAX {
            let mut m = vec![0f32; s * s];
            for i in 0..s {
                for j in 0..s {
                    if (i as i64 - j as i64).unsigned_abs() as usize > win {
                        m[i * s + j] = f32::NEG_INFINITY;
                    }
                }
            }
            Some(Tensor::from_vec_f32(m, (s, s))?.to_device(&self.device)?)
        } else {
            None
        };
        let attn = sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)?; // [1,nh,s,d]
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.o.forward(&ao) // [s,h]
    }

    /// One Qwen3 pre-norm encoder layer (plain residual, no AdaLN), window `win`.
    fn enc_layer(
        &self,
        l: &DetokLayer,
        x: &Tensor,
        s: usize,
        win: usize,
    ) -> crate::tensor::Result<Tensor> {
        let eps = self.rms_eps();
        let norm = x.rms_norm(&l.input_ln, eps)?;
        let x = x.add(&self.self_attn(l, &norm, s, win)?)?;
        let norm2 = x.rms_norm(&l.post_ln, eps)?;
        let gate = norm2.matmul_t(l.gate.weight()?)?;
        let up = norm2.matmul_t(l.up.weight()?)?;
        let ff = gate.silu()?.mul(&up)?;
        x.add(&ff.matmul_t(l.down.weight()?)?)
    }

    /// Run an embed-linear + N-layer encoder stack (alternating SWA-128/full on
    /// even/odd layers) + final norm. `embedded [S,H]` (on device) -> `[S,H]`.
    fn encode_stack(
        &self,
        embedded: Tensor,
        layers: &[DetokLayer],
        norm: &Tensor,
        s: usize,
    ) -> crate::tensor::Result<Vec<f32>> {
        let mut hid = embedded;
        for (i, l) in layers.iter().enumerate() {
            let win = if i % 2 == 0 {
                self.sliding_window
            } else {
                usize::MAX
            };
            hid = self.enc_layer(l, &hid, s, win)?;
        }
        hid.rms_norm(norm, self.rms_eps())?
            .flatten_all()?
            .to_vec1_f32()
    }

    /// Lyric path: `lyric_embed [S,1024]` -> embed Linear(+bias) -> 8L -> norm -> `[S,2048]`.
    pub fn lyric_forward(&self, lyric_embed: &[f32], s: usize) -> crate::tensor::Result<Vec<f32>> {
        let x = Tensor::from_vec_f32(lyric_embed.to_vec(), (s, 1024))?.to_device(&self.device)?;
        let emb = self.lyric_embed.forward(&x)?;
        self.encode_stack(emb, &self.lyric_layers, &self.lyric_norm, s)
    }

    /// Text path: `text_hidden [S,1024]` -> text_projector Linear (no bias) -> `[S,2048]`.
    pub fn text_forward(&self, text_hidden: &[f32], s: usize) -> crate::tensor::Result<Vec<f32>> {
        let x = Tensor::from_vec_f32(text_hidden.to_vec(), (s, 1024))?.to_device(&self.device)?;
        self.text_proj.forward(&x)?.flatten_all()?.to_vec1_f32()
    }

    /// Timbre path: `timbre_feats [S_ref,64]` -> embed Linear(64->2048)+bias -> 4L ->
    /// norm -> take frame[0] -> `[2048]`. (2B models: no CLS token; XL prepend the
    /// learned CLS and take its position-0 output - same "frame[0]" rule.)
    pub fn timbre_forward(
        &self,
        timbre_feats: &[f32],
        s_ref: usize,
    ) -> crate::tensor::Result<Vec<f32>> {
        let h = self.hidden;
        let x =
            Tensor::from_vec_f32(timbre_feats.to_vec(), (s_ref, 64))?.to_device(&self.device)?;
        let emb = self.timbre_embed.forward(&x)?; // [S_ref,H] on device
        let mut s = s_ref;
        let emb = if let Some(cls) = &self.timbre_cls {
            // XL: prepend the learned CLS token -> [1+S_ref, H].
            let cls_row = cls.reshape((1, h))?.to_device(&self.device)?;
            s += 1;
            Tensor::cat(&[&cls_row, &emb], 0)?
        } else {
            emb
        };
        let out = self.encode_stack(emb, &self.timbre_layers, &self.timbre_norm, s)?;
        Ok(out[0..h].to_vec()) // frame[0]
    }

    /// Full condition encoder: pack `cat(lyric, timbre[0:1], text_proj)` ->
    /// `enc_hidden [S_total, 2048]` (row-major). `S_total = S_lyric + 1 + S_text`
    /// when timbre is present.
    pub fn forward(
        &self,
        text_hidden: &[f32],
        s_text: usize,
        lyric_embed: &[f32],
        s_lyric: usize,
        timbre_feats: Option<(&[f32], usize)>,
    ) -> crate::tensor::Result<(Vec<f32>, usize)> {
        let h = self.hidden;
        let lyric = self.lyric_forward(lyric_embed, s_lyric)?;
        let text = self.text_forward(text_hidden, s_text)?;
        let timbre = match timbre_feats {
            Some((tf, s_ref)) => Some(self.timbre_forward(tf, s_ref)?),
            None => None,
        };
        let s_total = s_lyric + timbre.as_ref().map_or(0, |_| 1) + s_text;
        let mut out = Vec::with_capacity(s_total * h);
        out.extend_from_slice(&lyric);
        if let Some(t) = &timbre {
            out.extend_from_slice(t);
        }
        out.extend_from_slice(&text);
        Ok((out, s_total))
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

    // enc_hidden[66,2048] packs [lyric(11), timbre(1), text(54)]; validate the
    // lyric path vs rows 0..11 and the text-projector vs rows 12..66.
    #[test]
    #[ignore = "needs DiT GGUF (config.test HF hub) + /tmp/acedump_detok dumps - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_cond_vs_oracle() {
        let gguf =
            crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf");
        let m = CondModel::from_gguf(gguf.to_str().unwrap()).unwrap();
        let (enc, es) = load_dump("/tmp/acedump_detok/enc_hidden.bin"); // [66,2048]
        let (lyric_in, ls) = load_dump("/tmp/acedump_detok/lyric_embed.bin"); // [S_lyric,1024]
        let (text_in, ts) = load_dump("/tmp/acedump_detok/text_hidden.bin"); // [S_text,1024]
        let h = 2048;
        let (s_lyric, s_text) = (ls[0], ts[0]);
        assert_eq!(es, vec![s_lyric + 1 + s_text, h], "pack count");

        let lyric_out = m.lyric_forward(&lyric_in, s_lyric).unwrap();
        let lyric_ref = &enc[0..s_lyric * h];
        let c_lyric = cosine(&lyric_out, lyric_ref);
        println!("lyric cosine={c_lyric:.6} (S={s_lyric})");

        let text_out = m.text_forward(&text_in, s_text).unwrap();
        let text_ref = &enc[(s_lyric + 1) * h..(s_lyric + 1 + s_text) * h];
        let c_text = cosine(&text_out, text_ref);
        println!("text_proj cosine={c_text:.6} (S={s_text})");

        assert!(c_lyric > 0.999, "lyric cosine {c_lyric} too low");
        assert!(c_text > 0.999, "text_proj cosine {c_text} too low");
    }

    // Full cond encoder: feed dumped text_hidden + lyric_embed + timbre_feats ->
    // forward -> compare the whole packed enc_hidden [238,2048].
    #[test]
    #[ignore = "needs DiT GGUF (config.test HF hub) + /tmp/acedump_detok dumps - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_cond_full_vs_oracle() {
        let gguf =
            crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf");
        let m = CondModel::from_gguf(gguf.to_str().unwrap()).unwrap();
        let (enc, es) = load_dump("/tmp/acedump_detok/enc_hidden.bin");
        let (lyric_in, ls) = load_dump("/tmp/acedump_detok/lyric_embed.bin");
        let (text_in, ts) = load_dump("/tmp/acedump_detok/text_hidden.bin");
        let (timbre_in, tfs) = load_dump("/tmp/acedump_detok/timbre_feats.bin"); // [64] = S_ref 1
        let s_ref = tfs.iter().product::<usize>() / 64;
        let (out, s_total) = m
            .forward(&text_in, ts[0], &lyric_in, ls[0], Some((&timbre_in, s_ref)))
            .unwrap();
        assert_eq!(s_total, es[0], "S_total");
        let c = cosine(&out, &enc);
        let timbre_ref = &enc[ls[0] * 2048..(ls[0] + 1) * 2048];
        let timbre_out = m.timbre_forward(&timbre_in, s_ref).unwrap();
        let c_timbre = cosine(&timbre_out, timbre_ref);
        println!(
            "cond FULL enc_hidden cosine={c:.6} (S_total={s_total}); timbre cosine={c_timbre:.6}"
        );
        assert!(c > 0.999, "full cond cosine {c} too low");
    }
}
