//! ACE-Step 1.5 - FSQ (Finite Scalar Quantization) de-tokenizer support ( M1).
//!
//! The DiT's 128-channel `context` is built from the LM's audio codes:
//!   codes `[T_5Hz]` -> FSQ decode -> `[T_5Hz, 6]` -> tokenizer.quantizer.project_out
//!   `[6->2048]` -> detokenizer (embed + special_tokens broadcast + 2x standard-Qwen3
//!   encoder layers + proj_out `[2048->64]`) -> context_latents `[T_25Hz=5.T_5Hz, 64]`.
//! Weights live in the DiT GGUF (`tokenizer.*` / `detokenizer.*`). The Qwen3
//! encoder layer is the standard pre-norm block (input_layernorm -> qk-norm GQA
//! self-attn + RoPE, full attention -> post_attention_layernorm -> SwiGLU), distinct
//! from the DiT's AdaLN layer - built by reusing the self-attn/SwiGLU machinery.
//! This module starts with the FSQ value decode (the pure-math core).

/// FSQ levels for ACE-Step 1.5 (`fsq_input_levels`): 6 dims, one quantizer.
pub const FSQ_LEVELS: [usize; 6] = [8, 8, 8, 5, 5, 5];

/// Decode one FSQ integer index -> its 6 continuous values in [-1, 1] (oracle
/// `fsq_decode_index`): per dim `d`, `level = (index / stride_d) % L_d`, value =
/// `level / ((L_d-1)/2) - 1`, with `stride_d = Π_{k<d} L_k`. The total code space
/// is `ΠL = 8.8.8.5.5.5 = 64000`.
pub fn fsq_decode_index(index: usize, out: &mut [f32; 6]) {
    let mut stride = 1usize;
    for d in 0..6 {
        let l = FSQ_LEVELS[d];
        let level = (index / stride) % l;
        let half = (l - 1) as f32 / 2.0;
        out[d] = level as f32 / half - 1.0;
        stride *= l;
    }
}

/// Encode 6 continuous values (the `project_in` output) -> one FSQ integer index, the exact
/// inverse of [`fsq_decode_index`] (matches vector_quantize_pytorch FSQ + acestep.cpp
/// `fsq_encode_index`): per dim apply `tanh`, scale to a level `round((L-1).(t+1)/2)` clamped
/// to `[0,L-1]`, pack with cumulative stride `Π_{k<d} L_k`.
pub fn fsq_encode_index(raw: &[f32; 6]) -> u32 {
    let mut index = 0usize;
    let mut stride = 1usize;
    for d in 0..6 {
        let l = FSQ_LEVELS[d];
        let t = raw[d].tanh();
        let code = (((l as f32 - 1.0) * (t + 1.0) / 2.0 + 0.5).floor() as i64)
            .clamp(0, l as i64 - 1) as usize;
        index += code * stride;
        stride *= l;
    }
    index as u32
}

/// Decode a sequence of codes -> `[T.6]` row-major (token-major).
pub fn fsq_decode_codes(codes: &[u32]) -> Vec<f32> {
    let mut out = vec![0f32; codes.len() * 6];
    let mut buf = [0f32; 6];
    for (t, &c) in codes.iter().enumerate() {
        fsq_decode_index(c as usize, &mut buf);
        out[t * 6..t * 6 + 6].copy_from_slice(&buf);
    }
    out
}

use crate::tensor::{Device, Tensor};

// The de-tokenizer is a stack of standard Qwen3 encoder layers, so it is built from
// the shared block rather than a copy of it. The names it reads them under are kept.
pub(crate) use crate::inference::model::qwen3::encoder::{
    lin as detok_lin, load_t as detok_load_t, rope_tables,
};
pub use crate::inference::model::qwen3::encoder::{Layer as DetokLayer, Linear as DetokLinear};

/// The bidirectional Qwen3 stack an ACE-Step checkpoint carries, and the geometry it runs at.
///
/// The de-tokenizer and the tokenizer's attention pooler are the same stack read from two
/// prefixes of one file - same width, same head split, same rotary base, no mask either side.
/// What differs is only what each one projects into it and takes back out, so the stack is
/// held once and both towers reach it rather than each restating the arithmetic.
pub struct Pooler {
    pub layers: Vec<DetokLayer>, // 2 standard Qwen3 encoder layers
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub device: Device, // native
}

impl Pooler {
    /// The geometry both ACE-Step 1.5 towers are published at: 2048 wide, 16 query heads over
    /// 8 key heads of 128 channels, rotary base 1e6.
    fn acestep(layers: Vec<DetokLayer>, device: Device) -> Self {
        Self {
            layers,
            hidden: 2048,
            n_head: 16,
            n_kv: 8,
            head_dim: 128,
            rope_theta: 1e6,
            device,
        }
    }

    fn rms_eps(&self) -> f32 {
        1e-6
    }

    /// proj -> heads -> qk-norm (RMS over D x weight) -> NEOX RoPE -> `[1,heads,S,D]`
    /// on-device (the validated DiT q-path layout).
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
        let q = in_heads(proj, n_heads, s, d)?;
        let q = q.rms_norm(qknorm, self.rms_eps())?;
        let (cosv, sinv) = rope_tables(s, d, self.rope_theta);
        let cos = Tensor::from_vec_f32(cosv, (s, d / 2))?.to_device(&self.device)?;
        let sin = Tensor::from_vec_f32(sinv, (s, d / 2))?.to_device(&self.device)?;
        q.rope(&cos, &sin) // [1,heads,S,D]
    }

    /// Full bidirectional GQA self-attention: `x [S,H]` -> `[S,H]` post o_proj
    /// (qk-norm + NEOX RoPE, no mask - mirrors the DiT self-attn with win=∞).
    fn self_attn(&self, l: &DetokLayer, x: &Tensor, s: usize) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.qk_roped(&l.q, &l.q_norm, x, nh, s)?; // [1,nh,s,d]
        let k = self.qk_roped(&l.k, &l.k_norm, x, nkv, s)?; // [1,nkv,s,d]
        let v = in_heads(x.matmul_t(l.v.weight()?)?, nkv, s, d)?; // [1,nkv,s,d]
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?); // [1,nh,s,d]
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = sdpa(&q, &k, &v, None, false, scale, 1.0)?; // [1,nh,s,d]
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.o.forward(&ao) // [s,h]
    }

    /// Standard Qwen3 pre-norm layer: `hidden [S,H]` (on device) -> `[S,H]`.
    fn enc_layer(&self, l: &DetokLayer, x: &Tensor, s: usize) -> crate::tensor::Result<Tensor> {
        let eps = self.rms_eps();
        let norm = x.rms_norm(&l.input_ln, eps)?;
        let x = x.add(&self.self_attn(l, &norm, s)?)?; // plain residual
        let norm2 = x.rms_norm(&l.post_ln, eps)?;
        let gate = norm2.matmul_t(l.gate.weight()?)?;
        let up = norm2.matmul_t(l.up.weight()?)?;
        let ff = gate.silu()?.mul(&up)?; // SwiGLU
        x.add(&ff.matmul_t(l.down.weight()?)?)
    }

    /// Every layer in turn over `x [S,H]`, already on this stack's device.
    fn run(&self, x: &Tensor, s: usize) -> crate::tensor::Result<Tensor> {
        let mut h = x.clone();
        for l in &self.layers {
            h = self.enc_layer(l, &h, s)?;
        }
        Ok(h)
    }
}

/// `[S, heads.D]` as `[1, heads, S, D]`: the head axis is split out of the row and moved in
/// front of the positions, which is the layout the attention reads its operands in.
///
/// Every projection in this family lands in rows and is attended to in heads, so the move
/// between the two is written once here and the DiT reaches it for its own q, k and v.
pub(crate) fn in_heads(
    proj: Tensor,
    n_heads: usize,
    s: usize,
    d: usize,
) -> crate::tensor::Result<Tensor> {
    proj.reshape((s, n_heads, d))?
        .transpose(0, 1)?
        .unsqueeze(0)?
        .contiguous()
}

/// The FSQ de-tokenizer: codes -> context_latents (DiT context source).
pub struct DetokModel {
    pub fsq_proj: DetokLinear, // project_out [6->2048]
    pub embed: DetokLinear,    // embed_tokens [2048->2048]
    pub special_tok: Tensor,   // [P=5, H=2048] (per-frame positional bias)
    pub stack: Pooler,
    pub norm: Tensor,          // final rms weight [H]
    pub proj_out: DetokLinear, // [2048->64]
}

impl DetokModel {
    /// Load the detokenizer from acestep-v15-turbo-Q8_0.gguf (`tokenizer.*` /
    /// `detokenizer.*` tensors). Geometry mirrors the DiT (H2048/16h/8kv/hd128/θ1e6).
    pub fn from_gguf(path: &str) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let c = gguf_file::read_mapped_file(&f)?;
        // See `gguf_resident_bytes`: the directory is the authority on what this component
        // puts on the card, and the loader below expands every weight it reads.
        let device = {
            let sz =
                crate::inference::model::acestep::ops::gguf_resident_bytes(&c, &["detokenizer."]);
            crate::inference::place::plan::place_whole(
                sz,
                crate::inference::place::runtime_demand::load_runtime_floor(sz),
            )
        };
        let mut layers = Vec::with_capacity(2);
        for l in 0..2usize {
            let p = format!("detokenizer.layers.{l}");
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
        // special_tokens GGUF ne=[2048,5] -> substrate row-major [5,2048] (frame-major).
        let special_tok =
            detok_load_t(&c, &mut f, "detokenizer.special_tokens", &device)?.reshape((5, 2048))?;
        Ok(DetokModel {
            fsq_proj: detok_lin(&c, &mut f, "tokenizer.quantizer.project_out", true, &device)?,
            embed: detok_lin(&c, &mut f, "detokenizer.embed_tokens", true, &device)?,
            special_tok,
            stack: Pooler::acestep(layers, device.clone()),
            norm: detok_load_t(&c, &mut f, "detokenizer.norm.weight", &device)?,
            proj_out: detok_lin(&c, &mut f, "detokenizer.proj_out", true, &device)?,
        })
    }

    /// Decode LM audio `codes [T_5Hz]` -> context_latents `[T_25Hz.64]` row-major,
    /// frame-major (`out[t.64 + c]`, `t = g.5 + p`). Each 5Hz code expands to P=5
    /// frames at 25Hz via the per-token detok encoder.
    pub fn decode(&self, codes: &[u32]) -> crate::tensor::Result<Vec<f32>> {
        let (h, p) = (self.stack.hidden, 5usize);
        let dev = &self.stack.device;
        let mut out = vec![0f32; codes.len() * p * 64];
        let mut fsq = [0f32; 6];
        for (g, &code) in codes.iter().enumerate() {
            fsq_decode_index(code as usize, &mut fsq);
            // project_out [6->2048] -> embed_tokens [2048->2048]
            let fin = Tensor::from_vec_f32(fsq.to_vec(), (1, 6))?.to_device(dev)?;
            let quant = self.fsq_proj.forward(&fin)?;
            let embedded = self.embed.forward(&quant)?; // [1,H]
                                                        // broadcast the per-token embedding over P frames + add the per-frame special_tokens
            let hidden = embedded.broadcast_as((p, h))?.add(&self.special_tok)?; // [P,H]
            let hidden = self.stack.run(&hidden, p)?;
            // Final norm over [P,H], then proj_out [2048->64] -> 5 frames of 64 channels.
            let normed = hidden.rms_norm(&self.norm, self.stack.rms_eps())?;
            let frames = self.proj_out.forward(&normed)?;
            let fv: Vec<f32> = frames.flatten_all()?.to_vec1_f32()?; // [P,64]
            out[g * p * 64..(g + 1) * p * 64].copy_from_slice(&fv);
        }
        Ok(out)
    }
}

/// FSQ tokenizer ENCODER: VAE latents `[T_25Hz,64]` -> FSQ semantic code indices `[T_5Hz]`,
/// the inverse of [`DetokModel`]. Each group of 5 latent frames -> audio_acoustic_proj(64->2048)
/// -> embed_tokens -> prepend a learned CLS -> 2 bidirectional Qwen3 layers -> RMSNorm -> take the
/// CLS -> project_in(2048->6) -> [`fsq_encode_index`]. Reuses the detok Qwen3 layer architecture
/// (`tokenizer.attention_pooler.*` weights). Unblocks the cover / lego / extract task modes.
pub struct TokEncoder {
    pub proj: DetokLinear,   // tokenizer.audio_acoustic_proj [64->2048]
    pub embed: DetokLinear,  // tokenizer.attention_pooler.embed_tokens [2048->2048]
    pub special_tok: Tensor, // tokenizer.attention_pooler.special_token [1,2048] (CLS)
    pub norm: Tensor,        // tokenizer.attention_pooler.norm [2048]
    pub fsq_in: DetokLinear, // tokenizer.quantizer.project_in [2048->6]
    /// tokenizer.attention_pooler.layers.{0,1}
    pub stack: Pooler,
}

impl TokEncoder {
    /// Load the tokenizer encoder from an ACE-Step DiT GGUF (same file the DetokModel uses).
    pub fn from_gguf(path: &str) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let c = gguf_file::read_mapped_file(&f)?;
        let device = {
            let sz =
                crate::inference::model::acestep::ops::gguf_resident_bytes(&c, &["tokenizer."]);
            crate::inference::place::plan::place_whole(
                sz,
                crate::inference::place::runtime_demand::load_runtime_floor(sz),
            )
        };
        let mut layers = Vec::with_capacity(2);
        for l in 0..2usize {
            let p = format!("tokenizer.attention_pooler.layers.{l}");
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
        let special_tok = detok_load_t(
            &c,
            &mut f,
            "tokenizer.attention_pooler.special_token",
            &device,
        )?
        .reshape((1, 2048))?;
        Ok(TokEncoder {
            proj: detok_lin(&c, &mut f, "tokenizer.audio_acoustic_proj", true, &device)?,
            embed: detok_lin(
                &c,
                &mut f,
                "tokenizer.attention_pooler.embed_tokens",
                true,
                &device,
            )?,
            special_tok,
            norm: detok_load_t(
                &c,
                &mut f,
                "tokenizer.attention_pooler.norm.weight",
                &device,
            )?,
            fsq_in: detok_lin(&c, &mut f, "tokenizer.quantizer.project_in", true, &device)?,
            stack: Pooler::acestep(layers, device),
        })
    }

    /// Encode VAE latents `[T_25Hz.64]` (frame-major, `flat[t.64+c]`) -> FSQ codes `[T_5Hz]`.
    /// Groups of 5 frames -> 1 code; the tail is padded to a multiple of 5 with `silence`
    /// (`[>=pad.64]`, the GGUF `silence_latent`); pass empty when `T_25Hz` is already a multiple.
    pub fn encode(
        &self,
        latents: &[f32],
        t_25hz: usize,
        silence: &[f32],
    ) -> crate::tensor::Result<Vec<u32>> {
        let (h, p) = (self.stack.hidden, 5usize);
        let dev = &self.stack.device;
        let pad = (p - t_25hz % p) % p;
        let t_padded = t_25hz + pad;
        let t_5hz = t_padded / p;
        let mut input = vec![0f32; t_padded * 64];
        input[..t_25hz * 64].copy_from_slice(&latents[..t_25hz * 64]);
        if pad > 0 {
            input[t_25hz * 64..].copy_from_slice(&silence[..pad * 64]);
        }
        let cls = self.special_tok.reshape((1, h))?;
        let s = p + 1; // 1 CLS + 5 patches
        let mut codes = Vec::with_capacity(t_5hz);
        let mut raw = [0f32; 6];
        for g in 0..t_5hz {
            let grp = &input[g * p * 64..(g + 1) * p * 64];
            let tok_in = Tensor::from_vec_f32(grp.to_vec(), (p, 64))?.to_device(dev)?;
            let projected = self.proj.forward(&tok_in)?; // [5,2048]
            let embedded = self.embed.forward(&projected)?; // [5,2048]
            let hid = Tensor::cat(&[&cls, &embedded], 0)?; // [6,2048]
            let hid = self.stack.run(&hid, s)?;
            let normed = hid.rms_norm(&self.norm, self.stack.rms_eps())?;
            let cls_out = normed.narrow(0, 0, 1)?; // [1,2048] CLS
            let fsq_vals = self.fsq_in.forward(&cls_out)?; // [1,6]
            let fv = fsq_vals.flatten_all()?.to_vec1_f32()?;
            raw.copy_from_slice(&fv[..6]);
            codes.push(fsq_encode_index(&raw));
        }
        Ok(codes)
    }
}

/// The repository ACE-Step's GGUF checkpoints are published in.
pub const ACESTEP_REPO: &str = "models--Serveurperso--ACE-Step-1.5-GGUF";

/// One of those checkpoints, by file name.
pub fn acestep_gguf(name: &str) -> std::path::PathBuf {
    crate::inference::cache::hf::file(ACESTEP_REPO, name)
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    #[test]
    fn fsq_decode_correct() {
        // index 0 -> level 0 in every dim -> value -1.0 each.
        let mut o = [0f32; 6];
        fsq_decode_index(0, &mut o);
        assert_eq!(o, [-1.0; 6]);
        // top level of each dim: index = ΠL - 1 -> last level each -> +1.0 each.
        fsq_decode_index(64000 - 1, &mut o);
        for v in o {
            assert!((v - 1.0).abs() < 1e-6, "max index -> +1: {v}");
        }
        // mid level of dim0 (L=8): level 4 -> 4/3.5 - 1 = 0.142857; index=4 -> dim0 level4, rest 0.
        fsq_decode_index(4, &mut o);
        assert!((o[0] - (4.0 / 3.5 - 1.0)).abs() < 1e-6);
        assert_eq!(o[1], -1.0); // dim1 still level 0
                                // closed-form spot check across dims via the stride decomposition.
        let idx = 3 + 5 * 8 + 2 * 64; // dim0=3, dim1=5, dim2=2
        fsq_decode_index(idx, &mut o);
        assert!((o[0] - (3.0 / 3.5 - 1.0)).abs() < 1e-6);
        assert!((o[1] - (5.0 / 3.5 - 1.0)).abs() < 1e-6);
        assert!((o[2] - (2.0 / 3.5 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn fsq_decode_codes_shape() {
        let codes = vec![0u32, 100, 64000 - 1];
        let v = fsq_decode_codes(&codes);
        assert_eq!(v.len(), 3 * 6);
        assert_eq!(&v[0..6], &[-1.0f32; 6]); // code 0
    }

    // Oracle-validated path (needs the DiT GGUF + an acedump_detok dump from a
    // request with known codes). Format: int32 ndims, int32 shape[], f32 data.
    fn load_dump(path: &str) -> (Vec<f32>, Vec<usize>) {
        let b = std::fs::read(path).unwrap();
        let nd = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut shape = Vec::with_capacity(nd);
        for i in 0..nd {
            shape.push(i32::from_le_bytes(b[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize);
        }
        let off = 4 + nd * 4;
        let data: Vec<f32> = b[off..]
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
    #[ignore = "needs DiT GGUF (config acestep_models_dir) + /tmp/acedump_detok dump + /tmp/detok_codes.csv - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_detok_vs_oracle() {
        let gguf = super::acestep_gguf("acestep-v15-turbo-Q8_0.gguf");
        let m = DetokModel::from_gguf(gguf.to_str().unwrap()).unwrap();
        let codes: Vec<u32> = std::fs::read_to_string("/tmp/detok_codes.csv")
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let out = m.decode(&codes).unwrap(); // [T_25Hz.64] frame-major
        let (oref, shape) = load_dump("/tmp/acedump_detok/detok_output.bin"); // [320,64]
        assert_eq!(shape, vec![codes.len() * 5, 64], "frame count");
        assert_eq!(out.len(), oref.len());
        let cos = cosine(&out, &oref);
        let max_abs = out
            .iter()
            .zip(&oref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "detok cosine={cos:.6} max_abs_diff={max_abs:.5} (n={})",
            out.len()
        );
        assert!(cos > 0.999, "detok cosine {cos} too low");
    }
}
