//! A decoder of stated shapes with deterministic weights, and the step that drives it.
//!
//! The microbench and the precision tests build the same decoder from it, so a timing and a
//! tolerance are measured on one object.

use super::{Decoder, DecoderConfig};
use crate::tensor::{DType, Device, Result, Tensor, VarBuilder};
use std::collections::HashMap;

/// The shapes of a decoder.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub codebooks: usize,
    pub vocab: usize,
    /// Rows of the position table: the longest prompt plus steps a run takes.
    pub positions: usize,
}

impl Shape {
    pub fn config(&self) -> Result<DecoderConfig> {
        serde_json::from_value(serde_json::json!({
            "vocab_size": self.vocab,
            "max_position_embeddings": self.positions,
            "num_hidden_layers": self.layers,
            "ffn_dim": self.ffn,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.heads,
            "num_cross_attention_key_value_heads": self.heads,
            "activation_function": "gelu",
            "hidden_size": self.hidden,
            "scale_embedding": false,
            "num_codebooks": self.codebooks,
            "pad_token_id": self.vocab - 1,
            "bos_token_id": self.vocab,
            "eos_token_id": self.vocab - 1,
            "tie_word_embeddings": false,
            "rope_embeddings": false,
            "rope_theta": 10000.0,
        }))
        .map_err(|e| crate::tensor::Error::msg(format!("synthetic decoder config: {e}")))
    }

    /// Bytes of the projections one step reads, at `element` bytes each: per layer the 4
    /// self-attention projections, the query and output of the cross attention, and the 2
    /// feed-forward ones, then the heads.
    pub fn step_bytes(&self, element: usize) -> u64 {
        let (h, f) = (self.hidden as u64, self.ffn as u64);
        let layer = 4 * h * h + 2 * h * h + 2 * h * f;
        let heads = self.codebooks as u64 * self.vocab as u64 * h;
        (self.layers as u64 * layer + heads) * element as u64
    }
}

/// The `i`th value of stream `seed`, in [-1, 1).
fn unit(seed: u64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add(i as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// A host tensor of `dims` whose values span `scale` times [-1, 1), from stream `seed`.
pub fn noise(seed: u64, dims: &[usize], scale: f32) -> Result<Tensor> {
    let n: usize = dims.iter().product();
    let v = (0..n).map(|i| unit(seed, i) * scale).collect();
    Tensor::from_vec_f32(v, dims.to_vec())
}

fn constant(dims: &[usize], value: f32) -> Result<Tensor> {
    Tensor::from_vec_f32(vec![value; dims.iter().product()], dims.to_vec())
}

/// Tensors under construction, each from a stream of its own.
struct Builder {
    map: HashMap<String, Tensor>,
    seed: u64,
}

impl Builder {
    fn noise(&mut self, name: String, dims: &[usize], scale: f32) -> Result<()> {
        self.seed += 1;
        self.map.insert(name, noise(self.seed, dims, scale)?);
        Ok(())
    }

    /// A projection spans `1 / sqrt(in)`, so the stream keeps a stable scale through the layers.
    fn matrix(&mut self, name: String, out: usize, inp: usize) -> Result<()> {
        self.noise(name, &[out, inp], 3f32.sqrt() / (inp as f32).sqrt())
    }

    fn norm(&mut self, prefix: &str, width: usize) -> Result<()> {
        self.map
            .insert(format!("{prefix}.weight"), constant(&[width], 1.0)?);
        self.map
            .insert(format!("{prefix}.bias"), constant(&[width], 0.0)?);
        Ok(())
    }
}

/// The decoder's tensors under the names the checkpoint uses, below the `decoder` prefix.
pub fn weights(shape: &Shape) -> Result<HashMap<String, Tensor>> {
    let mut b = Builder {
        map: HashMap::new(),
        seed: 1,
    };
    let (h, f) = (shape.hidden, shape.ffn);
    let trunk = "model.decoder";
    for book in 0..shape.codebooks {
        b.noise(
            format!("{trunk}.embed_tokens.{book}.weight"),
            &[shape.vocab + 1, h],
            0.5,
        )?;
        b.matrix(format!("lm_heads.{book}.weight"), shape.vocab, h)?;
    }
    b.noise(
        format!("{trunk}.embed_positions.weights"),
        &[shape.positions, h],
        0.5,
    )?;
    for l in 0..shape.layers {
        let p = format!("{trunk}.layers.{l}");
        for attn in ["self_attn", "encoder_attn"] {
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                b.matrix(format!("{p}.{attn}.{name}.weight"), h, h)?;
            }
        }
        b.matrix(format!("{p}.fc1.weight"), f, h)?;
        b.matrix(format!("{p}.fc2.weight"), h, f)?;
        for norm in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ] {
            b.norm(&format!("{p}.{norm}"), h)?;
        }
    }
    b.norm(&format!("{trunk}.layer_norm"), h)?;
    Ok(b.map)
}

/// The decoder of `shape` with synthetic weights on `device`, its projections in half
/// precision when `half` is set and `device` is a card.
pub fn decoder(shape: &Shape, half: bool, device: &Device) -> Result<Decoder> {
    let vb = VarBuilder::from_tensors(weights(shape)?, DType::F32, device);
    Decoder::new(&shape.config()?, half, vb)
}

/// What a generation holds fixed: the text prefix and the encoder output, on `device`.
pub struct Conditioning {
    pub prompt: Tensor,
    pub encoder: Tensor,
}

impl Conditioning {
    pub fn new(shape: &Shape, prompt: usize, encoder: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            prompt: noise(7, &[1, prompt, shape.hidden], 1.0)?.to_device(device)?,
            encoder: noise(8, &[1, encoder, shape.hidden], 1.0)?.to_device(device)?,
        })
    }
}

/// One step of `dec` over `tokens`, one per codebook: the logits row of each codebook on the
/// host. Step 0 carries the prompt; step `n` sits `n` past it in the position table.
pub fn step(
    dec: &mut Decoder,
    cond: &Conditioning,
    tokens: &[u32],
    n: usize,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    let fed = Tensor::from_slice(tokens, (1, tokens.len(), 1), device)?;
    let prompt_len = cond.prompt.dim(1)?;
    let (prefix, at) = match n {
        0 => (Some(&cond.prompt), 0),
        _ => (None, prompt_len + n),
    };
    let logits = dec.forward(&fed, prefix, &cond.encoder, at)?;
    logits
        .iter()
        .map(|rows| {
            let last = rows.dim(1)? - 1;
            use crate::tensor::IndexOp;
            Ok(rows.i((0, last))?.to_vec_f32())
        })
        .collect()
}

/// The index of the largest value.
pub fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold((0, f32::MIN), |best, (i, &v)| match v > best.1 {
            true => (i as u32, v),
            false => best,
        })
        .0
}
