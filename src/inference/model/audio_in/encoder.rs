//! The speech encoder of an `mmproj` audio tower.
//!
//! Non-causal, pre-norm, LayerNorm: `mel [1, n_mels, T]` goes through a three-tap
//! convolution that keeps the length and a second one that halves it, each followed by
//! GELU; the learned positions are added; the blocks run full attention over the frames;
//! a final LayerNorm closes. Every dimension is read from the file's `clip.audio.*` keys
//! and checked against the shapes of the tensors.
//!
//! All tensors are dequantised to F32 at load: the encoder runs once per clip, and its
//! weights are small beside the language model's.

use super::source::TensorReader;
use crate::inference::model::mmproj::MmprojAudio;
use crate::tensor::layer::{same_length_1d, Conv1d, Conv1dConfig};
use crate::tensor::{Device, Error, Result, Tensor};

fn err(m: String) -> Error {
    Error(m)
}

/// `x [n, in]` times `w [out, in]` transposed, plus an optional bias.
pub(crate) fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.transpose(0, 1)?.contiguous()?)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

struct Block {
    ln1: (Tensor, Tensor),
    q: (Tensor, Tensor),
    k: Tensor,
    v: (Tensor, Tensor),
    out: (Tensor, Tensor),
    ln2: (Tensor, Tensor),
    up: (Tensor, Tensor),
    down: (Tensor, Tensor),
}

impl Block {
    fn load(r: &TensorReader<'_>, geo: &MmprojAudio, i: usize) -> Result<Self> {
        let (w, f) = (geo.width, geo.feed_forward);
        let p = format!("a.blk.{i}");
        let pair = |name: &str, rows: usize, cols: usize| -> Result<(Tensor, Tensor)> {
            Ok((
                r.matrix(&format!("{p}.{name}.weight"), rows, cols)?,
                r.vector(&format!("{p}.{name}.bias"))?,
            ))
        };
        let norm = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                r.vector(&format!("{p}.{name}.weight"))?,
                r.vector(&format!("{p}.{name}.bias"))?,
            ))
        };
        Ok(Self {
            ln1: norm("ln1")?,
            q: pair("attn_q", w, w)?,
            k: r.matrix(&format!("{p}.attn_k.weight"), w, w)?,
            v: pair("attn_v", w, w)?,
            out: pair("attn_out", w, w)?,
            ln2: norm("ln2")?,
            up: pair("ffn_up", f, w)?,
            down: pair("ffn_down", w, f)?,
        })
    }

    /// `x [seq, width]`, one sequence, full attention.
    fn forward(&self, x: &Tensor, geo: &MmprojAudio) -> Result<Tensor> {
        let (seq, heads, hd) = (x.dim(0)?, geo.heads, geo.head_dim());
        let h = x.layer_norm(&self.ln1.0, Some(&self.ln1.1), geo.eps)?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((seq, heads, hd))?.transpose(0, 1)?.contiguous()
        };
        let q = split(linear(&h, &self.q.0, Some(&self.q.1))?)?;
        let k = split(linear(&h, &self.k, None)?)?;
        let v = split(linear(&h, &self.v.0, Some(&self.v.1))?)?;
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (hd as f32).sqrt(), 0.0)?;
        let o = scores.softmax_last_dim()?.matmul(&v)?;
        let o = o.transpose(0, 1)?.contiguous()?.reshape((seq, geo.width))?;
        let x = x.add(&linear(&o, &self.out.0, Some(&self.out.1))?)?;
        let h = x.layer_norm(&self.ln2.0, Some(&self.ln2.1), geo.eps)?;
        let h = linear(&h, &self.up.0, Some(&self.up.1))?.gelu_erf()?;
        x.add(&linear(&h, &self.down.0, Some(&self.down.1))?)
    }
}

/// The encoder: convolutions, positions, blocks, final norm.
pub struct Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Tensor,
    blocks: Vec<Block>,
    post_ln: (Tensor, Tensor),
    geo: MmprojAudio,
    dev: Device,
}

impl Encoder {
    pub(crate) fn load(r: &TensorReader<'_>, geo: &MmprojAudio) -> Result<Self> {
        let taps = same_length_1d(3, 1);
        let halving = Conv1dConfig { stride: 2, ..taps };
        let conv = |i: usize, c_in: usize, cfg: Conv1dConfig| -> Result<Conv1d> {
            let w = r.get(&format!("a.conv1d.{i}.weight"))?;
            if w.dims() != [geo.width, c_in, 3] {
                return Err(err(format!(
                    "a.conv1d.{i}.weight: shape {:?}, the file's metadata implies {:?}",
                    w.dims(),
                    [geo.width, c_in, 3]
                )));
            }
            Ok(Conv1d::new(
                w,
                Some(r.vector(&format!("a.conv1d.{i}.bias"))?),
                cfg,
            ))
        };
        let positions = r.get("a.position_embd.weight")?;
        if positions.dim(1)? != geo.width {
            return Err(err(format!(
                "a.position_embd.weight: width {}, the file's metadata implies {}",
                positions.dim(1)?,
                geo.width
            )));
        }
        Ok(Self {
            conv1: conv(1, geo.n_mels, taps)?,
            conv2: conv(2, geo.width, halving)?,
            positions,
            blocks: (0..geo.blocks)
                .map(|i| Block::load(r, geo, i))
                .collect::<Result<Vec<_>>>()?,
            post_ln: (r.vector("a.post_ln.weight")?, r.vector("a.post_ln.bias")?),
            geo: geo.clone(),
            dev: r.dev.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Frames the encoder holds positions for, in mel frames.
    pub fn context_frames(&self) -> Result<usize> {
        Ok(self.positions.dim(0)? * 2)
    }

    /// `mel [1, n_mels, T]` to encoder frames `[ceil(T / 2), width]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(&mel.to_device(&self.dev)?)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let frames = x.dim(2)?;
        if frames > self.positions.dim(0)? {
            return Err(err(format!(
                "{frames} encoder frames exceed the {} the file holds positions for",
                self.positions.dim(0)?
            )));
        }
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((frames, self.geo.width))?;
        let mut x = x.add(&self.positions.narrow(0, 0, frames)?)?;
        for b in &self.blocks {
            x = b.forward(&x, &self.geo)?;
        }
        x.layer_norm(&self.post_ln.0, Some(&self.post_ln.1), self.geo.eps)
    }
}
