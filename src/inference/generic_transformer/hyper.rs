//! The wide residual: the stream a layer reads and writes is `streams` parallel copies of the
//! hidden state, mixed down to one input by a data-dependent gate before each sublayer and
//! written back with a per-stream weight.
//!
//! A block of such a model carries no norm of its own: the grouped RMS norm inside the mixer
//! is its input norm, and the final mixer is the output norm.

use super::config::meta_u32;
use super::gguf_file;
use super::{QMatMul, Result, Tensor};
use crate::tensor::{DType, D};

/// The wide residual's dimensions, from the file's `hyper_connection.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperMeta {
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// Rank of the mixer's low-rank gate projection.
    pub rank: usize,
}

impl HyperMeta {
    /// The metadata of `arch`, `None` when the file has no wide residual or declares fewer
    /// than two streams, which have nothing to mix.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let streams = meta_u32(content, arch, "hyper_connection.count")? as usize;
        let rank = meta_u32(content, arch, "hyper_connection.low_rank")? as usize;
        (streams > 1 && rank > 0).then_some(Self { streams, rank })
    }

    /// Width of the residual stream for a hidden state of `embedding_length`.
    pub fn width(&self, embedding_length: usize) -> usize {
        self.streams * embedding_length
    }
}

/// One mixer of the wide residual: the grouped norm, the low-rank gate and, for a sublayer
/// that writes back, the per-stream write weights.
///
/// The mixed input is `mean_s(xn_s * sigmoid(up . silu(down . xn / streams))_s)` with `xn`
/// the stream-wise RMS-normalised residual scaled by `gamma`; the write weights are
/// `2 sigmoid(inject . xn / streams)`, one per stream.
pub struct HyperMixer {
    /// `[streams * dim]` F32, the gamma of every stream (already `1 + w`).
    gamma: Tensor,
    down: QMatMul,
    up: QMatMul,
    inject: Option<QMatMul>,
    streams: usize,
    dim: usize,
    eps: f64,
}

impl HyperMixer {
    pub fn new(
        gamma: Tensor,
        down: QMatMul,
        up: QMatMul,
        inject: Option<QMatMul>,
        streams: usize,
        dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let gamma = gamma.to_dtype(DType::F32)?.reshape(streams * dim)?;
        Ok(Self {
            gamma,
            down,
            up,
            inject,
            streams,
            dim,
            eps,
        })
    }

    /// Stream-wise RMS norm of `r` `[b, seq, streams * dim]`, scaled by gamma.
    fn grouped_norm(&self, r: &Tensor) -> Result<Tensor> {
        grouped_rms(r, &self.gamma, self.streams, self.dim, self.eps)
    }

    /// The device the mixer's weights live on.
    pub fn device(&self) -> crate::tensor::Device {
        self.down.device()
    }

    /// The sublayer input `[b, seq, dim]`, and the write weights `[b, seq, streams]` when
    /// this mixer has them.
    pub fn mix(&self, r: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        self.mix_traced(r, |_, _| {})
    }

    /// [`Self::mix`], handing each intermediate to `sink` by name.
    pub fn mix_traced(
        &self,
        r: &Tensor,
        mut sink: impl FnMut(&str, &Tensor),
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b, seq, _) = r.dims3()?;
        let r = r.to_dtype(DType::F32)?;
        let xn = self.grouped_norm(&r)?;
        sink("normed", &xn);
        let n = self.streams as f64;
        let down = self.down.forward(&xn)?;
        sink("down", &down);
        let low = crate::tensor::ops::silu(&(down / n)?)?;
        let up = self.up.forward(&low)?;
        sink("up", &up);
        let gate = crate::tensor::ops::sigmoid(&up)?;
        let gated = (xn.clone() * gate)?.reshape((b, seq, self.streams, self.dim))?;
        let mixed = (gated.sum(2)? / n)?;
        sink("mixed", &mixed);
        let inj = match &self.inject {
            Some(w) => {
                let raw = w.forward(&xn)?;
                sink("inject", &raw);
                Some((crate::tensor::ops::sigmoid(&(raw / n)?)? * 2.0)?)
            }
            None => None,
        };
        Ok((mixed, inj))
    }
}

/// RMS norm of every stream of `r` `[b, seq, streams * dim]` apart, scaled by `gamma`
/// `[streams * dim]`.
pub fn grouped_rms(
    r: &Tensor,
    gamma: &Tensor,
    streams: usize,
    dim: usize,
    eps: f64,
) -> Result<Tensor> {
    let (b, seq, _) = r.dims3()?;
    let grouped = r.reshape((b, seq, streams, dim))?;
    let ms = grouped.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (ms + eps)?.sqrt()?.recip()?;
    let normed = grouped
        .broadcast_mul(&inv)?
        .reshape((b, seq, streams * dim))?;
    normed.broadcast_mul(&gamma.to_device(&r.device())?)
}

/// The two mixers of one block: the one ahead of the token mixer and the one ahead of the
/// feed-forward.
pub struct WideBlock {
    pub attn: HyperMixer,
    pub ffn: HyperMixer,
}

/// `r + out (x) inj`: every stream adds the sublayer's output `[b, seq, dim]` scaled by its
/// own write weight `[b, seq, streams]`.
pub fn combine(r: &Tensor, out: &Tensor, inj: &Tensor) -> Result<Tensor> {
    let (b, seq, wide) = r.dims3()?;
    let streams = inj.dim(D::Minus1)?;
    let dim = wide / streams;
    let out = out.to_dtype(DType::F32)?.reshape((b, seq, 1, dim))?;
    let w = inj.reshape((b, seq, streams, 1))?;
    let add = out.broadcast_mul(&w)?.reshape((b, seq, wide))?;
    r.to_dtype(DType::F32)? + add
}

/// The residual a token starts from: its embedding, repeated once per stream.
pub fn widen(embedding: &Tensor, streams: usize) -> Result<Tensor> {
    let parts: Vec<&Tensor> = (0..streams).map(|_| embedding).collect();
    Tensor::cat(&parts, D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(meta: &[(&str, u32)]) -> Content {
        let mut metadata = HashMap::new();
        for (k, v) in meta {
            metadata.insert(format!("a.{k}"), Value::U32(*v));
        }
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    #[test]
    fn a_file_without_streams_has_no_wide_residual() {
        assert_eq!(HyperMeta::from_gguf(&content(&[]), "a"), None);
        let one = content(&[
            ("hyper_connection.count", 1),
            ("hyper_connection.low_rank", 8),
        ]);
        assert_eq!(HyperMeta::from_gguf(&one, "a"), None);
    }

    #[test]
    fn the_streams_and_rank_are_read() {
        let c = content(&[
            ("hyper_connection.count", 4),
            ("hyper_connection.low_rank", 320),
        ]);
        let m = HyperMeta::from_gguf(&c, "a").unwrap();
        assert_eq!((m.streams, m.rank, m.width(2560)), (4, 320, 10240));
    }
}
