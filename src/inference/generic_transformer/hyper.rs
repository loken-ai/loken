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

    /// Whether this mixer's stages run as the fused launches: its weights are on a card.
    #[cfg(feature = "cuda")]
    pub fn on_card(&self) -> bool {
        self.gamma.device().is_cuda()
    }

    /// The grouped norm of `r`, in the launch a card has for it.
    #[cfg(feature = "cuda")]
    pub fn normed_on_card(&self, r: &Tensor) -> Result<Tensor> {
        crate::inference::moe_cuda::hc_norm(
            &r.to_dtype(DType::F32)?,
            &self.gamma,
            self.streams,
            self.dim,
            self.eps as f32,
        )
    }

    /// The gamma and epsilon of the norm: what the previous layer's write-back fuses in.
    #[cfg(feature = "cuda")]
    pub fn norm_params(&self) -> (&Tensor, f32) {
        (&self.gamma, self.eps as f32)
    }

    /// [`Self::mix`] from the already normalised residual `xn`, each stage a launch: the
    /// sublayer input (F16 when `f16_out`) and the raw write-back weights, which
    /// [`Self::combine_on_card`] turns into `2 sigmoid(. / streams)`.
    #[cfg(feature = "cuda")]
    pub fn mix_on_card(&self, xn: &Tensor, f16_out: bool) -> Result<(Tensor, Option<Tensor>)> {
        use crate::inference::moe_cuda::{hc_low, hc_mean_gate};
        let low = hc_low(&self.down.forward(xn)?, self.streams)?;
        let up = self.up.forward(&low)?;
        let mixed = hc_mean_gate(xn, &up, self.streams, self.dim, f16_out)?;
        let inject = self.inject.as_ref().map(|w| w.forward(xn)).transpose()?;
        Ok((mixed, inject))
    }

    /// Streams the mixer mixes.
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// Width of one stream.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The device the mixer's weights live on.
    pub fn device(&self) -> crate::tensor::Device {
        self.down.device()
    }

    /// The sublayer input `[b, seq, dim]`, and the write weights `[b, seq, streams]` when
    /// this mixer has them.
    pub fn mix(&self, r: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (_, seq, _) = r.dims3()?;
        let block = mixer_block_rows();
        if seq <= block {
            return self.mix_traced(r, |_, _| {});
        }
        // A long chunk goes through the mixer a block of rows at a time, so its temporaries
        // (several buffers as wide as the residual) stay the size of one block.
        let mut mixed = Vec::new();
        let mut inj = Vec::new();
        let mut at = 0;
        while at < seq {
            let n = block.min(seq - at);
            let (m, i) = self.mix_traced(&r.narrow(1, at, n)?, |_, _| {})?;
            mixed.push(m);
            inj.extend(i);
            at += n;
        }
        let refs: Vec<&Tensor> = mixed.iter().collect();
        let inj = if inj.is_empty() {
            None
        } else {
            Some(Tensor::cat(&inj.iter().collect::<Vec<_>>(), 1)?)
        };
        Ok((Tensor::cat(&refs, 1)?, inj))
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

/// Largest chunk a prompt is cut into, however many experts a model has.
pub(super) const CHUNK_CEILING: usize = 8192;
/// Buffers of `streams x dim` f32 a wide layer keeps live per token of a chunk: the residual, at a
/// combine the product it adds and the new residual, and the sublayers' own activations counted at
/// the same width (the mixers' temporaries are bounded apart, by [`mixer_block_rows`]).
const WIDE_LIVE_BUFFERS: usize = 5;
/// Buffers of `streams x dim` f32 a mixer holds for the rows of one block: the cast residual, the
/// squared copy, the normed copy, the scaled copy, the gate, the gated product and the low-rank
/// activations.
const MIXER_LIVE_BUFFERS: usize = 8;

/// Rows a mixer works on at once: two tiles of the grouped expert matmul, so the low-rank products
/// still fill their tiles while the mixer's temporaries stay a fixed size however long the chunk.
pub fn mixer_block_rows() -> usize {
    #[cfg(feature = "cuda")]
    let tile = crate::inference::moe_cuda::MMQ_X_MAX;
    #[cfg(not(feature = "cuda"))]
    let tile = 128;
    2 * tile
}

/// What a wide layer's prefill keeps on a card: per token of a chunk, the `streams x dim` f32
/// buffers live at once; apart from the chunk, the mixers' block temporaries; and the chunk it
/// runs best at, the one that gives each routed expert a full tile of the grouped matmul on
/// average, `tile x experts / top-k` tokens, between the ladder's usual top and [`CHUNK_CEILING`].
/// `experts` is `(count, used)`, `None` for a model without a routed mixture; its experts run on a
/// card for a batch of rows, and keep their own activations live per routed pair, `expert_ffn` wide.
pub fn prefill_footprint(
    streams: usize,
    dim: usize,
    experts: Option<(usize, usize)>,
    expert_ffn: usize,
) -> PrefillFootprint {
    #[cfg(feature = "cuda")]
    let tile = crate::inference::moe_cuda::MMQ_X_MAX;
    #[cfg(not(feature = "cuda"))]
    let tile = 1;
    let target_chunk = experts
        .map(|(n, used)| (n * tile / used.max(1)).clamp(2048, CHUNK_CEILING))
        .unwrap_or(0);
    let f = std::mem::size_of::<f32>();
    PrefillFootprint {
        bytes_per_token: streams * dim * f * WIDE_LIVE_BUFFERS
            + experts.map_or(0, |(_, used)| {
                super::expert_tier::prefill_bytes_per_token(used, dim, expert_ffn)
            }),
        fixed_bytes: mixer_block_rows() * streams * dim * f * MIXER_LIVE_BUFFERS,
        target_chunk,
    }
}

/// See [`prefill_footprint`].
#[derive(Debug, Clone, Copy)]
pub struct PrefillFootprint {
    pub bytes_per_token: usize,
    pub fixed_bytes: usize,
    pub target_chunk: usize,
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

/// [`combine`] for raw write-back weights, in one launch; with `next` the norm of the result
/// for the mixer that follows comes back too.
#[cfg(feature = "cuda")]
pub fn combine_on_card(
    r: &Tensor,
    out: &Tensor,
    inject_raw: &Tensor,
    next: Option<&HyperMixer>,
    streams: usize,
    dim: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    crate::inference::moe_cuda::hc_combine(
        r,
        out,
        inject_raw,
        next.map(|m| m.norm_params()),
        streams,
        dim,
    )
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
