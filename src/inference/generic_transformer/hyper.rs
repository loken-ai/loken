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
    down: Weight,
    up: Weight,
    inject: Option<Weight>,
    streams: usize,
    dim: usize,
    eps: f64,
}

/// Rows from which a mixer's projection runs as the card's half-precision product: the height of
/// the tile that product works in.
const HALF_PRODUCT_ROWS: usize = 64;

/// The relative distance from a mixer weight, from which holding it as int8 rows is not done:
/// the tolerance the tier's expert outputs are checked at.
#[cfg(feature = "cuda")]
const INT8_TOLERANCE: f64 = 2e-2;

/// A mixer's weight: a dense one on a card is held as int8 rows with a scale each when that
/// stays within [`INT8_TOLERANCE`] of it, else F16, which is exact for a stored BF16 weight;
/// either reads fewer bytes in a decode step. It is widened where a stage wants a matrix
/// product; any other weight is kept as it is.
enum Weight {
    Half(Tensor),
    /// `[n, k]` bytes that are signed, and the `n` scales.
    Int8 {
        q: Tensor,
        scale: Tensor,
    },
    Plain(QMatMul),
}

/// The rows of the dense F32 weight `t` as int8 with a scale for each group of 64 values, on `t`'s
/// card; `None` when that would move the weight by more than [`INT8_TOLERANCE`], or when a row is
/// not a whole number of groups.
#[cfg(feature = "cuda")]
pub(super) fn int8_rows(t: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
    let (n, k) = t.dims2()?;
    let group = crate::inference::moe_cuda::INT8_GROUP;
    if k % group != 0 {
        return Ok(None);
    }
    let host: Vec<f32> = t
        .to_device(&crate::tensor::Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let mut q = vec![0u8; n * k];
    let mut scale = vec![0f32; n * (k / group)];
    let (mut moved, mut norm) = (0f64, 0f64);
    for r in 0..n {
        for g in 0..k / group {
            let at = r * k + g * group;
            let values = &host[at..at + group];
            let peak = values.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let sc = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scale[r * (k / group) + g] = sc;
            for (i, &v) in values.iter().enumerate() {
                let code = (v / sc).round().clamp(-127.0, 127.0) as i8;
                q[at + i] = code as u8;
                let d = f64::from(v - f32::from(code) * sc);
                moved += d * d;
                norm += f64::from(v) * f64::from(v);
            }
        }
    }
    if moved.sqrt() > INT8_TOLERANCE * norm.sqrt() {
        return Ok(None);
    }
    let host_dev = crate::tensor::Device::Cpu;
    Ok(Some((
        Tensor::from_vec(q, (n, k), &host_dev)?.to_device(&t.device())?,
        Tensor::from_vec(scale, (n, k / group), &host_dev)?.to_device(&t.device())?,
    )))
}

impl Weight {
    /// The mixer's weights held together: as int8 rows when every one of them can be, since the
    /// fused launches read the three in one format, else each as [`Self::pack_exact`] has it.
    fn pack_all(
        down: QMatMul,
        up: QMatMul,
        inject: Option<QMatMul>,
    ) -> Result<(Self, Self, Option<Self>)> {
        #[cfg(feature = "cuda")]
        {
            let rows = |w: &QMatMul| -> Result<Option<(Tensor, Tensor)>> {
                match w.dense() {
                    Some(t)
                        if t.dtype() == DType::F32
                            && t.device().is_cuda()
                            && t.dims().len() == 2 =>
                    {
                        int8_rows(t)
                    }
                    _ => Ok(None),
                }
            };
            let (d, u) = (rows(&down)?, rows(&up)?);
            let i = inject.as_ref().map(rows).transpose()?;
            if let (Some(d), Some(u)) = (d, u) {
                if i.as_ref().is_none_or(Option::is_some) {
                    let i = i.map(|i| i.expect("checked above"));
                    return Ok((
                        Weight::Int8 { q: d.0, scale: d.1 },
                        Weight::Int8 { q: u.0, scale: u.1 },
                        i.map(|(q, scale)| Weight::Int8 { q, scale }),
                    ));
                }
            }
        }
        Ok((
            Self::pack_exact(down)?,
            Self::pack_exact(up)?,
            inject.map(Self::pack_exact).transpose()?,
        ))
    }

    fn pack_exact(w: QMatMul) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Some(t) = w.dense() {
            if t.dtype() == DType::F32 && t.device().is_cuda() && t.dims().len() == 2 {
                // CONVERSION: a dense mixer weight is held F16 on a card when that loses
                // nothing the weight had: a stored BF16 weight is exact in it but for the
                // values under its smallest normal, which move by less than 1e-7.
                let half = t.contiguous()?.to_dtype(DType::F16)?;
                let worst = (t - half.to_dtype(DType::F32)?)?
                    .abs()?
                    .flatten_all()?
                    .max_keepdim(0)?
                    .to_vec1::<f32>()?[0];
                if worst <= 1e-7 {
                    return Ok(Weight::Half(half));
                }
            }
        }
        Ok(Weight::Plain(w))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Weight::Plain(w) => w.forward(x),
            #[cfg(feature = "cuda")]
            Weight::Int8 { q, scale } => {
                // Rows enough for the half-precision product widen the weight to F16; fewer,
                // to the F32 the launches multiply in.
                let dims = x.dims().to_vec();
                let k = *dims.last().expect("an activation has a width");
                let rows = x.elem_count() / k;
                let x2 = x.reshape((rows, k))?;
                let half = rows >= HALF_PRODUCT_ROWS && x.device().is_cuda();
                let w = crate::inference::moe_cuda::hc_dequant(q, scale, !half)?;
                let y = if half {
                    x2.to_dtype(DType::F16)?.matmul_t(&w)?.to_dtype(x.dtype())?
                } else {
                    x2.matmul_t(&w)?.to_dtype(x.dtype())?
                };
                let mut out = dims;
                *out.last_mut().expect("an activation has a width") = q.dims()[0];
                y.reshape(out)
            }
            #[cfg(not(feature = "cuda"))]
            Weight::Int8 { .. } => crate::tensor::bail!("int8 mixer weights live on a card"),
            Weight::Half(w) => {
                let dims = x.dims().to_vec();
                let k = *dims.last().expect("an activation has a width");
                let rows = x.elem_count() / k;
                let x2 = x.reshape((rows, k))?;
                // Rows enough to fill a tile of the card's half-precision product use it: the
                // activation is rounded to half and the product accumulates in F32. Fewer rows,
                // a decoded token or the rows of a verify, keep the F32 product they had.
                let y = if rows >= HALF_PRODUCT_ROWS && x.device().is_cuda() {
                    x2.to_dtype(DType::F16)?.matmul_t(w)?.to_dtype(x.dtype())?
                } else {
                    x2.matmul_t(&w.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?
                };
                let mut out = dims;
                *out.last_mut().expect("an activation has a width") = w.dims()[0];
                y.reshape(out)
            }
        }
    }

    /// The weight as the dense tensor a fused launch reads, F16 or F32.
    fn dense(&self) -> Option<&Tensor> {
        match self {
            Weight::Half(w) => Some(w),
            Weight::Int8 { .. } => None,
            Weight::Plain(w) => w.dense(),
        }
    }

    /// The weight as a dense tensor, widened from int8 rows where it is held so.
    #[cfg(all(feature = "cuda", test))]
    fn widened(&self) -> Option<Tensor> {
        match self {
            Weight::Int8 { q, scale } => {
                crate::inference::moe_cuda::hc_dequant(q, scale, false).ok()
            }
            other => other.dense().cloned(),
        }
    }

    /// The weight as the fused launches read it: dense or as int8 rows.
    #[cfg(feature = "cuda")]
    fn mix(&self) -> Option<crate::inference::moe_cuda::MixWeight<'_>> {
        use crate::inference::moe_cuda::MixWeight;
        match self {
            Weight::Int8 { q, scale } => Some(MixWeight::Int8 { q, scale }),
            other => other.dense().map(MixWeight::Dense),
        }
    }

    fn device(&self) -> crate::tensor::Device {
        match self {
            Weight::Half(w) => w.device().clone(),
            Weight::Int8 { q, .. } => q.device().clone(),
            Weight::Plain(w) => w.device(),
        }
    }
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
        let (down, up, inject) = Weight::pack_all(down, up, inject)?;
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
        use crate::inference::moe_cuda::{hc_low, hc_mean_gate, hc_mix_fused_over};
        if let (Some(down), Some(up)) = (self.down.mix(), self.up.mix()) {
            let inject = self.inject.as_ref().map(|w| w.mix());
            if inject.as_ref().is_none_or(|w| w.is_some()) {
                let inject = inject.flatten();
                if let Some(done) =
                    hc_mix_fused_over(xn, down, up, inject, self.streams, self.dim, f16_out)?
                {
                    return Ok(done);
                }
            }
        }
        let low = hc_low(&self.down.forward(xn)?, self.streams)?;
        let up = self.up.forward(&low)?;
        let mixed = hc_mean_gate(xn, &up, self.streams, self.dim, f16_out)?;
        let inject = self.inject.as_ref().map(|w| w.forward(xn)).transpose()?;
        Ok((mixed, inject))
    }

    /// The `down` weight as a dense tensor, for a test of the fused launches.
    #[cfg(all(feature = "cuda", test))]
    pub(super) fn down_dense(&self) -> Option<Tensor> {
        self.down.widened()
    }

    /// The `up` weight as a dense tensor, for a test of the fused launches.
    #[cfg(all(feature = "cuda", test))]
    pub(super) fn up_dense(&self) -> Option<Tensor> {
        self.up.widened()
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

/// Rows of a prefill chunk a card's memory is planned for: the first rung of the chunk ladder.
pub(super) const PLANNED_CHUNK_ROWS: usize = 2048;
/// Buffers of `streams x dim` f32 a wide layer keeps live per token of a chunk: the residual, at a
/// combine the product it adds and the new residual, the casts and copies the sublayers make of
/// it, and their own activations counted at the same width (the mixers' temporaries are bounded
/// apart, by [`mixer_block_rows`]). Counted so that the model's bytes per token match the growth
/// of the card's pool over a prefill pass, which a pass of 354 and one of 668 rows put at 0.85 MB
/// a row.
const WIDE_LIVE_BUFFERS: usize = 15;
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
/// buffers live at once; apart from the chunk, the mixers' block temporaries; and the chunk the
/// card is planned for, [`PLANNED_CHUNK_ROWS`]. A longer prompt runs as several chunks, and a
/// layer whose experts stream in crosses them once for all of its chunks, so the memory a wider
/// chunk would take stays with the experts' cache. `experts` is `(count, used)`, `None` for a
/// model without a routed mixture; its experts run on a card for a batch of rows, and keep their
/// own activations live per routed pair, `expert_ffn` wide.
pub fn prefill_footprint(
    streams: usize,
    dim: usize,
    experts: Option<(usize, usize)>,
    expert_ffn: usize,
) -> PrefillFootprint {
    let target_chunk = experts.map_or(0, |_| PLANNED_CHUNK_ROWS);
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
