//! Shared attention and tensor helpers for the native model stack.
//!
//! Named for the family that first needed them, and used by every tower since: the tiled
//! SDPA here backs the image DiTs, the vision towers and the text encoders as well as the
//! audio ones. It is core, not a family - which is why it compiles into every build.
//!

use crate::tensor::{Result, Tensor, D};

/// What ONE COMPONENT of a checkpoint occupies on the card it is placed on.
///
/// The loaders used to answer this with the file's size on disk, and with a typed byte
/// count - 4 GB, 700 MB, 200 MB - whenever the file could not be stat'd. A figure nobody
/// measured cannot decide a placement, and the tensor directory is right there: it is read
/// before placement in every one of these loaders and accounts for every weight.
///
/// These loaders dequantise every weight they read and hand the card an f32 tensor, so
/// what goes resident is the component's element count times four - not the quantised
/// bytes, and not the file. Gating on the file was wrong in both directions at once, and
/// only worked because the two errors happened to cancel: the shared ACE-Step GGUF holds
/// the denoiser as well, so it over-states a small encoder by the rest of the file, while
/// under-stating that encoder's own expansion by four. A checkpoint whose split between
/// components differs breaks the coincidence, and the failure is an out-of-memory during
/// the load, on a card the planner had already approved.
///
/// `prefixes` selects the component's tensors. When none of them match - a renamed prefix,
/// a repacked checkpoint - the whole directory is charged rather than nothing: a demand of
/// zero reads to the planner as "needs no memory", which is the one answer that can never
/// be right.
pub(crate) fn gguf_resident_bytes(
    c: &crate::tensor::quantized::gguf_file::Content,
    prefixes: &[&str],
) -> u64 {
    /// Bytes per element once the loader has expanded the weight for the device.
    const RESIDENT_BYTES: u64 = 4;
    let matched: u64 = c
        .tensor_infos
        .iter()
        .filter(|(n, _)| prefixes.iter().any(|p| n.starts_with(p)))
        .map(|(_, i)| i.elem_count() as u64 * RESIDENT_BYTES)
        .sum();
    if matched > 0 {
        return matched;
    }
    c.tensor_infos
        .values()
        .map(|i| i.elem_count() as u64 * RESIDENT_BYTES)
        .sum()
}

/// Repeat KV heads for grouped-query attention - the substrate states it once.
pub(crate) use crate::tensor::ops::repeat_kv;

/// Scaled dot-product attention on the native stack. Drop-in for the compat
/// `ops::sdpa`: `scale` is applied before softmax; `softcapping != 1.0`
/// applies `tanh(scores/cap).cap`; `do_causal` adds a causal mask; otherwise an
/// optional additive `mask` is broadcast onto the scores. `q`: `[b,h,sq,d]`,
/// `k`/`v`: `[b,h,sk,d]`.
///
/// `pub` rather than `pub(crate)`: attention is the one primitive every transformer port
/// needs, and the alternative to sharing it is a second implementation of the most
/// numerically delicate loop in the stack, diverging silently from the tested one.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    // The tile follows the SEQUENCE: a fixed one bounds only one side of the transient.
    let seq = k
        .dims()
        .get(k.dims().len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    sdpa_tiled(
        q,
        k,
        v,
        mask,
        do_causal,
        scale,
        softcapping,
        query_tile_for(seq),
    )
}

/// [`sdpa`] with the two GEMMs on TENSOR CORES.
///
/// `sdpa_tiled_dt` has carried this since the image-edit work, where attention measured 71%
/// of a 1024-square forward - and until now not one call site in the repository asked for
/// it, so every DiT ran its attention as an F32 GEMM that the memory system, not the
/// arithmetic units, was limiting. On a video DiT that is the whole cost: attention is
/// quadratic in the token count, and a 30 s clip at 512 square is 123 904 tokens against
/// 21 504 for the 81-frame clip - 5.8 times the tokens and, measured, more than 26 times the
/// time, which is the signature of a quadratic term dominating everything else.
///
/// BF16 rather than F16 because it keeps F32's exponent range, which DiT activations need,
/// and cuBLAS accumulates BF16 products in F32 - so this is a narrower mantissa on the two
/// matmuls, not a narrower dynamic range. The softmax stays F32. It also halves the score
/// tile, which is the widest buffer in the forward.
///
/// Callers that need bit-exact F32 - parity harnesses, CPU paths, anything comparing against
/// a reference dump - must keep using [`sdpa`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_tc(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    let dims = q.dims();
    let seq = k
        .dims()
        .get(k.dims().len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    let heads = dims
        .get(dims.len().saturating_sub(3))
        .copied()
        .unwrap_or(REF_HEADS);
    let tile = tc_query_tile_for_heads(seq, heads);
    sdpa_tiled_dt(q, k, v, mask, do_causal, scale, softcapping, tile, true)
}

/// Head count the tile was tuned at. The score slab is `[tile, seq]` PER HEAD, so a model
/// three times as wide holds three times the slab for the same tile.
const REF_HEADS: usize = 12;

/// [`tc_query_tile_for`] for a model of `heads` attention heads.
///
/// Bounding the TILE bounds the slab only for one model width. A 14B video DiT has 40 heads
/// against the 12 the tile was tuned at, so the same tile holds 3.3x the scores - which is
/// how a 14B that fits its cards comfortably on paper OOMed on a cast. Scaling the tile by
/// the head count makes the resident slab a property of the SEQUENCE rather than of how
/// wide the model happens to be.
///
/// It costs almost nothing: the measured tile sweep at a video shape is 87.7 ms at 1024,
/// 90.4 ms at 384 and 95.1 ms at 256 - a few percent, against a spill to the host that
/// costs an order of magnitude.
pub(crate) fn tc_query_tile_for_heads(seq: usize, heads: usize) -> usize {
    let base = (query_tile_for(seq) * 2).min(QUERY_TILE);
    (base * REF_HEADS / heads.max(1)).clamp(MIN_QUERY_TILE, base)
}

/// Bytes per score element on the [`sdpa_tc`] path.
///
/// The scores and the softmax output are BF16 there, not F32. An estimate that charges four
/// bytes for a two-byte buffer reserves twice what the forward takes, and a reserve is
/// subtracted from the card before any block is placed - so over-stating it does not waste
/// memory, it pushes blocks onto the HOST. Measured: a 14B video DiT on two empty 16.6 GB
/// cards was handed a 4.3 GB budget and put eight of its forty blocks on the CPU.
pub(crate) fn tc_score_bytes() -> u64 {
    crate::tensor::DType::BF16.size_in_bytes() as u64
}

/// The query tile for a DENOISING DiT's attention, where the score slab is measured in
/// gigabytes and the tile decides whether a placement fits.
///
/// WHY IT IS EXPRESSED AGAINST `head_dim`. A tile written as a count of queries bounds
/// nothing: one slab is `heads * tile * seq * elem`, so a fixed query count gives a
/// slab that GROWS with the sequence - the thing the tiling was there to stop. Beside
/// the activation the same forward carries, `seq * dim * elem`, the ratio is
/// `tile / head_dim`, so a tile expressed as a multiple of `head_dim` makes the slab a
/// fixed multiple of an activation at any resolution, for any head count, on any card.
/// That multiple is what there is to choose, and being dimensionless it transfers.
///
/// WHY THE MULTIPLE IS TWO. Swept on an image DiT at a geometry where the slabs
/// dominate, images checked bit-for-bit at every point:
///
/// ```text
///     multiple   held per card   render   energy
///        8          6.88 GB      145.1 s  27.4 kJ
///        4          4.36 GB      141.7 s  26.9 kJ
///        2          2.62 GB      137.2 s  26.6 kJ
///        1          2.42 GB      144.1 s  27.3 kJ
/// ```
///
/// Two is the minimum of all three columns at once, not a compromise between them.
/// Above it, memory rises faster than the tile does - the allocator holds the slabs of
/// iterations whose frees the stream has not reached, so a bigger tile is retained
/// several times over. Below it the memory stops coming back while the time turns and
/// rises, the work per launch having become too small to cover the launch.
///
/// WHY IT IS A MINIMUM WITH THE EXISTING BOUND, and not a replacement for it. The
/// bound above it was won against real exhaustions on long clips, and expressed in the
/// same currency it sits at a multiple of 1.3 to 2.1 - the same place this sweep landed
/// from the other side. Where the two disagree is where its CEILING binds, which is at
/// image-length sequences. Taking the smaller of the two can only lower a tile, never
/// raise one, so no path that was won the hard way can move: it is a change to the
/// sequences nobody had measured, and to nothing else.
///
/// WHY ONLY A DiT MAY READ THIS. Lowering a tile makes callers tile that did not tile
/// before, and a caller whose slab is megabytes buys nothing for the launches it adds -
/// the ratio says "twelve activations, tile it" where the absolute says "it is twenty
/// megabytes, leave it alone". Every other caller of [`sdpa`] - the VAEs, the text and
/// vision encoders, the audio and speech paths - keeps [`query_tile_for`] until
/// somebody measures it, which is the same rule this whole derivation was written
/// under: the sequences that were measured move, the rest do not.
pub(crate) fn dit_query_tile(head_dim: usize, seq: usize, heads: usize) -> usize {
    /// The slab is worth this many of the activations the forward already carries.
    const SLAB_IN_ACTIVATIONS: usize = 2;
    (SLAB_IN_ACTIVATIONS * head_dim)
        .min(tc_query_tile_for_heads(seq, heads))
        .max(MIN_QUERY_TILE)
}

/// Query-tile size for [`sdpa`]. The full score matrix `[B,H,Sq,Sk]` is the
/// peak activation in dense non-causal attention (Wan/ACE-Step DiT 3D
/// self-attention), so it OOMs as the token count grows with larger/longer
/// video clips. Processing the query dimension in tiles caps the peak at
/// `[B,H,TILE,Sk]`; each query row's softmax is independent so the result is
/// bit-exact (see flux `scaled_dot_product_attention` / zimage
/// `attention_chunked`). A CUDA flash-attn kernel is deliberately NOT used: it produces
/// NaN/Inf on Blackwell (sm_120).
pub(crate) const QUERY_TILE: usize = 1024;

/// Smallest tile worth launching: below this the per-launch overhead outweighs the
/// memory saved, and no sequence this pipeline renders needs to go lower.
pub(crate) const MIN_QUERY_TILE: usize = 128;

/// The sequence beyond which the tile must start shrinking.
///
/// A FIXED tile bounds only one side of the transient: the peak is `[B, H, TILE, Sk]`, so
/// it still grows linearly with the sequence. That is fine for an image - a 1024^2 render
/// is ~21k tokens - and it is not fine for video, where the token count grows with the
/// clip: thirty seconds at 512^2 is ~124k tokens, and at a fixed tile of 1024 the score
/// buffer alone comes to eleven gigabytes, so the model fits no card and the whole denoise
/// falls to the host.
const TILE_SPAN: usize = 32768;

/// How many tokens of a feed-forward to compute at once.
///
/// Same reasoning as the query tile and the same exactness: a token's path through the
/// MLP does not depend on the others, so slicing is free. What it buys is the
/// `[tokens, 4 * dim]` intermediate, which on a long clip is the largest single buffer in
/// the forward - larger than the attention scores once those are tiled.
///
/// Returns `usize::MAX` below the span so the caller takes its single-call path unchanged.
pub(crate) fn ffn_chunk_for(tokens: usize) -> usize {
    if tokens <= TILE_SPAN {
        return usize::MAX;
    }
    TILE_SPAN.max(MIN_QUERY_TILE)
}

/// [`ffn_chunk_for`] bounded in BYTES rather than in tokens.
///
/// A token count is the wrong unit for a memory bound: the intermediate is
/// `chunk x mlp_ratio x dim`, so the same chunk costs four times as much on a model four
/// times as wide, and the same span that is comfortable at one resolution is gigabytes at
/// another. Measured at a 1024-square clip on a 14B: 5.4 GB of MLP intermediate on a card
/// that had 10.8 GB left for the entire forward.
///
/// The bound is expressed against the RESIDUAL STREAM - `tokens x dim` - because that is
/// the one buffer whose size is not a choice: it is what holding the sequence costs. Asking
/// the MLP intermediate not to exceed it is a ratio, derived from the architecture and the
/// request, and it needs no byte figure written anywhere.
pub(crate) fn ffn_chunk_bytes_bounded(tokens: usize, mlp_ratio: f64) -> usize {
    let by_tokens = ffn_chunk_for(tokens);
    // Below the span the caller takes its single-call path, and it must keep taking it:
    // slicing a short sequence adds launches to save a buffer that was never large.
    if by_tokens == usize::MAX {
        return usize::MAX;
    }
    // The intermediate is written and read at `mlp_ratio x dim` per token against the
    // stream's `dim`, so the chunk that matches it is `tokens / (2 x ratio)`.
    let by_bytes = ((tokens as f64) / (2.0 * mlp_ratio.max(1.0))).floor() as usize;
    by_tokens.min(by_bytes.max(MIN_QUERY_TILE))
}

/// The query tile for a sequence of `seq` keys, so the score transient stops growing.
///
/// Below [`TILE_SPAN`] this is exactly [`QUERY_TILE`], so every image-sized render behaves
/// as it did. Above it the tile shrinks in proportion, holding `tile * seq` at the value
/// it has at that span.
///
/// The tiling is exact IN ARITHMETIC - each query row's softmax is independent of the
/// others - so it changes the memory and the number of launches and nothing about what the
/// computation means. On the host that is bit-exactness too, and the tests below hold it
/// across several tile sizes and tail lengths.
///
/// It is NOT a promise that two DIFFERENT tiles give the same bits on a device that picks
/// its GEMM kernel by shape. Measured on an image DiT: changing the tile left three of four
/// geometries bit-identical and moved the fourth - the one whose last tile went from eleven
/// rows to two hundred and sixty-seven, which is a different kernel. The image is not wrong,
/// it is another sample of the same distribution. But a change of tile is a change of
/// sample, and that belongs in a release note rather than in a silent diff.
pub(crate) fn query_tile_for(seq: usize) -> usize {
    let seq = seq.max(1);
    (QUERY_TILE * TILE_SPAN / seq).clamp(MIN_QUERY_TILE, QUERY_TILE)
}

/// [`sdpa`] with an explicit query-tile size (the `tile` argument exists so the
/// bit-exactness unit test can force the tiled path on tiny inputs). When
/// `Sq <= tile` the single-pass path runs unchanged (bit-identical, zero
/// overhead for decode / short-sequence callers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
    tile: usize,
) -> Result<Tensor> {
    sdpa_tiled_dt(q, k, v, mask, do_causal, scale, softcapping, tile, false)
}

/// `sdpa_tiled` with an explicit GEMM precision. `bf16_gemm` runs the two matmuls
/// (scores and the value product) on tensor cores in BF16 while the softmax and the
/// score post-processing stay F32: BF16 has F32's exponent range (so activations of
/// ~1e9 are safe, unlike F16) and cuBLAS accumulates BF16 products in F32. On this
/// class of DiT attention that is the difference between a memory-bound F32 GEMM and
/// the tensor-core path - attention measured 71% of a 1024^2 edit's forward time.
/// Callers that need bit-exact F32 (parity harnesses, CPU paths) pass false.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_tiled_dt(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
    tile: usize,
    bf16_gemm: bool,
) -> Result<Tensor> {
    // BF16 only helps on a CUDA device with F32 inputs; anything else stays exact.
    let bf16_gemm = bf16_gemm && q.device().is_cuda() && q.dtype() == crate::tensor::DType::F32;
    let cast = |t: &Tensor| -> Result<Tensor> {
        if bf16_gemm {
            t.to_dtype(crate::tensor::DType::BF16)
        } else {
            Ok(t.clone())
        }
    };
    let kt = cast(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
    // Fold the softmax scale into Q ONCE instead of scaling the score matrix per
    // tile. Scaling is linear so the result is unchanged, but the score buffer is
    // [heads, tile, seq] - hundreds of MB at DiT sequence lengths - and this
    // attention is memory-bound on passes over it, not compute-bound: dropping one
    // full read+write of that buffer per tile is worth far more than the touch of Q.
    let qc_f32 = q.contiguous()?;
    let q_scaled = if scale == 1.0 {
        qc_f32.clone()
    } else {
        qc_f32.affine(scale, 0.0)?
    };
    let qc = cast(&q_scaled)?;
    let vc = cast(&v.contiguous()?)?;
    let sq = qc.dim(D::Minus2)?;

    // Process one query tile (global rows [q_off, q_off+tile_len)). Independent
    // of the other tiles, so concatenating the per-tile outputs reproduces the
    // single-pass result bit-for-bit.
    let attend = |qt: &Tensor, q_off: usize| -> Result<Tensor> {
        let sqt = qt.dim(D::Minus2)?;
        // Q already carries the scale (folded above).
        let scores = qt.matmul(&kt)?;
        // Nothing to add to the scores (no mask, no causal fill, no softcap) means the
        // softmax can consume the GEMM's BF16 output directly. That is the common DiT
        // case, and it removes the two widest passes of the whole forward: the upcast
        // to F32 before the softmax and the downcast after it, over a score tile that
        // is hundreds of MB. The BF16 softmax computes in F32 internally, so the result
        // is bit-identical to the cast-softmax-cast it replaces.
        let plain = mask.is_none() && !do_causal && (softcapping - 1.0).abs() <= f32::EPSILON;
        if bf16_gemm && plain {
            let attn = scores.softmax_last_dim()?.contiguous()?;
            return attn.matmul(&vc)?.to_dtype(crate::tensor::DType::F32);
        }
        let scores = if bf16_gemm {
            scores.to_dtype(crate::tensor::DType::F32)?
        } else {
            scores
        };
        let scores = if (softcapping - 1.0).abs() > f32::EPSILON {
            scores
                .affine(1.0 / softcapping, 0.0)?
                .tanh()?
                .affine(softcapping, 0.0)?
        } else {
            scores
        };
        let scores = if do_causal {
            let sk = scores.dim(D::Minus1)?;
            // Key offset uses the FULL query length so tile row r maps to global
            // query index q_off+r (key j visible iff j <= q_off+r+off).
            let off = sk - sq;
            let neg_inf = f32::NEG_INFINITY;
            let mut data = vec![0f32; sqt * sk];
            for i in 0..sqt {
                let gi = q_off + i;
                for j in 0..sk {
                    if j > gi + off {
                        data[i * sk + j] = neg_inf;
                    }
                }
            }
            let cmask = Tensor::from_vec_f32(data, (sqt, sk))?
                .to_device(&scores.device())?
                .to_dtype(scores.dtype())?;
            scores.broadcast_add(&cmask)?
        } else if let Some(m) = mask {
            // Slice the mask's query rows to this tile; key/batch/head axes
            // broadcast unchanged. A broadcast query dim (==1) is left as-is.
            if m.dim(D::Minus2)? == 1 {
                scores.broadcast_add(m)?
            } else {
                scores.broadcast_add(&m.narrow(D::Minus2, q_off, sqt)?)?
            }
        } else {
            scores
        };
        let attn = scores.softmax_last_dim()?;
        let attn = attn.contiguous()?;
        let out = if bf16_gemm {
            attn.to_dtype(crate::tensor::DType::BF16)?
                .matmul(&vc)?
                .to_dtype(crate::tensor::DType::F32)?
        } else {
            attn.matmul(&vc)?
        };
        Ok(out)
    };

    let sq = qc_f32.dim(D::Minus2)?;
    if sq <= tile {
        return attend(&qc, 0);
    }
    let mut outs = Vec::with_capacity(sq.div_ceil(tile));
    let mut off = 0;
    while off < sq {
        let t = (sq - off).min(tile);
        // The tile MUST be materialized contiguous: narrowing the query dim of a batched
        // [b,h,sq,d] tensor yields a strided view (each batch/head still strides over the FULL
        // sq), and the batched-matmul path reads batches at the narrowed tile's pitch - heads
        // beyond the first read shifted data, which surfaced as patch-aligned block artifacts
        // in any forward whose sequence exceeded the tile (Wan video at >=1024 tokens). The
        // CPU unit test never caught it (the CPU matmul resolves strided views correctly).
        outs.push(attend(&qc.narrow(D::Minus2, off, t)?.contiguous()?, off)?);
        off += t;
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    Tensor::cat(&refs, D::Minus2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic pseudo-random spread in roughly [-1, 1].
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// What the tiling DOES promise, held across tile sizes and across tail lengths:
    /// on the host, a query tile changes nothing about the bits.
    ///
    /// The tail is the part worth pinning. A tile that divides the sequence produces
    /// full tiles only; one that does not leaves a last tile of a different height, and
    /// on a device that selects its GEMM kernel by shape that is where two tile sizes
    /// stop agreeing - measured, on one geometry of four. The host has no such
    /// selection, so here every tail must give the same answer, and this test says so
    /// with tails of every length from a full tile down to one row.
    #[test]
    fn query_tiling_agrees_across_tile_sizes_and_tails_on_the_host() {
        let (b, h, sq, sk, d) = (1usize, 2usize, 12usize, 12usize, 8usize);
        let q = Tensor::from_vec_f32(data(b * h * sq * d, 81), (b, h, sq, d)).unwrap();
        let k = Tensor::from_vec_f32(data(b * h * sk * d, 82), (b, h, sk, d)).unwrap();
        let v = Tensor::from_vec_f32(data(b * h * sk * d, 83), (b, h, sk, d)).unwrap();
        let scale = 1.0f32 / (d as f32).sqrt();
        let reference = sdpa_tiled(&q, &k, &v, None, false, scale, 1.0, sq)
            .unwrap()
            .to_vec_f32();
        // 12 divides by 6, 4, 3 and 2; 5, 7, 8, 9 and 11 leave tails of 2, 5, 4, 3 and 1.
        for tile in [2usize, 3, 4, 5, 6, 7, 8, 9, 11, 12] {
            let out = sdpa_tiled(&q, &k, &v, None, false, scale, 1.0, tile)
                .unwrap()
                .to_vec_f32();
            assert_eq!(
                out,
                reference,
                "tile {tile} (tail {}) did not reproduce the untiled result",
                sq % tile
            );
        }
    }

    #[test]
    fn sdpa_query_tiling_is_bit_exact() {
        let (b, h, sq, sk, d) = (1usize, 2usize, 10usize, 10usize, 8usize);
        let q = Tensor::from_vec_f32(data(b * h * sq * d, 71), (b, h, sq, d)).unwrap();
        let k = Tensor::from_vec_f32(data(b * h * sk * d, 72), (b, h, sk, d)).unwrap();
        let v = Tensor::from_vec_f32(data(b * h * sk * d, 73), (b, h, sk, d)).unwrap();
        let scale = 1.0f32 / (d as f32).sqrt();

        // Additive masks the ACE-Step callers pass: a [1,1,Sq,Sk] explicit one
        // (must be sliced per tile) and a [1,1,1,Sk] broadcast one (kept as-is).
        let mfull = Tensor::from_vec_f32(data(sq * sk, 74), (1, 1, sq, sk)).unwrap();
        let mbcast = Tensor::from_vec_f32(data(sk, 75), (1, 1, 1, sk)).unwrap();

        let cases: [(Option<&Tensor>, bool); 4] = [
            (None, false),
            (None, true),
            (Some(&mfull), false),
            (Some(&mbcast), false),
        ];
        for (mask, causal) in cases {
            let single = sdpa_tiled(&q, &k, &v, mask, causal, scale, 1.0, sq).unwrap();
            let tiled = sdpa_tiled(&q, &k, &v, mask, causal, scale, 1.0, 3).unwrap();
            assert_eq!(single.dims(), tiled.dims());
            let (a, e) = (tiled.to_vec_f32(), single.to_vec_f32());
            for (x, y) in a.iter().zip(e.iter()) {
                assert_eq!(
                    x,
                    y,
                    "tiled sdpa mismatch (mask={}, causal={})",
                    mask.is_some(),
                    causal
                );
            }
        }
    }
}

#[cfg(test)]
mod gpu_tile_tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// The BF16 softmax must be BIT-identical to the cast-softmax-cast it replaced.
    ///
    /// This is not a tolerance test on purpose. The claim that justified removing two
    /// passes over the attention score tile is that widening bf16 is exact, the f32
    /// arithmetic is unchanged, and the single rounding at the end is the one the final
    /// cast already did. If that is true the outputs match exactly; if it is not, the
    /// change is a silent precision edit to every DiT in the tree, and "close enough"
    /// is exactly how that would get shipped.
    #[test]
    #[ignore = "needs a CUDA device"]
    fn bf16_softmax_is_bit_identical_to_the_f32_round_trip() {
        use crate::tensor::{DType, Device, Tensor};
        let Ok(dev) = Device::new_cuda(0) else {
            println!("no CUDA device; skipping");
            return;
        };
        // Shapes that exercise a real score tile, plus awkward widths that land the
        // block reduction on a partial tail.
        for (rows, cols) in [(64usize, 9216usize), (7, 1), (3, 255), (129, 513)] {
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            let mut v = Vec::with_capacity(rows * cols);
            for _ in 0..rows * cols {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // Spread across a wide exponent range - attention scores are not small.
                v.push(((seed >> 40) as f32 / 1024.0) - 1.0);
            }
            let x = Tensor::from_vec(v, (rows, cols), &dev).expect("x");
            let xb = x.to_dtype(DType::BF16).expect("to bf16");

            // The chain this replaced.
            let want = xb
                .to_dtype(DType::F32)
                .and_then(|t| t.softmax_last_dim())
                .and_then(|t| t.to_dtype(DType::BF16))
                .expect("reference chain");
            // The kernel.
            let got = xb.softmax_last_dim().expect("bf16 softmax");
            assert_eq!(got.dtype(), DType::BF16, "must stay bf16");

            let a = want
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = got
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{rows}x{cols}: {diff} elements differ from the f32 round-trip"
            );
        }
    }

    // GPU parity of the tiled SDPA at the exact Wan video shape (queries > QUERY_TILE, 4-D
    // batched, CUDA). The CPU test cannot catch CUDA-kernel stride behavior. Run:
    //   cargo test -p loken --profile fast gpu_sdpa_tile_parity -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_sdpa_tile_parity() {
        let dev = match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(d) => crate::tensor::Device::Cuda(d),
            Err(_) => {
                eprintln!("no cuda; skip");
                return;
            }
        };
        let (b, h, sq, sk, d) = (1usize, 12usize, 1280usize, 1280usize, 128usize);
        let mk = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
                })
                .collect()
        };
        let q = Tensor::from_vec_f32(mk(b * h * sq * d, 1), (b, h, sq, d))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let k = Tensor::from_vec_f32(mk(b * h * sk * d, 2), (b, h, sk, d))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let v = Tensor::from_vec_f32(mk(b * h * sk * d, 3), (b, h, sk, d))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let scale = 1.0f32 / (d as f32).sqrt();
        let single = sdpa_tiled(&q, &k, &v, None, false, scale, 1.0, sq)
            .unwrap()
            .to_vec_f32();
        let tiled = sdpa_tiled(&q, &k, &v, None, false, scale, 1.0, 1024)
            .unwrap()
            .to_vec_f32();
        // TRUE oracle: the same inputs on CPU (tiled-vs-single on GPU shares the same matmul,
        // so a size-dependent CUDA matmul defect would pass that parity while being wrong).
        let qc2 = q.to_device(&crate::tensor::Device::Cpu).unwrap();
        let kc2 = k.to_device(&crate::tensor::Device::Cpu).unwrap();
        let vc2 = v.to_device(&crate::tensor::Device::Cpu).unwrap();
        let cpu = sdpa_tiled(&qc2, &kc2, &vc2, None, false, scale, 1.0, sq)
            .unwrap()
            .to_vec_f32();
        let mut max_cpu_gpu = 0f32;
        for (a, e) in single.iter().zip(cpu.iter()) {
            let d = (a - e).abs();
            if d > max_cpu_gpu {
                max_cpu_gpu = d;
            }
        }
        println!("GPU-vs-CPU oracle: max|diff|={max_cpu_gpu:e}");
        let mut max_abs = 0f32;
        let mut first_bad = None;
        for (i, (a, e)) in tiled.iter().zip(single.iter()).enumerate() {
            let dd = (a - e).abs();
            if dd > max_abs {
                max_abs = dd;
            }
            if dd > 1e-4 && first_bad.is_none() {
                first_bad = Some((i, *a, *e));
            }
        }
        println!("GPU tile parity: max|diff|={max_abs:e} first_bad={first_bad:?}");
        assert!(
            max_abs < 1e-4,
            "tiled CUDA sdpa diverges: max|diff|={max_abs}"
        );
    }
}

#[cfg(test)]
mod query_tile_tests {
    use super::*;

    /// An image-sized sequence keeps the tile it has always had, so nothing that works
    /// today changes speed or launch count.
    #[test]
    fn short_sequences_are_untouched() {
        assert_eq!(query_tile_for(1), QUERY_TILE);
        assert_eq!(query_tile_for(4096), QUERY_TILE);
        assert_eq!(query_tile_for(21_504), QUERY_TILE, "a 1024^2 image render");
        assert_eq!(query_tile_for(TILE_SPAN), QUERY_TILE, "exactly at the span");
    }

    /// Past the span the transient stops growing: `tile * seq` holds at the value it has
    /// at the span, which is the whole point - a fixed tile bounds only one side of it.
    #[test]
    fn the_transient_stops_growing_with_the_clip() {
        let at_span = QUERY_TILE * TILE_SPAN;
        for seq in [40_000usize, 123_904, 500_000] {
            let t = query_tile_for(seq);
            assert!(t < QUERY_TILE, "seq {seq} should shrink the tile, got {t}");
            assert!(
                t * seq <= at_span || t == MIN_QUERY_TILE,
                "seq {seq}: tile {t} gives {} against a budget of {at_span}",
                t * seq
            );
        }
    }

    /// A 30 s clip at 512^2 is the case this exists for. At the fixed tile its score
    /// buffer was eleven gigabytes, which fits no card here; state the improvement as a
    /// number so a future change to the span cannot quietly undo it.
    #[test]
    fn a_thirty_second_clip_fits_a_card() {
        const TOKENS: usize = 123_904; // 481 frames at 512^2, 1.3B geometry
        const HEADS: usize = 12;
        let bytes = |tile: usize| 2u64 * tile as u64 * TOKENS as u64 * HEADS as u64 * 4;
        assert!(bytes(QUERY_TILE) > 10 * (1 << 30), "the case this fixes");
        assert!(
            bytes(query_tile_for(TOKENS)) < 4 * (1 << 30),
            "still {} GB",
            bytes(query_tile_for(TOKENS)) as f64 / (1u64 << 30) as f64
        );
    }

    /// The tile never reaches zero, whatever the sequence.
    #[test]
    fn the_tile_has_a_floor() {
        assert_eq!(query_tile_for(usize::MAX), MIN_QUERY_TILE);
        assert!(query_tile_for(0) >= MIN_QUERY_TILE);
    }
}

#[cfg(test)]
mod flash_headroom {
    //! How much is left on the table by materialising the score matrix at all.
    //!
    //! The tiled path writes the `[tile, seq]` scores to HBM, reads them for the softmax,
    //! writes them again and reads them for the second GEMM. A flash kernel keeps that block
    //! in registers and shared memory and never writes it. The repository already has one -
    //! `fused_kernels::flash_prefill_f16`, tensor-core, online-softmax - but it is CAUSAL, so
    //! it cannot serve a DiT's bidirectional attention as it stands.
    //!
    //! Before changing its signature, measure whether the change is worth making. A causal
    //! pass does half the score work of a full one, so twice the causal time is a fair (and
    //! slightly pessimistic, since it skips whole tiles) estimate of the non-causal cost.
    //!
    //!   cargo test -p loken --release flash_headroom -- --ignored --nocapture
    use crate::tensor::{Device, Tensor};

    #[test]
    #[ignore]
    fn tiled_scores_versus_flash_at_the_wan_shape() {
        let dev = match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(d) => Device::Cuda(d),
            Err(_) => {
                eprintln!("no cuda; skip");
                return;
            }
        };
        // One 81-frame clip's self-attention: 121/4+1 latent frames x 32 x 32 patches.
        let (b, h, s, hd) = (1usize, 12usize, 21_504usize, 128usize);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
                })
                .collect()
        };
        let shape = (b, h, s, hd);
        let n = b * h * s * hd;
        let q = Tensor::from_vec(mk(n), shape, &dev).unwrap();
        let k = Tensor::from_vec(mk(n), shape, &dev).unwrap();
        let v = Tensor::from_vec(mk(n), shape, &dev).unwrap();
        let scale = 1.0 / (hd as f32).sqrt();

        let sync = || {
            let _ = dev.synchronize();
        };
        // Warm both paths before timing: the first call pays allocation and module load.
        let _ = super::sdpa_tc(&q, &k, &v, None, false, scale, 1.0).unwrap();
        sync();
        let t0 = std::time::Instant::now();
        let _ = super::sdpa_tc(&q, &k, &v, None, false, scale, 1.0).unwrap();
        sync();
        let tiled = t0.elapsed().as_secs_f32();

        match crate::inference::kernel::fused::flash_prefill_f16(&q, &k, &v, h, h, scale, 0) {
            Ok(Some(_)) => {}
            other => {
                eprintln!("flash kernel unavailable at this shape: {other:?}");
                return;
            }
        }
        sync();
        let t1 = std::time::Instant::now();
        let _ = crate::inference::kernel::fused::flash_prefill_f16(&q, &k, &v, h, h, scale, 0);
        sync();
        let causal = t1.elapsed().as_secs_f32();

        eprintln!(
            "wan self-attn S={s} h={h} hd={hd}: tiled(bf16) {:.1} ms | flash(causal) {:.1} ms \
             -> non-causal estimate {:.1} ms = {:.1}x",
            tiled * 1e3,
            causal * 1e3,
            causal * 2.0 * 1e3,
            tiled / (causal * 2.0)
        );

        // What a 14B's forward is actually made of. Attention and the feed-forward scale
        // differently with width - attention with heads, the MLP with the square of the
        // model width - so the share that a better attention kernel could win has to be
        // measured at the width that matters, not inferred from the small model.
        for (heads, dim, ffn, tag) in [
            (12usize, 1536usize, 8960usize, "1.3B"),
            (40, 5120, 13824, "14B"),
        ] {
            let n = b * heads * s * hd;
            let q2 = Tensor::from_vec(mk(n), (b, heads, s, hd), &dev).unwrap();
            let k2 = Tensor::from_vec(mk(n), (b, heads, s, hd), &dev).unwrap();
            let v2 = Tensor::from_vec(mk(n), (b, heads, s, hd), &dev).unwrap();
            let _ = super::sdpa_tc(&q2, &k2, &v2, None, false, scale, 1.0).unwrap();
            sync();
            let t = std::time::Instant::now();
            let _ = super::sdpa_tc(&q2, &k2, &v2, None, false, scale, 1.0).unwrap();
            sync();
            let attn_ms = t.elapsed().as_secs_f64() * 1e3;
            // One block's feed-forward: two GEMMs, [S,dim]x[dim,ffn] and back.
            let x = Tensor::from_vec(mk(s * dim), (1, s, dim), &dev).unwrap();
            let w1 = Tensor::from_vec(mk(dim * ffn), (1, dim, ffn), &dev).unwrap();
            let w2 = Tensor::from_vec(mk(ffn * dim), (1, ffn, dim), &dev).unwrap();
            let run = || {
                let h = x
                    .to_dtype(crate::tensor::DType::BF16)
                    .unwrap()
                    .matmul(&w1.to_dtype(crate::tensor::DType::BF16).unwrap())
                    .unwrap();
                let _ = h
                    .matmul(&w2.to_dtype(crate::tensor::DType::BF16).unwrap())
                    .unwrap();
            };
            run();
            sync();
            let t = std::time::Instant::now();
            run();
            sync();
            let ffn_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{tag} block at S={s}: attention {attn_ms:6.1} ms | ffn {ffn_ms:6.1} ms  \
                 -> attention is {:.0}% of the pair, a perfect attention kernel caps at {:.2}x",
                100.0 * attn_ms / (attn_ms + ffn_ms),
                (attn_ms + ffn_ms) / ffn_ms
            );
        }

        // The score block is [tile, S] per head. Small enough and the softmax reads it back
        // out of L2 instead of HBM; too small and the GEMM shape and the launch count take
        // over. Nothing predicts where that crossover sits on a given card, so sweep it.
        for t in [64usize, 96, 128, 192, 256, 384, 512, 1024, 2048] {
            let _ = super::sdpa_tiled_dt(&q, &k, &v, None, false, scale, 1.0, t, true).unwrap();
            sync();
            let t0 = std::time::Instant::now();
            let _ = super::sdpa_tiled_dt(&q, &k, &v, None, false, scale, 1.0, t, true).unwrap();
            sync();
            let ms = t0.elapsed().as_secs_f32() * 1e3;
            let block_mb = (t * s * h * 2) as f64 / 1e6;
            eprintln!("  tile {t:>5}: {ms:7.1} ms   (score block {block_mb:.0} MB)");
        }
    }
}

#[cfg(test)]
mod flash_dit_tests {
    //! The DiT flash kernel: is it RIGHT, and is it faster than what it replaces.
    //!
    //!   cargo test -p loken --release flash_dit -- --ignored --nocapture
    use crate::tensor::{Device, Tensor};

    fn rand_vec(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// Against the tiled path, which is the reference this has to reproduce. Not bit-exact:
    /// the online softmax sums in a different order and the accumulator is rescaled as it
    /// goes, so the comparison is a relative error over the whole output.
    #[test]
    #[ignore]
    fn it_matches_the_tiled_path() {
        let dev = match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(d) => Device::Cuda(d),
            Err(_) => {
                eprintln!("no cuda; skip");
                return;
            }
        };
        let mut seed = 0x243F6A8885A308D3u64;
        // Shapes that exercise the awkward cases: a sequence that is not a multiple of the
        // block, one head, several heads, both head dimensions.
        for (b, h, s, hd) in [
            (1usize, 1usize, 64usize, 64usize),
            (1, 2, 130, 64),
            (1, 4, 256, 128),
            (1, 12, 1024, 128),
            (1, 3, 333, 128),
        ] {
            let n = b * h * s * hd;
            let q = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let k = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let v = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let scale = 1.0 / (hd as f32).sqrt();
            let want = super::sdpa_tc(&q, &k, &v, None, false, scale, 1.0)
                .unwrap()
                .to_vec_f32();
            let got = crate::inference::kernel::fused::flash_dit_bf16(&q, &k, &v, scale, 0, 0)
                .unwrap()
                .unwrap_or_else(|| panic!("kernel declined [{b},{h},{s},{hd}]"))
                .to_vec_f32();
            assert_eq!(want.len(), got.len());
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (a, c) in want.iter().zip(&got) {
                num += ((a - c) as f64).powi(2);
                den += (*a as f64).powi(2);
            }
            let rel = (num / den.max(1e-30)).sqrt();
            assert!(
                rel < 2e-2,
                "[{b},{h},{s},{hd}]: relative error {rel:.4} against the tiled path"
            );
            eprintln!("  [{b},{h},{s},{hd}] rel {rel:.5} ok");
        }
    }

    /// And is it worth having. Both video shapes, against the path in production.
    #[test]
    #[ignore]
    fn it_is_faster_at_the_video_shapes() {
        let dev = match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(d) => Device::Cuda(d),
            Err(_) => {
                eprintln!("no cuda; skip");
                return;
            }
        };
        let mut seed = 0x13198A2E03707344u64;
        let (b, s, hd) = (1usize, 21_504usize, 128usize);
        for (h, tag) in [(12usize, "1.3B"), (40, "14B")] {
            let n = b * h * s * hd;
            let q = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let k = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let v = Tensor::from_vec(rand_vec(&mut seed, n), (b, h, s, hd), &dev).unwrap();
            let scale = 1.0 / (hd as f32).sqrt();
            let sync = || {
                let _ = dev.synchronize();
            };
            let _ = super::sdpa_tc(&q, &k, &v, None, false, scale, 1.0).unwrap();
            sync();
            let t = std::time::Instant::now();
            let _ = super::sdpa_tc(&q, &k, &v, None, false, scale, 1.0).unwrap();
            sync();
            let tiled = t.elapsed().as_secs_f64() * 1e3;
            let _ =
                crate::inference::kernel::fused::flash_dit_bf16(&q, &k, &v, scale, 0, 0).unwrap();
            sync();
            let t = std::time::Instant::now();
            let _ =
                crate::inference::kernel::fused::flash_dit_bf16(&q, &k, &v, scale, 0, 0).unwrap();
            sync();
            let flash = t.elapsed().as_secs_f64() * 1e3;
            // And with the radial mask on: 21504 tokens is 21 latent frames of 32x32.
            let ft = 1024usize;
            let _ =
                crate::inference::kernel::fused::flash_dit_bf16(&q, &k, &v, scale, ft, 32).unwrap();
            sync();
            let t = std::time::Instant::now();
            let _ =
                crate::inference::kernel::fused::flash_dit_bf16(&q, &k, &v, scale, ft, 32).unwrap();
            sync();
            let radial = t.elapsed().as_secs_f64() * 1e3;
            let tflops = 4.0 * (s as f64) * (s as f64) * (hd * h) as f64 / 1e12;
            eprintln!(
                "{tag} S={s} h={h}: tiled {tiled:7.1} ms ({:.0} TFLOP/s) | \
                 flash {flash:7.1} ms ({:.0} TFLOP/s) -> {:.2}x | \
                 radial {radial:7.1} ms -> {:.2}x vs tiled",
                tflops / (tiled / 1e3),
                tflops / (flash / 1e3),
                tiled / flash,
                tiled / radial
            );
        }
    }
}

#[cfg(test)]
mod ffn_chunk_bytes_tests {
    use super::{ffn_chunk_bytes_bounded, ffn_chunk_for, MIN_QUERY_TILE};

    /// The intermediate must not outgrow the residual stream, whatever the width. A token
    /// bound cannot promise that: the same chunk costs four times as much on a model four
    /// times as wide, which is how a 14B came to hold 5.4 GB of it on a card with 10.8 free.
    #[test]
    fn the_intermediate_stays_within_the_stream_it_rides_beside() {
        for tokens in [40_000usize, 86_016, 200_000] {
            for ratio in [2.0f64, 4.0, 8.0] {
                let chunk = ffn_chunk_bytes_bounded(tokens, ratio).min(tokens);
                let stream = tokens as f64;
                let intermediate = 2.0 * ratio * chunk as f64;
                assert!(
                    intermediate <= stream * 1.001,
                    "tokens={tokens} ratio={ratio}: {intermediate} against a {stream} stream"
                );
            }
        }
    }

    /// A wider feed-forward gets a SMALLER slice, which is the whole point of counting
    /// bytes: the token bound gave every model the same one.
    #[test]
    fn a_wider_feed_forward_takes_a_smaller_slice() {
        let t = 86_016usize;
        let narrow = ffn_chunk_bytes_bounded(t, 2.0);
        let wide = ffn_chunk_bytes_bounded(t, 8.0);
        assert!(wide < narrow, "ratio 8 got {wide}, ratio 2 got {narrow}");
    }

    /// Short sequences keep the single-call path exactly as they had it - no extra launches
    /// and no behaviour change where memory was never the problem.
    #[test]
    fn a_short_sequence_is_untouched() {
        for tokens in [512usize, 4096, 20_000] {
            assert_eq!(ffn_chunk_for(tokens), usize::MAX);
            assert_eq!(ffn_chunk_bytes_bounded(tokens, 4.0), usize::MAX);
        }
    }

    /// And it never slices so thin that the launches cost more than the buffer saves.
    #[test]
    fn the_slice_has_a_floor() {
        assert!(ffn_chunk_bytes_bounded(40_000, 1000.0) >= MIN_QUERY_TILE);
    }
}

#[cfg(test)]
mod dit_query_tile_tests {
    use super::{dit_query_tile, query_tile_for, tc_query_tile_for_heads, MIN_QUERY_TILE};

    /// Shapes this repo actually renders: heads, head_dim, sequence.
    const SHAPES: &[(&str, usize, usize, usize)] = &[
        ("image DiT, 512 square", 30, 128, 1035),
        ("image DiT, 1024 square", 30, 128, 4107),
        ("image DiT, 1536 square", 30, 128, 9227),
        ("wide image DiT", 28, 120, 4096),
        ("video DiT, one window", 40, 128, 21504),
        ("video DiT, long clip", 40, 128, 124000),
        ("narrow model", 8, 64, 800),
        ("audio DiT", 12, 64, 32768),
    ];

    /// THE GATE. Taking the smaller of the derived tile and the bound the video paths
    /// were won with can only LOWER a tile - so no path that was won against a real
    /// exhaustion can move, whatever a later hand does to the derivation.
    #[test]
    fn no_shape_is_ever_given_a_bigger_tile_than_it_has_today() {
        for (what, heads, head_dim, seq) in SHAPES {
            let today = tc_query_tile_for_heads(*seq, *heads);
            let derived = dit_query_tile(*head_dim, *seq, *heads);
            assert!(
                derived <= today.max(MIN_QUERY_TILE),
                "{what}: {derived} against the {today} it has today"
            );
        }
    }

    /// Where the slab is the thing that decides a placement, the derived term binds and
    /// the slab comes out at two activations - the multiple that was swept.
    #[test]
    fn an_image_dit_gets_the_multiple_that_was_measured() {
        let (heads, head_dim, seq) = (30, 128, 9227);
        assert_eq!(dit_query_tile(head_dim, seq, heads), 2 * head_dim);
        let slab = heads * dit_query_tile(head_dim, seq, heads) * seq;
        let activation = seq * heads * head_dim;
        assert_eq!(slab / activation, 2);
    }

    /// ...and where the bound the clips were won with is the tighter of the two, it is
    /// the one that stands. This is the case the minimum exists for.
    #[test]
    fn a_long_clip_keeps_the_tile_it_was_won_with() {
        let (heads, head_dim, seq) = (40, 128, 124000);
        assert_eq!(
            dit_query_tile(head_dim, seq, heads),
            tc_query_tile_for_heads(seq, heads)
        );
        assert!(dit_query_tile(head_dim, seq, heads) < 2 * head_dim);
    }

    /// The floor that was already there still holds: a model narrow enough for two of
    /// its head_dim to fall under it does not get a tile too small to launch.
    #[test]
    fn a_narrow_model_does_not_fall_through_the_floor() {
        assert_eq!(dit_query_tile(32, 4096, 8), MIN_QUERY_TILE);
    }

    /// The callers this derivation is NOT for keep exactly what they had. Nothing in
    /// this module changes what [`query_tile_for`] answers - the VAEs, the encoders and
    /// the speech paths read that, and none of them has been measured.
    #[test]
    fn the_unmeasured_callers_are_untouched() {
        for seq in [1usize, 800, 4096, 21_504, 32_768, 124_000] {
            let expect = (super::QUERY_TILE * super::TILE_SPAN / seq.max(1))
                .clamp(MIN_QUERY_TILE, super::QUERY_TILE);
            assert_eq!(query_tile_for(seq), expect, "seq {seq}");
        }
    }
}
