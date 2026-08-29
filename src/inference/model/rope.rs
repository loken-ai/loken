//! Rotary position embeddings, and the table a decode reads them from.
//!
//! Plain RoPE rotates position `p` by `p.θ_i`. YaRN keeps that for the high-frequency
//! dimensions, which a model has already seen through, and interpolates the low-frequency
//! ones, which it has not - so a checkpoint trained at 4k answers coherently past it. Both
//! come out of the same call: `YarnParams::None` is the plain form.

use crate::tensor::{DType, Device, Result, Tensor};

pub const MAX_SEQ_LEN: usize = 4096;

/// RoPE precompute table length. The cos/sin tables must span every position a
/// request can reach (`index_pos + seq_len`), independent of the KV-cache
/// fallback cap above. A request with `num_ctx` up to 8192 (prompt + generation)
/// needs positions beyond 4096, so the tables are sized larger; this is what
/// removes the `narrow: 4096+512 > dim 0` ceiling that capped long-context
/// generation on the generic transformer. Cost is trivial - the table is
/// `ROPE_TABLE_SEQ_LEN x head_dim/2 x f32` (≈4 MB at 16384x128), allocated once
/// per device at load. Sized to the common server `num_ctx` ceiling with margin.
pub const ROPE_TABLE_SEQ_LEN: usize = 16384;

/// YaRN rope scaling parameters (DeepSeek/HF style). When present the rope
/// inverse frequencies interpolate between extrapolation (original) for the
/// high-frequency dims and interpolation (÷factor) for the low-frequency dims,
/// with a linear ramp over `[low, high]` derived from `beta_fast`/`beta_slow`.
/// The attention mscale cancels (mscale == mscale_all_dim) -> cos/sin unscaled.
/// Shared by the Mistral3 path and the GenericHeteroTransformer path so both
/// produce identical rope tables (devstral-small-2: factor=48, orig_ctx=8192).
#[derive(Clone, Copy, Debug)]
pub struct YarnParams {
    pub factor: f32,
    pub orig_ctx: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// The angular frequency each pair of a head's dimensions turns at.
///
/// A rotary embedding rotates the pair at dimension `i` by `p . base^(-2i/head_dim)` when it
/// sits at position `p`: the first pair turns once per position and the last barely at all, so
/// the same vector doubles as a position code at every scale the model was trained on. There
/// are `head_dim / 2` of them because a rotation takes two dimensions.
///
/// Returned bare rather than as a tensor: the families differ in the shape they broadcast it
/// against - one axis for a sequence, three for an image's row and column - and in nothing else.
/// Written as a reciprocal of a positive power rather than as a negative one: the two agree
/// mathematically and not always in the last bit, and a table that shifts by an ULP shifts
/// every angle a model was trained against.
pub fn inverse_frequencies(head_dim: usize, base: f32) -> Vec<f32> {
    // Counted in PAIRS: pair `p` occupies dimensions `2p` and `2p+1`, and the exponent is the
    // first of the two over the head's width.
    (0..head_dim.div_ceil(2))
        .map(|pair| 1f32 / base.powf((2 * pair) as f32 / head_dim as f32))
        .collect()
}

/// The same frequencies, accumulated in double precision and narrowed at the end.
///
/// Which of the two a family takes is not a style choice: an image model's table is built once
/// and every patch position is rotated by it, so the families below were validated against the
/// double-precision one and answer a fraction differently under the other.
pub fn inverse_frequencies_f64(head_dim: usize, base: f64) -> Vec<f32> {
    (0..head_dim.div_ceil(2))
        .map(|pair| 1f32 / base.powf((2 * pair) as f64 / head_dim as f64) as f32)
        .collect()
}

/// The plain rope table, `(cos, sin)` shaped `[max_seq, rotary_dim/2]` on `device`.
///
/// Row `p`, column `i` is the cosine and the sine of `p . θ_i`: the angle the `i`-th pair of a
/// head's dimensions has turned through at position `p`. A family that rotates only part of a
/// head passes the ROTATED width, not the head width - the dimensions it leaves alone simply
/// have no column here.
///
/// Angles and their cosines are formed here, in host f32, one position at a time.
/// [`precomput_freqs_cis_yarn`] answers the same table mathematically but forms it as a device
/// matmul of the positions against the frequencies and takes the cosine and sine there, and
/// the two do not agree in the last bit. That is why both exist: a router picks its experts by
/// a top-k over logits, where an angle off by an ULP is enough to select a different expert, so
/// a family stays on the form it was validated against rather than on whichever is nearer.
pub fn precomput_freqs_cis_host(
    rotary_dim: usize,
    max_seq: usize,
    base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = inverse_frequencies(rotary_dim, base);
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for p in 0..max_seq {
        for (i, &f) in inv.iter().enumerate() {
            let a = p as f32 * f;
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_seq, half), device)?,
        Tensor::from_vec(sin, (max_seq, half), device)?,
    ))
}

/// `precomput_freqs_cis` with optional YaRN context-extension scaling. With
/// `yarn == None` this is bit-identical to the plain rope table.
pub fn precomput_freqs_cis_yarn(
    head_dim: usize,
    freq_base: f32,
    yarn: Option<YarnParams>,
    table_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Number of precomputed positions. MUST cover the KV window the model was
    // loaded with, else prefilling past it fails with `narrow: pos+n > table_len`.
    let table_seq_len = table_seq_len.max(1);
    let theta: Vec<f32> = match yarn {
        None => inverse_frequencies(head_dim, freq_base),
        Some(y) => {
            let half = head_dim / 2;
            // dim index where a given number of rotations completes over orig_ctx
            let find_dim = |num_rot: f32| -> f32 {
                (head_dim as f32 * (y.orig_ctx / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * freq_base.ln())
            };
            let low = find_dim(y.beta_fast).floor().max(0.0);
            let high = find_dim(y.beta_slow).ceil().min((half - 1) as f32);
            let denom = (high - low).max(1e-3);
            // Counted in pairs like the plain table above, because the ramp is indexed by pair
            // while the exponent is indexed by dimension.
            (0..head_dim.div_ceil(2))
                .map(|pair| {
                    let pos_freq = freq_base.powf((2 * pair) as f32 / head_dim as f32);
                    let extrap = 1.0 / pos_freq; // original (high-freq dims)
                    let interp = 1.0 / (y.factor * pos_freq); // ÷factor (low-freq dims)
                    let ramp = ((pair as f32 - low) / denom).clamp(0.0, 1.0);
                    // ramp=0 (high freq) -> extrap; ramp=1 (low freq) -> interp
                    interp * ramp + extrap * (1.0 - ramp)
                })
                .collect()
        }
    };
    // The table is the outer product of the positions with the frequencies: one row per
    // position, one column per pair, each entry the angle that pair has turned through there.
    let frequencies = Tensor::new(theta.as_slice(), device)?.reshape((1, theta.len()))?;
    let positions = Tensor::arange_u32(0, table_seq_len as u32)?
        .to_device(device)?
        .to_dtype(DType::F32)?
        .reshape((table_seq_len, 1))?;
    let angles = positions.matmul(&frequencies)?;
    Ok((angles.cos()?, angles.sin()?))
}
