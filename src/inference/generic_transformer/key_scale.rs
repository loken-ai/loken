//! The channel scale a quantised key store holds its keys under.
//!
//! A layer that norms its queries and keys folds a scale per key lane into the two norm weights.
//! The K norm weight is divided by the scale, the Q norm weight is multiplied by it.
//! Every score is the same: a lane of the key is divided by what the same lane of the query is
//! multiplied by.
//! The scale is equal on the 2 lanes of a rotation pair, so the rotation turns the scaled pair as
//! it turns the plain one.
//!
//! The scale is the magnitude of the K norm weight, so the scaled lanes of a key carry
//! comparable magnitudes.
//! A block of a quantised store then shares its step among lanes of one size, and a lane with a
//! large weight no longer sets the step of the lanes beside it.
//!
//! Every store of a scaled layer holds scaled keys: the quantised store, the full-width cache it
//! moves to, the host stores and the rows a state cache keeps.
//! The queries are scaled wherever they are computed, as the weight carries the scale.
//!
//! A layer on a card without a K norm folds a power of two per key lane into its projections.
//! The K projection's rows and the K bias are divided by it, the Q rows of the heads that read the
//! lane and the Q bias are multiplied by it.
//! A power of two changes only exponents: every product of a query lane with a key lane is the
//! same, bit for bit, and only the rounding of the stored keys and of the read queries changes.
//! The scale balances the 2: it is the root of the key lane's size over the query lane's size,
//! both taken from the weights.

use super::mixer::Mixer;
use super::GenericTransformerLayer;
use crate::tensor::layer::RmsNorm;
use crate::tensor::quant_cpu::{exponent::scale_rows_pow2, to_float_bytes};
use crate::tensor::quantized::{QStorage, QTensor};
use crate::tensor::{DType, Device, Result, Tensor};

/// The lane a rotation turns together with `lane`, `None` for a lane it leaves alone.
///
/// The first `rotated` lanes of a head rotate.
/// With `adjacent`, lane `2i` pairs with lane `2i + 1`; otherwise lane `i` pairs with lane
/// `i + rotated / 2`.
pub(super) fn rotation_partner(lane: usize, rotated: usize, adjacent: bool) -> Option<usize> {
    let half = rotated / 2;
    if lane >= 2 * half {
        return None;
    }
    Some(match (adjacent, lane < half) {
        (true, _) => lane ^ 1,
        (false, true) => lane + half,
        (false, false) => lane - half,
    })
}

/// The scale of each lane of one head, from the K norm weight of that head.
///
/// A lane's magnitude is the absolute value of its weight.
/// The 2 lanes of a rotation pair take the root mean square of their 2 magnitudes.
/// The scales are divided by their root mean square over the head, so the scaled key keeps the
/// length the plain key has on average.
/// A lane whose pair carries no weight keeps the scale 1: its key lane is zero.
pub(super) fn head_scale(weight: &[f32], rotated: usize, adjacent: bool) -> Vec<f64> {
    let magnitude: Vec<f64> = weight.iter().map(|w| f64::from(*w).abs()).collect();
    let paired: Vec<f64> = (0..magnitude.len())
        .map(|i| match rotation_partner(i, rotated, adjacent) {
            Some(j) => ((magnitude[i].powi(2) + magnitude[j].powi(2)) / 2.0).sqrt(),
            None => magnitude[i],
        })
        .collect();
    let rms = (paired.iter().map(|s| s * s).sum::<f64>() / paired.len().max(1) as f64).sqrt();
    paired
        .iter()
        .map(|&s| {
            if s > 0.0 && rms.is_finite() {
                s / rms
            } else {
                1.0
            }
        })
        .collect()
}

/// The Q and K norm weights with the scale folded in, `None` when the weights do not fit the
/// heads.
///
/// `query` holds one head of `head_dim` lanes shared by every query head, or one per query head.
/// `key` holds one head shared by every key head, or one per key head.
/// A query head reads the key head of its group, and takes that head's scale.
/// A key weight per head beside a shared query weight has no single scale per query lane.
pub(super) fn fold(
    query: &[f32],
    key: &[f32],
    shape: (usize, usize, usize),
    rotated: usize,
    adjacent: bool,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (n_head, n_kv_head, head_dim) = shape;
    if head_dim == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
        return None;
    }
    let key_heads = key.len() / head_dim;
    let query_heads = query.len() / head_dim;
    let shapes_fit = key.len().is_multiple_of(head_dim)
        && query.len().is_multiple_of(head_dim)
        && (key_heads == 1 || key_heads == n_kv_head)
        && (query_heads == 1 || query_heads == n_head)
        && !(key_heads > 1 && query_heads == 1);
    if !shapes_fit {
        return None;
    }
    let scales: Vec<Vec<f64>> = key
        .chunks(head_dim)
        .map(|w| head_scale(w, rotated, adjacent))
        .collect();
    let key_out = key
        .chunks(head_dim)
        .zip(&scales)
        .flat_map(|(w, s)| w.iter().zip(s).map(|(&w, &s)| (f64::from(w) / s) as f32))
        .collect();
    let group = n_head / n_kv_head;
    let query_out = query
        .chunks(head_dim)
        .enumerate()
        .flat_map(|(h, w)| {
            let s = &scales[if key_heads == 1 { 0 } else { h / group }];
            w.iter().zip(s).map(|(&w, &s)| (f64::from(w) * s) as f32)
        })
        .collect();
    Some((query_out, key_out))
}

/// The size of each output lane of the projection `w` read through the input norm weight `g`:
/// the root of the sum of the squares of the lane's row times `g`, and of the lane's bias.
/// `None` when the shapes do not fit.
fn lane_sizes(w: &QTensor, g: &[f32], bias: Option<&[f32]>) -> Result<Option<Vec<f64>>> {
    let dims = w.shape().dims();
    let (rows, cols) = match dims {
        [rows, cols] => (*rows, *cols),
        _ => return Ok(None),
    };
    let dtype = w.dtype();
    if g.len() != cols
        || bias.is_some_and(|b| b.len() != rows)
        || !cols.is_multiple_of(dtype.block_size())
    {
        return Ok(None);
    }
    let row_bytes = cols / dtype.block_size() * dtype.type_size();
    let bytes = w.data()?;
    let mut row = vec![0f32; cols];
    let mut sizes = Vec::with_capacity(rows);
    for r in 0..rows {
        to_float_bytes(dtype, &bytes[r * row_bytes..(r + 1) * row_bytes], &mut row)?;
        let along: f64 = row
            .iter()
            .zip(g)
            .map(|(&w, &g)| (f64::from(w) * f64::from(g)).powi(2))
            .sum();
        let offset = bias.map_or(0.0, |b| f64::from(b[r]).powi(2));
        sizes.push((along + offset).sqrt());
    }
    Ok(Some(sizes))
}

/// The size of each key lane and of the query lanes that read it, `n_kv_head` heads of
/// `head_dim` lanes, as a rotated pair of lanes holds them.
///
/// `key` holds the size of each key lane, `query` the size of each query lane, `n_head` heads.
/// A key lane's query size is the root mean square over the query heads that read its head.
/// The 2 lanes of a rotation pair take the root mean square of their sizes, on each side.
fn paired_sizes(
    key: &[f64],
    query: &[f64],
    shape: (usize, usize, usize),
    rotated: usize,
    adjacent: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (n_head, n_kv_head, head_dim) = shape;
    let group = n_head / n_kv_head;
    let paired = |v: &[f64]| -> Vec<f64> {
        (0..head_dim)
            .map(|i| match rotation_partner(i, rotated, adjacent) {
                Some(j) => ((v[i].powi(2) + v[j].powi(2)) / 2.0).sqrt(),
                None => v[i],
            })
            .collect()
    };
    let (mut k, mut q) = (Vec::new(), Vec::new());
    for kv in 0..n_kv_head {
        k.extend(paired(&key[kv * head_dim..(kv + 1) * head_dim]));
        let read: Vec<f64> = (0..head_dim)
            .map(|i| {
                let sum: f64 = (0..group)
                    .map(|g| query[(kv * group + g) * head_dim + i].powi(2))
                    .sum();
                (sum / group as f64).sqrt()
            })
            .collect();
        q.extend(paired(&read));
    }
    (k, q)
}

/// The balanced scale of each key lane: the root of its key size over its query size, divided by
/// its root mean square over the head.
/// A lane with no size on either side takes the scale 1 before the division.
fn balanced_scales(k: &[f64], q: &[f64], head_dim: usize) -> Vec<f64> {
    let mut scales = Vec::with_capacity(k.len());
    for (kh, qh) in k.chunks(head_dim).zip(q.chunks(head_dim)) {
        let s: Vec<f64> = kh
            .iter()
            .zip(qh)
            .map(|(&k, &q)| {
                if k > 0.0 && q > 0.0 {
                    (k / q).sqrt()
                } else {
                    1.0
                }
            })
            .collect();
        let rms = (s.iter().map(|v| v * v).sum::<f64>() / head_dim as f64).sqrt();
        scales.extend(s.iter().map(|&v| v / rms));
    }
    scales
}

/// The base-2 logarithm of each scale rounded to the nearest integer, 0 for a scale without one.
fn rounded_exponents(scales: &[f64]) -> Vec<i32> {
    scales
        .iter()
        .map(|&s| {
            let r = s.log2();
            if r.is_finite() {
                r.round() as i32
            } else {
                0
            }
        })
        .collect()
}

/// The expected squared rounding error of the scores of a layer whose keys are stored, and whose
/// queries are read, in blocks of `block` lanes that each round with a step of their largest
/// value.
/// `k` and `q` are the lane sizes of [`paired_sizes`], `scales` what each key lane is divided by
/// and its query lanes multiplied by.
fn rounding_noise(k: &[f64], q: &[f64], scales: &[f64], block: usize) -> f64 {
    let stored: Vec<f64> = k.iter().zip(scales).map(|(k, s)| k / s).collect();
    let read: Vec<f64> = q.iter().zip(scales).map(|(q, s)| q * s).collect();
    stored
        .chunks(block)
        .zip(read.chunks(block))
        .map(|(kb, qb)| {
            let (k_step, q_step) = (
                kb.iter().fold(0f64, |m, v| m.max(*v)),
                qb.iter().fold(0f64, |m, v| m.max(*v)),
            );
            kb.iter()
                .zip(qb)
                .map(|(k, q)| (q * k_step).powi(2) + (k * q_step).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// The exponent of the power of two each key lane is divided by, `n_kv_head` heads of
/// `head_dim` lanes: the balanced scale of [`balanced_scales`] rounded by [`rounded_exponents`].
#[cfg(test)]
fn balanced_exponents(
    key: &[f64],
    query: &[f64],
    shape: (usize, usize, usize),
    rotated: usize,
    adjacent: bool,
) -> Vec<i32> {
    let (k, q) = paired_sizes(key, query, shape, rotated, adjacent);
    rounded_exponents(&balanced_scales(&k, &q, shape.2))
}

/// Whether the rounded scales remove more of a layer's expected score rounding error than the
/// rounding of the scales to powers of two moves it.
fn fold_pays(k: &[f64], q: &[f64], exponents: &[i32], scales: &[f64], block: usize) -> bool {
    let rounded: Vec<f64> = exponents.iter().map(|&e| 2f64.powi(e)).collect();
    let plain = rounding_noise(k, q, &vec![1.0; k.len()], block);
    let folded = rounding_noise(k, q, &rounded, block);
    let exact = rounding_noise(k, q, scales, block);
    plain - folded > (folded - exact).abs()
}

/// `t` on the host with row `r` multiplied by `2^exponents[r]`, `None` when a row cannot take
/// its factor exactly.
fn scaled_rows(t: &QTensor, exponents: &[i32]) -> Result<Option<QTensor>> {
    let row_len = t.shape().elem_count() / exponents.len().max(1);
    let bytes = t.data()?;
    let Some(out) = scale_rows_pow2(t.dtype(), &bytes, row_len, exponents) else {
        return Ok(None);
    };
    Ok(Some(host_tensor(out, t)?))
}

/// The values of a host vector, as F32.
fn vector_values(t: &QTensor) -> Result<Vec<f32>> {
    // CONVERSION: reads a vector once at load to derive the key scale; the stored weight is kept.
    t.dequantize(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

/// The Q and K projections and biases of an attention layer with the balanced key scale folded in.
pub(super) struct ProjectionFold {
    pub q: QTensor,
    pub k: QTensor,
    pub q_bias: Option<QTensor>,
    pub k_bias: Option<QTensor>,
}

/// The lanes of a head of `head_dim` lanes an attention layer rotates, from its partial width.
/// 0 for a layer that rotates none.
pub(super) fn rotated_lanes(head_dim: usize, rope_dim: Option<usize>, no_rope: bool) -> usize {
    match (no_rope, rope_dim.filter(|&d| d > 0 && d < head_dim)) {
        (true, _) => 0,
        (false, Some(d)) => d,
        (false, None) => head_dim,
    }
}

/// Folds the balanced key scale into the host projections `q` and `k` of `n_head` query heads,
/// and into their biases.
///
/// `input_norm` is the norm the projections read their input through, `None` for none.
/// `rotated` lanes of each head rotate, in adjacent pairs with `adjacent`.
/// `None` when a weight is not on the host, when the shapes do not fit, when the quantised key
/// kernels do not serve the head shape, when no lane takes a factor, or when a weight cannot
/// take its factors exactly.
pub(super) fn fold_projections(
    q: &QTensor,
    k: &QTensor,
    biases: (Option<&QTensor>, Option<&QTensor>),
    input_norm: Option<&QTensor>,
    n_head: usize,
    rope: (Option<usize>, bool, bool),
) -> Result<Option<ProjectionFold>> {
    if !(q.device().is_cpu() && k.device().is_cpu()) {
        return Ok(None);
    }
    let (q_dims, k_dims) = (q.shape().dims(), k.shape().dims());
    let (q_rows, k_rows) = match (q_dims, k_dims) {
        ([qr, qc], [kr, kc]) if qc == kc => (*qr, *kr),
        _ => return Ok(None),
    };
    if n_head == 0 || !q_rows.is_multiple_of(n_head) {
        return Ok(None);
    }
    let head_dim = q_rows / n_head;
    if head_dim == 0 || !k_rows.is_multiple_of(head_dim) {
        return Ok(None);
    }
    let n_kv_head = k_rows / head_dim;
    if n_kv_head == 0
        || !n_head.is_multiple_of(n_kv_head)
        || !crate::inference::cache::kv_core::kv_shape_supported(head_dim, n_head / n_kv_head)
    {
        return Ok(None);
    }
    let cols = q_dims[1];
    let g = match input_norm {
        Some(n) => vector_values(n)?,
        None => vec![1.0; cols],
    };
    let (q_bias, k_bias) = biases;
    let q_b = q_bias.map(vector_values).transpose()?;
    let k_b = k_bias.map(vector_values).transpose()?;
    let (Some(key), Some(query)) = (
        lane_sizes(k, &g, k_b.as_deref())?,
        lane_sizes(q, &g, q_b.as_deref())?,
    ) else {
        return Ok(None);
    };
    let (rope_dim, no_rope, adjacent) = rope;
    let rotated = rotated_lanes(head_dim, rope_dim, no_rope);
    let (k_sizes, q_sizes) = paired_sizes(
        &key,
        &query,
        (n_head, n_kv_head, head_dim),
        rotated,
        adjacent,
    );
    let scales = balanced_scales(&k_sizes, &q_sizes, head_dim);
    let exponents = rounded_exponents(&scales);
    let block = crate::inference::cache::kv_core::KV_BLOCK;
    if exponents.iter().all(|&e| e == 0)
        || !fold_pays(&k_sizes, &q_sizes, &exponents, &scales, block)
    {
        return Ok(None);
    }
    let group = n_head / n_kv_head;
    let (q_bytes, k_bytes) = (q.data()?, k.data()?);
    let (q_row, k_row) = (q_bytes.len() / q_rows, k_bytes.len() / k_rows);
    let (mut q_out, mut k_out) = (Vec::with_capacity(q_bytes.len()), Vec::new());
    let mut lanes = Vec::with_capacity(exponents.len());
    // Every exponent of a head moved by one shift scores the same.
    // Each head takes the shift nearest 0 whose factors its rows take exactly.
    // Past the span of an f16's exponents and digits no shift keeps a nonzero field.
    let span = f16_span();
    let shifts = || std::iter::once(0).chain((1..=span).flat_map(|i| [-i, i]));
    for kv in 0..n_kv_head {
        let head = &exponents[kv * head_dim..(kv + 1) * head_dim];
        let k_rows_of = &k_bytes[kv * head_dim * k_row..(kv + 1) * head_dim * k_row];
        let q_rows_of =
            &q_bytes[kv * group * head_dim * q_row..(kv + 1) * group * head_dim * q_row];
        let found = shifts().find_map(|c| {
            let k_e: Vec<i32> = head.iter().map(|&e| -(e + c)).collect();
            let q_e: Vec<i32> = (0..group)
                .flat_map(|_| head.iter().map(move |&e| e + c))
                .collect();
            let k_scaled = scale_rows_pow2(k.dtype(), k_rows_of, cols, &k_e)?;
            let q_scaled = scale_rows_pow2(q.dtype(), q_rows_of, cols, &q_e)?;
            Some((c, k_scaled, q_scaled))
        });
        let Some((c, k_scaled, q_scaled)) = found else {
            return Ok(None);
        };
        k_out.extend(k_scaled);
        q_out.extend(q_scaled);
        lanes.extend(head.iter().map(|&e| e + c));
    }
    let k_exponents: Vec<i32> = lanes.iter().map(|&e| -e).collect();
    let q_exponents: Vec<i32> = (0..n_head)
        .flat_map(|h| {
            let kv = h / group;
            lanes[kv * head_dim..(kv + 1) * head_dim].iter().copied()
        })
        .collect();
    let scaled = |t: Option<&QTensor>, e: &[i32]| -> Result<Option<Option<QTensor>>> {
        match t {
            None => Ok(Some(None)),
            Some(t) => Ok(scaled_rows(t, e)?.map(Some)),
        }
    };
    let (Some(q_bias), Some(k_bias)) =
        (scaled(q_bias, &q_exponents)?, scaled(k_bias, &k_exponents)?)
    else {
        return Ok(None);
    };
    Ok(Some(ProjectionFold {
        q: host_tensor(q_out, q)?,
        k: host_tensor(k_out, k)?,
        q_bias,
        k_bias,
    }))
}

/// The number of powers of two an f16 spans, from its smallest subnormal to its largest value.
fn f16_span() -> i32 {
    half::f16::MAX_EXP - half::f16::MIN_EXP + half::f16::MANTISSA_DIGITS as i32
}

/// `bytes` as a host tensor of the type and shape of `like`.
fn host_tensor(bytes: Vec<u8>, like: &QTensor) -> Result<QTensor> {
    // CONVERSION: rebuilds a weight from its own blocks, only their scale fields multiplied by a power of two.
    let storage = QStorage::from_data(std::borrow::Cow::Owned(bytes), &Device::Cpu, like.dtype())?;
    QTensor::new(storage, like.shape().clone())
}

/// A norm with `values` in place of its weight, at the weight's shape, type and device.
fn renormed(norm: &RmsNorm, values: Vec<f32>) -> Result<RmsNorm> {
    let w = norm.weight();
    let weight = Tensor::from_vec(values, w.shape(), &Device::Cpu)?
        .to_dtype(w.dtype())?
        .to_device(&w.device())?;
    Ok(RmsNorm::from_tensor(weight, norm.eps()))
}

/// The values of a norm's weight, as F32 on the host.
fn host_values(norm: &RmsNorm) -> Result<Vec<f32>> {
    norm.weight()
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

impl GenericTransformerLayer {
    /// Whether the store this layer holds its keys in is quantised.
    /// On a card it is the quantised cache the layer holds.
    /// On the host it is the Q8 store the configured cache format asks for.
    fn holds_quantised_keys(&self) -> bool {
        if self.cos.device().is_cuda() {
            #[cfg(feature = "gpu")]
            return self.quant_kv.is_some();
            #[cfg(not(feature = "gpu"))]
            return false;
        }
        self.host_q8_kv
    }

    /// Folds the key scale into the Q and K norm weights of an attention layer that holds its
    /// keys in a quantised store, and records it in `key_scaled`.
    /// A layer whose keys another layer reads, or that reads another layer's keys, keeps its
    /// weights: `shares_keys` says so.
    /// A layer without both norms keeps its weights.
    pub(super) fn fold_key_scale(&mut self, shares_keys: bool) -> Result<()> {
        if shares_keys
            || !matches!(self.mixer, Mixer::Attention)
            || self.populate_dual_kv
            || !self.holds_quantised_keys()
        {
            return Ok(());
        }
        let (Some(query), Some(key)) = (self.attn_q_norm.as_ref(), self.attn_k_norm.as_ref())
        else {
            return Ok(());
        };
        let rotated = match (self.no_rope, self.rope_dim) {
            (true, _) => 0,
            (false, 0) => self.head_dim,
            (false, d) => d,
        };
        let Some((q, k)) = fold(
            &host_values(query)?,
            &host_values(key)?,
            (self.n_head, self.n_kv_head, self.head_dim),
            rotated,
            self.flags.use_rope_i,
        ) else {
            return Ok(());
        };
        let (query, key) = (renormed(query, q)?, renormed(key, k)?);
        self.attn_q_norm = Some(query);
        self.attn_k_norm = Some(key);
        self.key_scaled = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    /// A deterministic spread of values in [-1, 1).
    fn values(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u64 + 1)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(40503));
                ((h >> 8) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// `x` rotated by `angle` times the pair's rank, pairs as `rotation_partner` names them.
    fn rotate(x: &[f64], angle: f64, rotated: usize, adjacent: bool) -> Vec<f64> {
        let mut out = x.to_vec();
        for i in 0..x.len() {
            let Some(j) = rotation_partner(i, rotated, adjacent) else {
                continue;
            };
            // The first lane of a pair takes cos and -sin, the second sin and cos.
            let first = if adjacent { i % 2 == 0 } else { i < j };
            let rank = if adjacent { i / 2 } else { i.min(j) };
            let (c, s) = (
                (angle * (rank + 1) as f64).cos(),
                (angle * (rank + 1) as f64).sin(),
            );
            out[i] = if first {
                x[i] * c - x[j] * s
            } else {
                x[j] * s + x[i] * c
            };
        }
        out
    }

    /// The Q8_0 rounding of `x` in blocks of 32: each block shares the step of its largest
    /// magnitude over 127, held as F16.
    fn q8_0(x: &[f64]) -> Vec<f64> {
        x.chunks(32)
            .flat_map(|b| {
                let max = b.iter().fold(0f64, |m, v| m.max(v.abs()));
                let d = f64::from(half::f16::from_f64(max / 127.0).to_f32());
                b.iter()
                    .map(move |v| if d > 0.0 { (v / d).round() * d } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn a_rotation_pairs_lanes_by_its_layout() {
        assert_eq!(rotation_partner(0, 8, false), Some(4));
        assert_eq!(rotation_partner(5, 8, false), Some(1));
        assert_eq!(rotation_partner(0, 8, true), Some(1));
        assert_eq!(rotation_partner(5, 8, true), Some(4));
        assert_eq!(rotation_partner(8, 8, false), None);
        assert_eq!(rotation_partner(3, 0, false), None);
    }

    #[test]
    fn a_pair_shares_one_scale_and_the_scales_keep_unit_mean_square() {
        for (rotated, adjacent) in [(64, false), (64, true), (32, false), (0, false)] {
            let w = values(64, 7);
            let s = head_scale(&w, rotated, adjacent);
            for i in 0..64 {
                if let Some(j) = rotation_partner(i, rotated, adjacent) {
                    assert_eq!(s[i], s[j], "lanes {i} and {j}, rotated {rotated}");
                }
            }
            let ms = s.iter().map(|v| v * v).sum::<f64>() / 64.0;
            assert!((ms - 1.0).abs() < 1e-12, "mean square {ms}");
        }
        assert_eq!(head_scale(&[0.0; 8], 8, false), vec![1.0; 8]);
    }

    /// Queries and keys normed, rotated and scored with the folded weights score as with the
    /// plain ones, for each rotation layout and for a weight per head.
    #[test]
    fn the_folded_weights_score_as_the_plain_ones() {
        let (n_head, n_kv, hd) = (4usize, 2usize, 64usize);
        for (rotated, adjacent, per_head) in [
            (64, false, false),
            (64, true, false),
            (32, false, false),
            (0, false, false),
            (64, false, true),
        ] {
            let heads = |n: usize| if per_head { n } else { 1 };
            let q_w = values(heads(n_head) * hd, 1);
            let k_w: Vec<f32> = values(heads(n_kv) * hd, 2)
                .iter()
                .enumerate()
                .map(|(i, v)| if i % hd == 3 { 40.0 } else { *v })
                .collect();
            let (q_f, k_f) = fold(&q_w, &k_w, (n_head, n_kv, hd), rotated, adjacent).unwrap();
            let (xq, xk) = (values(n_head * hd, 3), values(n_kv * hd, 4));
            for h in 0..n_head {
                let kv = h / (n_head / n_kv);
                let lanes = |w: &[f32], x: &[f32], head: usize, scaled_head: usize| -> Vec<f64> {
                    let w = &w[if w.len() == hd { 0 } else { scaled_head * hd }..][..hd];
                    (0..hd)
                        .map(|i| f64::from(w[i]) * f64::from(x[head * hd + i]))
                        .collect()
                };
                let rotated_pair = |qw: &[f32], kw: &[f32]| {
                    let q = rotate(&lanes(qw, &xq, h, h), 0.3, rotated, adjacent);
                    let k = rotate(&lanes(kw, &xk, kv, kv), 0.7, rotated, adjacent);
                    (q, k)
                };
                let (q, k) = rotated_pair(&q_w, &k_w);
                let (qf, kf) = rotated_pair(&q_f, &k_f);
                let (plain, folded) = (dot(&q, &k), dot(&qf, &kf));
                // Each folded weight is one F32 rounding off its exact value.
                let bound = 4.0 * f64::from(f32::EPSILON) * dot(&q, &q).sqrt() * dot(&k, &k).sqrt();
                assert!(
                    (plain - folded).abs() <= bound,
                    "head {h}, rotated {rotated}, adjacent {adjacent}: {plain} against {folded}"
                );
            }
        }
    }

    /// A quantised host weight of `rows` rows of `cols` values.
    fn host_weight(v: Vec<f32>, rows: usize, cols: usize, dtype: GgmlDType) -> QTensor {
        let t = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap();
        QTensor::quantize_onto(&t, dtype, &Device::Cpu).unwrap()
    }

    fn host_rows(t: &QTensor) -> Vec<f32> {
        t.dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Lanes of one size on both sides take no factor.
    /// A key pair 16 times the size of the other lanes takes a factor below 1, equal on its 2
    /// lanes, and the lanes of each head keep a root mean square scale near 1.
    #[test]
    fn balanced_exponents_pair_lanes_and_divide_the_large_ones() {
        let (n_head, n_kv, hd) = (4usize, 2usize, 64usize);
        let query = vec![1.0f64; n_head * hd];
        assert!(
            balanced_exponents(&vec![1.0; n_kv * hd], &query, (n_head, n_kv, hd), hd, false)
                .iter()
                .all(|&e| e == 0)
        );
        let mut key = vec![1.0f64; n_kv * hd];
        key[5] = 256.0;
        key[5 + hd / 2] = 256.0;
        let e = balanced_exponents(&key, &query, (n_head, n_kv, hd), hd, false);
        assert_eq!(e[5], e[5 + hd / 2]);
        assert!(e[5] > e[6], "{} against {}", e[5], e[6]);
        assert!(
            e[hd..].iter().all(|&x| x == 0),
            "the second head is uniform"
        );
    }

    /// A key pair 16 times the size of the other lanes pays for its fold: the rounded scales
    /// remove more expected rounding error than their rounding to powers of two moves.
    /// Scales that round to the scales they came from pay for any gain.
    #[test]
    fn an_outlier_pair_pays_for_its_fold() {
        let hd = 64usize;
        let q = vec![1.0f64; hd];
        let mut k = vec![1.0f64; hd];
        k[5] = 256.0;
        k[5 + hd / 2] = 256.0;
        let scales = balanced_scales(&k, &q, hd);
        let exponents = rounded_exponents(&scales);
        assert_ne!(exponents[5], 0);
        assert!(fold_pays(&k, &q, &exponents, &scales, 32));
        let plain = vec![0; hd];
        assert!(!fold_pays(&k, &q, &plain, &vec![1.0; hd], 32));
    }

    /// The fold multiplies each K row and bias by the power of two of its lane, and each Q row by
    /// the inverse power of the lane its head reads, bit for bit.
    /// The 2 lanes of a rotation pair take one power.
    #[test]
    fn the_projection_fold_scales_rows_by_the_powers_of_their_lanes() {
        let (n_head, n_kv, hd, cols) = (4usize, 2usize, 64usize, 64usize);
        let group = n_head / n_kv;
        let mut kw = values(n_kv * hd * cols, 3);
        for kv in 0..n_kv {
            for lane in [5, 5 + hd / 2] {
                for c in 0..cols {
                    kw[(kv * hd + lane) * cols + c] *= 16.0;
                }
            }
        }
        let q = host_weight(
            values(n_head * hd * cols, 4),
            n_head * hd,
            cols,
            GgmlDType::Q8_0,
        );
        let k = host_weight(kw, n_kv * hd, cols, GgmlDType::Q8_0);
        let kb = host_weight(values(n_kv * hd, 5), 1, n_kv * hd, GgmlDType::F32);
        let fold = fold_projections(
            &q,
            &k,
            (None, Some(&kb)),
            None,
            n_head,
            (None, false, false),
        )
        .unwrap()
        .expect("an outlier pair takes a factor");
        let (q0, k0, kb0) = (host_rows(&q), host_rows(&k), host_rows(&kb));
        let (q1, k1) = (host_rows(&fold.q), host_rows(&fold.k));
        let kb1 = host_rows(fold.k_bias.as_ref().unwrap());
        // The factor of each key lane, from the first nonzero value of its row.
        let factor: Vec<f32> = (0..n_kv * hd)
            .map(|r| {
                let c = (0..cols).find(|&c| k0[r * cols + c] != 0.0).unwrap();
                k1[r * cols + c] / k0[r * cols + c]
            })
            .collect();
        for r in 0..n_kv * hd {
            let f = factor[r];
            assert_eq!(
                f.to_bits() & 0x007f_ffff,
                0,
                "row {r}: {f} is a power of two"
            );
            let partner = rotation_partner(r % hd, hd, false).unwrap() + r / hd * hd;
            assert_eq!(f, factor[partner], "rows {r} and {partner}");
            for c in 0..cols {
                assert_eq!(k1[r * cols + c].to_bits(), (k0[r * cols + c] * f).to_bits());
            }
            assert_eq!(kb1[r].to_bits(), (kb0[r] * f).to_bits(), "bias {r}");
        }
        assert!(factor[5] < factor[6], "the large pair is divided the most");
        for h in 0..n_head {
            for lane in 0..hd {
                let f = factor[(h / group) * hd + lane];
                let r = h * hd + lane;
                for c in 0..cols {
                    assert_eq!(
                        (q1[r * cols + c] * f).to_bits(),
                        q0[r * cols + c].to_bits(),
                        "head {h} lane {lane}"
                    );
                }
            }
        }
        assert!(fold.q_bias.is_none());
    }

    /// A key rounded to Q8_0 scores as close under the fold as plain when its lanes carry one
    /// size, and closer than plain when one rotation pair's weight sets the step of its blocks.
    /// The error is the summed distance of the rounded scores to the exact ones.
    #[test]
    fn a_scaled_key_rounds_closer_when_one_pair_is_an_outlier() {
        let hd = 64usize;
        let error = |k_w: &[f32]| -> (f64, f64) {
            let q_w = vec![1.0f32; hd];
            let (q_f, k_f) = fold(&q_w, k_w, (1, 1, hd), hd, false).unwrap();
            let (mut plain, mut scaled) = (0f64, 0f64);
            for seed in 0..64u64 {
                let (xq, xk) = (values(hd, 100 + seed), values(hd, 200 + seed));
                let lanes = |w: &[f32], x: &[f32]| -> Vec<f64> {
                    (0..hd).map(|i| f64::from(w[i]) * f64::from(x[i])).collect()
                };
                let (q, k) = (lanes(&q_w, &xq), lanes(k_w, &xk));
                let (qs, ks) = (lanes(&q_f, &xq), lanes(&k_f, &xk));
                let exact = dot(&q, &k);
                plain += (dot(&q, &q8_0(&k)) - exact).abs();
                scaled += (dot(&qs, &q8_0(&ks)) - exact).abs();
            }
            (plain, scaled)
        };
        let uniform = vec![0.5f32; hd];
        let (plain, scaled) = error(&uniform);
        assert!(
            (plain - scaled).abs() <= 1e-6 * plain.max(1e-30),
            "uniform lanes: {plain} plain, {scaled} scaled"
        );
        let mut outlier = uniform.clone();
        outlier[3] = 40.0;
        outlier[3 + hd / 2] = 40.0;
        let (plain, scaled) = error(&outlier);
        eprintln!("an outlier pair: {plain} plain, {scaled} scaled");
        assert!(
            scaled < plain,
            "an outlier pair: {plain} plain, {scaled} scaled"
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod card_tests {
    use super::super::{tiny_model, GenericHeteroTransformer};
    use crate::inference::engine::llm_engine::KvQuant;
    use crate::tensor::{DType, Device, Tensor};

    const LAYERS: usize = 2;
    /// Tokens before the rows the paths compare.
    const PROMPT: usize = 24;
    /// Rows each path computes after the prompt.
    const ROWS: usize = 4;

    fn ids(toks: &[u32]) -> Tensor {
        Tensor::new(toks, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    fn rows(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn relative_l2(exact: &[f32], got: &[f32]) -> f64 {
        let (mut dd, mut ww) = (0f64, 0f64);
        for (&w, &g) in exact.iter().zip(got) {
            let (w, g) = (f64::from(w), f64::from(g));
            dd += (w - g) * (w - g);
            ww += w * w;
        }
        (dd / ww).sqrt()
    }

    /// The logits of the `ROWS` tokens after the prompt, by 3 paths: rows of 1 prefill of every
    /// token, decode steps after a prefill of the prompt, and 1 verify of the rows after it.
    fn paths(m: &mut GenericHeteroTransformer) -> [Vec<f32>; 3] {
        let vocab = tiny_model::QUANT_VOCAB as u32;
        let toks: Vec<u32> = (0..PROMPT + ROWS)
            .map(|i| ((i * 7 + i / 3) as u32) % vocab)
            .collect();
        let (prompt, next) = toks.split_at(PROMPT);
        let all = rows(&m.forward_all(&ids(&toks), 0).unwrap());
        let width = all.len() / toks.len();
        let prefill = all[(PROMPT - 1) * width..(PROMPT + ROWS - 1) * width].to_vec();
        // The prompt's last row leads the decode steps and the verify rows.
        let mut decode = rows(&m.forward(&ids(prompt), 0).unwrap());
        for (i, &t) in next[..ROWS - 1].iter().enumerate() {
            decode.extend(rows(&m.forward(&ids(&[t]), PROMPT + i).unwrap()));
        }
        let mut verify = rows(&m.forward(&ids(prompt), 0).unwrap());
        verify.extend(rows(
            &m.forward_all(&ids(&next[..ROWS - 1]), PROMPT).unwrap(),
        ));
        [prefill, decode, verify]
    }

    /// A model whose K norm weight has one large rotation pair, on a card: the layers of a load
    /// with a quantised cache store scaled keys, and its prefill rows, decode steps and verify
    /// rows are no farther from the full-width cache than those of the same cache with plain
    /// keys.
    /// A load with a full-width cache keeps the file's norm weights.
    #[test]
    fn scaled_keys_read_closer_to_the_full_width_cache_on_every_path() {
        if Device::new_cuda(0).is_err() {
            eprintln!("skipping: no usable card");
            return;
        }
        let head_dim = tiny_model::QUANT_EMBD / tiny_model::QUANT_HEADS;
        let mut key_norm = vec![1.0f32; head_dim];
        key_norm[3] = 24.0;
        key_norm[3 + head_dim / 2] = 24.0;
        let path = tiny_model::write_quant_with(
            "key-scale-card",
            LAYERS,
            tiny_model::QuantExtras {
                key_norm: Some(key_norm),
                ..Default::default()
            },
        );
        let load = |kv| tiny_model::load_on_cards_with_kv(&path, LAYERS, &[(0, 0)], kv);
        let mut full = load(KvQuant::Off);
        let mut scaled = load(KvQuant::Q8);
        let mut plain = load(KvQuant::Q8);
        assert!(full.layers.iter().all(|l| !l.key_scaled));
        assert!(scaled.layers.iter().all(|l| l.key_scaled));
        for (p, f) in plain.layers.iter_mut().zip(&full.layers) {
            p.attn_q_norm = f.attn_q_norm.clone();
            p.attn_k_norm = f.attn_k_norm.clone();
            p.key_scaled = false;
        }
        let want = paths(&mut full);
        let (got, base) = (paths(&mut scaled), paths(&mut plain));
        for (i, name) in ["prefill rows", "decode steps", "verify rows"]
            .iter()
            .enumerate()
        {
            let (d_scaled, d_plain) = (
                relative_l2(&want[i], &got[i]),
                relative_l2(&want[i], &base[i]),
            );
            eprintln!("{name}: {d_scaled} scaled, {d_plain} plain, off the full-width cache");
            assert!(
                d_scaled < d_plain,
                "{name}: scaled keys {d_scaled} off the full-width cache, plain keys {d_plain}"
            );
        }
        tiny_model::remove(&path);
    }

    /// A model without K norms whose K bias has one large rotation pair, on a card: a load with a
    /// quantised cache folds the key scale into its projections, and its prefill rows, decode
    /// steps and verify rows are no farther from the full-width cache than those of the same
    /// cache with the file's projections.
    /// A load with a full-width cache, and a load on the host, keep the file's projections.
    #[test]
    fn folded_projections_read_closer_to_the_full_width_cache_on_every_path() {
        if Device::new_cuda(0).is_err() {
            eprintln!("skipping: no usable card");
            return;
        }
        let head_dim = tiny_model::QUANT_EMBD / tiny_model::QUANT_HEADS;
        let mut key_bias = vec![0.05f32; head_dim];
        key_bias[3] = 24.0;
        key_bias[3 + head_dim / 2] = -24.0;
        let path = tiny_model::write_quant_with(
            "key-fold-card",
            LAYERS,
            tiny_model::QuantExtras {
                key_bias: Some(key_bias),
                ..Default::default()
            },
        );
        let host = tiny_model::load_q8_kv(&path, LAYERS);
        assert!(host.layers.iter().all(|l| !l.key_scaled));
        let load = |kv| tiny_model::load_on_cards_with_kv(&path, LAYERS, &[(0, 0)], kv);
        let mut full = load(KvQuant::Off);
        let mut scaled = load(KvQuant::Q8);
        let mut plain = load(KvQuant::Q8);
        assert!(full.layers.iter().all(|l| !l.key_scaled));
        assert!(scaled.layers.iter().all(|l| l.key_scaled));
        for (p, f) in plain.layers.iter_mut().zip(&full.layers) {
            p.attn_q = f.attn_q.clone();
            p.attn_k = f.attn_k.clone();
            p.attn_qkv = f.attn_qkv.clone();
            p.attn_q_bias = f.attn_q_bias.clone();
            p.attn_k_bias = f.attn_k_bias.clone();
            p.attn_qkv_bias = f.attn_qkv_bias.clone();
            p.key_scaled = false;
        }
        let want = paths(&mut full);
        let (got, base) = (paths(&mut scaled), paths(&mut plain));
        for (i, name) in ["prefill rows", "decode steps", "verify rows"]
            .iter()
            .enumerate()
        {
            let (d_scaled, d_plain) = (
                relative_l2(&want[i], &got[i]),
                relative_l2(&want[i], &base[i]),
            );
            eprintln!("{name}: {d_scaled} folded, {d_plain} plain, off the full-width cache");
            assert!(
                d_scaled < d_plain,
                "{name}: folded projections {d_scaled} off the full-width cache, plain {d_plain}"
            );
        }
        tiny_model::remove(&path);
    }
}
