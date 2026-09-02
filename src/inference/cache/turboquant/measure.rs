//! The measurement that decides whether the kernel is worth writing.
//!
//! Everything else in this module is machinery; this file is the judge. It takes real K
//! and V from a model that actually ran and reports, per layer, how far each candidate
//! storage lands from the values a full-width cache would have held - the rotated 3-bit
//! scheme, the same scheme without the rotation, and the Q4_0 cache the tree already
//! ships, each labelled with the bits per value it costs.
//!
//! Two things the table has to keep straight or it decides nothing:
//!
//! * **Equal data.** Every column is measured over the same tokens, heads and channels.
//!   The Q4_0 K layout groups along the token axis in windows of 32, so the token count is
//!   truncated to a multiple of 32 for all columns rather than for that one.
//! * **Equal budget.** Three bits plus a per-group f16 scale is not three bits. A column
//!   that costs 7 bits per value beating one that costs 4.5 is not a result, and the bit
//!   rate is carried next to the error so that it cannot be read as one.

use super::hadamard::HadamardRotation;
use super::quant::{dequantise_run, quantise_run, Scheme};
use crate::tensor::quant_cpu::{from_float_bytes, to_float_bytes};
use crate::tensor::quantized::GgmlDType;
use half::f16;

/// Block length of the Q4_0 comparison column, which is the format's own.
const Q4_BLOCK: usize = 32;

/// Seed for the measurement's rotation. Fixed so a re-run reproduces the table; a cache
/// would derive its own per-model seed the same way and store it.
pub const MEASUREMENT_ROTATION_SEED: u64 = 0x7175_616E_7431;

/// Relative reconstruction error: `||x - x_hat|| / ||x||`, accumulated in f64 so that the
/// figure is a property of the quantiser and not of the accumulator.
pub fn relative_error(reference: &[f32], reconstructed: &[f32]) -> f64 {
    assert_eq!(reference.len(), reconstructed.len(), "length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in reference.iter().zip(reconstructed.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        num += (a - b) * (a - b);
        den += a * a;
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// One candidate storage, with what it cost and what it lost.
#[derive(Clone, Debug)]
pub struct Column {
    /// Short name, used as the table heading.
    pub label: &'static str,
    /// Stored bits per value, side information amortised in.
    pub bits_per_value: f64,
    /// Relative reconstruction error against the original f32 values.
    pub relative_error: f64,
}

/// Every column, for one layer's K and V.
#[derive(Clone, Debug)]
pub struct LayerReport {
    pub layer: usize,
    pub tokens: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Values per scale for the 3-bit columns.
    pub values_per_group: usize,
    pub k: Vec<Column>,
    pub v: Vec<Column>,
}

/// Round a run through f16 - the incumbent cache's own storage, and therefore the floor
/// any quantised column is trying to approach.
fn through_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|x| f16::from_f32(*x).to_f32()).collect()
}

/// Round a run through Q4_0 blocks of 32 consecutive values.
fn through_q4_0(values: &[f32]) -> Vec<f32> {
    assert_eq!(
        values.len() % Q4_BLOCK,
        0,
        "Q4_0 needs whole blocks of {Q4_BLOCK}"
    );
    let bytes = from_float_bytes(GgmlDType::Q4_0, values).expect("q4_0 quantise");
    let mut out = vec![0.0f32; values.len()];
    to_float_bytes(GgmlDType::Q4_0, &bytes, &mut out).expect("q4_0 dequantise");
    out
}

/// Apply the rotation to every `head_dim`-long row of a `[.., head_dim]` run.
fn rotate_rows(values: &[f32], head_dim: usize, rotation: &HadamardRotation) -> Vec<f32> {
    let mut out = values.to_vec();
    for row in out.chunks_exact_mut(head_dim) {
        rotation.apply(row);
    }
    out
}

/// Undo the rotation on every row, so the error is measured in the space the model reads.
fn unrotate_rows(values: &[f32], head_dim: usize, rotation: &HadamardRotation) -> Vec<f32> {
    let mut out = values.to_vec();
    for row in out.chunks_exact_mut(head_dim) {
        rotation.apply_inverse(row);
    }
    out
}

/// Quantise and reconstruct through the rotated space, returning values in the original
/// basis.
///
/// The rotation is orthogonal, so the error norm is the same measured either side of it;
/// undoing it anyway keeps the reported number comparable with the columns that never
/// rotated, and makes the round trip a real round trip rather than a claim about one.
fn through_rotated_3bit(
    values: &[f32],
    head_dim: usize,
    group: usize,
    scheme: Scheme,
    rotation: &HadamardRotation,
) -> (Vec<f32>, f64) {
    let rotated = rotate_rows(values, head_dim, rotation);
    let q = quantise_run(&rotated, group, scheme);
    let bits = q.bits_per_value();
    let back = dequantise_run(&q);
    (unrotate_rows(&back, head_dim, rotation), bits)
}

/// Quantise and reconstruct in the original basis.
fn through_plain_3bit(values: &[f32], group: usize, scheme: Scheme) -> (Vec<f32>, f64) {
    let q = quantise_run(values, group, scheme);
    let bits = q.bits_per_value();
    (dequantise_run(&q), bits)
}

/// Transpose `[n_kv_heads, tokens, head_dim]` into `[n_kv_heads, head_dim, tokens]`.
///
/// The Q4_0 K column exists to reproduce the cache already in the tree, and that cache is
/// per-channel: one 32-value block spans 32 consecutive token positions of a single
/// (head, channel). Measuring it in the per-token layout instead would compare the 3-bit
/// scheme against a strawman rather than against the incumbent.
fn to_channel_major(values: &[f32], n_kv_heads: usize, tokens: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for h in 0..n_kv_heads {
        for t in 0..tokens {
            for c in 0..head_dim {
                out[h * head_dim * tokens + c * tokens + t] =
                    values[h * tokens * head_dim + t * head_dim + c];
            }
        }
    }
    out
}

/// The inverse of [`to_channel_major`].
fn to_token_major(values: &[f32], n_kv_heads: usize, tokens: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for h in 0..n_kv_heads {
        for c in 0..head_dim {
            for t in 0..tokens {
                out[h * tokens * head_dim + t * head_dim + c] =
                    values[h * head_dim * tokens + c * tokens + t];
            }
        }
    }
    out
}

/// Bits per value of the Q4_0 column: four bits of code plus one f16 scale per 32 values.
fn q4_0_bits_per_value() -> f64 {
    4.0 + 16.0 / Q4_BLOCK as f64
}

/// What the rotation does to a real tensor, separated from what the quantiser does to it.
///
/// Two questions, and neither is answerable from the reconstruction error alone. If
/// `apply_inverse` does not exactly undo `apply`, every reported error silently contains a
/// rotation error that has nothing to do with three bits. And if the rotation does not
/// actually flatten the rows, then any improvement attributed to it came from somewhere
/// else and the explanation on the label is wrong.
#[derive(Clone, Debug)]
pub struct RotationDiagnostics {
    /// Rows examined.
    pub rows: usize,
    /// Largest round-trip error over all rows, as a fraction of the row's RMS. Seven
    /// butterfly stages of f32 adds and one scaling put this near 1e-6, not at machine
    /// epsilon; what matters is that it is orders of magnitude below the quantisation
    /// error it would otherwise contaminate.
    pub max_round_trip_error: f64,
    /// Largest relative change in a row's L2 norm across the rotation. Orthogonality says
    /// zero.
    pub max_relative_norm_change: f64,
    /// Mean excess kurtosis per row before and after. A normal sits at zero; a row with a
    /// few dominant channels sits far above it. Statement 1 predicts a large drop.
    pub kurtosis_before: f64,
    pub kurtosis_after: f64,
    /// Mean crest factor per row (largest magnitude over RMS) before and after. This is
    /// the quantity that actually sets a shared group scale, so it is the one the design
    /// needs to move.
    pub crest_before: f64,
    pub crest_after: f64,
}

/// Excess kurtosis and crest factor of one row.
fn row_shape(row: &[f32]) -> (f64, f64) {
    let n = row.len() as f64;
    let mean: f64 = row.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    let mut peak = 0.0f64;
    let mut energy = 0.0f64;
    for &x in row {
        let d = x as f64 - mean;
        m2 += d * d;
        m4 += d * d * d * d;
        peak = peak.max((x as f64).abs());
        energy += (x as f64) * (x as f64);
    }
    m2 /= n;
    m4 /= n;
    let kurtosis = if m2 > 0.0 { m4 / (m2 * m2) - 3.0 } else { 0.0 };
    let rms = (energy / n).sqrt();
    let crest = if rms > 0.0 { peak / rms } else { 0.0 };
    (kurtosis, crest)
}

/// Measure the rotation on its own, over every `head_dim`-long row of `values`.
pub fn rotation_diagnostics(
    values: &[f32],
    head_dim: usize,
    rotation: &HadamardRotation,
) -> RotationDiagnostics {
    let mut d = RotationDiagnostics {
        rows: 0,
        max_round_trip_error: 0.0,
        max_relative_norm_change: 0.0,
        kurtosis_before: 0.0,
        kurtosis_after: 0.0,
        crest_before: 0.0,
        crest_after: 0.0,
    };
    for row in values.chunks_exact(head_dim) {
        let (k_before, c_before) = row_shape(row);
        let norm_before: f64 = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();

        let mut work = row.to_vec();
        rotation.apply(&mut work);

        let (k_after, c_after) = row_shape(&work);
        let norm_after: f64 = work.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();

        rotation.apply_inverse(&mut work);
        let rms = (norm_before / head_dim as f64).sqrt();
        if rms > 0.0 {
            let err = row
                .iter()
                .zip(work.iter())
                .map(|(a, b)| ((*a - *b) as f64).abs())
                .fold(0.0f64, f64::max);
            d.max_round_trip_error = d.max_round_trip_error.max(err / rms);
        }
        if norm_before > 0.0 {
            d.max_relative_norm_change = d
                .max_relative_norm_change
                .max((norm_after - norm_before).abs() / norm_before);
        }

        d.kurtosis_before += k_before;
        d.kurtosis_after += k_after;
        d.crest_before += c_before;
        d.crest_after += c_after;
        d.rows += 1;
    }
    if d.rows > 0 {
        let n = d.rows as f64;
        d.kurtosis_before /= n;
        d.kurtosis_after /= n;
        d.crest_before /= n;
        d.crest_after /= n;
    }
    d
}

/// Quantise K in the per-channel layout the incumbent cache uses, at three bits.
///
/// This is the variant the first round of measurement pointed at. Q4_0's advantage there
/// was never its bit width - it was its *grouping*: a block spanning 32 consecutive token
/// positions of a single (head, channel) never mixes a large channel with a small one, so
/// the outlier problem is gone before quantisation starts and all sixteen levels are spent
/// on resolution. Grouping along `head_dim` instead, as the rotated scheme did, spends its
/// levels covering a spread the rotation had to create in the first place.
///
/// So: keep the grouping that works, and ask whether the rotation still adds anything on
/// top of it. `rotate` selects that - with it, each token's key is rotated along `head_dim`
/// before the transpose, so the stored channels are rotated channels and the `q . k`
/// identity still holds; without it, this is plain 3-bit KIVI.
fn through_kivi_3bit(
    values: &[f32],
    n_kv_heads: usize,
    tokens: usize,
    head_dim: usize,
    group: usize,
    rotation: Option<&HadamardRotation>,
) -> (Vec<f32>, f64) {
    let src = match rotation {
        Some(r) => rotate_rows(values, head_dim, r),
        None => values.to_vec(),
    };
    let channel_major = to_channel_major(&src, n_kv_heads, tokens, head_dim);
    // Absmax, not the RMS scale the Lloyd-Max levels are matched to. That is a measured
    // choice, not a default: on real keys the absmax scale beat the matched one at every
    // group size tried (K, group 8: 0.133 against 0.172; group 32: 0.164 against 0.174).
    // The rotated coordinates are not Gaussian enough for a scale that trades saturation
    // against resolution to come out ahead, and a scheme grouped along the token axis is
    // further from Gaussian still.
    let q = quantise_run(&channel_major, group, Scheme::SymmetricLloydAbsMax);
    let bits = q.bits_per_value();
    let back = to_token_major(&dequantise_run(&q), n_kv_heads, tokens, head_dim);
    let back = match rotation {
        Some(r) => unrotate_rows(&back, head_dim, r),
        None => back,
    };
    (back, bits)
}

/// Build the full report for one layer.
///
/// `k` and `v` are `[n_kv_heads, tokens, head_dim]` in row-major order. Both `tokens` and
/// `head_dim` must be multiples of 32 and of `values_per_group` - the token axis because
/// the per-channel columns group along it, the channel axis because the per-token columns
/// do. A caller that cannot satisfy that should truncate before calling rather than have
/// the harness silently measure a different slice for different columns.
pub fn evaluate_layer(
    layer: usize,
    k: &[f32],
    v: &[f32],
    n_kv_heads: usize,
    tokens: usize,
    head_dim: usize,
    values_per_group: usize,
) -> LayerReport {
    let expect = n_kv_heads * tokens * head_dim;
    assert_eq!(k.len(), expect, "K is not [n_kv_heads, tokens, head_dim]");
    assert_eq!(v.len(), expect, "V is not [n_kv_heads, tokens, head_dim]");
    assert_eq!(tokens % Q4_BLOCK, 0, "token count must be a multiple of 32");
    assert_eq!(head_dim % Q4_BLOCK, 0, "head_dim must be a multiple of 32");
    assert_eq!(
        head_dim % values_per_group,
        0,
        "a 3-bit group must divide head_dim, or the per-token columns would straddle rows"
    );
    assert_eq!(
        tokens % values_per_group,
        0,
        "a 3-bit group must divide the token count, or the per-channel columns would straddle heads"
    );

    let rotation = HadamardRotation::new(head_dim, MEASUREMENT_ROTATION_SEED);
    let g = values_per_group;

    let mut k_cols = Vec::new();
    k_cols.push(Column {
        label: "f16",
        bits_per_value: 16.0,
        relative_error: relative_error(k, &through_f16(k)),
    });
    for (label, scheme) in [
        ("rot3 rms", Scheme::SymmetricLloydRms),
        ("rot3 absmax", Scheme::SymmetricLloydAbsMax),
    ] {
        let (back, bits) = through_rotated_3bit(k, head_dim, g, scheme, &rotation);
        k_cols.push(Column {
            label,
            bits_per_value: bits,
            relative_error: relative_error(k, &back),
        });
    }
    for (label, scheme) in [
        ("raw3 rms", Scheme::SymmetricLloydRms),
        ("raw3 absmax", Scheme::SymmetricLloydAbsMax),
    ] {
        let (back, bits) = through_plain_3bit(k, g, scheme);
        k_cols.push(Column {
            label,
            bits_per_value: bits,
            relative_error: relative_error(k, &back),
        });
    }
    // The variant the first round of measurement asked for: the incumbent's per-channel
    // grouping at three bits, with the rotation and without it.
    for (label, rot) in [("rot3 kivi", Some(&rotation)), ("raw3 kivi", None)] {
        let (back, bits) = through_kivi_3bit(k, n_kv_heads, tokens, head_dim, g, rot);
        k_cols.push(Column {
            label,
            bits_per_value: bits,
            relative_error: relative_error(k, &back),
        });
    }
    {
        // The incumbent, in its own per-channel layout.
        let channel_major = to_channel_major(k, n_kv_heads, tokens, head_dim);
        let back = to_token_major(&through_q4_0(&channel_major), n_kv_heads, tokens, head_dim);
        k_cols.push(Column {
            label: "q4_0 kivi",
            bits_per_value: q4_0_bits_per_value(),
            relative_error: relative_error(k, &back),
        });
    }

    let mut v_cols = Vec::new();
    v_cols.push(Column {
        label: "f16",
        bits_per_value: 16.0,
        relative_error: relative_error(v, &through_f16(v)),
    });
    // V's KIVI grouping is the per-token one it already had: a group is a run of channels
    // inside a single token, which is the layout the Q4_0 V column uses too. So these two
    // columns already are "3-bit under KIVI grouping", with the rotation and without it  -
    // labelled to say so, because "asym" named the codebook and not the grouping, and the
    // grouping is the thing under test.
    {
        let (back, bits) =
            through_rotated_3bit(v, head_dim, g, Scheme::AsymmetricUniform, &rotation);
        v_cols.push(Column {
            label: "rot3 kivi",
            bits_per_value: bits,
            relative_error: relative_error(v, &back),
        });
    }
    {
        let (back, bits) = through_plain_3bit(v, g, Scheme::AsymmetricUniform);
        v_cols.push(Column {
            label: "raw3 kivi",
            bits_per_value: bits,
            relative_error: relative_error(v, &back),
        });
    }
    v_cols.push(Column {
        label: "q4_0",
        bits_per_value: q4_0_bits_per_value(),
        relative_error: relative_error(v, &through_q4_0(v)),
    });

    LayerReport {
        layer,
        tokens,
        n_kv_heads,
        head_dim,
        values_per_group: g,
        k: k_cols,
        v: v_cols,
    }
}

/// The K or the V half of a report, so that the renderer walks both with one body.
fn half<'a>(r: &'a LayerReport, which: &str) -> &'a [Column] {
    if which == "K" {
        &r.k
    } else {
        &r.v
    }
}

/// Render a set of layer reports as two fixed-width tables.
pub fn format_table(reports: &[LayerReport]) -> String {
    let mut s = String::new();
    if reports.is_empty() {
        s.push_str("no layers measured\n");
        return s;
    }
    let head = &reports[0];
    s.push_str(&format!(
        "tokens={} n_kv_heads={} head_dim={} values_per_group={}\n",
        head.tokens, head.n_kv_heads, head.head_dim, head.values_per_group
    ));

    for which in ["K", "V"] {
        let cols = half(head, which);
        s.push_str(&format!(
            "\n{which}  relative error  (bits/value in brackets)\n"
        ));
        s.push_str(&format!("{:>5}", "layer"));
        for c in cols {
            s.push_str(&format!("  {:>13}", c.label));
        }
        s.push('\n');
        s.push_str(&format!("{:>5}", ""));
        for c in cols {
            s.push_str(&format!("  {:>13}", format!("[{:.2}b]", c.bits_per_value)));
        }
        s.push('\n');
        for r in reports {
            s.push_str(&format!("{:>5}", r.layer));
            for c in half(r, which) {
                s.push_str(&format!("  {:>13.5}", c.relative_error));
            }
            s.push('\n');
        }
        // A mean over layers, because a per-layer table is easy to read one row at a time
        // and the decision is about the whole cache.
        s.push_str(&format!("{:>5}", "mean"));
        for i in 0..cols.len() {
            let m: f64 = reports
                .iter()
                .map(|r| half(r, which)[i].relative_error)
                .sum::<f64>()
                / reports.len() as f64;
            s.push_str(&format!("  {:>13.5}", m));
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::cache::turboquant::rng::{gaussian_sample, next_uniform};

    /// A synthetic stand-in with the shape of a real K tensor: Gaussian, except that a few
    /// channels are systematically large. That per-channel outlier structure is the thing
    /// the rotation is aimed at, and it is what naive per-token Q4_0 of K failed on.
    fn outlier_channels(n_kv_heads: usize, tokens: usize, head_dim: usize, seed: u64) -> Vec<f32> {
        let mut values = gaussian_sample(n_kv_heads * tokens * head_dim, seed);
        let mut state = seed ^ 0x9999;
        let gains: Vec<f32> = (0..head_dim)
            .map(|_| {
                if next_uniform(&mut state) < 0.03 {
                    20.0
                } else {
                    1.0
                }
            })
            .collect();
        for row in values.chunks_exact_mut(head_dim) {
            for (x, g) in row.iter_mut().zip(gains.iter()) {
                *x *= *g;
            }
        }
        values
    }

    #[test]
    fn turboquant_relative_error_is_zero_on_an_identity_and_one_on_a_wipe() {
        // The harness must be able to say no. A metric that cannot report a total loss is
        // not measuring anything.
        let x = gaussian_sample(1024, 1);
        assert_eq!(relative_error(&x, &x), 0.0);
        let zeros = vec![0.0f32; x.len()];
        assert!((relative_error(&x, &zeros) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn turboquant_channel_major_transpose_round_trips() {
        // The Q4_0 K column depends on this being a permutation and nothing else. If it
        // scrambled anything, that column would report an error that is really a bug.
        let (h, t, d) = (3usize, 64usize, 32usize);
        let x = gaussian_sample(h * t * d, 42);
        let back = to_token_major(&to_channel_major(&x, h, t, d), h, t, d);
        assert_eq!(back, x);
    }

    #[test]
    fn turboquant_rotation_helps_on_channel_structured_data() {
        // The design's own claim, on synthetic data with the structure the design assumes.
        // If this fails the harness is wrong; if it passes and the real-tensor run fails,
        // the assumption about real tensors is wrong. The two are worth separating.
        let (h, t, d, g) = (2usize, 64usize, 128usize, 32usize);
        let k = outlier_channels(h, t, d, 2026);
        let v = gaussian_sample(h * t * d, 7);
        let r = evaluate_layer(0, &k, &v, h, t, d, g);
        let by = |cols: &Vec<Column>, label: &str| {
            cols.iter()
                .find(|c| c.label == label)
                .unwrap()
                .relative_error
        };
        let rot = by(&r.k, "rot3 rms");
        let raw_rms = by(&r.k, "raw3 rms");
        let raw_absmax = by(&r.k, "raw3 absmax");
        assert!(
            rot < raw_rms.min(raw_absmax),
            "rotation did not help on channel-structured K: rot {rot}, raw rms {raw_rms}, raw absmax {raw_absmax}"
        );
    }

    #[test]
    fn turboquant_rotation_diagnostics_isolate_the_rotation_from_the_quantiser() {
        let (h, t, d) = (2usize, 64usize, 128usize);
        let rot = HadamardRotation::new(d, MEASUREMENT_ROTATION_SEED);

        // On rows with a few dominant channels the rotation must be an exact involution
        // AND must visibly flatten them, or the reconstruction tables are attributing an
        // improvement to the wrong stage.
        let spiky = outlier_channels(h, t, d, 2026);
        let diag = rotation_diagnostics(&spiky, d, &rot);
        assert!(
            diag.max_round_trip_error < 1e-4,
            "round trip off by {:.3e}",
            diag.max_round_trip_error
        );
        assert!(
            diag.max_relative_norm_change < 1e-5,
            "norm drifted by {:.3e}",
            diag.max_relative_norm_change
        );
        assert!(
            diag.kurtosis_after < diag.kurtosis_before / 2.0,
            "kurtosis {:.2} -> {:.2} is not a flattening",
            diag.kurtosis_before,
            diag.kurtosis_after
        );
        assert!(
            diag.crest_after < diag.crest_before / 1.5,
            "crest {:.2} -> {:.2} is not a flattening",
            diag.crest_before,
            diag.crest_after
        );

        // And the instrument must be able to say no: on rows that are already Gaussian
        // there is nothing to flatten, so it must NOT report a flattening. A diagnostic
        // that shows an improvement on every input is not measuring the rotation.
        let flat = gaussian_sample(h * t * d, 1);
        let none = rotation_diagnostics(&flat, d, &rot);
        assert!(
            none.kurtosis_before.abs() < 0.6 && none.kurtosis_after.abs() < 0.6,
            "Gaussian rows moved from kurtosis {:.2} to {:.2}",
            none.kurtosis_before,
            none.kurtosis_after
        );
        assert!(
            (none.crest_after - none.crest_before).abs() < 0.5,
            "Gaussian rows moved from crest {:.2} to {:.2}",
            none.crest_before,
            none.crest_after
        );
    }

    #[test]
    fn turboquant_kivi_grouping_beats_channel_grouping_without_any_rotation() {
        // The claim the per-channel columns rest on: when the outliers are a property of
        // the channel and not of the token, grouping along the token axis removes them
        // outright, with no rotation involved. If this fails, the transpose in
        // `through_kivi_3bit` is wrong and its numbers are a bug rather than a result.
        let (h, t, d, g) = (2usize, 64usize, 128usize, 32usize);
        let k = outlier_channels(h, t, d, 2026);
        let v = gaussian_sample(h * t * d, 7);
        let r = evaluate_layer(0, &k, &v, h, t, d, g);
        let by = |cols: &Vec<Column>, label: &str| {
            cols.iter()
                .find(|c| c.label == label)
                .unwrap()
                .relative_error
        };
        let kivi = by(&r.k, "raw3 kivi");
        let per_channel = by(&r.k, "raw3 absmax");
        assert!(
            kivi < per_channel,
            "token-axis grouping {kivi} did not beat channel-axis grouping {per_channel}"
        );
        // And the two per-channel columns must cost the same, or the comparison is a bit
        // rate comparison wearing an accuracy label.
        let bits = |cols: &Vec<Column>, label: &str| {
            cols.iter()
                .find(|c| c.label == label)
                .unwrap()
                .bits_per_value
        };
        assert_eq!(bits(&r.k, "raw3 kivi"), bits(&r.k, "rot3 kivi"));
        assert_eq!(bits(&r.k, "raw3 kivi"), bits(&r.k, "raw3 absmax"));
    }

    #[test]
    fn turboquant_f16_column_is_far_below_every_three_bit_column() {
        // Sanity on the floor: f16 must be orders of magnitude better than three bits, or
        // the harness is comparing the wrong arrays.
        let (h, t, d, g) = (2usize, 32usize, 64usize, 32usize);
        let k = gaussian_sample(h * t * d, 3);
        let v = gaussian_sample(h * t * d, 4);
        let r = evaluate_layer(0, &k, &v, h, t, d, g);
        let f16_err = r.k[0].relative_error;
        assert!(f16_err < 1e-3, "f16 error {f16_err} is implausibly large");
        for c in r.k.iter().skip(1) {
            assert!(
                c.relative_error > f16_err * 10.0,
                "column {} at {} is not credibly worse than f16",
                c.label,
                c.relative_error
            );
        }
    }

    #[test]
    fn turboquant_table_renders_every_column() {
        let (h, t, d, g) = (1usize, 32usize, 32usize, 32usize);
        let k = gaussian_sample(h * t * d, 5);
        let v = gaussian_sample(h * t * d, 6);
        let table = format_table(&[evaluate_layer(0, &k, &v, h, t, d, g)]);
        for label in [
            "f16",
            "rot3 rms",
            "raw3 absmax",
            "rot3 kivi",
            "raw3 kivi",
            "q4_0 kivi",
        ] {
            assert!(table.contains(label), "table is missing column {label}");
        }
    }
}
