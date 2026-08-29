//! The host reference quantiser: group scales, 3-bit codes, and the stored layout.
//!
//! A group is a run of `values_per_group` consecutive values sharing one scale (and, for
//! V, one zero point). It is always a whole number of eight-value packing groups, so the
//! bit layout in [`super::pack`] never has to know what a scale covers.
//!
//! Storage cost per value is three bits plus the group's side information amortised over
//! the group: `3 + 16/g` bits for the symmetric K scheme, `3 + 32/g` for the asymmetric V
//! scheme. At `g = 8` that is 5 and 7 bits, which is *worse* than the 4.5 bits of the Q4_0
//! cache already in the tree - the eight-value group is the packing unit, not a sensible
//! scale unit, and any comparison that leaves the two conflated is comparing different
//! budgets. [`GroupQuantised::bits_per_value`] is reported next to every error figure for
//! that reason.

use super::codebook::LloydMaxCodebook;
use super::pack::{pack_run, unpack_run, GROUP_VALUES};
use half::f16;

/// How a group's values are mapped onto eight codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Symmetric, Lloyd-Max levels, group scale is the group's RMS.
    ///
    /// This is the scale the codebook is *derived* for: the levels minimise error against
    /// `N(0, 1)`, so the matching scale is the one that makes a group unit variance. It
    /// saturates at the outer level, about 2.15 sigma, which is cheap on a Gaussian and
    /// expensive on anything with a tail.
    ///
    /// **Measured, and not the default.** On real qwen3 keys it lost to
    /// [`Scheme::SymmetricLloydAbsMax`] at every group size tried - 0.172 against 0.133 at
    /// a group of 8, 0.174 against 0.164 at 32. The rotated coordinates are not Gaussian
    /// enough for the saturation trade to pay, and a group of 8 is far too small for its
    /// RMS to estimate anything reliably. It is kept as the control that shows this, not
    /// as a candidate.
    SymmetricLloydRms,
    /// Symmetric, Lloyd-Max levels, group scale placed so the group's largest magnitude
    /// lands exactly on the outer level.
    ///
    /// **This is the measured choice for K**, on the numbers above. Anyone re-deriving the
    /// RMS scale from the Lloyd-Max argument and wondering why it is not used should read
    /// this rather than repeat the experiment: the argument is sound and the data
    /// disagrees with it.
    ///
    /// It never saturates, so it wastes resolution whenever one value in a group is far
    /// from the rest - exactly the case that either the rotation, or a grouping along an
    /// axis the outliers do not vary along, is there to prevent.
    SymmetricLloydAbsMax,
    /// Asymmetric uniform over eight levels, with a stored zero point.
    ///
    /// V is averaged against softmax weights rather than dotted with Q, so it has no
    /// orthogonality argument to exploit and no reason to prefer a Gaussian-matched
    /// codebook; the plain min-to-max ramp is what the KIVI-style V path already does.
    AsymmetricUniform,
}

impl Scheme {
    /// Bits of side information a group carries: a scale, plus a zero point when the
    /// scheme is asymmetric.
    fn side_bits(&self) -> f64 {
        match self {
            Scheme::SymmetricLloydRms | Scheme::SymmetricLloydAbsMax => 16.0,
            Scheme::AsymmetricUniform => 32.0,
        }
    }
}

/// A quantised run: packed codes, plus one scale (and optionally one zero) per group.
#[derive(Clone, Debug)]
pub struct GroupQuantised {
    /// Three bytes per eight values, in the layout [`super::pack`] documents.
    pub packed: Vec<u8>,
    /// One scale per group, stored at the width a cache would store it.
    pub scales: Vec<f16>,
    /// One zero point per group; empty for the symmetric schemes.
    pub zeros: Vec<f16>,
    /// Values covered by one scale.
    pub values_per_group: usize,
    /// Values in the run.
    pub values: usize,
    /// The scheme that produced this run.
    pub scheme: Scheme,
}

impl GroupQuantised {
    /// Stored bits per value, side information included.
    pub fn bits_per_value(&self) -> f64 {
        3.0 + self.scheme.side_bits() / self.values_per_group as f64
    }

    /// Bytes actually held, which is what the figure above amortises.
    pub fn stored_bytes(&self) -> usize {
        self.packed.len() + 2 * self.scales.len() + 2 * self.zeros.len()
    }
}

/// Quantise a run of values.
///
/// `values.len()` must be a whole number of groups and `values_per_group` a whole number
/// of eight-value packing groups. Both are asserted rather than rounded: a run that did
/// not divide would silently give its last group a scale computed from padding.
pub fn quantise_run(values: &[f32], values_per_group: usize, scheme: Scheme) -> GroupQuantised {
    assert!(
        values_per_group % GROUP_VALUES == 0 && values_per_group > 0,
        "a group is a whole number of {GROUP_VALUES}-value packing groups"
    );
    assert_eq!(
        values.len() % values_per_group,
        0,
        "run length {} is not a whole number of {values_per_group}-value groups",
        values.len()
    );

    let codebook = LloydMaxCodebook::gaussian_3bit();
    let n_groups = values.len() / values_per_group;
    let mut indices = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(n_groups);
    let mut zeros = Vec::new();

    for group in values.chunks_exact(values_per_group) {
        match scheme {
            Scheme::SymmetricLloydRms | Scheme::SymmetricLloydAbsMax => {
                let raw_scale = match scheme {
                    Scheme::SymmetricLloydRms => {
                        let energy: f64 = group.iter().map(|x| (*x as f64) * (*x as f64)).sum();
                        (energy / group.len() as f64).sqrt() as f32
                    }
                    _ => {
                        let absmax = group.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                        absmax / codebook.saturation()
                    }
                };
                // Round-trip the scale through the width it is stored at, and quantise
                // against that value. A quantiser that used a wider scale than the one it
                // writes down would report an error the reader cannot reproduce.
                let scale = f16::from_f32(raw_scale);
                scales.push(scale);
                let s = scale.to_f32();
                if s <= 0.0 || !s.is_finite() {
                    // A dead group: every value is zero, so every code is the one nearest
                    // zero and the scale multiplies it away regardless.
                    let zero_code = codebook.index_of(0.0);
                    indices.extend(std::iter::repeat(zero_code).take(group.len()));
                } else {
                    let inv = 1.0 / s;
                    indices.extend(group.iter().map(|x| codebook.index_of(x * inv)));
                }
            }
            Scheme::AsymmetricUniform => {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for &x in group {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
                let zero = f16::from_f32(lo);
                let scale = f16::from_f32((hi - zero.to_f32()) / 7.0);
                scales.push(scale);
                zeros.push(zero);
                let (z, s) = (zero.to_f32(), scale.to_f32());
                if s <= 0.0 || !s.is_finite() {
                    indices.extend(std::iter::repeat(0u8).take(group.len()));
                } else {
                    let inv = 1.0 / s;
                    indices.extend(group.iter().map(|x| {
                        (((x - z) * inv).round()).clamp(0.0, 7.0) as u8
                    }));
                }
            }
        }
    }

    let mut packed = Vec::with_capacity(indices.len() / GROUP_VALUES * 3);
    pack_run(&indices, &mut packed);

    GroupQuantised {
        packed,
        scales,
        zeros,
        values_per_group,
        values: values.len(),
        scheme,
    }
}

/// Reconstruct a quantised run.
pub fn dequantise_run(q: &GroupQuantised) -> Vec<f32> {
    let codebook = LloydMaxCodebook::gaussian_3bit();
    let mut indices = Vec::with_capacity(q.values);
    unpack_run(&q.packed, &mut indices);
    indices.truncate(q.values);

    let mut out = Vec::with_capacity(q.values);
    for (g, group) in indices.chunks_exact(q.values_per_group).enumerate() {
        let s = q.scales[g].to_f32();
        match q.scheme {
            Scheme::SymmetricLloydRms | Scheme::SymmetricLloydAbsMax => {
                out.extend(group.iter().map(|&i| codebook.dequantise(i) * s));
            }
            Scheme::AsymmetricUniform => {
                let z = q.zeros[g].to_f32();
                out.extend(group.iter().map(|&i| z + i as f32 * s));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::cache::turboquant::measure::relative_error;
    use crate::inference::cache::turboquant::rng::gaussian_sample;

    #[test]
    fn turboquant_symmetric_scheme_hits_the_codebook_distortion_on_gaussian_data() {
        // THE PIPELINE VALIDATION. Feed the quantiser exactly what the theory assumes  - 
        // i.i.d. N(0, 1), already Gaussian, no rotation involved - and it must land on the
        // distortion the solver computed for itself: 0.034548, or 14.62 dB. Scaling,
        // packing and code assignment are all in the path, so if any of them is wrong this
        // number moves and every real-tensor figure this module reports is worthless.
        //
        // The theory's own condition is a *fixed* unit scale. This pipeline estimates the
        // scale per group, which is extra side information the Lloyd-Max analysis does not
        // have: a group that happened to draw small values gets a smaller scale and finer
        // resolution. So small groups legitimately beat the fixed-scale figure, and only
        // the large-group limit - where the per-group RMS converges to the global one  - 
        // is comparable with it. That limit is the check; the trend towards it is what
        // says the gain at small groups is adaptive scaling and not a leak.
        let values = gaussian_sample(1 << 18, 0xBEEF);
        let expected = LloydMaxCodebook::gaussian_3bit().distortion().sqrt();

        let groups = [32usize, 64, 256, 1024, 8192];
        let mut measured = Vec::new();
        for group in groups {
            let q = quantise_run(&values, group, Scheme::SymmetricLloydRms);
            let rel = relative_error(&values, &dequantise_run(&q));
            println!(
                "N(0,1) through the 3-bit pipeline, group {group:>5}: rel {rel:.6}  \
                 ({:.2} dB)  vs fixed-scale floor {expected:.6} (14.62 dB)",
                -20.0 * rel.log10()
            );
            measured.push(rel);
        }

        // The limit must reproduce the solver's own number. This is the validation: every
        // stage - scale, code assignment, bit packing, reconstruction - is in this path,
        // so if any of them is wrong this does not land.
        let limit = *measured.last().unwrap();
        assert!(
            (limit - expected).abs() <= expected * 0.02,
            "at a near-global scale the pipeline gave {limit}, not the codebook's own \
             {expected} - the loss is in the scale rule, the packing or the code \
             assignment, and every real-tensor number is then wrong"
        );

        // Nothing may sit far below the floor: adaptive scaling is worth a few per cent
        // over 32 values, not a different answer. A big drop would mean the reconstruction
        // is seeing something other than three bits.
        for (group, rel) in groups.iter().zip(measured.iter()) {
            assert!(
                *rel >= expected * 0.90,
                "group {group}: rel {rel} is far below the fixed-scale floor {expected}, \
                 which no amount of per-group scale adaptation buys"
            );
            assert!(
                *rel <= expected * 1.10,
                "group {group}: rel {rel} is more than 10% above the floor {expected}"
            );
        }

        // And the gain must shrink as the group grows, which is what identifies it as the
        // adaptive scale rather than as an error in the measurement.
        assert!(
            measured[0] < limit,
            "a 32-value scale ({}) did not beat a near-global one ({limit}), so the \
             advantage at small groups is not coming from scale adaptation",
            measured[0]
        );
    }

    #[test]
    fn turboquant_asymmetric_scheme_is_exact_on_a_ramp() {
        // Eight values evenly spaced between a group's min and max are exactly the eight
        // levels of an asymmetric uniform quantiser. If this is not near-exact the zero
        // point or the step is wrong.
        let values: Vec<f32> = (0..8).map(|i| -3.0 + i as f32).collect();
        let q = quantise_run(&values, 8, Scheme::AsymmetricUniform);
        let back = dequantise_run(&q);
        for (a, b) in values.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-2, "ramp not reproduced: {a} vs {b}");
        }
    }

    #[test]
    fn turboquant_quantiser_handles_a_dead_group() {
        // An all-zero group has no scale to speak of. It must come back as zeros rather
        // than as NaN, because a prompt's padding and a freshly allocated cache slot are
        // both all-zero.
        let values = vec![0.0f32; 64];
        for scheme in [
            Scheme::SymmetricLloydRms,
            Scheme::SymmetricLloydAbsMax,
            Scheme::AsymmetricUniform,
        ] {
            let back = dequantise_run(&quantise_run(&values, 32, scheme));
            assert!(
                back.iter().all(|x| *x == 0.0),
                "{scheme:?} did not reproduce a dead group"
            );
        }
    }

    #[test]
    fn turboquant_stored_size_matches_the_advertised_bit_rate() {
        let values = gaussian_sample(4096, 5);
        for (scheme, group) in [
            (Scheme::SymmetricLloydRms, 8usize),
            (Scheme::SymmetricLloydRms, 32),
            (Scheme::AsymmetricUniform, 8),
            (Scheme::AsymmetricUniform, 64),
        ] {
            let q = quantise_run(&values, group, scheme);
            let measured = q.stored_bytes() as f64 * 8.0 / q.values as f64;
            assert!(
                (measured - q.bits_per_value()).abs() < 1e-9,
                "{scheme:?} g={group}: stored {measured} bits/value, advertised {}",
                q.bits_per_value()
            );
        }
    }

    #[test]
    fn turboquant_absmax_beats_rms_when_a_group_has_an_outlier() {
        // The reason both scale rules exist. One value ten times the rest is exactly the
        // case an RMS scale saturates and an absmax scale does not.
        let mut values = gaussian_sample(4096, 11);
        for (i, x) in values.iter_mut().enumerate() {
            if i % 32 == 0 {
                *x *= 10.0;
            }
        }
        let rms = relative_error(
            &values,
            &dequantise_run(&quantise_run(&values, 32, Scheme::SymmetricLloydRms)),
        );
        let absmax = relative_error(
            &values,
            &dequantise_run(&quantise_run(&values, 32, Scheme::SymmetricLloydAbsMax)),
        );
        assert!(
            absmax < rms,
            "absmax {absmax} did not beat rms {rms} on outlier-bearing groups"
        );
    }
}
