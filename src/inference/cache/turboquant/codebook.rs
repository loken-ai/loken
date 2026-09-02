//! The 3-bit Lloyd-Max codebook for a standard normal, solved here rather than tabulated.
//!
//! After the Hadamard rotation a coordinate is close to Gaussian - that is what mixing a
//! whole block into every output coordinate does - so the right eight levels are the ones
//! that minimise mean squared error against `N(0, 1)`. Those are the Lloyd-Max levels, and
//! they are the fixed point of two conditions applied in turn:
//!
//! * every boundary sits midway between the two levels it separates, and
//! * every level sits at the conditional mean of the interval it represents.
//!
//! Eight constants copied out of a paper are eight constants nobody in this repository can
//! check. The iteration is short, it converges in well under a millisecond, and it is run
//! by a test, so the numbers are reproducible from the code that uses them.
//!
//! Where this applies, and where it does not
//! -----------------------------------------
//!
//! The **levels** are used. They are the reconstruction points every symmetric 3-bit path
//! here quantises to, and on Gaussian input they measurably beat eight evenly spaced ones,
//! which a test pins.
//!
//! The **scale matched to them** is not used. The derivation assumes unit variance, so the
//! scale it implies is the group's RMS - and on real attention keys that scale lost to a
//! plain absmax at every group size measured (0.172 against 0.133 at a group of 8, 0.174
//! against 0.164 at 32). Real coordinates, rotated or not, are not Gaussian enough for a
//! rule that trades saturation against resolution to come out ahead, and eight values are
//! far too few to estimate an RMS from. See [`super::quant::Scheme`], where both are kept
//! and only one is the default.
//!
//! So this file is right about the shape of the answer and wrong about its calibration,
//! and both halves of that are worth having written down.

use std::sync::OnceLock;

/// Number of quantisation levels: three bits.
pub const LEVELS: usize = 8;

/// Convergence tolerance on the largest single-level movement between iterations.
///
/// The iteration is a fixed-point map on eight f64 values and it contracts, so this is
/// reached in a few dozen passes; it is stated as a constant so a caller reading the
/// codebook knows how far the levels are trusted.
pub const CONVERGENCE_TOLERANCE: f64 = 1e-12;

/// Upper bound on iterations, so a solver that stopped contracting reports a failure
/// rather than spinning.
const MAX_ITERATIONS: usize = 1000;

/// The error function, accurate across the range the solver walks.
///
/// Two regimes, because one series cannot cover both: the Maclaurin series is exact-ish
/// and cheap while `|x|` is small, and loses digits to cancellation as `|x|` grows; the
/// continued fraction for the complementary function takes over where the series gives up.
/// The Lloyd-Max boundaries all sit below `|x| = 2`, so the series is what actually runs  -
/// the tail branch is here so that the function is a function rather than a special case,
/// and a test pins the two against each other where they meet.
fn erf(x: f64) -> f64 {
    const SWITCH: f64 = 3.0;
    if x.abs() < SWITCH {
        erf_series(x)
    } else {
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        s * (1.0 - erfc_continued_fraction(x.abs()))
    }
}

/// `erf(x) = (2/sqrt(pi)) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))`, summed until the terms
/// stop changing the total.
fn erf_series(x: f64) -> f64 {
    let mut term_num = x; // x^(2n+1) / n!, sign folded in
    let mut sum = x;
    for n in 1..200 {
        term_num *= -x * x / (n as f64);
        let term = term_num / (2 * n + 1) as f64;
        sum += term;
        if term.abs() <= 1e-18 * sum.abs().max(1e-300) {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// `erfc(x)` for `x > 0` from its continued fraction, evaluated from the tail inwards.
///
/// The fraction is `exp(-x^2)/sqrt(pi) / (x + a1/(x + a2/(x + ...)))` with `a_n = n/2`. It
/// converges quickly once `x` is past 2, which is the only place it is called.
fn erfc_continued_fraction(x: f64) -> f64 {
    let mut f = 0.0f64;
    for n in (1..=80usize).rev() {
        f = (n as f64 * 0.5) / (x + f);
    }
    (-x * x).exp() / std::f64::consts::PI.sqrt() / (x + f)
}

/// The standard normal density.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The standard normal cumulative distribution.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Eight levels and the seven decision boundaries between them.
#[derive(Clone, Debug)]
pub struct LloydMaxCodebook {
    /// Reconstruction levels, ascending, in units of the group's scale.
    levels: [f32; LEVELS],
    /// Midpoint boundaries, ascending; `boundaries[i]` separates level `i` from `i + 1`.
    boundaries: [f32; LEVELS - 1],
    /// Mean squared error of the quantiser against `N(0, 1)`, in the same units squared.
    /// This is the number that says what three bits can buy before any of the rest of the
    /// pipeline is involved.
    distortion: f64,
    /// Iterations the solve took to reach [`CONVERGENCE_TOLERANCE`].
    iterations: usize,
}

impl LloydMaxCodebook {
    /// The codebook, solved once per process and shared thereafter.
    pub fn gaussian_3bit() -> &'static Self {
        static CODEBOOK: OnceLock<LloydMaxCodebook> = OnceLock::new();
        CODEBOOK.get_or_init(|| {
            Self::solve(CONVERGENCE_TOLERANCE).expect("Lloyd-Max iteration failed to converge")
        })
    }

    /// Run the Lloyd-Max iteration to `tolerance`.
    ///
    /// Symmetry of the result is not assumed anywhere: all eight levels are free, and the
    /// fact that they come out in `±` pairs is a property of the solution rather than of
    /// the solver.
    pub fn solve(tolerance: f64) -> Result<Self, String> {
        // Start from levels spread evenly across the bulk of the distribution. Any
        // reasonable ordered start lands on the same fixed point; this one is easy to read.
        let mut levels: [f64; LEVELS] = std::array::from_fn(|i| {
            let t = (i as f64 + 0.5) / LEVELS as f64;
            -3.0 + 6.0 * t
        });

        let mut iterations = 0usize;
        let mut converged = false;
        while iterations < MAX_ITERATIONS {
            iterations += 1;

            // Boundary condition: each boundary is the midpoint of its two levels.
            let mut bounds = [0.0f64; LEVELS + 1];
            bounds[0] = f64::NEG_INFINITY;
            bounds[LEVELS] = f64::INFINITY;
            for i in 1..LEVELS {
                bounds[i] = 0.5 * (levels[i - 1] + levels[i]);
            }

            // Centroid condition: each level is the conditional mean of its interval.
            // The numerator is closed form - the integral of `x * phi(x)` over `[a, b]` is
            // `phi(a) - phi(b)` - so only the interval's probability needs the CDF.
            let mut moved = 0.0f64;
            for i in 0..LEVELS {
                let (a, b) = (bounds[i], bounds[i + 1]);
                let pdf_a = if a.is_finite() { normal_pdf(a) } else { 0.0 };
                let pdf_b = if b.is_finite() { normal_pdf(b) } else { 0.0 };
                let cdf_a = if a.is_finite() { normal_cdf(a) } else { 0.0 };
                let cdf_b = if b.is_finite() { normal_cdf(b) } else { 1.0 };
                let mass = cdf_b - cdf_a;
                if mass <= 0.0 {
                    return Err(format!("interval {i} collapsed to zero probability"));
                }
                let centroid = (pdf_a - pdf_b) / mass;
                moved = moved.max((centroid - levels[i]).abs());
                levels[i] = centroid;
            }

            if moved < tolerance {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(format!(
                "Lloyd-Max did not reach {tolerance:e} in {MAX_ITERATIONS} iterations"
            ));
        }

        // Distortion of a centroid quantiser: `E[x^2] - sum_i p_i c_i^2`, which for the
        // standard normal is `1 - sum_i p_i c_i^2`.
        let mut bounds = [0.0f64; LEVELS + 1];
        bounds[0] = f64::NEG_INFINITY;
        bounds[LEVELS] = f64::INFINITY;
        for i in 1..LEVELS {
            bounds[i] = 0.5 * (levels[i - 1] + levels[i]);
        }
        let mut energy = 0.0f64;
        for i in 0..LEVELS {
            let (a, b) = (bounds[i], bounds[i + 1]);
            let cdf_a = if a.is_finite() { normal_cdf(a) } else { 0.0 };
            let cdf_b = if b.is_finite() { normal_cdf(b) } else { 1.0 };
            energy += (cdf_b - cdf_a) * levels[i] * levels[i];
        }

        Ok(Self {
            levels: std::array::from_fn(|i| levels[i] as f32),
            boundaries: std::array::from_fn(|i| (0.5 * (levels[i] + levels[i + 1])) as f32),
            distortion: 1.0 - energy,
            iterations,
        })
    }

    /// The eight reconstruction levels, ascending.
    pub fn levels(&self) -> &[f32; LEVELS] {
        &self.levels
    }

    /// The mean squared error against `N(0, 1)`.
    pub fn distortion(&self) -> f64 {
        self.distortion
    }

    /// Iterations the solve took.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The largest level in magnitude - the point past which the quantiser saturates, and
    /// therefore the number that says how much of a Gaussian tail three bits gives away.
    pub fn saturation(&self) -> f32 {
        self.levels[LEVELS - 1].abs().max(self.levels[0].abs())
    }

    /// The index whose level is nearest `x`, where `x` is already in units of the group
    /// scale.
    ///
    /// The seven boundaries are the decision rule, so this is a scan over seven
    /// comparisons rather than eight distances - the same answer, and the shape a device
    /// kernel would want later.
    pub fn index_of(&self, x: f32) -> u8 {
        let mut idx = 0u8;
        for b in self.boundaries.iter() {
            if x >= *b {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// The level an index reconstructs to.
    pub fn dequantise(&self, index: u8) -> f32 {
        self.levels[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turboquant_erf_matches_known_values() {
        // Reference values to sixteen digits; the series branch is exercised below 3 and
        // the continued fraction above it.
        let cases = [
            (0.5f64, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
            (4.0, 0.999_999_984_582_742_1),
        ];
        for (x, want) in cases {
            let got = erf(x);
            assert!(
                (got - want).abs() < 1e-12,
                "erf({x}) = {got}, expected {want}"
            );
            assert!((erf(-x) + want).abs() < 1e-12, "erf is not odd at {x}");
        }
    }

    #[test]
    fn turboquant_erf_branches_agree_where_they_meet() {
        // An instrument that cannot disagree with itself is not an instrument. The series
        // and the continued fraction are independent computations of the same quantity;
        // they must give the same answer either side of the switch.
        //
        // The tolerance is the series' cancellation, not the fraction's convergence: at
        // x = 3.6 the largest term of the series is about 1e4 times the sum, so four
        // digits of the double are gone before the two are compared. That is the reason
        // the switch sits at 3 and the codebook only ever asks below 2.
        for x in [2.6f64, 2.8, 3.0, 3.2, 3.6] {
            let series = erf_series(x);
            let tail = 1.0 - erfc_continued_fraction(x);
            assert!(
                (series - tail).abs() < 1e-12,
                "erf branches disagree at {x}: {series} vs {tail}"
            );
        }
    }

    #[test]
    fn turboquant_lloyd_max_converges_and_is_symmetric() {
        let cb = LloydMaxCodebook::solve(CONVERGENCE_TOLERANCE).expect("converged");
        assert!(
            cb.iterations() < MAX_ITERATIONS,
            "took {} iterations",
            cb.iterations()
        );

        // Symmetry was never imposed on the solver, so it is a real check on the solution.
        for i in 0..LEVELS / 2 {
            let lo = cb.levels()[i];
            let hi = cb.levels()[LEVELS - 1 - i];
            assert!(
                (lo + hi).abs() < 1e-5,
                "levels {i} and {} are not a +/- pair: {lo} vs {hi}",
                LEVELS - 1 - i
            );
        }

        // Ascending, and therefore usable as a boundary scan.
        for i in 1..LEVELS {
            assert!(cb.levels()[i] > cb.levels()[i - 1], "levels not ascending");
        }

        // The published optimum for an eight-level Gaussian quantiser is a mean squared
        // error near 0.03454, i.e. about 14.6 dB of signal-to-noise. Landing elsewhere
        // means the fixed point found is not the Lloyd-Max one.
        let d = cb.distortion();
        assert!(
            (d - 0.034_54).abs() < 5e-4,
            "distortion {d} is not the known optimum"
        );
        let snr_db = -10.0 * d.log10();
        assert!(
            (14.2..15.0).contains(&snr_db),
            "SNR {snr_db} dB out of range"
        );

        // Printed, not just asserted: the eight numbers are the module's one piece of
        // published data and a reader should be able to see them without a debugger.
        println!(
            "Lloyd-Max 3-bit / N(0,1): {} iterations, distortion {d:.6} ({snr_db:.2} dB)",
            cb.iterations()
        );
        println!(
            "  levels     {}",
            cb.levels()
                .iter()
                .map(|x| format!("{x:+.6}"))
                .collect::<Vec<_>>()
                .join("  ")
        );
        println!(
            "  boundaries {}",
            cb.boundaries
                .iter()
                .map(|x| format!("{x:+.6}"))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }

    #[test]
    fn turboquant_lloyd_max_is_a_fixed_point() {
        // Re-running the conditions on the converged levels must not move them. This is
        // the definition of the answer, checked rather than assumed.
        let cb = LloydMaxCodebook::solve(CONVERGENCE_TOLERANCE).expect("converged");
        let again = LloydMaxCodebook::solve(1e-14).expect("converged");
        for i in 0..LEVELS {
            assert!(
                (cb.levels()[i] - again.levels()[i]).abs() < 1e-6,
                "tightening the tolerance moved level {i}"
            );
        }
    }

    #[test]
    fn turboquant_codebook_round_trips_its_own_levels() {
        let cb = LloydMaxCodebook::gaussian_3bit();
        for i in 0..LEVELS as u8 {
            assert_eq!(
                cb.index_of(cb.dequantise(i)),
                i,
                "level {i} did not map back"
            );
        }
        // Anything past the outer levels saturates rather than wrapping.
        assert_eq!(cb.index_of(1e9), (LEVELS - 1) as u8);
        assert_eq!(cb.index_of(-1e9), 0);
    }

    #[test]
    fn turboquant_codebook_beats_a_uniform_quantiser_on_gaussian_data() {
        // The claim that justifies solving anything: for Gaussian input the Lloyd-Max
        // levels cost less error than eight evenly spaced ones at the same bit width.
        let cb = LloydMaxCodebook::gaussian_3bit();
        let sample = super::super::rng::gaussian_sample(200_000, 0x1234_5678);

        let mut err_lloyd = 0.0f64;
        let mut err_uniform = 0.0f64;
        // A uniform quantiser has to choose a clip point; 3 sigma over 8 levels is the
        // usual choice and is generous to it.
        let step = 6.0f32 / (LEVELS as f32 - 1.0);
        for &x in &sample {
            let q = cb.dequantise(cb.index_of(x));
            err_lloyd += ((x - q) as f64).powi(2);

            let idx = (((x + 3.0) / step).round()).clamp(0.0, LEVELS as f32 - 1.0);
            let u = -3.0 + idx * step;
            err_uniform += ((x - u) as f64).powi(2);
        }
        assert!(
            err_lloyd < err_uniform,
            "Lloyd-Max {err_lloyd} did not beat uniform {err_uniform}"
        );
    }
}
