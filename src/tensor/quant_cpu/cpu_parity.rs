//! The float carriers against a definition that is not one of them.
//!
//! `f32`, `f16` and `bf16` reach their dot product through `BlockFormat`, whose
//! `dot` and `dot_scalar` are the SAME function for these three - so the parity
//! net that covers the quantised formats compares a kernel with itself here and
//! can only ever pass. The judge has to come from outside: each carrier's
//! values are exactly representable in f64, so the reference is the same dot
//! product summed in f64 over the values the carrier actually holds.
//!
//! That gap is not hypothetical. `load_bf16` read bf16 bit patterns with
//! `_mm256_cvtph_ps`, an f16 converter - 1.0 came back as 1.875 - and every
//! test in the crate passed, because nothing here compared the vectorised
//! carrier to anything but itself.

#[cfg(test)]
mod tests {
    use crate::tensor::quant_cpu::*;
    use half::{bf16, f16};

    /// Values with both signs, several octaves, and no round numbers - a
    /// carrier that dropped its low mantissa bits would still pass on powers
    /// of two.
    fn signal(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(seed.wrapping_mul(374_761_393));
                let unit = (h >> 8) as f32 / (1 << 24) as f32 - 0.5;
                unit * (1 << (i % 6)) as f32 * 1.031_7
            })
            .collect()
    }

    /// The dot product of what the carrier HOLDS, summed where no rounding can
    /// hide a wrong load.
    fn reference<T: BlockFormat>(a: &[T], b: &[T]) -> f64 {
        let (mut av, mut bv) = (vec![0f32; a.len()], vec![0f32; b.len()]);
        T::dequantize(a, &mut av);
        T::dequantize(b, &mut bv);
        av.iter().zip(&bv).map(|(&x, &y)| x as f64 * y as f64).sum()
    }

    fn carrier_agrees<T: BlockFormat<ActivationBlock = T>>(name: &str, tolerance: f64) {
        // Lengths on and off the 32-value vector step, so the tail loop is
        // exercised too - it is a separate piece of code from the kernel.
        for n in [1usize, 8, 31, 32, 33, 96, 257] {
            for seed in [1u32, 77] {
                let mut a = vec![T::zeros(); n];
                let mut b = vec![T::zeros(); n];
                T::quantize(&signal(seed, n), &mut a);
                T::quantize(&signal(seed ^ 0x9e37_79b9, n), &mut b);

                let got = T::dot(&a, &b) as f64;
                let want = reference(&a, &b);
                let slack = tolerance * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= slack,
                    "{name}: dot {got} against the f64 sum of the same values {want} \
                     (n = {n}, seed {seed}, tolerance {slack:e})"
                );
            }
        }
    }

    #[test]
    fn the_f32_carrier_sums_the_values_it_holds() {
        carrier_agrees::<f32>("f32", 1e-6);
    }

    #[test]
    fn the_f16_carrier_sums_the_values_it_holds() {
        carrier_agrees::<f16>("f16", 1e-3);
    }

    #[test]
    fn the_bf16_carrier_sums_the_values_it_holds() {
        carrier_agrees::<bf16>("bf16", 1e-2);
    }

    /// The load in isolation: a dot product can absorb a wrong lane in its sum,
    /// a single value cannot.
    ///
    /// The length is 64 and not 8 on purpose. The kernel runs over whole steps
    /// of 32 and leaves the remainder to a scalar tail, so a short vector never
    /// reaches the vectorised load at all - a test of eight values passed
    /// against the broken load without touching it.
    #[test]
    fn every_bf16_lane_reads_back_as_itself() {
        const N: usize = 64;
        let values: Vec<f32> = signal(3, N);
        let mut held = vec![bf16::ZERO; N];
        bf16::quantize(&values, &mut held);

        for (i, &v) in held.iter().enumerate() {
            let mut one = vec![bf16::ZERO; N];
            one[i] = bf16::ONE;
            // `Σ held.onehot` is the i-th value, whatever the kernel does to
            // the other lanes.
            let got = bf16::dot(&held, &one);
            let want = v.to_f32();
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "bf16 lane {i}: read back as {got}, holds {want}"
            );
        }
    }
}
