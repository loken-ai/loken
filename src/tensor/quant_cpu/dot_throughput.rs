//! What each dot product costs, per format.
//!
//! A decode step is bandwidth-bound: it reads the whole weight once and does a
//! few operations per byte. So the number that matters is bytes of weight per
//! second, and the working set has to be larger than the last-level cache or
//! the measurement reports the cache instead of the kernel.
//!
//! Not a gate - it asserts nothing. It exists so a change to these kernels can
//! be answered with a number rather than an intention. Run it with
//! `cargo test --release dot_throughput -- --ignored --nocapture`.

#[cfg(test)]
mod tests {
    use crate::tensor::quant_cpu::*;
    use std::time::Instant;

    /// Weight bytes per pass. Comfortably past a 32 MB L3 so the read comes
    /// from memory, which is where a decode step reads it from.
    const WORKING_SET: usize = 96 << 20;

    fn signal(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * (1 << (i % 5)) as f32
            })
            .collect()
    }

    fn measure<T: BlockFormat>(name: &str) {
        let block_bytes = std::mem::size_of::<T>();
        let blocks = WORKING_SET / block_bytes;
        let n = blocks * T::BLOCK_LEN;

        let mut weights = vec![T::zeros(); blocks];
        T::quantize(&signal(1, n), &mut weights);
        let mut acts = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
        T::ActivationBlock::quantize(&signal(2, n), &mut acts);

        // `black_box` on the operands, every pass. Without it the three calls
        // have identical arguments and no visible effect, so the optimiser is
        // free to run one and triple the result - which it did, and only for
        // some formats: the first reading of this had q5_K three times faster
        // per block than q4_K while doing strictly more work.
        let pass = || {
            let w = std::hint::black_box(&weights[..]);
            let a = std::hint::black_box(&acts[..]);
            std::hint::black_box(T::dot(w, a))
        };

        // One untimed pass so the comparison is between kernels, not between a
        // cold page table and a warm one.
        let mut sink = pass();

        const PASSES: usize = 3;
        let start = Instant::now();
        for _ in 0..PASSES {
            sink += pass();
        }
        let secs = start.elapsed().as_secs_f64();

        let bytes = (blocks * block_bytes * PASSES) as f64;
        println!(
            "{name:>6}  {:>7.1} GB/s  {:>7.2} ns/block  ({blocks} blocks of {block_bytes} B, \
             checksum {sink:.3e})",
            bytes / secs / 1e9,
            secs / (blocks * PASSES) as f64 * 1e9,
        );
    }

    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn what_each_dot_product_costs() {
        println!(
            "weight bytes read per second, {} MB working set",
            WORKING_SET >> 20
        );
        measure::<BlockQ4_0>("q4_0");
        measure::<BlockQ4_1>("q4_1");
        measure::<BlockQ5_0>("q5_0");
        measure::<BlockQ5_1>("q5_1");
        measure::<BlockQ8_0>("q8_0");
        measure::<BlockQ8_1>("q8_1");
        measure::<BlockMxFp4>("mxfp4");
        measure::<BlockQ2K>("q2_K");
        measure::<BlockQ3K>("q3_K");
        measure::<BlockQ4K>("q4_K");
        measure::<BlockQ5K>("q5_K");
        measure::<BlockQ6K>("q6_K");
        measure::<BlockQ8K>("q8_K");
        measure::<f32>("f32");
        measure::<half::f16>("f16");
        measure::<half::bf16>("bf16");
    }
}
