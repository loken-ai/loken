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

    /// [`measure`] for a format that is read and never written: its weights are the bytes of a
    /// pattern, which any block decodes to some value.
    fn measure_read<T: BlockFormat>(name: &str) {
        let block_bytes = std::mem::size_of::<T>();
        let blocks = WORKING_SET / block_bytes;
        let n = blocks * T::BLOCK_LEN;
        let mut weights = vec![T::zeros(); blocks];
        let mut state = 0x9E37_79B9u32;
        let raw = unsafe {
            std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, blocks * block_bytes)
        };
        for (i, b) in raw.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 24) as u8;
            // Keep the leading half-precision scale of each block finite and small.
            if i % block_bytes < 2 {
                *b = if i % block_bytes == 0 { 0x00 } else { 0x2C };
            }
        }
        let mut acts = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
        T::ActivationBlock::quantize(&signal(2, n), &mut acts);
        let pass = || {
            let w = std::hint::black_box(&weights[..]);
            let a = std::hint::black_box(&acts[..]);
            std::hint::black_box(T::dot(w, a))
        };
        let mut sink = pass();
        const PASSES: usize = 3;
        let start = Instant::now();
        for _ in 0..PASSES {
            sink += pass();
        }
        let secs = start.elapsed().as_secs_f64();
        let bytes = (blocks * block_bytes * PASSES) as f64;
        println!(
            "{name:>6}  {:>7.1} GB/s  {:>7.2} ns/block  ({blocks} blocks of {block_bytes} B, checksum {sink:.3e})",
            bytes / secs / 1e9,
            secs / (blocks * PASSES) as f64 * 1e9,
        );
    }

    /// [`measure_read`] over the cores together: the weights in one shard a thread, each thread
    /// dotting its shard, for the thread counts a pool is sized to. What the DRAM and the cores
    /// give the expert products a layer request runs.
    fn measure_read_together<T: BlockFormat>(name: &str) {
        let block_bytes = std::mem::size_of::<T>();
        let blocks = WORKING_SET / block_bytes;
        let n = blocks * T::BLOCK_LEN;
        let mut weights = vec![T::zeros(); blocks];
        let mut state = 0x9E37_79B9u32;
        let raw = unsafe {
            std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, blocks * block_bytes)
        };
        for (i, b) in raw.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 24) as u8;
            if i % block_bytes < 2 {
                *b = if i % block_bytes == 0 { 0x00 } else { 0x2C };
            }
        }
        let mut acts = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
        T::ActivationBlock::quantize(&signal(2, n), &mut acts);
        let per_block = T::BLOCK_LEN / T::ActivationBlock::BLOCK_LEN;
        let physical = num_cpus::get_physical().max(1);
        let logical = num_cpus::get().max(1);
        for threads in [1usize, 2, 3, 4, 6, 8, physical, logical] {
            let shard = blocks.div_ceil(threads);
            const PASSES: usize = 3;
            let start = Instant::now();
            let sink: f32 = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let lo = (t * shard).min(blocks);
                        let hi = ((t + 1) * shard).min(blocks);
                        let w = &weights[lo..hi];
                        let a = &acts[lo * per_block..hi * per_block];
                        scope.spawn(move || {
                            let mut s = 0f32;
                            for _ in 0..PASSES {
                                let w = std::hint::black_box(w);
                                let a = std::hint::black_box(a);
                                s += std::hint::black_box(T::dot(w, a));
                            }
                            s
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum()
            });
            let secs = start.elapsed().as_secs_f64();
            let bytes = (blocks * block_bytes * PASSES) as f64;
            println!(
                "{name:>7} x{threads:<2} {:>6.1} GB/s together (checksum {sink:.3e})",
                bytes / secs / 1e9
            );
        }
    }

    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn what_the_cores_reach_together() {
        measure_read_together::<BlockIq2S>("iq2_s");
        measure_read_together::<BlockQ8_0>("q8_0");
        measure_read_together::<f32>("f32");
    }

    /// Random blocks of a read-only format and `n` activations quantised from signals.
    fn read_only_rows<T: BlockFormat>(
        blocks: usize,
        acts: usize,
    ) -> (Vec<T>, Vec<Vec<T::ActivationBlock>>) {
        let block_bytes = std::mem::size_of::<T>();
        let n = blocks * T::BLOCK_LEN;
        let mut weights = vec![T::zeros(); blocks];
        let mut state = 0x9E37_79B9u32;
        let raw = unsafe {
            std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, blocks * block_bytes)
        };
        for (i, b) in raw.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 24) as u8;
            if i % block_bytes < 2 {
                *b = if i % block_bytes == 0 { 0x00 } else { 0x2C };
            }
        }
        let ys = (0..acts)
            .map(|t| {
                let mut a = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
                T::ActivationBlock::quantize(&signal(3 + t as u32, n), &mut a);
                a
            })
            .collect();
        (weights, ys)
    }

    fn many_is_each<T: BlockFormat>(name: &str) {
        for acts in [1usize, 2, 3, 8, 11] {
            let (weights, ys) = read_only_rows::<T>(10, acts);
            let refs: Vec<&[T::ActivationBlock]> = ys.iter().map(Vec::as_slice).collect();
            let mut got = vec![0f32; acts];
            T::dot_many(&weights, &refs, &mut got);
            for (t, y) in ys.iter().enumerate() {
                let want = T::dot(&weights, y);
                assert_eq!(
                    got[t].to_bits(),
                    want.to_bits(),
                    "{name}: activation {t} of {acts}"
                );
            }
        }
    }

    /// Several activations over the same rows give, each, the dot of that activation alone,
    /// to the bit, in every format the host decodes once for them.
    #[test]
    fn a_dot_over_several_activations_is_each_dot() {
        many_is_each::<BlockIq2S>("iq2_s");
        many_is_each::<BlockIq3Xxs>("iq3_xxs");
        many_is_each::<BlockIq3S>("iq3_s");
        many_is_each::<BlockIq4Xs>("iq4_xs");
        many_is_each::<BlockIq4Nl>("iq4_nl");
        many_is_each::<BlockQ2_0>("q2_0");
    }

    fn measure_many<T: BlockFormat>(name: &str) {
        // An expert's gate rows: 512 rows of 2560 weights.
        let (rows, width) = (512usize, 2560usize);
        let per_row = width / T::BLOCK_LEN;
        let (weights, ys) = read_only_rows::<T>(rows * per_row, 4);
        for acts in [1usize, 2, 3, 4] {
            let refs: Vec<&[T::ActivationBlock]> = ys[..acts].iter().map(Vec::as_slice).collect();
            let time = |f: &dyn Fn(usize, &mut [f32])| -> f64 {
                let mut out = vec![0f32; acts];
                let start = Instant::now();
                for _ in 0..3 {
                    for r in 0..rows {
                        f(r, &mut out[..]);
                        std::hint::black_box(&out);
                    }
                }
                start.elapsed().as_secs_f64() * 1e6 / 3.0
            };
            let each = time(&|row: usize, o: &mut [f32]| {
                let w = &weights[row * per_row..(row + 1) * per_row];
                for (t, y) in refs.iter().enumerate() {
                    o[t] = T::dot(w, y);
                }
            });
            let many = time(&|row: usize, o: &mut [f32]| {
                let w = &weights[row * per_row..(row + 1) * per_row];
                T::dot_many(w, &refs, o);
            });
            println!(
                "{name:>7} x{acts}: {each:7.0} us each, {many:7.0} us together ({:.2}x)",
                each / many
            );
        }
    }

    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn what_decoding_once_for_several_activations_saves() {
        measure_many::<BlockIq2S>("iq2_s");
        measure_many::<BlockIq3Xxs>("iq3_xxs");
        measure_many::<BlockIq4Nl>("iq4_nl");
        measure_many::<BlockQ2_0>("q2_0");
    }

    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn what_each_read_only_dot_product_costs() {
        println!("read-only formats, weight bytes read per second");
        measure_read::<BlockIq2S>("iq2_s");
        measure_read::<BlockIq3S>("iq3_s");
        measure_read::<BlockIq3Xxs>("iq3_xxs");
        measure_read::<BlockIq4Xs>("iq4_xs");
        measure_read::<BlockIq4Nl>("iq4_nl");
        measure_read::<BlockQ2_0>("q2_0");
        measure::<BlockQ4K>("q4_K");
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
