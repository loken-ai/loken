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

    /// Weight bytes per pass of the in-cache comparison: inside L2, so the kernel is the limit.
    const CACHE_SET: usize = 192 << 10;

    /// One kernel, timed over a cache-resident set, in GB/s of weights.
    fn in_cache<T: BlockFormat>(kernel: fn(&[T], &[T::ActivationBlock]) -> f32) -> f64 {
        let blocks = CACHE_SET / std::mem::size_of::<T>();
        let n = blocks * T::BLOCK_LEN;
        let mut weights = vec![T::zeros(); blocks];
        T::quantize(&signal(1, n), &mut weights);
        let mut acts = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
        T::ActivationBlock::quantize(&signal(2, n), &mut acts);
        let pass = || {
            let w = std::hint::black_box(&weights[..]);
            let a = std::hint::black_box(&acts[..]);
            std::hint::black_box(kernel(w, a))
        };
        let mut sink = pass();
        const PASSES: usize = 20_000;
        let start = Instant::now();
        for _ in 0..PASSES {
            sink += pass();
        }
        let secs = start.elapsed().as_secs_f64();
        std::hint::black_box(sink);
        (blocks * std::mem::size_of::<T>() * PASSES) as f64 / secs / 1e9
    }

    fn tiers<T: BlockFormat>(
        name: &str,
        avx2: fn(&[T], &[T::ActivationBlock]) -> f32,
        avx512: fn(&[T], &[T::ActivationBlock]) -> f32,
    ) {
        // Alternate the two tiers and keep the best of three.
        let (mut a2, mut a5) = (0f64, 0f64);
        for _ in 0..3 {
            a2 = a2.max(in_cache(avx2));
            a5 = a5.max(in_cache(avx512));
        }
        println!(
            "{name:>6}  avx2 {a2:>7.1} GB/s   avx512 {a5:>7.1} GB/s   x{:.2}",
            a5 / a2
        );
    }

    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn avx2_against_avx512_in_cache() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        println!(
            "weight bytes per second, {} KB working set (L2)",
            CACHE_SET >> 10
        );
        tiers::<BlockQ4K>("q4_K", avx::vec_dot_q4k_q8k, avx512::vec_dot_q4k_q8k);
    }

    /// The Q4_K prefill GEMM group kernel, AVX2 against each strip width of the AVX-512 one, in
    /// cache: GMAC/s over an 8-column group by 8 activation rows.
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn q4k_gemm_group_tiers() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (k, n, m) = (2048usize, 8usize, 8usize);
        let nb = k / QK_K;
        let wf = signal(3, n * k);
        let mut wq = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            BlockQ4K::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q4k::repack(&wq, n, nb);
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            BlockQ8K::quantize(&signal(10 + r as u32, k), &mut aq[r * nb..(r + 1) * nb]);
        }
        let refs: Vec<&[BlockQ8K]> = (0..m).map(|r| &aq[r * nb..(r + 1) * nb]).collect();
        let time = |f: &dyn Fn(&mut [f32])| {
            let mut out = [0f32; 64];
            f(&mut out);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..2000 {
                    f(std::hint::black_box(&mut out));
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(out);
            (n * m * k) as f64 * 2000.0 / best / 1e9
        };
        let avx2 = time(&|o| unsafe { repack_q4k::gemm_group_avx2_rt::<1>(&x8, &refs, nb, o) });
        println!("q4_K gemm group, {m} rows x {n} cols x k={k}");
        println!("  avx2 rt1        {avx2:>7.1} GMAC/s");
        macro_rules! strip {
            ($rt:literal) => {{
                let v = time(&|o| unsafe {
                    avx512::repack_q4k::gemm_group_avx512_rt::<$rt>(&x8, &refs, nb, o)
                });
                println!("  avx512 rt{}      {v:>7.1} GMAC/s   x{:.2}", $rt, v / avx2);
            }};
        }
        strip!(1);
        strip!(2);
        strip!(4);
        strip!(8);
    }

    /// The Q6_K prefill GEMM group kernel, AVX2 against AVX-512, in cache.
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn q6k_gemm_group_tiers() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (k, n, m) = (2048usize, 8usize, 8usize);
        let nb = k / QK_K;
        let wf = signal(3, n * k);
        let mut wq = vec![BlockQ6K::zeros(); n * nb];
        for r in 0..n {
            BlockQ6K::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q6k::repack(&wq, n, nb);
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            BlockQ8K::quantize(&signal(10 + r as u32, k), &mut aq[r * nb..(r + 1) * nb]);
        }
        let refs: Vec<&[BlockQ8K]> = (0..m).map(|r| &aq[r * nb..(r + 1) * nb]).collect();
        let time = |f: &dyn Fn(&mut [f32])| {
            let mut out = [0f32; 64];
            f(&mut out);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..2000 {
                    out = [0f32; 64];
                    f(std::hint::black_box(&mut out));
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(out);
            (n * m * k) as f64 * 2000.0 / best / 1e9
        };
        let avx2 = time(&|o| unsafe { repack_q6k::gemm_group_avx2(&x8, &refs, nb, o) });
        let avx512 = time(&|o| unsafe { avx512::repack_q6k::gemm_group_avx512(&x8, &refs, nb, o) });
        println!("q6_K gemm group, {m} rows x {n} cols x k={k}");
        println!("  avx2   {avx2:>7.1} GMAC/s");
        println!("  avx512 {avx512:>7.1} GMAC/s   x{:.2}", avx512 / avx2);
    }

    /// Every other x8 prefill GEMM, AVX2 against AVX-512, in cache: 8 rows x 8 columns x k=2048.
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn x8_gemm_group_tiers() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        use crate::tensor::quant_cpu::{
            repack_mxfp4_x8, repack_q4_0, repack_q5_0, repack_q5k, repack_q8_0, repack_q8_0_x8,
            BlockMxFp4, BlockQ4_0, BlockQ5K, BlockQ5_0, BlockQ8_0,
        };
        let (k, n, m) = (2048usize, 8usize, 8usize);
        let rate = |f: &dyn Fn(&mut [f32; 64])| {
            let mut out = [0f32; 64];
            f(&mut out);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..2000 {
                    out = [0f32; 64];
                    f(std::hint::black_box(&mut out));
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(out);
            (n * m * k) as f64 * 2000.0 / best / 1e9
        };
        let show = |name: &str, avx2: f64, avx512: f64| {
            println!(
                "{name:<10} avx2 {avx2:>7.1}  avx512 {avx512:>7.1} GMAC/s   x{:.2}",
                avx512 / avx2
            );
        };
        let wf = signal(3, n * k);
        let q8_acts = |nb: usize| {
            let mut aq = vec![BlockQ8_0::zeros(); m * nb];
            for r in 0..m {
                BlockQ8_0::quantize(&signal(10 + r as u32, k), &mut aq[r * nb..(r + 1) * nb]);
            }
            aq
        };

        // Q5_K
        let nb = k / QK_K;
        let mut wq = vec![BlockQ5K::zeros(); n * nb];
        for r in 0..n {
            BlockQ5K::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q5k::repack(&wq, n, nb);
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            BlockQ8K::quantize(&signal(10 + r as u32, k), &mut aq[r * nb..(r + 1) * nb]);
        }
        let refs: Vec<&[BlockQ8K]> = (0..m).map(|r| &aq[r * nb..(r + 1) * nb]).collect();
        show(
            "q5_K",
            rate(&|o| unsafe { repack_q5k::gemm_group_avx2(&x8, &refs, nb, o) }),
            rate(&|o| unsafe { avx512::repack_q5k::gemm_group_avx512(&x8, &refs, nb, o) }),
        );

        // Block-major 32-weight formats.
        let nb = k / 32;
        let aq = q8_acts(nb);
        let refs: Vec<&[BlockQ8_0]> = (0..m).map(|r| &aq[r * nb..(r + 1) * nb]).collect();
        let mut wq = vec![BlockQ4_0::zeros(); n * nb];
        for r in 0..n {
            BlockQ4_0::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let g8 = repack_q4_0::repack(&wq, n, nb);
        show(
            "q4_0",
            rate(&|o| unsafe { repack_q4_0::gemm_group_avx2(&g8, &refs, nb, o) }),
            rate(&|o| unsafe { avx512::repack_inline::gemm_group_q4_0(&g8, &refs, nb, o) }),
        );
        let mut wq = vec![BlockQ5_0::zeros(); n * nb];
        for r in 0..n {
            BlockQ5_0::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let g8 = repack_q5_0::repack(&wq, n, nb);
        show(
            "q5_0",
            rate(&|o| unsafe { repack_q5_0::gemm_group_avx2(&g8, &refs, nb, o) }),
            rate(&|o| unsafe { avx512::repack_inline::gemm_group_q5_0(&g8, &refs, nb, o) }),
        );
        let mut wq = vec![BlockQ8_0::zeros(); n * nb];
        for r in 0..n {
            BlockQ8_0::quantize(&wf[r * k..(r + 1) * k], &mut wq[r * nb..(r + 1) * nb]);
        }
        let g8 = repack_q8_0::repack(&wq, n, nb);
        show(
            "q8_0",
            rate(&|o| unsafe { repack_q8_0::gemm_group_avx2(&g8, &refs, nb, o) }),
            rate(&|o| unsafe { avx512::repack_inline::gemm_group_q8_0(&g8, &refs, nb, o) }),
        );

        // Column-interleaved x8 layouts, 4-row activation groups (16 rows = 4 groups).
        let rows = 16usize;
        let ax4 = repack_mxfp4_x8::quantize_act(&signal(10, rows * k), rows, k);
        let c = |i: usize| &ax4[i * nb..(i + 1) * nb];
        let rate16 = |f: &dyn Fn(&mut [f32; 128])| {
            let mut out = [0f32; 128];
            f(&mut out);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..2000 {
                    f(std::hint::black_box(&mut out));
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(out);
            (n * rows * k) as f64 * 2000.0 / best / 1e9
        };
        let mut raw = vec![BlockQ8_0::zeros(); n * nb];
        for r in 0..n {
            BlockQ8_0::quantize(&wf[r * k..(r + 1) * k], &mut raw[r * nb..(r + 1) * nb]);
        }
        let w = repack_q8_0_x8::repack(&raw, n, nb);
        show(
            "q8_0 x8",
            rate16(&|o| unsafe {
                repack_q8_0_x8::gemm_group16_avx2(c(0), c(1), c(2), c(3), &w, nb, o)
            }),
            rate16(&|o| unsafe {
                avx512::repack_x8::gemm_group16_q8_0(c(0), c(1), c(2), c(3), &w, nb, o)
            }),
        );
        let mut raw = vec![BlockMxFp4::zeros(); n * nb];
        for r in 0..n {
            BlockMxFp4::quantize(&wf[r * k..(r + 1) * k], &mut raw[r * nb..(r + 1) * nb]);
        }
        let w = repack_mxfp4_x8::repack(&raw, n, nb);
        show(
            "mxfp4 x8",
            rate16(&|o| unsafe {
                repack_mxfp4_x8::gemm_group16_avx2(c(0), c(1), c(2), c(3), &w, nb, o)
            }),
            rate16(&|o| unsafe {
                avx512::repack_x8::gemm_group16_mxfp4(c(0), c(1), c(2), c(3), &w, nb, o)
            }),
        );
    }

    /// The f32, f16 and bf16 dot carriers, AVX2 against AVX-512, in cache (k = 4096).
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn float_carriers_tiers() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        use crate::tensor::quant_cpu::cpu;
        use half::{bf16, f16};
        let k = 4096usize;
        let (a, b) = (signal(1, k), signal(2, k));
        let (ah, bh): (Vec<f16>, Vec<f16>) = (
            a.iter().map(|&v| f16::from_f32(v)).collect(),
            b.iter().map(|&v| f16::from_f32(v)).collect(),
        );
        let (ab, bb): (Vec<bf16>, Vec<bf16>) = (
            a.iter().map(|&v| bf16::from_f32(v)).collect(),
            b.iter().map(|&v| bf16::from_f32(v)).collect(),
        );
        let rate = |f: &dyn Fn(&mut f32)| {
            let mut best = f64::MAX;
            let mut c = 0f32;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..200_000 {
                    f(std::hint::black_box(&mut c));
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(c);
            k as f64 * 200_000.0 / best / 1e9
        };
        let show = |name: &str, avx2: f64, avx512: f64| {
            println!(
                "{name:<5} avx2 {avx2:>7.1}  avx512 {avx512:>7.1} GMAC/s   x{:.2}",
                avx512 / avx2
            );
        };
        unsafe {
            show(
                "f32",
                rate(&|c| cpu::vectorised::vec_dot_f32(a.as_ptr(), b.as_ptr(), c, k)),
                rate(&|c| avx512::float::vec_dot_f32(a.as_ptr(), b.as_ptr(), c, k)),
            );
            show(
                "f16",
                rate(&|c| cpu::vectorised::vec_dot_f16(ah.as_ptr(), bh.as_ptr(), c, k)),
                rate(&|c| avx512::float::vec_dot_f16(ah.as_ptr(), bh.as_ptr(), c, k)),
            );
            show(
                "bf16",
                rate(&|c| cpu::vectorised::vec_dot_bf16(ab.as_ptr(), bb.as_ptr(), c, k)),
                rate(&|c| avx512::float::vec_dot_bf16(ab.as_ptr(), bb.as_ptr(), c, k)),
            );
        }
    }
}
