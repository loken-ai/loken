//! The 4-bit AWQ Marlin GEMM of one layer's 4 projections against the weight dequantised to f16
//! once into a transient buffer and a dense f16 GEMM, as a function of the number of rows.
//! Marlin streams the packed weight and is bound by compute for a prefill-size batch.
//! The dense GEMM runs on the tensor cores at its own rate, and the dequantisation is one pass
//! over the weight whatever the rows.
//! Each variant is timed under the host clock around stream synchronisations.
//! The weights are the synthetic ones of `marlin-rows`, in the checkpoint layout the dequantise
//! function takes and repacked by the loader's routine for Marlin.
//! Several copies of each weight, together larger than twice the card's second-level cache, are
//! called in turn so that every call reads its weights from card memory.

use super::marlin_rows::ProjectionShape;

/// The row counts timed when `--rows-list` is absent.
pub const DEFAULT_ROWS: [usize; 7] = [32, 64, 130, 260, 512, 1024, 2048];

/// The variants of a projection, in the order the table and the record list them.
pub const VARIANTS: [&str; 4] = [
    "marlin",
    "dequantise only",
    "dense f16 only",
    "dequantise + dense f16",
];

/// One measured row count of one projection: the median of each variant, in microseconds.
/// The dequantisation does not depend on the rows, so `dequant_us` is the same at every row count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub rows: usize,
    pub marlin_us: f64,
    pub dequant_us: f64,
    pub dense_us: f64,
    pub both_us: f64,
}

/// Tera floating-point operations per second of a `[rows, k] x [k, n]` product done in `us`
/// microseconds: 2 operations per multiply-add. 0 for a time at or below 0.
pub fn tflops(rows: usize, k: usize, n: usize, us: f64) -> f64 {
    if us <= 0.0 {
        return 0.0;
    }
    2.0 * rows as f64 * k as f64 * n as f64 / us / 1e6
}

/// The speed of the variant timed `other_us` relative to the one timed `base_us`: above 1 when
/// it is faster. 0 for a time at or below 0.
pub fn speed_vs(base_us: f64, other_us: f64) -> f64 {
    if base_us <= 0.0 || other_us <= 0.0 {
        return 0.0;
    }
    base_us / other_us
}

/// The smallest row count from which `other_us` is below `base_us` and stays below at every
/// larger row count measured. `None` when the largest row count is not faster.
/// `rows`, `base_us` and `other_us` are parallel and in any order.
pub fn crossover(rows: &[usize], base_us: &[f64], other_us: &[f64]) -> Option<usize> {
    let mut points: Vec<(usize, bool)> = rows
        .iter()
        .zip(base_us.iter().zip(other_us))
        .map(|(&r, (&b, &o))| (r, o < b))
        .collect();
    points.sort_by_key(|p| p.0);
    let tail = points.iter().rev().take_while(|p| p.1).count();
    if tail == 0 {
        return None;
    }
    Some(points[points.len() - tail].0)
}

/// The table of one projection: rows, the median of each variant, the speed of the dequantise
/// and dense pair relative to Marlin and the rate of the dense GEMM alone.
pub fn projection_table(p: &ProjectionShape, points: &[Point]) -> String {
    let mut out = format!(
        "{} {}x{}\n{:>6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}  {:>12}\n",
        p.name,
        p.k,
        p.n,
        "rows",
        VARIANTS[0],
        VARIANTS[1],
        VARIANTS[2],
        VARIANTS[3],
        "vs marlin",
        "dense TFLOPS"
    );
    for pt in points {
        out.push_str(&format!(
            "{:>6}  {:>11.1}  {:>11.1}  {:>11.1}  {:>11.1}  {:>8.2}  {:>12.1}\n",
            pt.rows,
            pt.marlin_us,
            pt.dequant_us,
            pt.dense_us,
            pt.both_us,
            speed_vs(pt.marlin_us, pt.both_us),
            tflops(pt.rows, p.k, p.n, pt.dense_us)
        ));
    }
    out
}

/// The final lines: for each row count the layer time of Marlin and of the dequantise and dense
/// pair with the speed of the second relative to the first, then the row count where the second
/// becomes faster.
pub fn layer_lines(rows: &[usize], marlin_us: &[f64], both_us: &[f64]) -> String {
    let mut out = String::new();
    for ((&r, &m), &b) in rows.iter().zip(marlin_us).zip(both_us) {
        out.push_str(&format!(
            "layer, 4 projections, m={r}: {} {m:.1} us, {} {b:.1} us, speed {:.2}\n",
            VARIANTS[0],
            VARIANTS[3],
            speed_vs(m, b)
        ));
    }
    out.push_str(&crossover_line(rows, marlin_us, both_us));
    out
}

/// The row count from which the dequantise and dense pair beats Marlin, or `none up to <max>`.
pub fn crossover_line(rows: &[usize], marlin_us: &[f64], both_us: &[f64]) -> String {
    match crossover(rows, marlin_us, both_us) {
        Some(r) => format!("{} is faster than {} from m={r}", VARIANTS[3], VARIANTS[0]),
        None => format!(
            "{} is faster than {}: none up to {}",
            VARIANTS[3],
            VARIANTS[0],
            rows.iter().copied().max().unwrap_or(0)
        ),
    }
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::inference::quantized_cuda::awq_dequant_f16_storage;
    use crate::microbench::marlin_rows::{
        copies_for, layer_projections, layer_totals, synthetic_awq, synthetic_awq_words,
        DEFAULT_HEADS, DEFAULT_INTER, DEFAULT_KV_HEADS,
    };
    use crate::microbench::stats::{run_for, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::cuda_ext::tensor_from_cuda_storage;
    use crate::tensor::gpu_runtime::driver::sys;
    use crate::tensor::marlin::MarlinAwqLayer;
    use crate::tensor::{Device, Tensor};
    use half::f16;
    use std::time::{Duration, Instant};

    /// A weight in the checkpoint layout, on the card, as the dequantise function takes it.
    struct Checkpoint {
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
    }

    /// The f16 weight `[k, n]` the crate's AWQ prefill path builds from `w`, in a new buffer.
    fn dequantise(
        w: &Checkpoint,
        p: &ProjectionShape,
        group_size: usize,
    ) -> Result<Tensor, String> {
        let storage =
            awq_dequant_f16_storage(&w.qweight, &w.qzeros, &w.scales, p.n, p.k, group_size)
                .map_err(|e| format!("{}: dequantise: {e}", p.name))?;
        tensor_from_cuda_storage(storage, (p.k, p.n))
            .map_err(|e| format!("{}: dequantise: {e:?}", p.name))
    }

    pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("dequant-gemm needs a card")?;
        let o = cx.options;
        let rows = if o.rows_list.is_empty() {
            DEFAULT_ROWS.to_vec()
        } else {
            o.rows_list.clone()
        };
        if rows.contains(&0) {
            return Err("--rows-list needs row counts above zero".to_string());
        }
        let max_rows = rows.iter().copied().max().unwrap_or(1);
        let projections = layer_projections(
            cx.shapes.hidden,
            o.inter.unwrap_or(DEFAULT_INTER),
            o.heads.unwrap_or(DEFAULT_HEADS),
            o.kv_heads.unwrap_or(DEFAULT_KV_HEADS),
            o.rope_head_dim(),
        );
        for p in &projections {
            p.check(o.group_size)?;
        }
        let dev = &card.dev;
        let device = Device::Cuda(dev.clone());
        let stream = dev.stream().clone();
        let l2 = card
            .dev
            .context()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .map_err(|e| format!("card cache size: {e:?}"))?
            .max(0) as usize;
        let target = Duration::from_secs_f64(o.seconds);
        let fail = Fail::default();

        let mut measured = Vec::new();
        let mut marlin_medians: Vec<Vec<f64>> = Vec::new();
        let mut both_medians: Vec<Vec<f64>> = Vec::new();
        for p in &projections {
            let name = format!("{} {}x{}", p.name, p.k, p.n);
            let (qweight, qzeros, scales) = synthetic_awq_words(p.k, p.n, o.group_size);
            let packed_bytes = (qweight.len() * 4 + qzeros.len() * 4 + scales.len() * 2) as u64;
            let dense_bytes = (p.k * p.n * 2) as u64;
            let groups = p.k / o.group_size;

            let repacked = synthetic_awq(p.k, p.n, o.group_size)?;
            let layers = (0..copies_for(l2, repacked.device_bytes()))
                .map(|_| {
                    MarlinAwqLayer::new(dev, &repacked).map_err(|e| format!("{}: {}", p.name, e.0))
                })
                .collect::<Result<Vec<_>, _>>()?;
            drop(repacked);

            let upload = |what: &str| -> Result<Checkpoint, String> {
                let tensors = (
                    Tensor::from_slice(&qweight, (p.k, p.n / 8), &device),
                    Tensor::from_slice(&qzeros, (groups, p.n / 8), &device),
                    Tensor::from_slice(&scales, (groups, p.n), &device),
                );
                match tensors {
                    (Ok(qweight), Ok(qzeros), Ok(scales)) => Ok(Checkpoint {
                        qweight,
                        qzeros,
                        scales,
                    }),
                    (a, b, c) => Err(format!(
                        "{} {what}: {:?} {:?} {:?}",
                        p.name,
                        a.err(),
                        b.err(),
                        c.err()
                    )),
                }
            };
            let sources = (0..copies_for(l2, packed_bytes as usize))
                .map(|_| upload("checkpoint weight"))
                .collect::<Result<Vec<_>, _>>()?;
            let dense = (0..copies_for(l2, dense_bytes as usize))
                .map(|_| dequantise(&sources[0], p, o.group_size))
                .collect::<Result<Vec<_>, _>>()?;

            let a_host: Vec<f16> = (0..max_rows * p.k)
                .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
                .collect();
            let a = stream
                .clone_htod(&a_host)
                .map_err(|e| format!("{} input: {e:?}", p.name))?;
            // Safety: the kernel writes the output before anything reads it.
            let mut c = unsafe { stream.alloc::<f16>(max_rows * p.n) }
                .map_err(|e| format!("{} output: {e:?}", p.name))?;

            let variant = format!("{name} {}", VARIANTS[1]);
            let mut next = 0usize;
            let samples = run_for(target, || {
                let source = &sources[next % sources.len()];
                next += 1;
                fail.ok(stream.synchronize());
                let t0 = Instant::now();
                let w = dequantise(source, p, o.group_size);
                drop_ok(&fail, w);
                fail.ok(stream.synchronize());
                t0.elapsed().as_secs_f64() * 1e6
            });
            fail.check(&variant)?;
            let dequant_us = Summary::of(&samples).median_us;
            measured.push(Measured {
                variant,
                bytes: Some(packed_bytes + dense_bytes),
                samples_us: samples,
                note: None,
            });

            let mut points = Vec::new();
            for &m in &rows {
                let x = Tensor::from_slice(&a_host[..m * p.k], (m, p.k), &device)
                    .map_err(|e| format!("{name} m={m} input: {e:?}"))?;

                let variant = format!("{name} m={m} {}", VARIANTS[0]);
                let mut next = 0usize;
                let samples = run_for(target, || {
                    let layer = &layers[next % layers.len()];
                    next += 1;
                    fail.ok(stream.synchronize());
                    let t0 = Instant::now();
                    fail.ok(layer.forward_into(dev, &a, m, &mut c));
                    fail.ok(stream.synchronize());
                    t0.elapsed().as_secs_f64() * 1e6
                });
                fail.check(&variant)?;
                let marlin_us = Summary::of(&samples).median_us;
                measured.push(Measured {
                    variant,
                    bytes: Some(packed_bytes),
                    samples_us: samples,
                    note: None,
                });

                let variant = format!("{name} m={m} {}", VARIANTS[2]);
                let mut next = 0usize;
                let samples = run_for(target, || {
                    let w = &dense[next % dense.len()];
                    next += 1;
                    fail.ok(stream.synchronize());
                    let t0 = Instant::now();
                    let y = x.matmul(w);
                    fail.ok(stream.synchronize());
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    fail.ok(y.map(drop));
                    us
                });
                fail.check(&variant)?;
                let dense_us = Summary::of(&samples).median_us;
                measured.push(Measured {
                    variant,
                    bytes: Some(dense_bytes),
                    samples_us: samples,
                    note: None,
                });

                let variant = format!("{name} m={m} {}", VARIANTS[3]);
                let mut next = 0usize;
                let samples = run_for(target, || {
                    let source = &sources[next % sources.len()];
                    next += 1;
                    fail.ok(stream.synchronize());
                    let t0 = Instant::now();
                    let y = dequantise(source, p, o.group_size)
                        .and_then(|w| x.matmul(&w).map_err(|e| format!("{name}: matmul: {e:?}")));
                    fail.ok(stream.synchronize());
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    drop_ok(&fail, y);
                    us
                });
                fail.check(&variant)?;
                let both_us = Summary::of(&samples).median_us;
                measured.push(Measured {
                    variant,
                    bytes: Some(packed_bytes + 2 * dense_bytes),
                    samples_us: samples,
                    note: None,
                });

                points.push(Point {
                    rows: m,
                    marlin_us,
                    dequant_us,
                    dense_us,
                    both_us,
                });
            }
            print!("{}", projection_table(p, &points));
            println!();
            marlin_medians.push(points.iter().map(|pt| pt.marlin_us).collect());
            both_medians.push(points.iter().map(|pt| pt.both_us).collect());
        }

        let marlin_totals = layer_totals(&marlin_medians);
        let both_totals = layer_totals(&both_medians);
        for ((&m, &marlin), &both) in rows.iter().zip(&marlin_totals).zip(&both_totals) {
            measured.push(Measured {
                variant: format!("layer, 4 projections, m={m}, {}", VARIANTS[0]),
                bytes: None,
                samples_us: vec![marlin],
                note: None,
            });
            measured.push(Measured {
                variant: format!("layer, 4 projections, m={m}, {}", VARIANTS[3]),
                bytes: None,
                samples_us: vec![both],
                note: Some(format!(
                    "{:.2} times the speed of marlin",
                    speed_vs(marlin, both)
                )),
            });
        }
        println!("{}", layer_lines(&rows, &marlin_totals, &both_totals));
        Ok(measured)
    }

    /// Note the error of `r` and drop its value.
    fn drop_ok<T>(fail: &Fail, r: Result<T, String>) {
        fail.ok(r.map(drop));
    }
}

#[cfg(feature = "gpu")]
pub use device::run;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microbench::marlin_rows::layer_projections;

    #[test]
    fn the_rate_counts_2_operations_per_multiply_add() {
        // 2 * 1000 * 1000 * 1000 operations in 1000 us is 2 TFLOPS.
        assert!((tflops(1000, 1000, 1000, 1000.0) - 2.0).abs() < 1e-9);
        assert_eq!(tflops(1, 1, 1, 0.0), 0.0);
        assert_eq!(tflops(1, 1, 1, -1.0), 0.0);
    }

    #[test]
    fn the_speed_is_above_1_when_the_other_variant_is_faster() {
        assert_eq!(speed_vs(200.0, 100.0), 2.0);
        assert_eq!(speed_vs(100.0, 200.0), 0.5);
        assert_eq!(speed_vs(0.0, 100.0), 0.0);
        assert_eq!(speed_vs(100.0, 0.0), 0.0);
    }

    #[test]
    fn the_crossover_is_the_row_count_from_which_the_other_stays_faster() {
        let rows = [32, 64, 130, 260, 512];
        let marlin = [10.0, 20.0, 30.0, 40.0, 50.0];
        let other = [20.0, 25.0, 20.0, 30.0, 40.0];
        assert_eq!(crossover(&rows, &marlin, &other), Some(130));
        // A win at 64 followed by a loss at 130 is not a crossover.
        let other = [20.0, 15.0, 35.0, 30.0, 40.0];
        assert_eq!(crossover(&rows, &marlin, &other), Some(260));
        let other = [20.0, 25.0, 35.0, 45.0, 55.0];
        assert_eq!(crossover(&rows, &marlin, &other), None);
        let other = [5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(crossover(&rows, &marlin, &other), Some(32));
    }

    #[test]
    fn the_crossover_reads_the_rows_in_any_order() {
        let rows = [512, 32, 130];
        let marlin = [50.0, 10.0, 30.0];
        let other = [40.0, 20.0, 20.0];
        assert_eq!(crossover(&rows, &marlin, &other), Some(130));
        assert_eq!(crossover(&[], &[], &[]), None);
        assert_eq!(crossover(&[64], &[10.0], &[10.0]), None);
    }

    #[test]
    fn the_crossover_line_names_the_row_count_or_the_largest_one() {
        let rows = [32, 2048];
        let line = crossover_line(&rows, &[10.0, 100.0], &[20.0, 50.0]);
        assert!(line.ends_with("from m=2048"), "{line}");
        let line = crossover_line(&rows, &[10.0, 100.0], &[20.0, 150.0]);
        assert!(line.ends_with("none up to 2048"), "{line}");
    }

    #[test]
    fn the_layer_lines_give_each_row_count_then_the_crossover() {
        let text = layer_lines(&[32, 130], &[100.0, 300.0], &[200.0, 150.0]);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("m=32") && lines[0].contains("0.50"));
        assert!(lines[1].contains("m=130") && lines[1].contains("2.00"));
        assert!(lines[2].ends_with("from m=130"));
    }

    #[test]
    fn the_table_gives_the_speed_against_marlin_and_the_dense_rate() {
        let p = layer_projections(4096, 12288, 32, 8, 128)[1];
        let t = projection_table(
            &p,
            &[Point {
                rows: 1024,
                marlin_us: 400.0,
                dequant_us: 100.0,
                dense_us: 100.0,
                both_us: 200.0,
            }],
        );
        assert!(t.contains("o 4096x4096"));
        assert!(t.contains("2.00"));
        // 2 * 1024 * 4096 * 4096 operations in 100 us.
        assert!(t.contains("343.6"), "{t}");
    }

    #[test]
    fn the_default_rows_ascend() {
        assert!(DEFAULT_ROWS.windows(2).all(|w| w[0] < w[1]));
    }
}
