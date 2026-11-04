//! The 4-bit AWQ Marlin GEMM of one layer's 4 projections as a function of the number of rows.
//! The launcher cuts a batch into tiles of `MAX_M_PER_CHUNK` rows and each tile reads the
//! weights again, so the time per call is expected to follow the passes over the weights,
//! `ceil(rows / MAX_M_PER_CHUNK)`. Each variant is one row count. The weights are synthetic,
//! repacked by the loader's own routine, and the call is the wrapper a served layer uses.
//! Several copies of each weight, together larger than the card's second-level cache, are
//! called in turn so that every call streams its weights from card memory.

use super::stats::gbs;
use crate::tensor::marlin::{
    // CONVERSION: imported for the synthetic repack of a microbench.
    repack_awq_to_marlin,
    MarlinAwqRepacked,
    MARLIN_GROUP_SIZE,
    MARLIN_TILE,
    MAX_M_PER_CHUNK,
    MIN_THREAD_N,
};
use half::f16;

/// The feed-forward length of a layer when `--inter` is absent.
pub const DEFAULT_INTER: usize = 12288;

/// The query heads of a layer when `--heads` is absent.
pub const DEFAULT_HEADS: usize = 32;

/// The key-value heads of a layer when `--kv-heads` is absent.
pub const DEFAULT_KV_HEADS: usize = 8;

/// One projection of a layer: `k` inputs to `n` outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionShape {
    pub name: &'static str,
    pub k: usize,
    pub n: usize,
}

/// The 4 projections of a layer: the fused query, key and value, the output, the fused gate and
/// up, and the down.
pub fn layer_projections(
    hidden: usize,
    inter: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> [ProjectionShape; 4] {
    [
        ProjectionShape {
            name: "qkv",
            k: hidden,
            n: (heads + 2 * kv_heads) * head_dim,
        },
        ProjectionShape {
            name: "o",
            k: heads * head_dim,
            n: hidden,
        },
        ProjectionShape {
            name: "gate+up",
            k: hidden,
            n: 2 * inter,
        },
        ProjectionShape {
            name: "down",
            k: inter,
            n: hidden,
        },
    ]
}

impl ProjectionShape {
    /// `Err` naming the projection when the repack or the kernel cannot take its shape.
    pub fn check(&self, group_size: usize) -> Result<(), String> {
        let bad = |why: String| Err(format!("{} {}x{}: {why}", self.name, self.k, self.n));
        if group_size != MARLIN_GROUP_SIZE {
            return bad(format!(
                "group size {group_size}, only {MARLIN_GROUP_SIZE} runs"
            ));
        }
        if self.k == 0 || !self.k.is_multiple_of(group_size) || !self.k.is_multiple_of(MARLIN_TILE)
        {
            return bad(format!(
                "k is not a multiple of the group size {group_size}"
            ));
        }
        if self.n == 0 || !self.n.is_multiple_of(MIN_THREAD_N) {
            return bad(format!("n is not a multiple of {MIN_THREAD_N}"));
        }
        Ok(())
    }
}

/// The passes over the weights a call of `rows` rows makes: one per tile of the launcher.
pub fn passes(rows: usize) -> usize {
    rows.div_ceil(MAX_M_PER_CHUNK)
}

/// Copies of a weight of `bytes` that together exceed twice the cache of `l2` bytes.
pub fn copies_for(l2: usize, bytes: usize) -> usize {
    (2 * l2).div_ceil(bytes.max(1)).max(1)
}

/// The row counts timed when `--rows-list` is absent.
pub const DEFAULT_ROWS: [usize; 10] = [1, 8, 16, 32, 33, 64, 65, 130, 260, 512];

/// The packed words, zero points and scales of an AWQ linear of fixed-pattern values, in the
/// checkpoint layout: `qweight [k, n / 8]`, `qzeros [k / group_size, n / 8]` and
/// `scales [k / group_size, n]`.
/// Every scale is finite and above zero.
pub fn synthetic_awq_words(
    k: usize,
    n: usize,
    group_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f16>) {
    let (pcols, groups) = (n / 8, k / group_size);
    let word =
        |i: usize, salt: u32| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt) as i32;
    let qweight: Vec<i32> = (0..k * pcols).map(|i| word(i, 1)).collect();
    let qzeros: Vec<i32> = (0..groups * pcols).map(|i| word(i, 7)).collect();
    let scales: Vec<f16> = (0..groups * n)
        .map(|i| f16::from_f32(0.01 + (i % 7) as f32 * 0.001))
        .collect();
    (qweight, qzeros, scales)
}

/// An AWQ linear of fixed-pattern values, repacked by the loader's routine.
/// Every scale is finite and above zero.
pub fn synthetic_awq(k: usize, n: usize, group_size: usize) -> Result<MarlinAwqRepacked, String> {
    let (qweight, qzeros, scales) = synthetic_awq_words(k, n, group_size);
    // CONVERSION: repacks synthetic words of a microbench, no stored weight.
    repack_awq_to_marlin(&qweight, &qzeros, &scales, k, n, group_size).map_err(|e| e.0)
}

/// One measured row count of one projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub rows: usize,
    pub median_us: f64,
}

/// The table of one projection: rows, passes, median time, time per pass and the weight bytes
/// streamed per second.
pub fn projection_table(p: &ProjectionShape, weight_bytes: u64, points: &[Point]) -> String {
    let mut out = format!(
        "{} {}x{}, weights {} B\n{:>6}  {:>6}  {:>11}  {:>11}  {:>8}\n",
        p.name, p.k, p.n, weight_bytes, "rows", "passes", "median us", "us / pass", "GB/s"
    );
    for pt in points {
        let n = passes(pt.rows);
        out.push_str(&format!(
            "{:>6}  {:>6}  {:>11.1}  {:>11.1}  {:>8.1}\n",
            pt.rows,
            n,
            pt.median_us,
            pt.median_us / n as f64,
            gbs(weight_bytes * n as u64, pt.median_us)
        ));
    }
    out
}

/// The time of the whole layer at each row count: the sum over its projections.
/// `per_projection[i][j]` is the median of projection `i` at row count `j`.
pub fn layer_totals(per_projection: &[Vec<f64>]) -> Vec<f64> {
    let len = per_projection.first().map_or(0, Vec::len);
    (0..len)
        .map(|j| per_projection.iter().map(|p| p[j]).sum())
        .collect()
}

/// The time at row count `num` over the time at row count `den`, when both were measured.
pub fn ratio(rows: &[usize], totals: &[f64], num: usize, den: usize) -> Option<f64> {
    let at = |r: usize| rows.iter().position(|&x| x == r).map(|i| totals[i]);
    match (at(num), at(den)) {
        (Some(a), Some(b)) if b > 0.0 => Some(a / b),
        _ => None,
    }
}

/// The final line: the layer time at each row count, then the ratio of the compared rows to one
/// tile.
pub fn layer_line(rows: &[usize], totals: &[f64], compare: usize) -> String {
    let times: Vec<String> = rows
        .iter()
        .zip(totals)
        .map(|(r, t)| format!("m={r} {t:.1} us"))
        .collect();
    let tail = match ratio(rows, totals, compare, MAX_M_PER_CHUNK) {
        Some(x) => format!(
            "m={compare} over m={MAX_M_PER_CHUNK}: {x:.2} (passes {} over {})",
            passes(compare),
            passes(MAX_M_PER_CHUNK)
        ),
        None => format!(
            "m={compare} over m={MAX_M_PER_CHUNK}: not measured, both must be in --rows-list"
        ),
    };
    format!("layer, 4 projections: {}. {tail}", times.join(", "))
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::microbench::stats::{run_for, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::gpu_runtime::driver::sys;
    use crate::tensor::marlin::MarlinAwqLayer;
    use std::time::Duration;

    pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("marlin-rows needs a card")?;
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
        let mut medians: Vec<Vec<f64>> = Vec::new();
        let mut layer_bytes = 0u64;
        for p in &projections {
            let repacked = synthetic_awq(p.k, p.n, o.group_size)?;
            let weight_bytes = repacked.device_bytes() as u64;
            layer_bytes += weight_bytes;
            let layers = (0..copies_for(l2, weight_bytes as usize))
                .map(|_| {
                    MarlinAwqLayer::new(dev, &repacked).map_err(|e| format!("{}: {}", p.name, e.0))
                })
                .collect::<Result<Vec<_>, _>>()?;
            drop(repacked);
            let a_host: Vec<f16> = (0..max_rows * p.k)
                .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
                .collect();
            let a = stream
                .clone_htod(&a_host)
                .map_err(|e| format!("{} input: {e:?}", p.name))?;
            // Safety: the kernel writes the output before anything reads it.
            let mut c = unsafe { stream.alloc::<f16>(max_rows * p.n) }
                .map_err(|e| format!("{} output: {e:?}", p.name))?;

            let mut points = Vec::new();
            for &m in &rows {
                let variant = format!("{} {}x{} m={m} passes={}", p.name, p.k, p.n, passes(m));
                let mut next = 0usize;
                let samples = run_for(target, || {
                    let layer = &layers[next % layers.len()];
                    next += 1;
                    fail.ok(stream.synchronize());
                    let t0 = std::time::Instant::now();
                    fail.ok(layer.forward_into(dev, &a, m, &mut c));
                    fail.ok(stream.synchronize());
                    t0.elapsed().as_secs_f64() * 1e6
                });
                fail.check(&variant)?;
                points.push(Point {
                    rows: m,
                    median_us: Summary::of(&samples).median_us,
                });
                measured.push(Measured {
                    variant,
                    bytes: Some(weight_bytes * passes(m) as u64),
                    samples_us: samples,
                    note: None,
                });
            }
            print!("{}", projection_table(p, weight_bytes, &points));
            println!();
            medians.push(points.iter().map(|pt| pt.median_us).collect());
        }

        let totals = layer_totals(&medians);
        for (&m, &total) in rows.iter().zip(&totals) {
            let note = (m == o.compare_rows)
                .then(|| ratio(&rows, &totals, m, MAX_M_PER_CHUNK))
                .flatten()
                .map(|x| format!("{x:.2} times the layer at m={MAX_M_PER_CHUNK}"));
            measured.push(Measured {
                variant: format!("layer, 4 projections, m={m}, passes={}", passes(m)),
                bytes: Some(layer_bytes * passes(m) as u64),
                samples_us: vec![total],
                note,
            });
        }
        println!("{}\n", layer_line(&rows, &totals, o.compare_rows));
        Ok(measured)
    }
}

#[cfg(feature = "gpu")]
pub use device::run;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tile_of_rows_is_one_pass_and_one_more_row_is_another() {
        assert_eq!(passes(1), 1);
        assert_eq!(passes(MAX_M_PER_CHUNK), 1);
        assert_eq!(passes(MAX_M_PER_CHUNK + 1), 2);
        assert_eq!(passes(130), 130usize.div_ceil(MAX_M_PER_CHUNK));
    }

    #[test]
    fn the_layer_has_4_projections_at_the_given_shapes() {
        let p = layer_projections(4096, 12288, 32, 8, 128);
        assert_eq!((p[0].k, p[0].n), (4096, 6144));
        assert_eq!((p[1].k, p[1].n), (4096, 4096));
        assert_eq!((p[2].k, p[2].n), (4096, 24576));
        assert_eq!((p[3].k, p[3].n), (12288, 4096));
        for s in &p {
            assert!(s.check(MARLIN_GROUP_SIZE).is_ok());
        }
    }

    #[test]
    fn a_shape_the_kernel_cannot_take_is_refused_by_name() {
        let odd = ProjectionShape {
            name: "odd",
            k: 4096,
            n: 100,
        };
        assert!(odd.check(MARLIN_GROUP_SIZE).unwrap_err().contains("odd"));
        assert!(layer_projections(4096, 12288, 32, 8, 128)[0]
            .check(64)
            .is_err());
    }

    #[test]
    fn the_copies_exceed_twice_the_cache() {
        assert_eq!(copies_for(100, 30), 7);
        assert_eq!(copies_for(0, 30), 1);
        assert_eq!(copies_for(100, 1000), 1);
    }

    #[test]
    fn the_synthetic_weight_has_the_loader_sizes_and_finite_scales() {
        let r = synthetic_awq(256, 64, MARLIN_GROUP_SIZE).unwrap();
        assert_eq!(r.b_q.len(), 256 / 16 * 64 * 2);
        assert_eq!(r.scales.len(), 2 * 64);
        assert!(r
            .scales
            .iter()
            .all(|s| s.to_f32().is_finite() && s.to_f32() > 0.0));
        assert_eq!(r.zp.len(), 2 * 64 / 8);
    }

    #[test]
    fn the_layer_time_is_the_sum_and_the_ratio_reads_two_row_counts() {
        let totals = layer_totals(&[vec![1.0, 2.0, 10.0], vec![3.0, 4.0, 20.0]]);
        assert_eq!(totals, vec![4.0, 6.0, 30.0]);
        let rows = [1, MAX_M_PER_CHUNK, 130];
        assert_eq!(ratio(&rows, &totals, 130, MAX_M_PER_CHUNK), Some(5.0));
        assert_eq!(ratio(&rows, &totals, 7, MAX_M_PER_CHUNK), None);
        assert!(layer_line(&rows, &totals, 130).contains("5.00"));
        assert!(layer_line(&rows, &totals, 7).contains("not measured"));
    }

    #[test]
    fn the_table_gives_the_time_per_pass_and_the_streaming_rate() {
        let p = layer_projections(4096, 12288, 32, 8, 128)[0];
        let t = projection_table(
            &p,
            1_000_000_000,
            &[Point {
                rows: 33,
                median_us: 2_000_000.0,
            }],
        );
        assert!(t.contains("qkv 4096x6144"));
        assert!(t.contains("1000000.0"));
        assert!(t.contains("1.0"));
    }
}
