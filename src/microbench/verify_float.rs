//! The rows of a draft verify, and a decode step, through a float weight on a card.
//! The weights are an F32 and an F16 projection of `--hidden` by `--hidden`, the F16 up and down
//! projections of `--inter` by `--hidden` and `--hidden` by `--inter`, and the float head held
//! in F16, `--vocab` by `--hidden`.
//! The row counts are the values of `--block`, 1 variant each.
//!
//! The variants of each weight and row count:
//! - `library`: the product the card ran before the float mat-vec kernel.
//!   1 row took the decode kernel: the F16-weight mat-vec for an F16 projection, the library's
//!   product for the others. More rows took the library's product.
//! - `library row by row`: that 1-row product for each row.
//! - `mat-vec`: the projection's forward, or the head's product, under the decode order a
//!   verify holds: the float mat-vec kernel for every row at once.
//!
//! Several copies of each weight, together larger than twice the card's second-level cache, are
//! called in turn so that every call reads its weights from card memory.
//! Each half weight first measures the variants of the float product, as a model's warm-up does.

/// The variants of a weight at a row count, in the order the table and the record list them.
pub const VARIANTS: [&str; 3] = ["library", "library row by row", "mat-vec"];

/// One measured row count of one weight: the median of each variant, in microseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub rows: usize,
    pub us: [f64; 3],
}

/// The time of `new_us` over the time of `old_us`: above 1 when the new variant is slower.
/// 0 for a time at or below 0.
pub fn cost_vs(old_us: f64, new_us: f64) -> f64 {
    if old_us <= 0.0 || new_us <= 0.0 {
        return 0.0;
    }
    new_us / old_us
}

/// The table of one weight: rows, the median of each variant and the time of the mat-vec over
/// the time of the library's product.
pub fn weight_table(name: &str, points: &[Point]) -> String {
    let mut out = format!(
        "{name}\n{:>6}  {:>12}  {:>20}  {:>12}  {:>8}\n",
        "rows", VARIANTS[0], VARIANTS[1], VARIANTS[2], "ratio"
    );
    for pt in points {
        out.push_str(&format!(
            "{:>6}  {:>12.1}  {:>20.1}  {:>12.1}  {:>8.2}\n",
            pt.rows,
            pt.us[0],
            pt.us[1],
            pt.us[2],
            cost_vs(pt.us[0], pt.us[2])
        ));
    }
    out
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::microbench::marlin_rows::{copies_for, DEFAULT_INTER};
    use crate::microbench::stats::{run_for, timed, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::gpu_runtime::driver::sys;
    use crate::tensor::quantized::{half_head, DecodeOrder, QMatMul};
    use crate::tensor::{DType, Device, Result, Tensor};
    use std::time::Duration;

    /// How a weight is held and multiplied.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        F32Projection,
        F16Projection,
        Head,
    }

    /// A weight as the server holds it.
    struct Held {
        kind: Kind,
        w: Tensor,
        qmm: QMatMul,
    }

    impl Held {
        /// The product of the rows `x` the card ran before the float mat-vec kernel.
        fn library(&self, x: &Tensor) -> Result<Tensor> {
            let rows = x.dim(0)?;
            match self.kind {
                Kind::F32Projection => x.matmul_t(&self.w),
                Kind::Head => x.to_dtype(DType::F16)?.matmul_t(&self.w),
                Kind::F16Projection => {
                    // A launch the F16-weight mat-vec fails falls to the library's product.
                    if let (1, Ok(Some(y))) = (rows, x.gemv_f16w(&self.w, None)) {
                        return Ok(y);
                    }
                    let y = x.to_dtype(DType::F16)?.matmul_t_f16_f32(&self.w)?;
                    y.ok_or_else(|| crate::tensor::Error::msg("no F16 library product"))
                }
            }
        }

        /// The product of the rows `x` as the server makes it now.
        fn served(&self, x: &Tensor) -> Result<Tensor> {
            match self.kind {
                Kind::Head => half_head(x, &self.w),
                _ => self.qmm.forward(x),
            }
        }
    }

    pub fn run(cx: &Context) -> std::result::Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("verify-float needs a card")?;
        let o = cx.options;
        if o.block.contains(&0) {
            return Err("--block needs row counts above zero".to_string());
        }
        let (hidden, vocab) = (cx.shapes.hidden, o.vocab);
        let inter = o.inter.unwrap_or(DEFAULT_INTER);
        let dev = Device::Cuda(card.dev.clone());
        let tensor = |e: crate::tensor::Error| format!("{e:?}");
        let l2 = card
            .dev
            .context()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .map_err(|e| format!("card cache size: {e:?}"))?
            .max(0) as usize;
        // A weight `[n, k]` of `dtype` on the card, filled a block of `k` rows at a time.
        let weight = |n: usize, k: usize, dtype: DType| -> std::result::Result<Tensor, String> {
            let w = Tensor::zeros_on((n, k), dtype, &dev).map_err(tensor)?;
            let mut at = 0;
            while at < n {
                let m = k.min(n - at);
                let part = Tensor::randn(0f32, 1.0, (m, k), &dev)
                    .and_then(|t| t.to_dtype(dtype))
                    .map_err(tensor)?;
                w.slice_set(&part, 0, at).map_err(tensor)?;
                at += m;
            }
            Ok(w)
        };
        let shapes = [
            ("F32 projection", hidden, hidden, Kind::F32Projection),
            ("F16 projection", hidden, hidden, Kind::F16Projection),
            ("F16 up projection", inter, hidden, Kind::F16Projection),
            ("F16 down projection", hidden, inter, Kind::F16Projection),
            ("F16 head", vocab, hidden, Kind::Head),
        ];
        let target = Duration::from_secs_f64(o.seconds);
        let fail = Fail::default();
        let mut measured = Vec::new();
        for (what, n, k, kind) in shapes {
            let name = format!("{what} {n}x{k}");
            let dtype = match kind {
                Kind::F32Projection => DType::F32,
                _ => DType::F16,
            };
            let bytes = (n * k * dtype.size_in_bytes()) as u64;
            let copies = (0..copies_for(l2, bytes as usize))
                .map(|_| {
                    let w = weight(n, k, dtype)?;
                    let qmm = match kind {
                        Kind::F32Projection => QMatMul::Tensor(w.clone()),
                        _ => QMatMul::TensorF16(w.clone()),
                    };
                    Ok(Held { kind, w, qmm })
                })
                .collect::<std::result::Result<Vec<_>, String>>()?;
            // Whether the F16-weight mat-vec takes 1 row of this weight.
            let gemv_note = match kind {
                Kind::F16Projection => {
                    let x = Tensor::randn(0f32, 1.0, (1, k), &dev).map_err(tensor)?;
                    match x.gemv_f16w(&copies[0].w, None) {
                        Ok(Some(_)) => None,
                        Ok(None) => Some("the F16-weight mat-vec declines 1 row".to_string()),
                        Err(e) => Some(format!("the F16-weight mat-vec fails 1 row: {e:?}")),
                    }
                }
                _ => None,
            };
            if let Some(note) = &gemv_note {
                println!("{name}: {note}, the library row variants take the library's product");
            }
            // The server measures the variants of the float product while a model warms up.
            if kind != Kind::F32Projection {
                let _choose = crate::inference::moe_cuda::ChooseFloatVariants::enter();
                let _order = DecodeOrder::enter();
                let x = Tensor::randn(0f32, 1.0, (1, k), &dev).map_err(tensor)?;
                copies[0].served(&x).map_err(tensor)?;
                if let Some(v) = crate::inference::moe_cuda::measured_float_variant(&copies[0].w) {
                    println!(
                        "{name}: the float mat-vec keeps variant {} for 1 row, {} up to 8, {} for more",
                        v.one, v.group, v.more
                    );
                }
            }
            let mut points = Vec::new();
            for &m in &o.block {
                let x = Tensor::randn(0f32, 1.0, (m, k), &dev).map_err(tensor)?;
                let rows: Vec<Tensor> = (0..m)
                    .map(|r| x.narrow(0, r, 1).and_then(|t| t.force_contiguous()))
                    .collect::<Result<_>>()
                    .map_err(tensor)?;
                let mut us = [0f64; 3];
                for (v, variant_name) in VARIANTS.iter().enumerate() {
                    let variant = format!("{name} m={m} {variant_name}");
                    let mut next = 0usize;
                    let samples = run_for(
                        target,
                        timed(|| {
                            let w = &copies[next % copies.len()];
                            next += 1;
                            let y = match v {
                                0 => fail.some(w.library(&x)).map(|y| vec![y]),
                                1 => fail.some(rows.iter().map(|r| w.library(r)).collect()),
                                _ => {
                                    let _order = DecodeOrder::enter();
                                    fail.some(w.served(&x)).map(|y| vec![y])
                                }
                            };
                            fail.ok(dev.synchronize());
                            drop(y);
                        }),
                    );
                    fail.check(&variant)?;
                    us[v] = Summary::of(&samples).median_us;
                    let reads = if v == 1 { m as u64 } else { 1 };
                    measured.push(Measured {
                        variant,
                        bytes: Some(bytes * reads),
                        samples_us: samples,
                        note: gemv_note.clone().filter(|_| v < 2),
                    });
                }
                points.push(Point { rows: m, us });
            }
            print!("{}", weight_table(&name, &points));
            println!();
        }
        Ok(measured)
    }
}

#[cfg(feature = "gpu")]
pub use device::run;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ratio_is_above_1_when_the_new_variant_is_slower() {
        assert_eq!(cost_vs(100.0, 200.0), 2.0);
        assert_eq!(cost_vs(200.0, 100.0), 0.5);
        assert_eq!(cost_vs(0.0, 100.0), 0.0);
        assert_eq!(cost_vs(100.0, 0.0), 0.0);
    }

    #[test]
    fn the_table_gives_each_row_count_and_its_ratio() {
        let t = weight_table(
            "F16 head 151936x4096",
            &[Point {
                rows: 4,
                us: [100.0, 400.0, 250.0],
            }],
        );
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("F16 head"));
        assert!(lines[2].contains("400.0") && lines[2].ends_with("2.50"));
    }
}
