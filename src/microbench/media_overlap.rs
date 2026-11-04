//! Two denoise steps of 2 requests on 1 card: one after the other, on 2 streams, or as 1 step of
//! twice the rows. A step is a stack of dense f16 GEMMs of a transformer block at the shapes
//! given, each variant timed from the launch of both steps to the end of both.
//! The 2 streams belong to 2 devices opened on the card, so they share its context and run
//! together when the card has room for it.

/// How many times faster `other_us` is than `serial_us`: above 1 when it finishes sooner.
pub fn speed_vs(serial_us: f64, other_us: f64) -> f64 {
    if other_us <= 0.0 {
        0.0
    } else {
        serial_us / other_us
    }
}

/// The note of a variant: its median against the 2 steps one after the other.
pub fn note_of(serial_us: f64, other_us: f64) -> String {
    let speed = speed_vs(serial_us, other_us);
    format!(
        "{:.2}x the speed of 2 steps one after the other, {:+.1}%",
        speed,
        (other_us / serial_us - 1.0) * 100.0
    )
}

#[cfg(feature = "gpu")]
pub use gpu::run;

#[cfg(feature = "gpu")]
mod gpu {
    use super::note_of;
    use crate::microbench::stats::{run_for, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::cuda::CudaDevice;
    use crate::tensor::{Device, Tensor};
    use half::f16;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// The projections of one block: fused q, k and v, the output, the feed-forward up and down.
    fn projections(hidden: usize, inter: usize) -> [(usize, usize); 4] {
        [
            (hidden, 3 * hidden),
            (hidden, hidden),
            (hidden, inter),
            (inter, hidden),
        ]
    }

    /// The weights of a stack on one device, and its input rows.
    struct Stack {
        dev: Arc<CudaDevice>,
        weights: Vec<[Tensor; 4]>,
        x: Vec<Tensor>,
    }

    fn weight(device: &Device, k: usize, n: usize) -> Result<Tensor, String> {
        let host: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i % 13) as f32 - 6.0) * 0.002))
            .collect();
        Tensor::from_slice(&host, (k, n), device).map_err(|e| format!("weight: {e:?}"))
    }

    fn rows_of(device: &Device, rows: usize, k: usize) -> Result<Tensor, String> {
        let host: Vec<f16> = (0..rows * k)
            .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        Tensor::from_slice(&host, (rows, k), device).map_err(|e| format!("rows: {e:?}"))
    }

    impl Stack {
        fn new(
            dev: Arc<CudaDevice>,
            layers: usize,
            hidden: usize,
            inter: usize,
            rows: usize,
        ) -> Result<Self, String> {
            let device = Device::Cuda(dev.clone());
            let shapes = projections(hidden, inter);
            let mut weights = Vec::with_capacity(layers);
            for _ in 0..layers {
                weights.push([
                    weight(&device, shapes[0].0, shapes[0].1)?,
                    weight(&device, shapes[1].0, shapes[1].1)?,
                    weight(&device, shapes[2].0, shapes[2].1)?,
                    weight(&device, shapes[3].0, shapes[3].1)?,
                ]);
            }
            let x = [hidden, hidden, hidden, inter]
                .iter()
                .map(|&k| rows_of(&device, rows, k))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self { dev, weights, x })
        }

        /// Queue layer `l` of a step on this device's stream.
        /// The outputs go to `keep`, so none is freed, which would wait for the card, before
        /// the step ends.
        fn layer(&self, l: usize, fail: &Fail, keep: &mut Vec<Tensor>) {
            for (w, x) in self.weights[l].iter().zip(&self.x) {
                if let Some(y) = fail.some(x.matmul(w)) {
                    keep.push(y);
                }
            }
        }

        fn sync(&self, fail: &Fail) {
            fail.ok(self.dev.stream().synchronize());
        }
    }

    pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("media-overlap needs a card")?;
        let o = cx.options;
        let hidden = cx.shapes.hidden;
        let inter = o.inter.unwrap_or(4 * hidden);
        let layers = o.layers;
        if o.rows == 0 || layers == 0 {
            return Err("--rows and --layers need values above zero".to_string());
        }
        let ordinal = card.dev.ordinal();
        let a = Stack::new(card.dev.clone(), layers, hidden, inter, o.rows)?;
        let b = Stack::new(
            CudaDevice::new(ordinal).map_err(|e| format!("second stream: {e:?}"))?,
            layers,
            hidden,
            inter,
            o.rows,
        )?;
        let wide = Stack::new(card.dev.clone(), layers, hidden, inter, 2 * o.rows)?;
        let target = Duration::from_secs_f64(o.seconds);
        let fail = Fail::default();
        let flops = |rows: usize| -> u64 {
            projections(hidden, inter)
                .iter()
                .map(|&(k, n)| 2 * (rows * k * n) as u64)
                .sum::<u64>()
                * layers as u64
        };

        let timed = |variant: &str, body: &mut dyn FnMut()| -> Result<f64, String> {
            let samples = run_for(target, || {
                a.sync(&fail);
                b.sync(&fail);
                let t0 = Instant::now();
                body();
                t0.elapsed().as_secs_f64() * 1e6
            });
            fail.check(variant)?;
            Ok(Summary::of(&samples).median_us)
        };
        let step = |s: &Stack, keep: &mut Vec<Tensor>| {
            for l in 0..layers {
                s.layer(l, &fail, keep);
            }
        };

        let alone = timed("1 step", &mut || {
            let mut keep = Vec::new();
            step(&a, &mut keep);
            a.sync(&fail);
        })?;
        let serial = timed("2 steps one after the other", &mut || {
            let mut keep = Vec::new();
            step(&a, &mut keep);
            step(&a, &mut keep);
            a.sync(&fail);
        })?;
        let two = timed("2 steps on 2 streams", &mut || {
            let (mut keep_a, mut keep_b) = (Vec::new(), Vec::new());
            for l in 0..layers {
                a.layer(l, &fail, &mut keep_a);
                b.layer(l, &fail, &mut keep_b);
            }
            a.sync(&fail);
            b.sync(&fail);
        })?;
        let batched = timed("1 step of 2 requests batched", &mut || {
            let mut keep = Vec::new();
            step(&wide, &mut keep);
            wide.sync(&fail);
        })?;

        let row = |variant: &str, median: f64, flops: u64, versus_serial: bool| {
            let tflops = if median > 0.0 {
                flops as f64 / median / 1e6
            } else {
                0.0
            };
            let note = if versus_serial {
                format!("{}, {tflops:.1} TFLOPS", note_of(serial, median))
            } else {
                format!("{tflops:.1} TFLOPS")
            };
            Measured {
                variant: variant.to_string(),
                bytes: None,
                samples_us: vec![median],
                note: Some(note),
            }
        };
        Ok(vec![
            row("1 step", alone, flops(o.rows), false),
            row(
                "2 steps one after the other",
                serial,
                2 * flops(o.rows),
                false,
            ),
            row("2 steps on 2 streams", two, 2 * flops(o.rows), true),
            row(
                "1 step of 2 requests batched",
                batched,
                flops(2 * o.rows),
                true,
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_variant_that_finishes_sooner_is_above_1() {
        assert!((speed_vs(200.0, 160.0) - 1.25).abs() < 1e-12);
        assert_eq!(speed_vs(200.0, 0.0), 0.0);
    }

    #[test]
    fn the_note_gives_the_speed_and_the_change() {
        let n = note_of(200.0, 100.0);
        assert!(n.contains("2.00x") && n.contains("-50.0%"), "{n}");
    }
}
