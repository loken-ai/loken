//! One decode step of a Parler-TTS decoder at given shapes, its projections held in F32 and in
//! half precision. The step is the whole stack: the embeddings, every layer, the norm and the
//! heads, and the logits row of each codebook read back to the host, as a generation reads them.

#[cfg(all(feature = "gpu", feature = "audio"))]
pub use gpu::run;

#[cfg(all(feature = "gpu", feature = "audio"))]
mod gpu {
    use crate::inference::model::parler::synthetic::{self, Conditioning, Shape};
    use crate::inference::model::parler::Decoder;
    use crate::microbench::stats::{gbs, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::Device;
    use std::time::{Duration, Instant};

    /// Steps timed together in one sample, after a prompt pass that is not timed.
    const STEPS_PER_SAMPLE: usize = 16;

    /// The cross attention reads this many encoder positions.
    const ENCODER_POSITIONS: usize = 32;

    pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("parler-step needs a card")?;
        let o = cx.options;
        let heads = o.heads.ok_or("parler-step needs --heads")?;
        let shape = Shape {
            hidden: cx.shapes.hidden,
            ffn: o.inter.ok_or("parler-step needs --inter")?,
            layers: o.layers,
            heads,
            codebooks: o.codebooks,
            vocab: o.vocab,
            positions: o.prompt_tokens + STEPS_PER_SAMPLE + 1,
        };
        if shape.hidden % heads != 0 || shape.layers == 0 || shape.codebooks == 0 {
            return Err(
                "parler-step: --hidden must divide by --heads, layers and codebooks above 0".into(),
            );
        }
        let device = Device::Cuda(card.dev.clone());
        let cond = Conditioning::new(&shape, o.prompt_tokens, ENCODER_POSITIONS, &device)
            .map_err(|e| format!("conditioning: {e:?}"))?;
        let target = Duration::from_secs_f64(o.seconds);
        let fail = Fail::default();
        let mut out = Vec::new();
        for (variant, half, element) in [("F32 weights", false, 4), ("F16 weights", true, 2)] {
            let mut dec = synthetic::decoder(&shape, half, &device)
                .map_err(|e| format!("{variant}: build: {e:?}"))?;
            let tokens = vec![0u32; shape.codebooks];
            let mut sample = |dec: &mut Decoder| -> f64 {
                dec.clear_kv_cache();
                fail.ok(synthetic::step(dec, &cond, &tokens, 0, &device).map(|_| ()));
                let t = Instant::now();
                for n in 1..=STEPS_PER_SAMPLE {
                    fail.ok(synthetic::step(dec, &cond, &tokens, n, &device).map(|_| ()));
                }
                t.elapsed().as_secs_f64() * 1e6 / STEPS_PER_SAMPLE as f64
            };
            let warm = Instant::now();
            while warm.elapsed() * 10 < target {
                sample(&mut dec);
            }
            let mut samples = Vec::new();
            let timed = Instant::now();
            while timed.elapsed() < target || samples.is_empty() {
                samples.push(sample(&mut dec));
            }
            fail.check(variant)?;
            let median = Summary::of(&samples).median_us;
            let bytes = shape.step_bytes(element);
            out.push(Measured {
                variant: variant.to_string(),
                bytes: Some(bytes),
                samples_us: samples,
                note: Some(format!(
                    "per step, {:.1} GB/s over the projections",
                    gbs(bytes, median)
                )),
            });
        }
        Ok(out)
    }
}
