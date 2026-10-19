//! Half-precision residency for model weights.
//!
//! A projection matrix or a convolution kernel is held in F16 on its device when every value
//! fits F16's range, and in BF16 when one does not: BF16 has F32's range. Vectors - norms,
//! biases, per-channel scales - stay F32. The GEMMs that read a half weight accumulate in F32.

use crate::tensor::layer::Linear;
use crate::tensor::{DType, Device, Result, Tensor};

/// The dtype a weight matrix whose largest magnitude is `absmax` is held in.
pub fn resident_dtype(absmax: f32) -> DType {
    if absmax.is_finite() && absmax <= half::f16::MAX.to_f32() {
        DType::F16
    } else {
        DType::BF16
    }
}

/// `t` converted to the dtype [`resident_dtype`] picks for its values, on the device it is on.
pub fn to_resident(t: &Tensor) -> Result<Tensor> {
    let absmax = t.to_vec_f32().iter().fold(0f32, |m, x| {
        if x.is_nan() {
            f32::INFINITY
        } else {
            m.max(x.abs())
        }
    });
    t.to_dtype(resident_dtype(absmax))
}

/// A projection from its published `[out, in]` weight, held at its resident dtype on `dev`,
/// with an F32 bias.
pub fn linear(weight: &Tensor, bias: Option<&Tensor>, dev: &Device) -> Result<Linear> {
    let w = to_resident(weight)?.to_device(dev)?;
    let b = bias
        .map(|b| b.to_dtype(DType::F32)?.to_device(dev))
        .transpose()?;
    Linear::from_published(w, b)
}

/// Timed runs of each precision in [`half_matmul_pays`], after one that is not timed; the
/// fastest of them is the one compared.
const MATMUL_PROBE_RUNS: usize = 3;

/// Whether `dev` multiplies `rows` rows by a `[width, width]` projection held in half precision
/// at least as fast as by the same projection held in F32. Measured on the card once per card
/// and shape, and kept; `false` off a card.
pub fn half_matmul_pays(dev: &Device, rows: usize, width: usize) -> Result<bool> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static MEASURED: OnceLock<Mutex<HashMap<(usize, usize, usize), bool>>> = OnceLock::new();
    let crate::tensor::DeviceLocation::Cuda { gpu_id } = dev.location() else {
        return Ok(false);
    };
    let key = (gpu_id, rows, width);
    let measured = MEASURED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&pays) = measured.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(pays);
    }
    let weight = Tensor::zeros((width, width), DType::F32)?;
    let full = Linear::new(weight.to_device(dev)?, None)?;
    let half = linear(&weight, None, dev)?;
    let x = Tensor::zeros_on((1, rows, width), DType::F32, dev)?;
    let fastest = |l: &Linear| -> Result<std::time::Duration> {
        l.forward(&x)?;
        dev.synchronize()?;
        let mut best = std::time::Duration::MAX;
        for _ in 0..MATMUL_PROBE_RUNS {
            let t = std::time::Instant::now();
            l.forward(&x)?;
            dev.synchronize()?;
            best = best.min(t.elapsed());
        }
        Ok(best)
    };
    let pays = fastest(&half)? <= fastest(&full)?;
    measured
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, pays);
    Ok(pays)
}

/// Bytes the tensor `name` of shape `dims` takes once loaded for a card. A projection matrix
/// (`*.weight` with two axes) and a weight-normed kernel direction (`*.weight_v`) are held in
/// half precision; a weight-norm magnitude (`*.weight_g`) is folded into its kernel and takes
/// nothing; every other tensor stays F32.
pub fn card_bytes(name: &str, dims: &[usize]) -> u64 {
    let elems = dims.iter().product::<usize>() as u64;
    let width = |d: DType| d.size_in_bytes() as u64;
    if name.ends_with(".weight_g") {
        0
    } else if name.ends_with(".weight_v") || (name.ends_with(".weight") && dims.len() == 2) {
        elems * width(DType::F16)
    } else {
        elems * width(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_weight_in_range_is_held_f16_and_one_beyond_it_bf16() {
        assert_eq!(resident_dtype(0.0), DType::F16);
        assert_eq!(resident_dtype(65504.0), DType::F16);
        assert_eq!(resident_dtype(7.0e4), DType::BF16);
        assert_eq!(resident_dtype(f32::INFINITY), DType::BF16);
    }

    #[test]
    fn a_bf16_source_out_of_f16_range_stays_bf16() {
        let t = Tensor::from_vec_f32(vec![1.0, -2.0e5], (1usize, 2usize))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(to_resident(&t).unwrap().dtype(), DType::BF16);
        let t = Tensor::from_vec_f32(vec![1.0, -2.0], (1usize, 2usize))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(to_resident(&t).unwrap().dtype(), DType::F16);
    }

    #[test]
    fn a_half_linear_keeps_an_f32_stream_and_its_values() {
        let (o, i) = (5usize, 7usize);
        let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.37).sin()).collect();
        let b: Vec<f32> = (0..o).map(|k| k as f32 * 0.1).collect();
        let x: Vec<f32> = (0..3 * i).map(|k| ((k as f32) * 0.11).cos()).collect();
        let wt = Tensor::from_vec_f32(w.clone(), (o, i)).unwrap();
        let bt = Tensor::from_vec_f32(b.clone(), o).unwrap();
        let lin = linear(&wt, Some(&bt), &Device::Cpu).unwrap();
        assert_eq!(lin.weight().unwrap().dtype(), DType::F16);
        let y = lin
            .forward(&Tensor::from_vec_f32(x.clone(), (3usize, i)).unwrap())
            .unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let y = y.to_vec_f32();
        for r in 0..3 {
            for c in 0..o {
                let want: f32 = b[c] + (0..i).map(|k| x[r * i + k] * w[c * i + k]).sum::<f32>();
                assert!(
                    (y[r * o + c] - want).abs() < 1e-2,
                    "{} vs {want}",
                    y[r * o + c]
                );
            }
        }
    }

    /// A conv and a convT kernel of the shared Oobleck builder: F32 on the host whatever is
    /// asked, F16 on a card when asked, and the F16 kernels give the F32 kernels' output.
    #[test]
    fn oobleck_kernels_are_half_on_a_card_only() {
        use crate::inference::model::acestep::vae::ConvW;
        let (c_out, c_in, k) = (6usize, 4usize, 4usize);
        let w: Vec<f32> = (0..c_out * c_in * k)
            .map(|i| ((i as f32) * 0.21).sin())
            .collect();
        let b: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.05).collect();
        let build = |convt: bool, half: bool, dev: &Device| {
            ConvW::from_fused(w.clone(), (c_out, c_in, k), b.clone(), convt, half, dev).unwrap()
        };
        assert_eq!(build(false, true, &Device::Cpu).wt.dtype(), DType::F32);
        assert_eq!(build(true, true, &Device::Cpu).wt.dtype(), DType::F32);
        let Ok(dev) = Device::new_cuda(0) else { return };
        let t = 9usize;
        let x: Vec<f32> = (0..c_in * t).map(|i| ((i as f32) * 0.17).cos()).collect();
        let x = Tensor::from_vec_f32(x, (1usize, c_in, t))
            .unwrap()
            .to_device(&dev)
            .unwrap();
        for convt in [false, true] {
            let (h, f) = (build(convt, true, &dev), build(convt, false, &dev));
            assert_eq!(h.wt.dtype(), DType::F16);
            assert_eq!(f.wt.dtype(), DType::F32);
            let run = |c: &ConvW| {
                if convt {
                    c.conv_t(&x, 1, 2).unwrap().to_vec_f32()
                } else {
                    c.conv(&x, 1, 1).unwrap().to_vec_f32()
                }
            };
            let (yh, yf) = (run(&h), run(&f));
            assert_eq!(yh.len(), yf.len());
            for (a, b) in yh.iter().zip(&yf) {
                assert!((a - b).abs() < 1e-2, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn card_bytes_halve_matrices_and_kernels_and_drop_magnitudes() {
        assert_eq!(card_bytes("blk.proj.weight", &[4, 8]), 64);
        assert_eq!(card_bytes("blk.conv.weight_v", &[4, 8, 3]), 192);
        assert_eq!(card_bytes("blk.conv.weight_g", &[4, 1, 1]), 0);
        assert_eq!(card_bytes("blk.norm.weight", &[8]), 32);
        assert_eq!(card_bytes("blk.conv.weight", &[4, 8, 1]), 128);
        assert_eq!(card_bytes("blk.table", &[6, 8]), 192);
    }
}

#[cfg(all(test, feature = "gpu", feature = "audio"))]
mod probes {
    //! Hardware probes: real checkpoints and the first CUDA card. Each renders one fixed
    //! seed, prints where every part sits with its bytes and the card memory in use, and
    //! leaves the audio in `results/half/` for [`compare_with_reference`]. Run one by name:
    //!   cargo test --release --lib NAME -- --ignored --nocapture --test-threads=1
    use crate::inference::serve::progress::placement;
    use crate::tensor::Device;
    use std::sync::{Arc, Mutex};

    const OUT_DIR: &str = "results/half";

    /// Bytes in use on the first card, or none when there is no card.
    fn card_used() -> Option<u64> {
        let dev = Device::new_cuda(0).ok()?;
        let (free, total) = crate::tensor::cuda_ext::mem_get_info(&dev).ok()?;
        Some((total - free) as u64)
    }

    /// Runs `render` with a placement reporter that prints every part as it is placed,
    /// with the card memory in use at that moment.
    fn watched<T>(render: impl FnOnce() -> T) -> T {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = seen.clone();
        let report: placement::SharedPlacementFn = Arc::new(move |part, runs| {
            if runs.is_empty() {
                return;
            }
            let bytes: u64 = runs.iter().map(|r| r.bytes).sum();
            let line = format!(
                "part {part}: {:?} bytes {bytes} card used {:?}",
                runs.iter()
                    .map(|r| (r.device, r.layer_start, r.layer_end))
                    .collect::<Vec<_>>(),
                card_used()
            );
            eprintln!("{line}");
            log.lock().unwrap().push(line);
        });
        let _scope = placement::publish(report);
        render()
    }

    fn save(name: &str, pcm: &[f32]) {
        std::fs::create_dir_all(OUT_DIR).unwrap();
        let bytes: Vec<u8> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{OUT_DIR}/{name}.f32"), bytes).unwrap();
    }

    fn load(path: &str) -> Option<Vec<f32>> {
        let raw = std::fs::read(path).ok()?;
        Some(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    }

    #[test]
    #[ignore = "hardware probe: needs the stable-audio weights and a CUDA card"]
    fn stable_audio_render_probe() {
        use crate::inference::model::stable_audio as sa;
        if sa::model_dir().is_none() {
            return;
        }
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .try_init();
        eprintln!("card used before: {:?}", card_used());
        let t0 = std::time::Instant::now();
        let (pcm, _) = watched(|| {
            sa::render(
                "gentle rain on a tin roof",
                "",
                6.0,
                8,
                7.0,
                42,
                |ph, i, n| {
                    eprintln!("{ph} {i}/{n} at {:?}", t0.elapsed());
                    Ok(())
                },
            )
        })
        .unwrap();
        let cold = t0.elapsed();
        eprintln!("card used resident: {:?}", card_used());
        let t1 = std::time::Instant::now();
        let (warm_pcm, _) = sa::render(
            "gentle rain on a tin roof",
            "",
            6.0,
            8,
            7.0,
            42,
            |_, _, _| Ok(()),
        )
        .unwrap();
        let warm = t1.elapsed();
        eprintln!("stable-audio cold {cold:?} warm {warm:?}");
        if let Some(parts) = sa::resident_parts() {
            for (name, runs) in parts {
                let bytes: u64 = runs.iter().map(|r| r.bytes).sum();
                eprintln!(
                    "resident {name}: {bytes} bytes on {:?}",
                    runs.first().map(|r| r.device)
                );
            }
        }
        assert_eq!(pcm.len(), warm_pcm.len());
        save("stable_audio", &pcm);
    }

    #[test]
    #[ignore = "hardware probe: needs the EzAudio weights and a CUDA card"]
    fn ezaudio_render_probe() {
        let t0 = std::time::Instant::now();
        let (pcm, _) = watched(|| {
            crate::inference::model::ezaudio::pipeline::render(
                "a dog barking in the distance",
                4.0,
                10,
                3.0,
                42,
                1.0,
                "",
            )
        })
        .unwrap();
        eprintln!("ezaudio render {:?}", t0.elapsed());
        save("ezaudio", &pcm);
    }

    #[test]
    #[ignore = "hardware probe: needs the ACE-Step weights and a CUDA card"]
    fn acestep_render_probe() {
        std::fs::create_dir_all(OUT_DIR).unwrap();
        let wav = format!("{OUT_DIR}/acestep.wav");
        let cfg: crate::inference::model::acestep::music::MusicConfig =
            serde_json::from_value(serde_json::json!({
                "caption": "calm acoustic guitar instrumental",
                "lyrics": "[instrumental]",
                "seed": 42,
                "codes_per_section": 50,
                "out": wav,
            }))
            .unwrap();
        let t0 = std::time::Instant::now();
        watched(|| crate::inference::model::acestep::music::render(&cfg)).unwrap();
        eprintln!("acestep render {:?}", t0.elapsed());
        let bytes = std::fs::read(&wav).unwrap();
        let (planar, _, _) =
            crate::inference::model::acestep::vae::decode_wav_s16le(&bytes).unwrap();
        save("acestep", &planar);
    }

    /// Correlation and signal-to-error ratio of each family's render against the
    /// reference render left beside it as `<family>.ref.f32`.
    #[test]
    #[ignore = "reads the renders the probes above left in results/half"]
    fn compare_with_reference() {
        for family in ["stable_audio", "ezaudio", "acestep"] {
            let (Some(ours), Some(reference)) = (
                load(&format!("{OUT_DIR}/{family}.f32")),
                load(&format!("{OUT_DIR}/{family}.ref.f32")),
            ) else {
                continue;
            };
            let n = ours.len().min(reference.len());
            let (o, r) = (&ours[..n], &reference[..n]);
            let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let (mo, mr) = (mean(o), mean(r));
            let (mut cov, mut vo, mut vr, mut err) = (0f64, 0f64, 0f64, 0f64);
            for (a, b) in o.iter().zip(r) {
                let (da, db) = (*a as f64 - mo, *b as f64 - mr);
                cov += da * db;
                vo += da * da;
                vr += db * db;
                err += (*a as f64 - *b as f64).powi(2);
            }
            let corr = cov / (vo * vr).sqrt().max(1e-30);
            let sig: f64 = r.iter().map(|x| (*x as f64).powi(2)).sum();
            let snr = 10.0 * (sig / err.max(1e-30)).log10();
            eprintln!(
                "{family}: {n} samples, correlation {corr:.5}, snr {snr:.2} dB, lengths {} / {}",
                ours.len(),
                reference.len()
            );
        }
    }
}
