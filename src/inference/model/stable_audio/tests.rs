//! The ignored cases here need real weights, a device, or a reference dump on
//! this machine; nothing about them is automatic. Run one by name with
//!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
use super::*;

/// Where the torch mirror left its dumps.
fn ref_dir() -> String {
    std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR")
}

/// One dump, read as the f32 values torch wrote: little-endian, no header, no shape.
///
/// The shape is not in the file, so every caller states the one it expects and the length
/// check below is what catches a disagreement - see [`Divergence::of`].
fn read_dump(dir: &str, name: &str) -> Vec<f32> {
    std::fs::read(format!("{dir}/{name}"))
        .expect(name)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The checkpoint every stage of the pipeline is loaded from.
fn ckpt() -> String {
    format!(
        "{}/stable-audio/model.safetensors",
        crate::inference::cache::hf::models_dir()
    )
}

/// Oracle dumps are channel-major `[C, T]`; every forward here takes `[T, C]`.
fn to_tc(channel_major: Vec<f32>, channels: usize, frames: usize) -> Tensor {
    let ct = Tensor::from_vec_f32(channel_major, (channels, frames)).unwrap();
    ct.transpose(0, 1).unwrap().contiguous().unwrap()
}

/// And back into the layout the dump is in, so the comparison is against torch's own order.
fn from_tc(x: &Tensor) -> Vec<f32> {
    let ct = x.transpose(0, 1).unwrap().contiguous().unwrap();
    ct.to_vec_f32()
}

/// Root-mean-square level of a clip: what tells rendered audio from silence.
fn rms(pcm: &[f32]) -> f64 {
    (pcm.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / pcm.len().max(1) as f64).sqrt()
}

/// How far an output sits from its reference, by every measure these tests judge on.
///
/// Scaffolding, not judgement: it measures and never decides. Each test states its own
/// thresholds next to the reason for them - the decoder's SnakeBeta stages amplify benign f32
/// reordering noise and the encoder's do not, so one shared verdict would have to be the
/// loosest of them and would stop failing for the stages that are actually tight.
struct Divergence {
    max_abs: f64,
    max_rel: f64,
    /// The worst relative gap among the samples that are ALSO absolutely off by more than
    /// `mixed_floor`. A reference value near zero cannot carry a relative bound on its own,
    /// and a sample that is off in only one of the two senses is noise, not divergence.
    worst_mixed: f64,
    /// Signal power over error power, in dB: the aggregate a per-sample maximum cannot see.
    snr_db: f64,
}

impl Divergence {
    /// `rel_floor` bounds the denominator of the relative gap; `mixed_floor` is the absolute
    /// gap a sample must also exceed before it counts towards `worst_mixed`. Pass
    /// `f64::INFINITY` for the latter when a test does not use that criterion.
    fn of(ours: &[f32], reference: &[f32], rel_floor: f64, mixed_floor: f64) -> Self {
        assert_eq!(ours.len(), reference.len(), "shape mismatch");
        let (mut max_abs, mut max_rel, mut worst_mixed) = (0f64, 0f64, 0f64);
        let (mut err_pow, mut sig_pow) = (0f64, 0f64);
        for (o, r) in ours.iter().zip(reference) {
            let d = (*o as f64 - *r as f64).abs();
            let rel = d / (r.abs() as f64).max(rel_floor);
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(rel);
            if d > mixed_floor {
                worst_mixed = worst_mixed.max(rel);
            }
            err_pow += d * d;
            sig_pow += (*r as f64) * (*r as f64);
        }
        Self {
            max_abs,
            max_rel,
            worst_mixed,
            snr_db: 10.0 * (sig_pow / err_pow.max(1e-30)).log10(),
        }
    }
}

/// t5-base conditioner parity vs the torch mirror (oracle_t5.py): tokenizer
/// ids must match exactly, encoder output within f32 noise.
#[test]
#[ignore]
fn t5_base_encoder_matches_torch() {
    use crate::inference::model::t5::flan::{t5_tokenize, T5Encoder};
    let dir = ref_dir();
    let t5_dir = format!(
        "{}/stable-audio/t5-base",
        crate::inference::cache::hf::models_dir()
    );
    // The ids are int32, not float, so they are read here rather than through `read_dump`.
    let ref_ids: Vec<u32> = std::fs::read(format!("{dir}/t5_ids.i32"))
        .expect("t5_ids.i32")
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
        .collect();
    let reference = read_dump(&dir, "t5_ref_out.f32");
    let prompt = "128 BPM tech house drum loop with deep bass and crisp hats";
    let ids = t5_tokenize(&t5_dir, prompt).unwrap();
    assert_eq!(ids, ref_ids, "tokenizer ids diverge from torch");
    let enc = T5Encoder::from_dir(&t5_dir).unwrap();
    let out = enc.encode(&ids).unwrap().to_vec_f32();
    let d = Divergence::of(&out, &reference, 1e-3, f64::INFINITY);
    eprintln!(
        "t5-base parity: max_rel {:.3e} | max_abs {:.3e}",
        d.max_rel, d.max_abs
    );
    assert!(
        d.max_rel < 1e-3,
        "t5-base encoder diverges: {:.3e}",
        d.max_rel
    );
}

/// DiT parity vs the torch mirror (oracle_dit.py): fixed latent [64,64],
/// t=0.5, the t5 embeds + number-conditioner embeds as cross tokens.
#[test]
#[ignore]
fn dit_matches_torch() {
    let dir = ref_dir();
    let ckpt = ckpt();
    let dev = Device::Cpu;

    // Number conditioners must match the oracle's dumps exactly first.
    for (which, file, val) in [
        ("seconds_start", "dit_sec_start.f32", 0.0f32),
        ("seconds_total", "dit_sec_total.f32", 30.0f32),
    ] {
        let nc = SaoNumberConditioner::load(&ckpt, which, &dev).unwrap();
        let ours = nc.forward(val).unwrap().to_vec_f32();
        let d = Divergence::of(&ours, &read_dump(&dir, file), 1e-3, f64::INFINITY);
        eprintln!("{which} embed max_abs = {:.3e}", d.max_abs);
        assert!(
            d.max_abs < 1e-5,
            "{which} conditioner diverges: {:.3e}",
            d.max_abs
        );
    }

    let dit = load_dit(&ckpt, &dev).expect("load dit");
    let t_len = 64usize;
    let x = to_tc(read_dump(&dir, "dit_in_64x64.f32"), 64, t_len);
    let cross_v = read_dump(&dir, "dit_cross.f32");
    let s = cross_v.len() / DIT_COND_DIM;
    let cross = Tensor::from_vec_f32(cross_v, (s, DIT_COND_DIM)).unwrap();
    let glob = Tensor::from_vec_f32(read_dump(&dir, "dit_global.f32"), (1usize, DIT_DIM)).unwrap();
    let out = from_tc(&dit.forward(&x, 0.5, &cross, &glob).unwrap());
    let reference = read_dump(&dir, "dit_ref_out.f32");
    let d = Divergence::of(&out, &reference, 1e-3, f64::INFINITY);
    eprintln!(
        "dit parity: max_rel {:.3e} | max_abs {:.3e} | SNR {:.1} dB",
        d.max_rel, d.max_abs, d.snr_db
    );
    assert!(d.snr_db > 70.0, "dit diverges: SNR {:.1} dB", d.snr_db);
}

/// End-to-end sampler parity: real DiT + VDenoiser + polyexponential
/// sigmas + DPM++(2M), 8 deterministic steps (oracle_sampler.py).
#[test]
#[ignore]
fn sampler_matches_torch() {
    let dir = ref_dir();
    let dit = load_dit(&ckpt(), &Device::Cpu).expect("load dit");
    let t_len = 64usize;
    let noise = to_tc(read_dump(&dir, "smp_noise_64x64.f32"), 64, t_len);
    let cross_v = read_dump(&dir, "dit_cross.f32");
    let s = cross_v.len() / DIT_COND_DIM;
    let cross = Tensor::from_vec_f32(cross_v, (s, DIT_COND_DIM)).unwrap();
    let glob = Tensor::from_vec_f32(read_dump(&dir, "dit_global.f32"), (1usize, DIT_DIM)).unwrap();

    let sigmas = sigmas_polyexponential(8, 0.3, 500.0, 1.0);
    let ref_sigmas = read_dump(&dir, "smp_sigmas.f32");
    for (a, b) in sigmas.iter().zip(&ref_sigmas) {
        assert!(
            (a - b).abs() <= 1e-3 * b.abs().max(1.0),
            "sigma schedule diverges: {a} vs {b}"
        );
    }
    let model = |x: &Tensor, t: f32| -> Result<Tensor> { dit.forward(x, t, &cross, &glob) };
    let out = sample_dpmpp_2m(&model, &noise, &sigmas, |_ph, i, n| {
        eprintln!("  step {i}/{n}");
        Ok(())
    })
    .unwrap();
    let out = from_tc(&out);
    let reference = read_dump(&dir, "smp_ref_out.f32");
    let d = Divergence::of(&out, &reference, 1e-3, f64::INFINITY);
    eprintln!(
        "sampler parity: max_abs {:.3e} | SNR {:.1} dB",
        d.max_abs, d.snr_db
    );
    assert!(d.snr_db > 60.0, "sampler diverges: SNR {:.1} dB", d.snr_db);
}

/// End-to-end smoke: render a short clip through the full pipeline
/// (t5 -> conditioners -> DiT+CFG -> DPM++ -> Oobleck decode) and check
/// the audio is non-degenerate. Writes the WAV next to SAO_REF_DIR.
#[test]
#[ignore]
fn render_smoke() {
    let dir = std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR");
    let t_cold = std::time::Instant::now();
    let (pcm, sr) = render(
        "128 BPM tech house drum loop with deep bass and crisp hats",
        "",
        10.0,
        25,
        7.0,
        42,
        |_ph, i, n| {
            eprintln!("  step {i}/{n}");
            Ok(())
        },
    )
    .expect("render");
    assert_eq!(sr, SAMPLE_RATE);
    assert!(pcm.iter().all(|x| x.is_finite()), "non-finite samples");
    let n = pcm.len();
    assert_eq!(n, (10.0 * SAMPLE_RATE as f64) as usize * 2, "length");
    let rms = (pcm.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).sqrt();
    let peak = pcm.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!(
        "render smoke: rms {rms:.4} peak {peak:.4} cold {:?}",
        t_cold.elapsed()
    );
    assert!(rms > 0.01, "audio is near-silent (rms {rms:.5})");
    // Warm request: the resident cache must make a second render cheap.
    let t_warm = std::time::Instant::now();
    let (pcm2, _) = render(
        "gentle rain on a tin roof",
        "",
        5.0,
        10,
        7.0,
        7,
        |_, _, _| Ok(()),
    )
    .expect("warm render");
    eprintln!("warm render (5 s, 10 steps): {:?}", t_warm.elapsed());
    assert!(pcm2.iter().all(|x| x.is_finite()));
    // Persist for listening.
    let mut wav: Vec<u8> = Vec::with_capacity(44 + n * 2);
    let (data_bytes, byte_rate) = ((n * 2) as u32, sr * 4);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&sr.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&4u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for s in &pcm {
        wav.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(format!("{dir}/render_smoke.wav"), wav).unwrap();
}

/// Wall-time probe: one DiT forward at the render's working size, on the
/// pipeline's chosen device. Diagnoses where the ~4 s/step goes.
#[test]
#[ignore]
fn dit_perf_probe() {
    let ckpt = format!(
        "{}/stable-audio/model.safetensors",
        crate::inference::cache::hf::models_dir()
    );
    let sz = std::fs::metadata(&ckpt).map(|m| m.len()).unwrap_or(5 << 30);
    let device = crate::inference::model::acestep::vae::vae_best_device(sz);
    eprintln!("device: {device:?}");
    let t0 = std::time::Instant::now();
    let dit = load_dit(&ckpt, &device).unwrap();
    eprintln!("load: {:?}", t0.elapsed());
    let t_len = 216usize;
    let x = Tensor::from_vec_f32(vec![0.1f32; t_len * 64], (t_len, 64usize))
        .unwrap()
        .to_device(&device)
        .unwrap();
    let cross = Tensor::from_vec_f32(vec![0.05f32; 19 * DIT_COND_DIM], (19usize, DIT_COND_DIM))
        .unwrap()
        .to_device(&device)
        .unwrap();
    let glob = Tensor::from_vec_f32(vec![0.02f32; DIT_DIM], (1usize, DIT_DIM))
        .unwrap()
        .to_device(&device)
        .unwrap();
    for i in 0..4 {
        let t1 = std::time::Instant::now();
        let out = dit.forward(&x, 0.5, &cross, &glob).unwrap();
        let _ = out.to_vec_f32(); // force sync
        eprintln!("forward {i}: {:?}", t1.elapsed());
    }
}

/// Wall-time probe: Oobleck decode of a 10 s latent on CPU.
#[test]
#[ignore]
fn decoder_perf_probe() {
    let ckpt = format!(
        "{}/stable-audio/model.safetensors",
        crate::inference::cache::hf::models_dir()
    );
    let sz = std::fs::metadata(&ckpt).map(|m| m.len()).unwrap_or(5 << 30);
    let device = crate::inference::model::acestep::vae::vae_best_device(sz);
    eprintln!("device: {device:?}");
    let t0 = std::time::Instant::now();
    let dec = load_decoder_on(&ckpt, "pretransform.model.decoder", &device).unwrap();
    eprintln!("load: {:?}", t0.elapsed());
    let t_len = 216usize;
    let lat = Tensor::from_vec_f32(vec![0.1f32; 64 * t_len], (64usize, t_len))
        .unwrap()
        .to_device(&device)
        .unwrap();
    let t1 = std::time::Instant::now();
    let out = dec.decode(&lat).unwrap();
    let _ = out.to_vec_f32();
    eprintln!("decode {:?} -> {:?}", t1.elapsed(), out.dims());
}

/// Encoder parity vs the torch mirror (oracle_encoder.py): fixed smooth
/// stereo audio in, mean|scale latents out.
#[test]
#[ignore]
fn oobleck_encoder_matches_torch() {
    let dir = std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR");
    let read = |n: &str| -> Vec<f32> {
        std::fs::read(format!("{dir}/{n}"))
            .expect(n)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let ckpt = format!(
        "{}/stable-audio/model.safetensors",
        crate::inference::cache::hf::models_dir()
    );
    let enc = load_encoder_on(&ckpt, "pretransform.model.encoder", &Device::Cpu).unwrap();
    let n = 32 * 2048;
    let audio = Tensor::from_vec_f32(read(&format!("enc_in_2x{n}.f32")), (2usize, n)).unwrap();
    let out = enc.encode_raw(&audio).unwrap().to_vec_f32();
    let reference = read("enc_ref_out.f32");
    assert_eq!(out.len(), reference.len(), "shape mismatch");
    let (mut max_rel, mut max_abs, mut worst_mixed) = (0f64, 0f64, 0f64);
    let (mut err_pow, mut sig_pow) = (0f64, 0f64);
    for (o, r) in out.iter().zip(&reference) {
        let d = (*o as f64 - *r as f64).abs();
        let rel = d / (r.abs() as f64).max(1e-3);
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(d);
        if d > 1e-4 {
            worst_mixed = worst_mixed.max(rel);
        }
        err_pow += d * d;
        sig_pow += (*r as f64) * (*r as f64);
    }
    let snr_db = 10.0 * (sig_pow / err_pow.max(1e-30)).log10();
    eprintln!(
        "encoder parity: max_rel {max_rel:.3e} | max_abs {max_abs:.3e} | \
         worst mixed(abs>1e-4) rel {worst_mixed:.3e} | SNR {snr_db:.1} dB"
    );
    assert!(worst_mixed < 1e-2, "encoder diverges: {worst_mixed:.3e}");
    assert!(snr_db > 70.0, "encoder SNR too low: {snr_db:.1} dB");
}

/// Audio-to-audio smoke: a low-noise variation must stay close to its
/// source clip; a high-noise one must diverge from it.
#[test]
#[ignore]
fn render_variation_smoke() {
    let (base, _) = render(
        "128 BPM tech house drum loop with deep bass",
        "",
        5.0,
        10,
        7.0,
        42,
        |_, _, _| Ok(()),
    )
    .expect("base render");
    let envelope = |pcm: &[f32]| -> Vec<f32> {
        pcm.chunks(4410 * 2)
            .map(|c| {
                (c.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / c.len() as f64).sqrt()
                    as f32
            })
            .collect()
    };
    let corr = |a: &[f32], b: &[f32]| -> f32 {
        let n = a.len().min(b.len());
        let (ma, mb) = (
            a[..n].iter().sum::<f32>() / n as f32,
            b[..n].iter().sum::<f32>() / n as f32,
        );
        let (mut num, mut da, mut db) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let (x, y) = (a[i] - ma, b[i] - mb);
            num += x * y;
            da += x * x;
            db += y * y;
        }
        num / (da.sqrt() * db.sqrt()).max(1e-9)
    };
    let e_base = envelope(&base);
    let run = |level: f32, seed: u64| -> Vec<f32> {
        let (pcm, _) = render_with_init(
            "128 BPM tech house drum loop with deep bass",
            "",
            5.0,
            10,
            7.0,
            seed,
            Some((base.as_slice(), level)),
            |_, _, _| Ok(()),
        )
        .expect("variation");
        assert!(pcm.iter().all(|x| x.is_finite()));
        envelope(&pcm)
    };
    let close = corr(&e_base, &run(1.0, 7));
    let far = corr(&e_base, &run(100.0, 7));
    eprintln!("variation envelope corr: level 1.0 -> {close:.3}, level 100 -> {far:.3}");
    assert!(
        close > 0.6,
        "low-noise variation lost the source (corr {close:.3})"
    );
    assert!(
        close > far,
        "noise level does not modulate variation strength"
    );
}

/// Long-clip render: the 30 s case used to OOM in the conv1d im2col
/// transient (a multi-GB single allocation); with column tiling it must
/// complete within an ordinary card's budget.
#[test]
#[ignore]
fn render_long_clip_no_oom() {
    let t0 = std::time::Instant::now();
    let (pcm, sr) = render(
        "evolving ambient pad with slow filter sweep",
        "",
        30.0,
        12,
        7.0,
        42,
        |_, _, _| Ok(()),
    )
    .expect("30 s render");
    assert_eq!(pcm.len(), (30.0 * sr as f64) as usize * 2);
    assert!(pcm.iter().all(|x| x.is_finite()));
    let rms =
        (pcm.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
    eprintln!("30 s render: rms {rms:.4} in {:?}", t0.elapsed());
    assert!(rms > 0.005, "near-silent output");
}

/// The no-OOM guarantee under real pressure: warm the pipeline, then hog
/// almost all remaining VRAM on its device and render a clip whose decode
/// transients exceed what is left. The op-level nets must degrade
/// (reclaim/retry, then CPU bounce) instead of failing the render.
#[test]
#[ignore]
fn render_under_vram_pressure_no_oom() {
    // Warm load (residency picks its device with normal free VRAM).
    let (_, _) = render("short noise burst", "", 2.0, 4, 1.0, 1, |_, _, _| Ok(())).expect("warm");
    let dev = {
        let slot = resident_slot().lock().unwrap_or_else(|e| e.into_inner());
        slot.as_ref().expect("resident").device.clone()
    };
    // Hog: leave ~1 GB free on the pipeline's device.
    let (free, total) = crate::tensor::cuda_ext::mem_get_info(&dev).expect("mem info");
    eprintln!(
        "device before hog: free {} MB / total {} MB",
        free >> 20,
        total >> 20
    );
    let leave = 1usize << 30;
    let hog_elems = free.saturating_sub(leave) / 4;
    let _hog = Tensor::zeros_on((hog_elems.max(1),), DType::F32, &dev).expect("hog alloc");
    let (free2, _) = crate::tensor::cuda_ext::mem_get_info(&dev).expect("mem info");
    eprintln!("device after hog: free {} MB", free2 >> 20);

    // 20 s decode peaks well beyond 1 GB of transients: without the OOM
    // nets this failed hard; with them it must complete.
    let t0 = std::time::Instant::now();
    let (pcm, _) = render(
        "evolving ambient pad, soft texture",
        "",
        20.0,
        6,
        4.0,
        9,
        |_, _, _| Ok(()),
    )
    .expect("render under pressure must not OOM");
    assert!(pcm.iter().all(|x| x.is_finite()));
    let rms =
        (pcm.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
    eprintln!("pressure render: rms {rms:.4} in {:?}", t0.elapsed());
    assert!(rms > 0.001, "degenerate output under pressure");
}

/// Folded-weight parity: our weight-norm fold vs torch's effective weight.
#[test]
#[ignore]
fn head_weight_fold_matches_torch() {
    let dir = std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR");
    let read = |n: &str| -> Vec<f32> {
        std::fs::read(format!("{dir}/{n}"))
            .expect(n)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let dec = load_decoder(
        &format!(
            "{}/stable-audio/model.safetensors",
            crate::inference::cache::hf::models_dir()
        ),
        "pretransform.model.decoder",
    )
    .unwrap();
    let ours_w = dec.head.weight.to_vec_f32();
    let ref_w = read("head_w_ref.f32");
    assert_eq!(ours_w.len(), ref_w.len());
    let mut mx = 0f64;
    for (a, b) in ours_w.iter().zip(&ref_w) {
        mx = mx.max((*a as f64 - *b as f64).abs() / (b.abs() as f64).max(1e-6));
    }
    eprintln!("head WEIGHT fold max_rel = {mx:.3e}");
    let ours_b = dec.head.bias.as_ref().unwrap().to_vec_f32();
    let ref_b = read("head_b_ref.f32");
    let mut mb = 0f64;
    for (a, b) in ours_b.iter().zip(&ref_b) {
        mb = mb.max((*a as f64 - *b as f64).abs() / (b.abs() as f64).max(1e-6));
    }
    eprintln!("head BIAS max_rel = {mb:.3e}");
    assert!(mx < 1e-5 && mb < 1e-6, "fold diverges");
}

/// Stage-by-stage divergence probe vs the torch mirror's dumps.
#[test]
#[ignore]
fn oobleck_decoder_stage_probe() {
    let dir = std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR");
    let read = |n: &str| -> Vec<f32> {
        std::fs::read(format!("{dir}/{n}"))
            .expect(n)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let dec = load_decoder(
        &format!(
            "{}/stable-audio/model.safetensors",
            crate::inference::cache::hf::models_dir()
        ),
        "pretransform.model.decoder",
    )
    .unwrap();
    let lat = read("dec_in_64x32.f32");
    let mut h = Tensor::from_vec_f32(lat, (64usize, 32usize)).unwrap();
    let cmp = |tag: &str, ours: &Tensor, file: &str| {
        let o = ours.to_vec_f32();
        let r = read(file);
        assert_eq!(o.len(), r.len(), "{tag} shape");
        let mut mx = 0f64;
        for (a, b) in o.iter().zip(&r) {
            mx = mx.max((*a as f64 - *b as f64).abs() / (b.abs() as f64).max(1e-3));
        }
        eprintln!("{tag}: max_rel {mx:.3e}");
    };
    h = dec.head.forward(&h).unwrap();
    cmp("stage0 head", &h, "dec_stage0.f32");
    for (bi, b) in dec.blocks.iter().enumerate() {
        h = b.forward(&h).unwrap();
        let idx = bi + 1;
        if [1usize, 3, 5].contains(&idx) {
            cmp(
                &format!("stage{idx} block"),
                &h,
                &format!("dec_stage{idx}.f32"),
            );
        }
    }
    h = dec.tail_snake.forward(&h).unwrap();
    cmp("stage6 snake", &h, "dec_stage6.f32");
}

/// Parity against the torch mirror of the decoder:
/// same checkpoint weights, same fixed latent.
///
/// Criterion is mixed abs/rel + SNR, not pure relative: the decoder's
/// SnakeBeta stages (sin^2 with large alpha) amplify benign f32
/// reordering noise, and torch itself drifts by ~5e-4 max_rel between
/// thread configurations on this very graph. A sample only fails if it
/// is BOTH absolutely and relatively off; the SNR gate bounds the
/// aggregate audio error far below audibility.
#[test]
#[ignore]
fn oobleck_decoder_matches_torch() {
    let dir = std::env::var("SAO_REF_DIR").expect("set SAO_REF_DIR");
    let read = |n: &str| -> Vec<f32> {
        std::fs::read(format!("{dir}/{n}"))
            .expect(n)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let lat = read("dec_in_64x32.f32");
    let reference = read("dec_ref_out.f32");
    let dec = load_decoder(
        &format!(
            "{}/stable-audio/model.safetensors",
            crate::inference::cache::hf::models_dir()
        ),
        "pretransform.model.decoder",
    )
    .expect("load decoder");
    let latent = Tensor::from_vec_f32(lat, (64usize, 32usize)).unwrap();
    let out = dec.decode(&latent).unwrap().to_vec_f32();
    assert_eq!(out.len(), reference.len(), "shape mismatch");
    let (mut max_rel, mut max_abs, mut worst_mixed) = (0f64, 0f64, 0f64);
    let (mut err_pow, mut sig_pow) = (0f64, 0f64);
    for (o, r) in out.iter().zip(&reference) {
        let d = (*o as f64 - *r as f64).abs();
        let rel = d / (r.abs() as f64).max(1e-3);
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(d);
        if d > 1e-5 {
            worst_mixed = worst_mixed.max(rel);
        }
        err_pow += d * d;
        sig_pow += (*r as f64) * (*r as f64);
    }
    let snr_db = 10.0 * (sig_pow / err_pow.max(1e-30)).log10();
    eprintln!(
        "decoder parity: max_rel {max_rel:.3e} | max_abs {max_abs:.3e} | \
         worst mixed(abs>1e-5) rel {worst_mixed:.3e} | SNR {snr_db:.1} dB"
    );
    assert!(
        worst_mixed < 1e-3,
        "decoder diverges beyond mixed abs/rel tolerance: {worst_mixed:.3e}"
    );
    assert!(snr_db > 70.0, "decoder SNR too low: {snr_db:.1} dB");
}
