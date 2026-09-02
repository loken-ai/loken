/// Claiming MMQ support and being able to feed it are the same claim.
///
/// The tile kernel reads the activation in a per-weight-type scale layout, so a dtype
/// listed as supported whose layout does not resolve fails at the first multi-row
/// prefill and falls back to materialising the weight in F32 - 8 to 16 times the size
/// it was quantised to shrink, which is how a tight multi-GPU split OOMs after a
/// successful load. Nothing announces it. Twice now the two sides disagreed: Q5_0 had a
/// compiled tile kernel that was never dispatched, and Q2K had both kernels compiled
/// with only the Rust declaration of its quantize launcher missing. Both were found
/// from the damage rather than from the code, so pin the agreement instead of the
/// dtypes: whatever the list says, both launchers must resolve for every entry.
#[cfg(feature = "cuda")]
#[test]
fn every_dtype_mmq_claims_to_support_can_actually_be_fed() {
    use super::GgmlDType::*;
    let all = [
        F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, MxFp4,
    ];
    let mut supported = 0;
    for dt in all {
        if !super::mmq::mmq_supports(dt) {
            continue;
        }
        supported += 1;
        assert!(
            super::mmq::mmq_quantize_launcher(dt).is_ok(),
            "{dt:?} is listed as MMQ-supported but has no activation quantize layout, \
             so it silently takes the dequant-to-F32 path"
        );
        assert!(
            super::mmq::mmq_launcher(dt).is_ok(),
            "{dt:?} is listed as MMQ-supported but has no tile kernel"
        );
    }
    assert!(
        supported > 0,
        "the supported list emptied out - the check would pass vacuously"
    );
    // Naming Q2K keeps this from passing vacuously for the very dtype it was written
    // for: dropping Q2K from the list would otherwise skip it in the loop above and
    // the test would still be green while 2-bit weights went back to the dequant path.
    assert!(
        super::mmq::mmq_supports(Q2K),
        "Q2K left the MMQ list - 2-bit weights are back on the dequant-to-F32 path"
    );
}

/// Moving a weight between devices must preserve it exactly. The host blocks are the
/// source of truth on both sides, so a round trip is not an approximation - if this
/// drifts, a model climbing back onto a GPU would come back subtly different from the
/// one that was running a moment earlier.
#[test]
fn a_quantized_weight_survives_a_device_move() {
    use super::{GgmlDType, QTensor};
    use crate::tensor::{Device, Tensor};
    let n = 512usize;
    let vals: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) / 7.0).collect();
    let t = Tensor::from_vec(vals, (2, 256), &Device::Cpu).unwrap();
    let q = QTensor::quantize_onto(&t, GgmlDType::Q8_0, &Device::Cpu).unwrap();
    let before: Vec<f32> = q
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let moved = q.to_device(&Device::Cpu).unwrap();
    let after: Vec<f32> = moved
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a, b, "a device move must not change a single value");
    }
    assert_eq!(moved.dtype(), q.dtype());
    assert_eq!(moved.shape().dims(), q.shape().dims());
}

/// The adapter must apply ON TOP of a weight that stays QUANTISED.
///
/// That is the whole claim: LoRA on a quantised base without dequantising it. If the
/// delta silently did nothing the base projection would still work, every shape would
/// check out, and a user's adapter would simply have no effect - so this measures the
/// difference rather than asserting the call succeeded.
#[test]
fn a_lora_applies_over_a_quantised_weight() {
    use super::super::lora::LoraDelta;
    use super::super::{Device, Tensor};

    let (k, n) = (256usize, 32usize);
    // Q8_0 needs whole blocks; 256 is a multiple of every block size in play.
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
    let bytes = super::super::quant_cpu::from_float_bytes(GgmlDType::Q8_0, &w).unwrap();
    let qt = QHostTensor::from_bytes(&bytes, GgmlDType::Q8_0, vec![n, k]).unwrap();
    let mut mm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &Device::Cpu).unwrap();

    let x =
        Tensor::from_vec_f32((0..k).map(|i| (i % 5) as f32 * 0.1).collect(), vec![1, k]).unwrap();
    let base = mm.forward(&x).unwrap().to_vec_f32();
    assert_eq!(base.len(), n);
    assert_eq!(mm.lora_count(), 0);

    // down [k, r] and up [r, n] all ones, scale s: every output gains
    // s * r * sum(x).
    let r = 2usize;
    let sum_x: f32 = (0..k).map(|i| (i % 5) as f32 * 0.1).sum();
    let scale = 0.25f32;
    mm.add_lora(LoraDelta {
        down: Tensor::from_vec_f32(vec![1.0; k * r], vec![k, r]).unwrap(),
        up: Tensor::from_vec_f32(vec![1.0; r * n], vec![r, n]).unwrap(),
        scale,
    })
    .unwrap();
    assert_eq!(mm.lora_count(), 1);

    let got = mm.forward(&x).unwrap().to_vec_f32();
    let expected_delta = scale * r as f32 * sum_x;
    for i in 0..n {
        let d = got[i] - base[i];
        assert!(
            (d - expected_delta).abs() < 1e-2,
            "out[{i}] moved by {d}, expected {expected_delta}"
        );
    }

    // Clearing restores the quantised base exactly - the blob was never touched.
    mm.clear_lora();
    assert_eq!(mm.forward(&x).unwrap().to_vec_f32(), base);
}

/// A mismatched adapter is refused rather than applied to the wrong axis.
#[test]
fn a_wrongly_shaped_lora_is_refused_on_a_quantised_weight() {
    use super::super::lora::LoraDelta;
    use super::super::{Device, Tensor};
    let (k, n) = (256usize, 32usize);
    let bytes =
        super::super::quant_cpu::from_float_bytes(GgmlDType::Q8_0, &vec![0.1f32; n * k]).unwrap();
    let qt = QHostTensor::from_bytes(&bytes, GgmlDType::Q8_0, vec![n, k]).unwrap();
    let mut mm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &Device::Cpu).unwrap();
    let bad = LoraDelta {
        down: Tensor::from_vec_f32(vec![1.0; 8 * 2], vec![8, 2]).unwrap(),
        up: Tensor::from_vec_f32(vec![1.0; 2 * n], vec![2, n]).unwrap(),
        scale: 1.0,
    };
    assert!(mm.add_lora(bad).is_err(), "a k mismatch must be refused");
}
use super::*;

/// Every block format the CPU encoder and the kernels are expected to serve.
///
/// One list, walked by each test below. Three copies of it meant a format could be added to
/// `GgmlDType`, given an encoder and a kernel, and still be exercised by none of them - the
/// lists were what decided, and nothing kept them in step.
const BLOCK_QUANTISED: &[GgmlDType] = &[
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

/// Those of them the repacked GEMM has a path for, plus the one format only it serves.
const REPACKED: &[GgmlDType] = &[
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q4_0,
    GgmlDType::Q5_0,
    GgmlDType::Q8_0,
    GgmlDType::MxFp4,
];

/// GGUF blobs from the local model store, smallest first.
fn find_ggufs() -> Vec<std::path::PathBuf> {
    let dir = crate::config::Config::load_test()
        .get_ollama_models_dir()
        .join("blobs");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let mut candidates: Vec<_> = rd
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let mut f = std::fs::File::open(&p).ok()?;
            let mut magic = [0u8; 4];
            std::io::Read::read_exact(&mut f, &mut magic).ok()?;
            (magic == gguf_file::layout::MAGIC_BYTES).then_some((e.metadata().ok()?.len(), p))
        })
        .collect();
    candidates.sort();
    candidates.into_iter().map(|(_, p)| p).collect()
}

/// Native QKernelMatMul (rows=1 mmvq GEMV, rows>1 MMQ) on a real Q4_K weight
/// vs an f64 reference matmul over the dequantized weights. The kernels
/// quantize the activations to q8_1 internally, so the bound is the
/// quantization-error envelope (~1e-2 relative), enough to catch any
/// layout/transpose/blob-offset bug.
#[cfg(feature = "cuda")]
#[test]
fn qmatmul_matches_oracle() {
    use crate::tensor::{cuda::CudaDevice, Device, Tensor};
    let Ok(dev) = CudaDevice::new(0) else { return };
    // locate a real 2-D Q4_K weight
    let mut found = None;
    'outer: for path in find_ggufs().iter().take(4) {
        let mut f = std::fs::File::open(path).unwrap();
        let Ok(content) = gguf_file::Content::read(&mut f) else {
            continue;
        };
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in names {
            let info = &content.tensor_infos[name];
            if info.ggml_dtype == GgmlDType::Q4K
                && info.shape.dims().len() == 2
                && info.elem_count() < 30_000_000
            {
                found = Some((path.clone(), name.clone()));
                break 'outer;
            }
        }
    }
    let Some((path, name)) = found else { return };
    let mut f = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut f).unwrap();
    let qt = content.host_tensor(&mut f, &name).unwrap();
    let (n, k) = (qt.dims[0], qt.dims[1]);
    let wf = qt.dequantize_f32().unwrap(); // [n, k] rows
    let qm = QKernelMatMul::from_qtensor(qt, dev.clone()).unwrap();

    let device = Device::Cuda(dev);
    for rows in [1usize, 3, 32] {
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
            .collect();
        let nx = Tensor::from_vec_f32(x.clone(), vec![rows, k])
            .unwrap()
            .to_device(&device)
            .unwrap();
        let got = qm.forward(&nx).unwrap().to_vec_f32();
        assert_eq!(got.len(), rows * n);
        // reference: y[r, o] = Σ_j x[r, j] * w[o, j] in f64. The error
        // budget is the activation-quantization envelope, which scales
        // with Σ|x.w| (the magnitude BEFORE cancellation), not with |y|.
        for r in 0..rows {
            for o in 0..n {
                let mut acc = 0f64;
                let mut mag = 0f64;
                for j in 0..k {
                    let t = x[r * k + j] as f64 * wf[o * k + j] as f64;
                    acc += t;
                    mag += t.abs();
                }
                let w = acc as f32;
                let g = got[r * n + o];
                let tol = (5e-3 * mag) as f32 + 1e-3;
                assert!(
                    (g - w).abs() <= tol,
                    "{name} rows={rows} [{r},{o}]: {g} vs {w} (tol {tol})"
                );
            }
        }
    }
    eprintln!("QKernelMatMul reference check on {name} [{n}x{k}] rows=1(gemv)+3+32(mmq) OK");
}

/// Q8_0 GPU mmvq parity at K=896: qwen2.5:0.5b's tied lm_head is
/// the token_embd Q8_0 tensor used as a QKernelMatMul with K=hidden=896 (native
/// loader reverses dims -> n=vocab=151936, k=896). A wrong lm_head gives
/// garbage logits -> garbage tokens from otherwise-correct hidden states.
/// Q5_0 mmvq was proven correct at K=896; Q8_0 at K=896 was NOT tested.
/// Compares the GPU GEMV to an f64 reference on a SUBSET of output rows.
#[test]
fn q8_0_gpu_mmvq_parity_k896() {
    use crate::tensor::{cuda::CudaDevice, Device, Tensor};
    let Ok(dev) = CudaDevice::new(0) else { return };
    let mut found = None;
    'outer: for path in find_ggufs().iter() {
        let mut f = std::fs::File::open(path).unwrap();
        let Ok(content) = gguf_file::Content::read(&mut f) else {
            continue;
        };
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in names {
            let info = &content.tensor_infos[name];
            // Prefer the lm_head/embedding (huge n) - the real decode logits
            // path - over small attn weights that also have K=896.
            if info.ggml_dtype == GgmlDType::Q8_0
                && info.shape.dims().len() == 2
                && info.shape.dims()[1] == 896
                && info.shape.dims()[0] >= 100_000
            {
                found = Some((path.clone(), name.clone()));
                break 'outer;
            }
        }
    }
    let Some((path, name)) = found else {
        eprintln!("no Q8_0 K=896 weight; skip");
        return;
    };
    let mut f = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut f).unwrap();
    let qt = content.host_tensor(&mut f, &name).unwrap();
    let (n, k) = (qt.dims[0], qt.dims[1]);
    assert_eq!(k, 896);
    let wf = qt.dequantize_f32().unwrap(); // [n, k]
    let qm = QKernelMatMul::from_qtensor(qt, dev.clone()).unwrap();
    let device = Device::Cuda(dev);
    let x: Vec<f32> = (0..k).map(|i| ((i % 71) as f32) * 0.027 - 0.95).collect();
    let nx = Tensor::from_vec_f32(x.clone(), vec![1usize, k])
        .unwrap()
        .to_device(&device)
        .unwrap();
    let got = qm.forward(&nx).unwrap().to_vec_f32();
    let check_n = n.min(3000); // subset of the vocab rows
    let mut max_rel = 0f64;
    for o in 0..check_n {
        let (mut acc, mut mag) = (0f64, 0f64);
        for j in 0..k {
            let t = x[j] as f64 * wf[o * k + j] as f64;
            acc += t;
            mag += t.abs();
        }
        let g = got[o] as f64;
        let rel = (g - acc).abs() / (mag + 1e-6);
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            (g - acc).abs() <= 5e-3 * mag + 1e-3,
            "Q8_0 {name} [{n}x{k}] o={o}: gpu {g} vs ref {acc} (mag {mag})"
        );
    }
    eprintln!("Q8_0 GPU mmvq K=896 parity OK on {name} [{n}x{k}] checked {check_n} rows max_rel={max_rel:.2e}");
}

/// Q5_0 GPU mmvq parity (confound-separator): qwen2.5:0.5b is the
/// only Q5_0-dominant model and produces deterministic GPU garbage. This
/// isolates whether the Q5_0 decode mmvq kernel itself is wrong (vs an
/// attention/hidden=896 issue) by comparing the GPU GEMV to an f64 reference
/// over the dequantized weight, on a real Q5_0 weight (K=896 from the model).
#[test]
fn q5_0_gpu_mmvq_matches_oracle() {
    use crate::tensor::{cuda::CudaDevice, Device, Tensor};
    let Ok(dev) = CudaDevice::new(0) else { return };
    let mut found = None;
    'outer: for path in find_ggufs().iter() {
        let mut f = std::fs::File::open(path).unwrap();
        let Ok(content) = gguf_file::Content::read(&mut f) else {
            continue;
        };
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in names {
            let info = &content.tensor_infos[name];
            if info.ggml_dtype == GgmlDType::Q5_0
                && info.shape.dims().len() == 2
                && info.elem_count() < 30_000_000
            {
                found = Some((path.clone(), name.clone()));
                break 'outer;
            }
        }
    }
    let Some((path, name)) = found else {
        eprintln!("no Q5_0 2-D weight found; skipping");
        return;
    };
    let mut f = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut f).unwrap();
    let qt = content.host_tensor(&mut f, &name).unwrap();
    let (n, k) = (qt.dims[0], qt.dims[1]);
    let wf = qt.dequantize_f32().unwrap();
    let qm = QKernelMatMul::from_qtensor(qt, dev.clone()).unwrap();
    let device = Device::Cuda(dev);
    let rows = 1usize; // decode GEMV - the path qwen2.5:0.5b uses
    let x: Vec<f32> = (0..rows * k)
        .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
        .collect();
    let nx = Tensor::from_vec_f32(x.clone(), vec![rows, k])
        .unwrap()
        .to_device(&device)
        .unwrap();
    let got = qm.forward(&nx).unwrap().to_vec_f32();
    let mut max_rel = 0f64;
    for o in 0..n {
        let (mut acc, mut mag) = (0f64, 0f64);
        for j in 0..k {
            let t = x[j] as f64 * wf[o * k + j] as f64;
            acc += t;
            mag += t.abs();
        }
        let g = got[o] as f64;
        let tol = 5e-3 * mag + 1e-3;
        let rel = (g - acc).abs() / (mag + 1e-6);
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            (g - acc).abs() <= tol,
            "Q5_0 {name} [{n}x{k}] o={o}: gpu {g} vs ref {acc} (tol {tol}, mag {mag})"
        );
    }
    eprintln!(
        "Q5_0 GPU mmvq parity OK on {name} [{n}x{k}] K%512={} max_rel={max_rel:.2e}",
        k % 512
    );
}

/// risk #7): the CPU QKernelMatMul must run the lifted k_quants
/// dot engine (NOT the dequant-f32 fallback) for the served dtypes  -
/// Q4K/Q6K/Q8_0/MxFp4 - and stay inside the quantization-error envelope
/// of an f64 reference matmul over the dequantized weights. Synthetic
/// weights, no model store needed.
#[test]
fn cpu_qmatmul_vec_dot_matches_oracle() {
    use crate::tensor::{quant_cpu, Device, Tensor};
    let (n, k) = (16usize, 512usize);
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i % 113) as f32) * 0.018 - 0.99)
        .collect();
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::MxFp4,
        GgmlDType::Q4_0,
    ] {
        let bytes = quant_cpu::from_float_bytes(dtype, &wf).unwrap();
        let qt = QHostTensor::from_bytes(&bytes, dtype, vec![n, k]).unwrap();
        let wdq = qt.dequantize_f32().unwrap(); // exact blocks the engine sees
        let qm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &Device::Cpu).unwrap();

        for rows in [1usize, 4] {
            let x: Vec<f32> = (0..rows * k)
                .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
                .collect();
            let nx = Tensor::from_vec_f32(x.clone(), vec![rows, k]).unwrap();
            let got = qm.forward(&nx).unwrap().to_vec_f32();
            assert_eq!(got.len(), rows * n);
            // tolerance = the activation-quantization envelope, scaled
            // by Σ|x.w| (magnitude before cancellation), not by |y|.
            for r in 0..rows {
                for o in 0..n {
                    let mut acc = 0f64;
                    let mut mag = 0f64;
                    for j in 0..k {
                        let t = x[r * k + j] as f64 * wdq[o * k + j] as f64;
                        acc += t;
                        mag += t.abs();
                    }
                    let w = acc as f32;
                    let g = got[r * n + o];
                    let tol = (5e-3 * mag) as f32 + 1e-3;
                    assert!(
                        (g - w).abs() <= tol,
                        "{dtype:?} rows={rows} [{r},{o}]: {g} vs {w} (tol {tol})"
                    );
                }
            }
        }
        // dot path proven: the dequant fallback cache must never have
        // been built on CPU for an engine-supported dtype.
        assert!(
            qm.dequant.get().is_none(),
            "{dtype:?}: CPU QKernelMatMul fell back to the dequant matmul"
        );
    }
}

/// Every format the GPU mat-vec kernel serves, against the weights it actually holds.
///
/// Two of the ten had a GPU parity test, and both needed a real GGUF on disk to find a
/// tensor of the right dtype - so on a machine without the right model they returned early
/// and looked like they had passed. These quantise a synthetic matrix with the CPU encoder
/// (itself pinned to ggml by `oracle_parity`), so there is nothing to find and nothing to
/// skip but the card.
///
/// The reference is an f64 dot product over the DEQUANTISED weights, not the original
/// floats: the kernel is answerable for what the blocks hold, not for the rounding that put
/// them there. The tolerance is the activation-quantisation envelope scaled by Σ|x.w|, the
/// magnitude before cancellation - a row that sums to near zero would otherwise demand a
/// relative accuracy no int8 activation can give.
///
/// WHAT IT DOES NOT DO, measured rather than assumed: it exercises whichever kernel
/// `QKernelMatMul::forward` chooses, not a named one. Perturbing `mmvq_gguf.cu`'s shared
/// warp reduction moves the 32-value formats here; perturbing the k-quant mat-vec entries in
/// that same file moves nothing, so the k-quants reach the card another way. That is the
/// right thing to pin - it is the served path - but it means this cannot be used to prove a
/// particular file is correct. The kernel names are built with `format!`, so grep cannot
/// answer that question either.
#[test]
fn every_gpu_mmvq_format_matches_the_weights_it_holds() {
    use crate::tensor::{cuda::CudaDevice, quant_cpu, Device, Tensor};
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the GPU mat-vec kernels are NOT covered by this run");
        return;
    };
    let device = Device::Cuda(dev);

    // Two shapes on purpose, because they reach DIFFERENT kernels. The x8-interleaved
    // repack GEMM takes over at four rows or more when `n` divides by eight; `n = 12`
    // makes it decline, and the tiled MMQ family gets the work instead. Testing only one
    // leaves the other family with no coverage at all - which is what was happening.
    for n in [16usize, 12] {
        let k = 1024usize;
        let wf: Vec<f32> = (0..n * k)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(374_761_393);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.4
            })
            .collect();

        for &dtype in BLOCK_QUANTISED {
            let bytes = quant_cpu::from_float_bytes(dtype, &wf)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let qt = QHostTensor::from_bytes(&bytes, dtype, vec![n, k])
                .unwrap_or_else(|e| panic!("{dtype:?}: from_bytes: {e}"));
            let wdq = qt.dequantize_f32().unwrap();
            let qm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &device)
                .unwrap_or_else(|e| panic!("{dtype:?}: upload: {e}"));

            // One row is the decode GEMV, which mat-vec exists for. Sixteen crosses into the
            // tiled matmul, which is a DIFFERENT kernel with its own dot products - testing
            // only one row leaves that half unexercised.
            let mut worst = 0f64;
            for rows in [1usize, 16] {
                let x: Vec<f32> = (0..rows * k)
                    .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
                    .collect();
                let nx = Tensor::from_vec_f32(x.clone(), vec![rows, k])
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let got = qm.forward(&nx).unwrap().to_vec_f32();
                assert_eq!(got.len(), rows * n, "{dtype:?} n={n}: wrong output length");

                for r in 0..rows {
                    for o in 0..n {
                        let (mut acc, mut mag) = (0f64, 0f64);
                        for j in 0..k {
                            let t = x[r * k + j] as f64 * wdq[o * k + j] as f64;
                            acc += t;
                            mag += t.abs();
                        }
                        // Sized to what the arithmetic actually costs, not to the pre-
                        // cancellation magnitude with room to spare. Both sides read the same
                        // quantised weights, so that error is common and drops out; what does
                        // not is the ACTIVATION, which the kernel quantises to Q8_1 and this
                        // reference does not. That is one 8-bit step per term, and it shows:
                        // the worst relative error across all twenty cells is 4.9e-4, and it
                        // barely moves between formats. Twice that leaves headroom and still
                        // sees a wrong dot product - at 5e-3 a five percent error in the
                        // accumulation passed unnoticed, which is how this was found.
                        let tol = 1e-3 * mag + 1e-3;
                        let err = (got[r * n + o] as f64 - acc).abs();
                        worst = worst.max(err / (mag + 1e-9));
                        assert!(
                        err <= tol,
                        "{dtype:?} n={n} rows={rows} r={r} col {o}: gpu {} against the blocks' \
                         own value {acc} (error {err:.4e}, tolerance {tol:.4e})",
                        got[r * n + o]
                    );
                    }
                }
            }
            eprintln!("{dtype:?} n={n}: agrees at 1 and 16 rows, worst relative error {worst:.2e}");
        }
    }
}

/// The seven formats that own an interleaved repack GEMM, checked through the dispatch that
/// chooses it rather than against the repack in isolation.
///
/// `repack.rs` names a block type, a repack, a GEMM and a cache cell per format, and nothing
/// checks that a row of that table is internally consistent - a format pointed at its
/// neighbour's repack still compiles and still returns numbers.
///
/// The judge is the GEMV, not a dequantised reference. A dequantised reference has to be given
/// a tolerance, and the only scale available to size one is the sum of the term magnitudes
/// *before* they cancel - which on these operands is some thirty times the result, so a 2%
/// error in the output sits comfortably inside it and the test says yes to anything. Running
/// the same weights and the same activation row through both kernels removes that problem
/// entirely: the quantisation error is common to both and cancels in the difference, leaving
/// only the accumulation order, and the two must agree to the carrier's own precision.
#[test]
fn every_cpu_repack_gemm_matches_the_gemv_on_the_same_weights() {
    use crate::tensor::{quant_cpu, DType, Device, Tensor};

    let k = 1024usize;
    // Four rows is the repack threshold and three is below it, so the same activation row
    // reaches a different kernel in each call. n=16 lets the repack take the four-row call;
    // n=12 makes it decline, which is how the GEMV gets covered on its own terms too.
    for n in [16usize, 12] {
        let wf: Vec<f32> = (0..n * k)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(374_761_393);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.4
            })
            .collect();

        for &dtype in REPACKED {
            let bytes = quant_cpu::from_float_bytes(dtype, &wf)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let qt = QHostTensor::from_bytes(&bytes, dtype, vec![n, k])
                .unwrap_or_else(|e| panic!("{dtype:?}: from_bytes: {e}"));
            let qm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &Device::Cpu)
                .unwrap_or_else(|e| panic!("{dtype:?}: build: {e}"));

            let x: Vec<f32> = (0..4 * k)
                .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
                .collect();
            let run = |rows: usize| {
                let t = Tensor::from_vec_f32(x[..rows * k].to_vec(), vec![rows, k])
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
                assert_eq!(
                    t.dtype(),
                    DType::F16,
                    "the f16 branch is what we mean to reach"
                );
                let out = qm.forward(&t).unwrap();
                assert_eq!(
                    out.dtype(),
                    DType::F16,
                    "{dtype:?}: output must follow the input"
                );
                out.to_vec_f32()
            };
            let tiled = run(4);
            let gemv = run(3);
            assert_eq!(tiled.len(), 4 * n, "{dtype:?} n={n}: wrong output length");

            // Both outputs are f16, whose step is about a thousandth of the value, and the two
            // kernels sum in different orders. The floor is set from the largest output in the
            // row so a column that cancelled to near zero is not held to a relative bound it
            // cannot meet.
            let mut worst = 0f64;
            for r in 0..3 {
                let scale = (0..n).map(|o| tiled[r * n + o].abs()).fold(0f32, f32::max) as f64;
                for o in 0..n {
                    let (a, b) = (tiled[r * n + o] as f64, gemv[r * n + o] as f64);
                    let tol = 3e-3 * a.abs() + 4e-3 * scale;
                    worst = worst.max((a - b).abs() / (scale + 1e-9));
                    assert!(
                        (a - b).abs() <= tol,
                        "{dtype:?} n={n} r={r} col {o}: four-row kernel {a} against the \
                         one-row kernel {b} on the same weights (tolerance {tol:.4e})"
                    );
                }
            }
            eprintln!("{dtype:?} n={n}: the two kernels agree, worst {worst:.2e} of row scale");
        }
    }
}

/// The GPU's two kernel families, judged against each other on the same weights.
///
/// `every_gpu_mmvq_format_matches_the_weights_it_holds` judges both against a dequantised
/// reference, and cannot be sharp: the only scale available to size its tolerance is Σ|x.w|,
/// the magnitude before cancellation, which on these operands is some thirty times the result.
/// It measures a worst error of 3e-4 against a threshold of 5e-3, and a kernel whose output
/// was scaled by 1.02 would still land inside it. It catches a format read as the wrong
/// format; it does not catch a mis-set scale.
///
/// This one takes the difference instead. One row goes to the mat-vec, sixteen to the tiled
/// matmul, and the weight quantisation is identical in both - so it drops out, and what is
/// left is two int8 activation quantisations of the same values. The two must land within a
/// thousandth or so of the row's own scale. On sm_120 the tiled family is the MMA path, which
/// no test reached before this one.
#[test]
fn the_gpu_tiled_matmul_agrees_with_the_mat_vec_on_the_same_weights() {
    use crate::tensor::{cuda::CudaDevice, quant_cpu, Device, Tensor};
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the GPU kernels are NOT covered by this run");
        return;
    };
    let device = Device::Cuda(dev);

    let k = 1024usize;
    for n in [16usize, 12] {
        let wf: Vec<f32> = (0..n * k)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(374_761_393);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.4
            })
            .collect();

        for &dtype in BLOCK_QUANTISED {
            let bytes = quant_cpu::from_float_bytes(dtype, &wf)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let qt = QHostTensor::from_bytes(&bytes, dtype, vec![n, k])
                .unwrap_or_else(|e| panic!("{dtype:?}: from_bytes: {e}"));
            let qm = QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &device)
                .unwrap_or_else(|e| panic!("{dtype:?}: upload: {e}"));

            // An activation both families quantise IDENTICALLY, so their difference is
            // arithmetic order and nothing else. Each 32-block opens with +1.27 and -1.27, so
            // both derive the same scale 1.27/127, and every value in the block is an exact
            // multiple of it. Left as an arbitrary ramp the two quantise independently and
            // disagree by 0.7% of the row scale - real, harmless, and enough to hide a
            // mis-scaled kernel behind the tolerance it would force.
            let x: Vec<f32> = (0..16 * k)
                .map(|i| match i % 32 {
                    0 => 1.27,
                    1 => -1.27,
                    _ => (((i * 37 + i / 32) % 255) as f32 - 127.0) * 0.01,
                })
                .collect();
            let run = |rows: usize| {
                let t = Tensor::from_vec_f32(x[..rows * k].to_vec(), vec![rows, k])
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                qm.forward(&t).unwrap().to_vec_f32()
            };
            let tiled = run(16);
            let matvec = run(1);
            assert_eq!(tiled.len(), 16 * n, "{dtype:?} n={n}: wrong output length");

            // The row's own largest output is the scale: a column that cancelled to near zero
            // cannot be held to a relative bound, and the two families quantise the activation
            // independently, so the floor has to cover one int8 step either way.
            let scale = matvec.iter().map(|v| v.abs()).fold(0f32, f32::max) as f64;
            let mut worst = 0f64;
            for o in 0..n {
                let (a, b) = (tiled[o] as f64, matvec[o] as f64);
                worst = worst.max((a - b).abs() / (scale + 1e-9));
                // Four thousandths of the row scale: twice the worst the ten formats
                // actually show, and a fifth of what a kernel scaled by 1.02 would produce.
                assert!(
                    (a - b).abs() <= 4e-3 * scale,
                    "{dtype:?} n={n} col {o}: tiled matmul {a} against the mat-vec {b} on the \
                     same weights (row scale {scale})"
                );
            }
            eprintln!("{dtype:?} n={n}: the two families agree, worst {worst:.2e} of row scale");
        }
    }
}

/// Every kernel name this code can ask the quantised module for, asked for.
///
/// That module is one NVRTC translation unit compiled at run time, and nothing loaded it in
/// the test suite: a name that drifted from its definition, or a definition removed because it
/// looked unused, would surface on a user's first quantised attention rather than here. The
/// names are gathered from the source rather than listed again, because a second list is the
/// thing that goes stale.
#[test]
fn every_kernel_name_the_quantised_module_is_asked_for_resolves() {
    use crate::tensor::cuda::CudaDevice;
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the quantised module is NOT covered by this run");
        return;
    };

    // The shape of a kernel name in this family: the prefixes the module actually exports,
    // followed by the format, head dimension and query count that select a specialisation.
    let pattern = regex_lite_prefixes();
    let mut names: Vec<String> = Vec::new();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/inference/quantized_cuda");
    // The names are read from the callers, so this cannot run where the source is not - a
    // test binary shipped to another machine, for instance. Say so rather than fail: the
    // machine that has the source is the one that can answer.
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no source tree at {dir:?}; the quantised module is NOT covered by this run");
        return;
    };
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("readable source");
        for raw in text.split('"').skip(1).step_by(2) {
            if pattern.iter().any(|p| raw.starts_with(p))
                && raw
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                names.push(raw.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    assert!(
        names.len() > 100,
        "expected the attention and flash families to be named here, found {}",
        names.len()
    );

    // The dequantisers and the KV-staging quantisers are reached through generated names, so
    // they are stated here - they are the eight entry points the rest of the engine needs.
    for k in [
        "quantize_q4_0",
        "quantize_q4_0_f16",
        "quantize_q8_0",
        "quantize_q8_0_f16",
        "dequantize_block_q4_K_f32",
        "dequantize_block_q5_K_f32",
        "dequantize_block_q6_K_f32",
        "dequantize_block_q8_0_f32",
        // Reached through the custom-module path rather than `quantized_fn`, which is how
        // one of them went missing without this test noticing.
        "quantize_q8_1",
        "quantize_q8_0_dev_slot",
        "quantize_q8_0_kv_paired",
        "quantize_q8_0_kv_paired_dev_slot",
    ] {
        names.push(k.to_string());
    }

    let mut missing: Vec<&str> = Vec::new();
    for name in &names {
        if dev.quantized_fn(name).is_err() {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {} names do not resolve in the quantised module: {:?}",
        missing.len(),
        names.len(),
        &missing[..missing.len().min(12)]
    );
    eprintln!(
        "{} kernel names resolve in the quantised module",
        names.len()
    );
}

/// The prefixes that mark a string literal as a kernel name rather than an error message.
fn regex_lite_prefixes() -> &'static [&'static str] {
    &["attn_score_", "attn_output_", "flash_splitk_", "kv_"]
}

/// What a GGUF container is, pinned to the byte.
///
/// The reader and the writer now take the layout from one declaration, so the failure they
/// can no longer have is disagreeing with each other - and the failure they can now have is
/// agreeing on something new. A field order, a width or the dimension reversal edited in that
/// single place would move both sides in step, the round trip would still pass, and every
/// file already on disk would stop loading. So the offsets below are spelled out rather than
/// derived, and the round trip is only the second half of the test.
#[test]
fn gguf_container_bytes_are_what_the_reader_expects() {
    use crate::tensor::gguf_write::{write_gguf, GgufEntry};

    let data: Vec<u8> = (0..24u8).collect(); // 2x3 F32 = 24 bytes
    let entries = vec![GgufEntry {
        name: "a".to_string(),
        dims: vec![2, 3],
        dtype: GgmlDType::F32,
        data: data.clone(),
    }];
    let name = format!("loken-gguf-layout-{}.gguf", std::process::id());
    let path = std::env::temp_dir().join(name);
    write_gguf(&path, &entries).unwrap();
    let raw = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());

    // Header: magic, version, tensor count, key/value count.
    assert_eq!(&raw[0..4], b"GGUF".as_slice(), "magic");
    assert_eq!(u32_at(4), 3, "version");
    assert_eq!(u64_at(8), 1, "tensor count");
    assert_eq!(u64_at(16), 0, "key/value count");

    // Descriptor: name, dim count, the dims innermost-first, dtype id, offset.
    assert_eq!(u64_at(24), 1, "name length");
    assert_eq!(raw[32], b'a', "name bytes");
    assert_eq!(u32_at(33), 2, "dim count");
    assert_eq!(u64_at(37), 3, "innermost dim comes first");
    assert_eq!(u64_at(45), 2, "outermost dim comes last");
    assert_eq!(u32_at(53), GgmlDType::F32.to_u32(), "dtype id");
    assert_eq!(u64_at(57), 0, "payload offset");

    // The payload starts at the first 32-byte boundary past the 65-byte directory, each
    // tensor is padded up to the next one, and every gap is zeroed.
    assert_eq!(raw.len(), 128, "file length");
    assert!(raw[65..96].iter().all(|b| *b == 0), "directory padding");
    assert_eq!(&raw[96..120], &data[..], "payload");
    assert!(raw[120..128].iter().all(|b| *b == 0), "payload padding");

    // And the reader lands on exactly those fields.
    let mut cursor = std::io::Cursor::new(&raw[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    assert_eq!(content.magic, gguf_file::VersionedMagic::GgufV3);
    assert_eq!(content.tensor_data_offset, 96);
    let info = &content.tensor_infos["a"];
    assert_eq!(info.shape.dims(), [2usize, 3].as_slice(), "row-major dims");
    assert_eq!(info.ggml_dtype, GgmlDType::F32);
    assert_eq!(info.offset, 0);
    assert_eq!(info.size_in_bytes(), data.len());
    let host = content.host_tensor(&mut cursor, "a").unwrap();
    assert_eq!(
        host.data(),
        &data[..],
        "tensor bytes survive the round trip"
    );
}

// -- A quantised forward on a device that allocates nothing -------------------

/// A counted quantised matmul must produce the tensor the real one does.
///
/// This is the gate the whole measurement rests on. A reserve read off a dry run is
/// worth exactly as much as the agreement between the shape the dry arm reports and the
/// shape the kernels write - and the dry arm cannot derive it independently, or it
/// becomes the second description of the forward that this mechanism exists to remove.
/// Every activation dtype the real path serves is run twice here, once computing and
/// once counting.
#[test]
fn a_counted_quantized_matmul_answers_the_shape_the_real_one_does() {
    use super::{GgmlDType, QKernelMatMul, QTensor};
    use crate::tensor::{DType, Device, Tensor};
    let (n, k) = (64usize, 256usize);
    let vals: Vec<f32> = (0..n * k)
        .map(|i| ((i % 53) as f32 - 26.0) / 11.0)
        .collect();
    let w = Tensor::from_vec(vals, (n, k), &Device::Cpu).unwrap();
    let q = QTensor::quantize_onto(&w, GgmlDType::Q4K, &Device::Cpu).unwrap();
    let blocks = q.native_qtensor().clone();

    let dry = Device::dry();
    let counted = QKernelMatMul::from_qtensor_on(blocks.clone(), &dry).unwrap();
    let real = QKernelMatMul::from_qtensor_on(blocks, &Device::Cpu).unwrap();

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for rows in [1usize, 7, 64] {
            let xr = Tensor::zeros((rows, k), DType::F32)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let xd = Tensor::dry(&dry, dtype, (rows, k)).unwrap();
            let a = real.forward(&xr).unwrap();
            let b = counted.forward(&xd).unwrap();
            assert_eq!(a.dims(), b.dims(), "{dtype:?} rows={rows}: shape");
            assert_eq!(a.dtype(), b.dtype(), "{dtype:?} rows={rows}: dtype");
        }
    }
}

/// The weight is charged where the upload happens, and given back where it is freed.
///
/// A placement decision reads two numbers off one run - what the weights hold and what
/// the forward adds on top - and they are told apart by WHEN each is charged. A weight
/// counted lazily at the first matmul lands inside the forward's window and is read as
/// activation, which is the difference between a model that fits a card and one whose
/// blocks are pushed onto the host.
#[test]
fn a_counted_quantized_weight_costs_its_padded_blob_from_the_moment_it_is_placed() {
    use super::{GgmlDType, QKernelMatMul, QTensor, BLOB_TAIL_PAD_BYTES};
    use crate::tensor::{DType, Device, Tensor};
    let (n, k) = (64usize, 256usize);
    let vals: Vec<f32> = (0..n * k).map(|i| (i % 29) as f32 * 0.07 - 1.0).collect();
    let w = Tensor::from_vec(vals, (n, k), &Device::Cpu).unwrap();
    let q = QTensor::quantize_onto(&w, GgmlDType::Q4K, &Device::Cpu).unwrap();
    let blocks = q.native_qtensor().clone();
    let want = (blocks.data().len() + BLOB_TAIL_PAD_BYTES) as u64;

    let dry = Device::dry();
    let led = dry.dry_ledger().unwrap().clone();
    let mm = QKernelMatMul::from_qtensor_on(blocks, &dry).unwrap();
    assert_eq!(
        led.live_bytes(),
        want,
        "the weight was not charged at placement, so it will be counted as activation"
    );

    // One forward against a resident weight: what it adds is the activation, and the
    // weight is not paid for twice. The input is in hand before the window opens - it
    // is what the caller brings, not what the matmul leaves behind.
    let x = Tensor::dry(&dry, DType::F32, (1usize, k)).unwrap();
    let resident = led.open_window();
    let y = mm.forward(&x).unwrap();
    let forward = led.peak_bytes() - resident;
    assert_eq!(
        forward,
        (n * 4) as u64,
        "a decode row's output is the only thing the matmul leaves at this layer"
    );
    drop(y);
    drop(x);
    drop(mm);
    assert_eq!(
        led.live_bytes(),
        0,
        "the weight kept its room after being dropped"
    );
}

/// A counted weight refuses an activation that is not counted with it.
///
/// The reason is the same one that makes two dry devices two devices: a matmul whose
/// operands live in different places is not a matmul the hardware would run, and
/// answering it with a plausible shape would report a peak for a forward that cannot
/// happen.
#[test]
fn a_counted_weight_refuses_a_real_activation() {
    use super::{GgmlDType, QKernelMatMul, QTensor};
    use crate::tensor::{DType, Device, Tensor};
    let (n, k) = (32usize, 256usize);
    let vals: Vec<f32> = (0..n * k).map(|i| (i % 13) as f32 * 0.1).collect();
    let w = Tensor::from_vec(vals, (n, k), &Device::Cpu).unwrap();
    let q = QTensor::quantize_onto(&w, GgmlDType::Q8_0, &Device::Cpu).unwrap();
    let mm = QKernelMatMul::from_qtensor_on(q.native_qtensor().clone(), &Device::dry()).unwrap();
    let x = Tensor::zeros((1usize, k), DType::F32).unwrap();
    assert!(
        mm.forward(&x).is_err(),
        "a counted weight served a real activation"
    );
    let other = Device::dry();
    let elsewhere = Tensor::dry(&other, DType::F32, (1usize, k)).unwrap();
    assert!(
        mm.forward(&elsewhere).is_err(),
        "a counted weight served an activation from another ledger"
    );
}

/// The dense entries a GGUF carries beside its blocks are counted, not decoded.
///
/// `token_embd` is the case that matters: at a 150k vocabulary its F32 form is over a
/// gigabyte, and a dry run that materialises it to answer a placement question has
/// allocated the very thing it promised not to - on the host, where no ledger sees it.
#[test]
fn a_dry_builder_reads_a_dense_entry_as_a_shape() {
    use super::{GgmlDType, QVarBuilder};
    use crate::tensor::{DType, Device};
    let dims = vec![8usize, 32];
    let vals: Vec<f32> = (0..8 * 32).map(|i| i as f32 * 0.5).collect();
    let bytes = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::F32, &vals).unwrap();
    let entry = crate::tensor::gguf_write::GgufEntry {
        name: "norm.weight".to_string(),
        dtype: GgmlDType::F32,
        dims: dims.clone(),
        data: bytes,
    };
    let dry = Device::dry();
    let led = dry.dry_ledger().unwrap().clone();
    let vb = QVarBuilder::from_quantized_entries(vec![entry], &dry).unwrap();
    let before = led.live_bytes();
    let t = vb.get_f32(dims.clone(), "norm.weight").unwrap();
    assert_eq!(t.dims(), dims.as_slice());
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(
        led.live_bytes() - before,
        (8 * 32 * 4) as u64,
        "a dense entry must cost its dense form on the ledger and nothing on the host"
    );
    assert!(
        !led.read_absent_data(),
        "the builder read values a dry run does not have"
    );
}

/// The blob an ollama tag names, or `None` when the manifest is there and the weights
/// are not - a manifest without its layer is an empty measurement, never a small one.
#[cfg(test)]
fn ollama_blob(tag: &str) -> Option<std::path::PathBuf> {
    let dir = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, ver) = tag.split_once(':').unwrap_or((tag, "latest"));
    let text = std::fs::read_to_string(
        dir.join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(ver),
    )
    .ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let digest = v["layers"]
        .as_array()?
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")?["digest"]
        .as_str()?
        .replace(':', "-");
    let blob = dir.join("blobs").join(digest);
    blob.exists().then_some(blob)
}

/// What a GGUF checkpoint would take from a card, run rather than estimated.
///
/// The load and the forward are the real ones - the same constructor the server calls,
/// the same plan shape, the same tensors in the same order - on a device that hands out
/// nothing and keeps a running total. What comes back is two numbers per model: what is
/// still held once the load is done, and what one forward adds on top of that.
///
/// Ignored by default because it needs the checkpoints on disk. Run it with
/// `cargo test --lib --release costs_a_card -- --ignored --nocapture`.
#[test]
#[ignore]
fn what_a_gguf_load_and_one_forward_would_cost_a_card() {
    use crate::inference::engine::llm_engine::KvQuant;
    use crate::inference::generic_transformer::GenericHeteroTransformer;
    use crate::inference::place::layer_executor::HeteroPlan;
    use crate::tensor::{Device, Tensor};
    use std::collections::HashMap;

    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    for tag in [
        "qwen3:1.7b",
        "granite3-moe:1b",
        "olmoe:latest",
        "llama3.2:1b",
    ] {
        let Some(path) = ollama_blob(tag) else {
            eprintln!("{tag:<18} no weights on disk - skipped");
            continue;
        };
        for tokens in [1usize, 512] {
            let file = std::fs::File::open(&path).unwrap();
            let file_size = file.metadata().unwrap().len();
            let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
            let content = super::gguf_file::Content::read_mapped(
                &mut std::io::Cursor::new(&mmap[..]),
                mmap.clone(),
            )
            .unwrap();
            let num_layers = content
                .metadata
                .iter()
                .find(|(k, _)| *k == "block_count" || k.ends_with(".block_count"))
                .and_then(|(_, v)| v.to_u32().ok())
                .unwrap() as usize;

            let dry = Device::dry();
            let led = dry.dry_ledger().unwrap().clone();
            let mut devices: HashMap<usize, Device> = HashMap::new();
            devices.insert(0, dry.clone());
            // One card with room for anything. The question here is what the model
            // holds, not whether it fits - a budget that forced a split would measure a
            // different arrangement from the one the numbers are compared against.
            let plan = HeteroPlan::calculate_with_kv(
                num_layers,
                file_size,
                &[(0usize, 1u64 << 40)],
                &[],
                1.0,
                0,
            );
            let mut model = match GenericHeteroTransformer::from_gguf_with_kv_quant(
                content,
                &mmap[..],
                &devices,
                &plan,
                KvQuant::Off,
                Some(4096),
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{tag:<18} load refused: {e}");
                    break;
                }
            };
            let weights = led.live_bytes();
            // The load's staging is behind us; what follows is one forward against a
            // card that already holds the weights.
            let resident = led.open_window();
            let ids: Vec<u32> = vec![0; tokens];
            let x = Tensor::from_vec(ids, (1usize, tokens), &Device::Cpu).unwrap();
            match model.forward(&x, 0) {
                Ok(out) => {
                    let peak = led.peak_bytes();
                    drop(out);
                    eprintln!(
                        "{tag:<18} tokens {tokens:<4} file {:>9.1} counted {:>9.1} \
                         forward {:>9.1} MiB  allocations {} blind {}",
                        mib(file_size),
                        mib(weights),
                        mib(peak - resident),
                        led.allocations(),
                        led.read_absent_data(),
                    );
                }
                Err(e) => {
                    eprintln!(
                        "{tag:<18} tokens {tokens:<4} file {:>9.1} counted {:>9.1} MiB  \
                         forward refused: {e}",
                        mib(file_size),
                        mib(weights),
                    );
                }
            }
        }
    }
}
