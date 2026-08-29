//! The known-answer test that decides whether a card may run the quantised kernels.
//!
//! A GPU that returns wrong numbers from a kernel is worse than one that refuses it: the
//! model answers, and the answer is nonsense. Each device is asked, once, to compute a matmul
//! whose result is known; a card that fails is gated off that kernel family for the life of
//! the process and takes the dequantised path instead.

use super::*;

#[cfg(feature = "cuda")]
pub(super) fn kat_bad(
    set: &std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>>,
    ordinal: usize,
) -> bool {
    set.get()
        .map(|m| {
            m.lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&ordinal)
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn kat_gate(
    set: &std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>>,
    ordinal: usize,
) {
    set.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(ordinal);
}

/// Boot known-answer test for the CUDA quantized matmul families on one device.
///
/// Builds a small deterministic weight, quantizes it once, runs the SAME public forward
/// on the CPU (the reference dequant arithmetic) and on the device at the row counts
/// that select each family - one row for the MMVQ GEMV, a wide batch for the MMQ tiled
/// GEMM - and gates any family whose answer is non-finite or far from the reference.
/// Returns (probe name, passed) per probe for the caller to log.
#[cfg(feature = "cuda")]
pub fn kernel_known_answer_test(device: &crate::tensor::Device) -> Vec<(&'static str, bool)> {
    let crate::tensor::Device::Cuda(qdev) = device else {
        return Vec::new();
    };
    let ordinal = qdev.ordinal();
    let mut report = Vec::new();
    let (out_f, k) = (16usize, 256usize);
    // Deterministic xorshift values in [-1, 1), the same recipe the repack parity
    // tests use - no RNG dependency, identical on every machine.
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8388608.0) - 1.0
    };
    let w: Vec<f32> = (0..out_f * k).map(|_| next()).collect();
    let x1: Vec<f32> = (0..k).map(|_| next()).collect();
    let x8: Vec<f32> = (0..8 * k).map(|_| next()).collect();
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        let probe = |xs: &[f32], rows: usize| -> Result<f32> {
            let wt = crate::tensor::Tensor::from_vec_f32(w.clone(), vec![out_f, k])?;
            let make = |dev: &crate::tensor::Device| -> Result<QKernelMatMul> {
                let qt = QTensor::quantize(&wt, dtype)?;
                QKernelMatMul::from_qtensor_on(qt.native_qtensor().clone(), dev)
            };
            let x_cpu = crate::tensor::Tensor::from_vec_f32(xs.to_vec(), vec![rows, k])?;
            let want = make(&crate::tensor::Device::Cpu)?
                .forward(&x_cpu)?
                .to_vec_f32();
            let x_dev = x_cpu.to_device(device)?;
            let got = make(device)?
                .forward(&x_dev)?
                .to_device(&crate::tensor::Device::Cpu)?
                .to_vec_f32();
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (a, b) in want.iter().zip(got.iter()) {
                if !b.is_finite() {
                    return Ok(f32::INFINITY);
                }
                num += f64::from(a - b) * f64::from(a - b);
                den += f64::from(*a) * f64::from(*a);
            }
            Ok((num / den.max(1e-12)).sqrt() as f32)
        };
        let name_gemv: &'static str = match dtype {
            GgmlDType::Q4K => "mmvq/q4_k",
            _ => "mmvq/q8_0",
        };
        let name_gemm: &'static str = match dtype {
            GgmlDType::Q4K => "mmq/q4_k",
            _ => "mmq/q8_0",
        };
        let ok1 = matches!(probe(&x1, 1), Ok(rel) if rel < 1e-2);
        if !ok1 {
            kat_gate(&KAT_BAD_MMVQ, ordinal);
        }
        report.push((name_gemv, ok1));
        let ok8 = matches!(probe(&x8, 8), Ok(rel) if rel < 1e-2);
        if !ok8 {
            if mmq_supports(dtype) {
                kat_gate(&KAT_BAD_MMQ, ordinal);
            } else {
                kat_gate(&KAT_BAD_MMVQ, ordinal);
            }
        }
        report.push((name_gemm, ok8));
    }
    report
}
