//! The quantized CPU matmul answers inside the quantization envelope at every SIMD level the
//! host can run, and a lower level really is selected.
//!
//! This is its own test binary so it can cap the dispatch level without another test seeing it:
//! the level is process-wide, and the library tests assume the host's best.

use loken::cpu::{self, CpuLevel};
use loken::tensor::quantized::{GgmlDType, QHostTensor, QKernelMatMul};
use loken::tensor::{quant_cpu, Device, Tensor};
use std::sync::Arc;

/// The dispatch level is process-wide, so the tests that change it take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

const DTYPES: [GgmlDType; 7] = [
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8_0,
    GgmlDType::MxFp4,
    GgmlDType::Q4_0,
    GgmlDType::Q5_0,
];

/// `y = x . w` in f64 over the weights the engine actually holds, and the magnitude it sums
/// before cancellation, which scales the tolerance.
fn reference(x: &[f32], w: &[f32], k: usize, o: usize) -> (f64, f64) {
    let (mut acc, mut mag) = (0f64, 0f64);
    for j in 0..k {
        let t = x[j] as f64 * w[o * k + j] as f64;
        acc += t;
        mag += t.abs();
    }
    (acc, mag)
}

fn check_every_dtype(level: CpuLevel) {
    let (n, k) = (16usize, 512usize);
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i % 113) as f32) * 0.018 - 0.99)
        .collect();
    for dtype in DTYPES {
        let bytes = quant_cpu::from_float_bytes(dtype, &wf).unwrap();
        let qt = QHostTensor::from_bytes(&bytes, dtype, vec![n, k]).unwrap();
        let held = qt.dequantize_f32().unwrap();
        let qm = QKernelMatMul::from_qtensor_on(Arc::new(qt), &Device::Cpu).unwrap();
        // 1 is the decode GEMV, 4 and 9 the tiled prefill GEMM with a ragged last tile.
        for rows in [1usize, 4, 9] {
            let x: Vec<f32> = (0..rows * k)
                .map(|i| ((i % 71) as f32) * 0.027 - 0.95)
                .collect();
            let nx = Tensor::from_vec_f32(x.clone(), vec![rows, k]).unwrap();
            let got = qm.forward(&nx).unwrap().to_vec_f32();
            assert_eq!(got.len(), rows * n);
            for r in 0..rows {
                for o in 0..n {
                    let (want, mag) = reference(&x[r * k..(r + 1) * k], &held, k, o);
                    let tol = 5e-3 * mag + 1e-3;
                    let g = got[r * n + o] as f64;
                    assert!(
                        (g - want).abs() <= tol,
                        "{level:?} {dtype:?} rows={rows} [{r},{o}]: {g} vs {want} (tol {tol})"
                    );
                }
            }
        }
    }
}

#[test]
fn every_supported_level_answers_inside_the_envelope() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let levels = cpu::supported_levels();
    assert_eq!(levels.first(), Some(&CpuLevel::Generic));
    for level in levels {
        cpu::set_level_cap(Some(level));
        assert_eq!(cpu::cpu_level(), level, "the cap was not applied");
        check_every_dtype(level);
    }
    cpu::set_level_cap(None);
    assert_eq!(cpu::cpu_level(), cpu::hardware_level());
}

#[test]
fn a_cap_above_the_hardware_does_not_raise_the_level() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    cpu::set_level_cap(Some(CpuLevel::Avx512));
    assert!(cpu::cpu_level() <= cpu::hardware_level());
    cpu::set_level_cap(None);
}
