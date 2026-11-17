//! The rows of a prompt through one Q5_0 projection on the host cores, from memory: the tiled
//! GEMM over the Q5_0 blocks against the tiled Q8_0 GEMM over the same values held as Q8_0
//! blocks. Run at `--rows` rows, `--hidden` inputs and `--inter` outputs.

use super::stats::run_for;
use super::{Context, Measured};
use crate::tensor::quant_cpu::{
    cast_blocks, from_float_bytes, matmul_q5_0_repacked_tiled, matmul_q8_0_x8_tiled, repack_q5_0,
    repack_q5_0_wide, BlockQ5_0,
};
use crate::tensor::quantized::GgmlDType;
use std::time::Duration;

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let (k, n, rows) = (cx.shapes.hidden, cx.options.expert_inter(), cx.options.rows);
    let block = GgmlDType::Q5_0.block_size();
    if !k.is_multiple_of(block) || !n.is_multiple_of(8) || rows == 0 {
        return Err(format!(
            "{rows} rows of {k} inputs and {n} outputs do not tile in blocks of {block} and groups of 8"
        ));
    }
    let nb = k / block;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 7) % 61) as f32 / 61.0 - 0.5)
        .collect();
    let stored = from_float_bytes(GgmlDType::Q5_0, &values).map_err(|e| e.to_string())?;
    let blocks: &[BlockQ5_0] = cast_blocks(&stored).map_err(|e| e.to_string())?;
    let x: Vec<f32> = (0..rows * k)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let mut dst = vec![0f32; rows * n];
    let weight_bytes = stored.len() as u64;
    let note = Some(format!(
        "{rows} rows x {k} inputs x {n} outputs, {} MB of Q5_0 blocks",
        weight_bytes / 1_000_000
    ));

    // CONVERSION: repacks synthetic blocks of a microbench, no stored weight.
    let g8 = repack_q5_0::repack(blocks, n, nb);
    let narrow = run_for(target, || {
        let t = std::time::Instant::now();
        let _ = matmul_q5_0_repacked_tiled((rows, k, n), &x, &g8, &mut dst);
        t.elapsed().as_secs_f64() * 1e6
    });
    let narrow_out = dst.clone();
    drop(g8);

    // CONVERSION: repacks synthetic blocks of a microbench, no stored weight.
    let x8 = repack_q5_0_wide::repack(blocks, n, nb);
    let wide = run_for(target, || {
        let t = std::time::Instant::now();
        let _ = matmul_q8_0_x8_tiled((rows, k, n), &x, &x8, &mut dst);
        t.elapsed().as_secs_f64() * 1e6
    });
    let differ = narrow_out
        .iter()
        .zip(&dst)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3 * a.abs().max(1.0))
        .count();
    Ok(vec![
        Measured {
            variant: "Q5_0 blocks, tiled Q5_0 GEMM".to_string(),
            bytes: Some(weight_bytes),
            samples_us: narrow,
            note: note.clone(),
        },
        Measured {
            variant: "the same values as Q8_0 blocks, tiled Q8_0 GEMM".to_string(),
            bytes: Some(weight_bytes),
            samples_us: wide,
            note: Some(format!(
                "{} outputs of {} differ from the first variant's past 1e-3",
                differ,
                rows * n
            )),
        },
    ])
}
