//! One routed expert computed on the host cores from memory: gate, up and down products of a
//! decode row, the floor of what a missing expert costs once its pages are in memory. Run at
//! the dimensions of the shapes file, on weights of a fixed pattern.

use super::stats::run_for;
use super::{Context, Measured};
use crate::inference::offload::cpu_experts::bench_miss_ms;
use crate::tensor::quantized::GgmlDType;
use std::time::Duration;

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let dim = cx.shapes.hidden;
    let inter = cx.options.inter;
    let gate = format_of(&cx.options.gate_format)?;
    let down = format_of(&cx.options.down_format)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    // Each sample is a few repeats of the three products, so the clock is not the measure.
    const REPEATS: usize = 8;
    let mut failed = false;
    let samples = run_for(target, || {
        match bench_miss_ms(dim, inter, gate, down, REPEATS) {
            Some(ms) => ms * 1000.0,
            None => {
                failed = true;
                0.0
            }
        }
    });
    if failed {
        return Err(format!(
            "the kernels do not run dimensions {dim} x {inter} in {} and {}",
            cx.options.gate_format, cx.options.down_format
        ));
    }
    let bytes = (inter * dim / gate.block_size() * gate.type_size()) * 2
        + dim * inter / down.block_size() * down.type_size();
    let mut rows = vec![Measured {
        variant: format!(
            "one expert on the cores, {} gate and up, {} down",
            cx.options.gate_format, cx.options.down_format
        ),
        bytes: Some(bytes as u64),
        samples_us: samples,
        note: Some("the same weights every call, so they stay in the caches".to_string()),
    }];
    rows.extend(distinct(cx, gate, down, bytes, target));
    Ok(rows)
}

/// Experts a decode's layer leaves the host, each call on weights not read since a full cycle
/// of the set, so they come from memory: through the pool's regions, and one per thread.
fn distinct(
    cx: &Context,
    gate: GgmlDType,
    down: GgmlDType,
    bytes: usize,
    target: Duration,
) -> Vec<Measured> {
    use crate::tensor::quant_cpu::{matmul_bytes_multi, matmul_bytes_multi_serial};
    const SET: usize = 96;
    const PER_CALL: usize = 4;
    let (dim, inter) = (cx.shapes.hidden, cx.options.inter);
    let gate_len = inter * (dim / gate.block_size()) * gate.type_size();
    let down_len = dim * (inter / down.block_size()) * down.type_size();
    let weights: Vec<[Vec<u8>; 3]> = (0..SET)
        .map(|i| {
            [
                vec![(i % 251) as u8; gate_len],
                vec![(i % 241) as u8; gate_len],
                vec![(i % 239) as u8; down_len],
            ]
        })
        .collect();
    let (x, h) = (vec![0.05f32; dim], vec![0.05f32; inter]);
    let mut next = 0usize;
    let mut pick = || -> Vec<&[Vec<u8>; 3]> {
        let at = next;
        next = (next + PER_CALL) % SET;
        (0..PER_CALL).map(|k| &weights[(at + k) % SET]).collect()
    };
    let mut gate_o = vec![vec![0f32; inter]; PER_CALL];
    let mut up_o = vec![vec![0f32; inter]; PER_CALL];
    let mut out_o = vec![vec![0f32; dim]; PER_CALL];
    let xs: Vec<&[f32]> = vec![&x; PER_CALL];
    let hs: Vec<&[f32]> = vec![&h; PER_CALL];
    let note = Some(format!(
        "{PER_CALL} experts per call out of {SET} distinct ones, {} MB each",
        bytes / 1_000_000
    ));
    let mut variants = Vec::new();

    let pooled = run_for(target, || {
        let ex = pick();
        let g: Vec<&[u8]> = ex.iter().map(|e| e[0].as_slice()).collect();
        let u: Vec<&[u8]> = ex.iter().map(|e| e[1].as_slice()).collect();
        let d: Vec<&[u8]> = ex.iter().map(|e| e[2].as_slice()).collect();
        let t = std::time::Instant::now();
        let mut o: Vec<&mut [f32]> = gate_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(gate, dim, inter, &g, &xs, &mut o);
        let mut o: Vec<&mut [f32]> = up_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(gate, dim, inter, &u, &xs, &mut o);
        let mut o: Vec<&mut [f32]> = out_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(down, inter, dim, &d, &hs, &mut o);
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: "4 experts through the pool, 3 regions".to_string(),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: pooled,
        note: note.clone(),
    });

    let threaded = run_for(target, || {
        let ex = pick();
        let t = std::time::Instant::now();
        std::thread::scope(|s| {
            for ((e, go), (uo, oo)) in ex
                .iter()
                .zip(gate_o.iter_mut())
                .zip(up_o.iter_mut().zip(out_o.iter_mut()))
            {
                let (x, h) = (&x, &h);
                s.spawn(move || {
                    let _ = matmul_bytes_multi_serial(gate, dim, inter, &[&e[0]], &[x], &mut [go]);
                    let _ = matmul_bytes_multi_serial(gate, dim, inter, &[&e[1]], &[x], &mut [uo]);
                    let _ = matmul_bytes_multi_serial(down, inter, dim, &[&e[2]], &[h], &mut [oo]);
                });
            }
        });
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: "4 experts, one thread each".to_string(),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: threaded,
        note: note.clone(),
    });

    let one = run_for(target, || {
        let ex = pick();
        let t = std::time::Instant::now();
        for (e, (go, (uo, oo))) in ex
            .iter()
            .zip(gate_o.iter_mut().zip(up_o.iter_mut().zip(out_o.iter_mut())))
        {
            let _ = matmul_bytes_multi_serial(gate, dim, inter, &[&e[0]], &[&x], &mut [go]);
            let _ = matmul_bytes_multi_serial(gate, dim, inter, &[&e[1]], &[&x], &mut [uo]);
            let _ = matmul_bytes_multi_serial(down, inter, dim, &[&e[2]], &[&h], &mut [oo]);
        }
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: "4 experts, one after the other on one thread".to_string(),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: one,
        note,
    });
    variants
}

fn format_of(name: &str) -> Result<GgmlDType, String> {
    match name {
        "iq2_xxs" => Ok(GgmlDType::Iq2Xxs),
        "q2_k" => Ok(GgmlDType::Q2K),
        "q4_k" => Ok(GgmlDType::Q4K),
        "q6_k" => Ok(GgmlDType::Q6K),
        "q8_0" => Ok(GgmlDType::Q8_0),
        other => Err(format!("format {other} is not one the bench knows")),
    }
}
