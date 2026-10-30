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
    Ok(vec![Measured {
        variant: format!(
            "one expert on the cores, {} gate and up, {} down",
            cx.options.gate_format, cx.options.down_format
        ),
        bytes: Some(bytes as u64),
        samples_us: samples,
        note: Some("weights in cache or memory as the allocation leaves them".to_string()),
    }])
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
