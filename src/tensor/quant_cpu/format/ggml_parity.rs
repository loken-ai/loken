//! Every weight format loken writes, held to ggml's encoder on the same rows.
//!
//! The fixture holds synthetic rows, an importance per column, and for each format the blocks ggml
//! wrote with that importance, the first rows of those blocks as ggml reads them, and the error
//! ggml's blocks leave with and without the importance. A format passes when loken reads ggml's
//! blocks to the same values, and its own blocks leave at most two percent more error than ggml's,
//! weighted and plain.

use super::*;

/// How far loken's error may sit above ggml's on the same rows.
const MARGIN: f64 = 1.02;

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/tests/vectors/ggml_encode.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn hex(v: &serde_json::Value) -> Vec<u8> {
    let s = v.as_str().unwrap();
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    hex(v)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// `sqrt(sum w (got - x)^2 / sum w x^2)`, `w` one weight per column.
fn error(x: &[f32], got: &[f32], w: &[f32]) -> f64 {
    let cols = w.len();
    let (mut num, mut den) = (0f64, 0f64);
    for (i, (&g, &t)) in got.iter().zip(x).enumerate() {
        let wi = w[i % cols] as f64;
        num += wi * (g as f64 - t as f64).powi(2);
        den += wi * (t as f64).powi(2);
    }
    (num / den).sqrt()
}

/// Every format the fixture holds, by its ggml name.
fn dtype_of(name: &str) -> GgmlDType {
    crate::inference::load::requantize::parse_dtype(name)
        .unwrap_or_else(|| panic!("{name}: loken does not write it"))
}

/// One format's blocks for `x`, weighed by `importance` when the format takes one.
fn encode(dtype: GgmlDType, x: &[f32], importance: Option<&[f32]>, cols: usize) -> Vec<u8> {
    match importance {
        Some(imp) => {
            let mut out = Vec::new();
            for row in x.chunks(cols) {
                out.extend(
                    crate::inference::load::calibrated::quantize_projection(
                        dtype,
                        row,
                        1,
                        cols,
                        Some(&(imp.to_vec(), Vec::new(), 0)),
                        false,
                        crate::inference::load::calibrated::Recipe {
                            min_rows: usize::MAX,
                            damping: 1.0,
                        },
                    )
                    .unwrap(),
                );
            }
            out
        }
        None => crate::tensor::quant_cpu::from_float_bytes(dtype, x).unwrap(),
    }
}

#[test]
fn every_format_encodes_as_well_as_ggml_and_reads_its_blocks() {
    let v = fixture();
    let (rows, cols, read) = (
        v["rows"].as_u64().unwrap() as usize,
        v["cols"].as_u64().unwrap() as usize,
        v["read"].as_u64().unwrap() as usize,
    );
    let x = floats(&v["x"]);
    let importance = floats(&v["importance"]);
    let ones = vec![1f32; cols];
    assert_eq!(x.len(), rows * cols);
    let mut failures = Vec::new();
    for (name, f) in v["formats"].as_object().unwrap() {
        let dtype = dtype_of(name);

        // ggml's blocks, read by loken, are the values ggml reads.
        let blocks = hex(&f["blocks"]);
        let mut back = vec![0f32; rows * cols];
        crate::tensor::quant_cpu::to_float_bytes(dtype, &blocks, &mut back).unwrap();
        let expected = floats(&f["expected"]);
        if back[..read * cols] != expected[..] {
            failures.push(format!("{name}: ggml's blocks read back to other values"));
        }

        let guided = crate::inference::load::calibrated::guided(dtype);
        let mut got = vec![0f32; rows * cols];
        let ours = encode(dtype, &x, guided.then_some(&importance[..]), cols);
        crate::tensor::quant_cpu::to_float_bytes(dtype, &ours, &mut got).unwrap();
        let (ours_w, theirs_w) = (
            error(&x, &got, &importance),
            f["weighted_error"].as_f64().unwrap(),
        );
        let ours = encode(dtype, &x, None, cols);
        crate::tensor::quant_cpu::to_float_bytes(dtype, &ours, &mut got).unwrap();
        let (ours_p, theirs_p) = (
            error(&x, &got, &ones),
            f["unweighted_error"].as_f64().unwrap(),
        );
        eprintln!(
            "{name:8} weighted {ours_w:.5} ggml {theirs_w:.5} ({:+.2}%)   plain {ours_p:.5} ggml {theirs_p:.5} ({:+.2}%)",
            100.0 * (ours_w / theirs_w - 1.0),
            100.0 * (ours_p / theirs_p - 1.0)
        );
        if ours_w > theirs_w * MARGIN {
            failures.push(format!(
                "{name}: weighted {ours_w} against ggml's {theirs_w}"
            ));
        }
        if ours_p > theirs_p * MARGIN {
            failures.push(format!("{name}: plain {ours_p} against ggml's {theirs_p}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
