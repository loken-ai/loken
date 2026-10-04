//! A weight changes representation only where the code says so, and says why.
//!
//! Dequantising, requantising, widening or repacking a stored weight costs resident bytes,
//! speed or numerics, and none of them fails: the model loads and answers. Several were found
//! one at a time by their symptom. This gate reads the tree for the call sites that change a
//! stored representation and requires each to carry a `CONVERSION: <reason>` comment on its
//! line or in the comment block above it. The sites that do not yet are counted per file in
//! `CONVERSION_DEBT`, and the count only goes down.
//!
//! Counted calls: the dequantisers, the quantisers, the raw-bytes storage constructor and the
//! repacks, everywhere in production code; and `.to_dtype(` in the files that load weights,
//! where it widens a stored tensor rather than casting an activation.

use super::placement_invariants::{code_only, fleet_sources, production};

/// Calls that turn a stored weight into another representation, wherever they appear.
const CONVERSION_CALLS: [&str; 7] = [
    ".dequantize(",
    ".dequantize_f16(",
    ".dequantize_f32(",
    "quantize_onto(",
    "QTensor::quantize(",
    "QStorage::from_data(",
    "repack_",
];

/// The cast, counted only in the files that read checkpoints into resident weights.
const CAST_CALL: &str = ".to_dtype(";

/// Files whose casts act on stored weights.
const WEIGHT_LOADERS: [&str; 5] = [
    "inference/generic_transformer/loader.rs",
    "inference/engine/llm_engine/loading.rs",
    "inference/engine/llm_engine/load.rs",
    "tensor/safetensors_io.rs",
    "tensor/varbuilder.rs",
];

/// How many lines above a call its exemption may sit.
const EXEMPTION_REACH: usize = 8;

/// The marker an exempted site carries.
const MARKER: &str = "CONVERSION:";

/// Whether a line's code (comment removed) is a call this gate counts.
fn is_conversion(rel: &str, line: &str) -> bool {
    let file = rel.rsplit('/').next().unwrap_or(rel);
    if file.starts_with("tests_") || file == "kat.rs" {
        return false;
    }
    let code = code_only(line);
    let t = code.trim_start();
    if t.starts_with("fn ")
        || t.starts_with("pub fn ")
        || t.starts_with("pub(crate) fn ")
        || t.starts_with("use ")
        || t.starts_with("mod ")
        || t.starts_with("pub mod ")
        || t.starts_with("pub(crate) mod ")
    {
        return false;
    }
    if CONVERSION_CALLS.iter().any(|c| code.contains(c)) {
        return true;
    }
    WEIGHT_LOADERS.contains(&rel) && code.contains(CAST_CALL)
}

/// The unannotated conversion sites of one file's production text, as line numbers.
fn unannotated(rel: &str, text: &str) -> Vec<usize> {
    let lines: Vec<&str> = production(text).lines().collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !is_conversion(rel, l) {
            continue;
        }
        let from = i.saturating_sub(EXEMPTION_REACH);
        let exempt = l.contains(MARKER)
            || lines[from..i]
                .iter()
                .any(|p| p.trim_start().starts_with("//") && p.contains(MARKER));
        if !exempt {
            out.push(i + 1);
        }
    }
    out
}

fn counts() -> std::collections::BTreeMap<String, usize> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = std::collections::BTreeMap::new();
    for path in fleet_sources() {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let n = unannotated(&rel, &text).len();
        if n > 0 {
            out.insert(rel, n);
        }
    }
    out
}

/// Files with conversion sites that carry no `CONVERSION:` reason yet, with how many. The
/// list only shrinks: annotate a site, or remove the conversion, and lower the count.
/// Regenerate with `cargo test conversion_debt_listing -- --ignored --nocapture`.
const CONVERSION_DEBT: &[(&str, usize)] = &[
    ("inference/generic_transformer/loader.rs", 27),
    ("inference/generic_transformer/projection.rs", 2),
    ("inference/load/calibration.rs", 1),
    ("inference/load/requantize.rs", 1),
    ("inference/model/acestep/dit/mod.rs", 1),
    ("inference/model/acestep/music.rs", 1),
    ("inference/model/acestep/vae.rs", 1),
    ("inference/model/deepseek_v41/engram.rs", 1),
    ("inference/model/deepseek_v41/load.rs", 1),
    ("inference/model/flux/opencl.rs", 3),
    ("inference/model/moondream/vision.rs", 1),
    ("inference/model/pixtral/mod.rs", 1),
    ("inference/model/qwen25/vision.rs", 1),
    ("inference/model/qwen3/encoder.rs", 1),
    ("inference/model/qwen35/vision.rs", 1),
    ("inference/model/qwen_image/textenc.rs", 1),
    ("inference/model/wan/dit/layers.rs", 1),
    ("inference/moe_cuda_cpu/mod.rs", 25),
    ("inference/offload/cuda.rs", 1),
    ("inference/offload/weight_stream.rs", 1),
    ("inference/serve/tp_model.rs", 3),
    ("tensor/layer/norm.rs", 1),
    ("tensor/quant_cpu/avx/superblock.rs", 1),
    ("tensor/quant_cpu/drivers.rs", 37),
    ("tensor/quant_cpu/mod.rs", 1),
    ("tensor/quant_cpu/repack_q8_0_x8.rs", 3),
    ("tensor/quantized/kernel_matmul.rs", 45),
    ("tensor/quantized/kernel_matmul/repack.rs", 10),
    ("tensor/quantized/matmul.rs", 2),
    ("tensor/quantized/qtensor.rs", 3),
    ("tensor/quantized/varbuilder.rs", 1),
    ("tensor/safetensors_io.rs", 1),
    ("tensor/varbuilder.rs", 1),
];

#[test]
fn the_scanner_counts_an_unannotated_site_and_skips_an_annotated_one() {
    let bare = "fn f() {\n    let w = q.dequantize(&dev)?;\n}\n";
    assert_eq!(unannotated("x.rs", bare), vec![2]);
    let above = "fn f() {\n    // CONVERSION: the kernel reads f32 only.\n    let w = q.dequantize(&dev)?;\n}\n";
    assert!(unannotated("x.rs", above).is_empty());
    let same = "let w = q.dequantize(&dev)?; // CONVERSION: needed\n";
    assert!(unannotated("x.rs", same).is_empty());
    let far = format!(
        "// CONVERSION: stale\n{}let w = q.dequantize(&dev)?;\n",
        "let a = 1;\n".repeat(EXEMPTION_REACH + 1)
    );
    assert_eq!(unannotated("x.rs", &far).len(), 1);
    let prose = "// the dequantize( call is slow\nfn dequantize(&self) {}\n";
    assert!(unannotated("x.rs", prose).is_empty());
    let cast = "let t = w.to_dtype(DType::F32)?;\n";
    assert!(unannotated("tensor/ops/x.rs", cast).is_empty());
    assert_eq!(unannotated("tensor/varbuilder.rs", cast), vec![1]);
    let tests = "fn a() {}\n#[cfg(test)]\nmod t { fn b() { q.dequantize(&d); } }\n";
    assert!(unannotated("x.rs", tests).is_empty());
}

#[test]
fn every_conversion_site_states_its_reason() {
    let found = counts();
    let debt: std::collections::BTreeMap<String, usize> = CONVERSION_DEBT
        .iter()
        .map(|(f, n)| (f.to_string(), *n))
        .collect();
    let mut problems = Vec::new();
    for (file, now) in &found {
        match debt.get(file) {
            None => problems.push(format!(
                "{file}: {now} conversion site(s) without a `{MARKER}` reason"
            )),
            Some(was) if now > was => problems.push(format!("{file}: {now} grew from {was}")),
            Some(was) if now < was => {
                problems.push(format!("{file}: {now} shrank from {was}; lower its debt"))
            }
            Some(_) => {}
        }
    }
    for file in debt.keys().filter(|f| !found.contains_key(*f)) {
        problems.push(format!("{file}: clean now; remove it from CONVERSION_DEBT"));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
#[ignore]
fn conversion_debt_listing() {
    for (f, n) in counts() {
        println!("    (\"{f}\", {n}),");
    }
}

#[test]
#[ignore]
fn conversion_site_listing() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for path in fleet_sources() {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for n in unannotated(&rel, &text) {
            println!(
                "SITE {rel}:{n}: {}",
                text.lines().nth(n - 1).unwrap_or("").trim()
            );
        }
    }
}
