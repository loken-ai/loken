//! A model's shape is read from its checkpoint, never written in the code.
//!
//! A layer count, a width, a head count or a vocabulary size written as a constant makes
//! the loader serve one checkpoint and silently misread any other of its family. The gate
//! reads every production file that builds models and counts the constants that name a
//! dimension; the files that still carry them are listed with their counts, and the list
//! may only shrink.

use super::placement_invariants::{code_only, fleet_sources, production};

/// Directories whose code builds models from checkpoints.
const MODEL_CODE: [&str; 6] = [
    "inference/model/",
    "inference/engine/",
    "inference/media/",
    "inference/codec/",
    "inference/serve/",
    "api/",
];

/// Words a constant's name carries when it states a dimension of a model.
const DIMENSION_WORDS: [&str; 10] = [
    "LAYERS",
    "HIDDEN",
    "DIM",
    "HEAD",
    "VOCAB",
    "CONTEXT",
    "INTERMEDIATE",
    "MLP",
    "EMBED",
    "CHANNELS",
];

/// Whether a line declares an integer constant whose name states a model dimension.
fn declares_a_dimension(line: &str) -> bool {
    let t = line.trim_start();
    let t = t
        .strip_prefix("pub(crate) ")
        .or_else(|| t.strip_prefix("pub(super) "))
        .or_else(|| t.strip_prefix("pub "))
        .unwrap_or(t);
    let Some(rest) = t.strip_prefix("const ") else {
        return false;
    };
    let Some((name, ty_value)) = rest.split_once(':') else {
        return false;
    };
    let Some((ty, value)) = ty_value.split_once('=') else {
        return false;
    };
    let integer_type = ["usize", "u32", "u64", "i32", "i64"].contains(&ty.trim());
    let literal = value
        .trim()
        .trim_end_matches(';')
        .chars()
        .all(|c| c.is_ascii_digit() || c == '_');
    integer_type && literal && names_a_dimension(name.trim())
}

/// Words that make a name a bound a request is checked against: `IMAGE_MAX_DIM` caps
/// what a caller may ask for and says nothing about the model serving it.
const BOUND_WORDS: [&str; 2] = ["MAX", "MIN"];

/// Whether a constant's name states a dimension: one of its `_`-separated words begins
/// with a dimension word (`HEADS`, `EMBEDDING`), and none of them makes it a bound.
fn names_a_dimension(name: &str) -> bool {
    let words: Vec<&str> = name.split('_').collect();
    let dimension = words
        .iter()
        .any(|w| DIMENSION_WORDS.iter().any(|d| w.starts_with(d)));
    dimension && !words.iter().any(|w| BOUND_WORDS.contains(w))
}

/// Field names a config struct gives a model dimension.
const DIMENSION_FIELDS: [&str; 14] = [
    "hidden_size",
    "hidden",
    "dim",
    "n_layers",
    "num_layers",
    "num_hidden_layers",
    "n_head",
    "n_heads",
    "num_heads",
    "num_attention_heads",
    "n_kv_head",
    "head_dim",
    "vocab_size",
    "intermediate_size",
];

/// Whether a line initialises a dimension field of a config with a literal.
fn fills_a_dimension_field(line: &str) -> bool {
    let t = line.trim();
    let Some((field, value)) = t.split_once(':') else {
        return false;
    };
    let value = value.trim().trim_end_matches(',');
    !value.is_empty()
        && value.chars().all(|c| c.is_ascii_digit() || c == '_')
        && DIMENSION_FIELDS.contains(&field.trim())
}

fn dimension_constants() -> std::collections::BTreeMap<String, usize> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = std::collections::BTreeMap::new();
    for path in fleet_sources() {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        if !MODEL_CODE.iter().any(|d| rel.starts_with(d)) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let n = production(&text)
            .lines()
            .filter(|l| {
                let c = code_only(l);
                declares_a_dimension(c) || fills_a_dimension_field(c)
            })
            .count();
        if n > 0 {
            out.insert(rel, n);
        }
    }
    out
}

/// Files that still write a model dimension as a constant, with how many. The list only
/// shrinks: a file that reads its shape from the checkpoint leaves it.
const GEOMETRY_DEBT: &[(&str, usize)] = &[
    ("inference/codec/dac.rs", 2),
    ("inference/codec/melband.rs", 4),
    ("inference/engine/image_engine/load_flux.rs", 4),
    ("inference/engine/image_engine/load_others.rs", 1),
    ("inference/engine/image_engine/mod.rs", 1),
    ("inference/engine/image_engine/params.rs", 1),
    ("inference/engine/qwen_image_engine.rs", 1),
    ("inference/media/midi.rs", 1),
    ("inference/model/acestep/cond.rs", 3),
    ("inference/model/acestep/fsq.rs", 3),
    ("inference/model/acestep/textenc.rs", 3),
    ("inference/model/boogu/dit.rs", 3),
    ("inference/model/clip/vision.rs", 3),
    ("inference/model/deepseek_v41/mod.rs", 2),
    ("inference/model/flux/common.rs", 1),
    ("inference/model/flux2/dit.rs", 2),
    ("inference/model/kyutai/mimi.rs", 1),
    ("inference/model/mixformer/mod.rs", 2),
    ("inference/model/moondream/quantized.rs", 1),
    ("inference/model/moondream/vision.rs", 3),
    ("inference/model/piper/mod.rs", 2),
    ("inference/model/pixtral/mod.rs", 3),
    ("inference/model/pocket_tts/mod.rs", 1),
    ("inference/model/qwen25/vision.rs", 2),
    ("inference/model/qwen3vl/textenc.rs", 8),
    ("inference/model/qwen_image/dit.rs", 2),
    ("inference/model/sdpa.rs", 1),
    ("inference/model/sdxl/pipeline.rs", 2),
    ("inference/model/sdxl/text.rs", 10),
    ("inference/model/sdxl/unet.rs", 4),
    ("inference/model/ultravox/mod.rs", 1),
    ("inference/model/voxtral/mod.rs", 3),
    ("inference/model/wan/dit/config.rs", 2),
    ("inference/model/wan/dit/layers.rs", 7),
    ("inference/model/zimage/dit/mod.rs", 2),
    ("inference/serve/spec_kv_cache.rs", 1),
];

#[test]
fn a_constant_names_a_dimension_only_when_it_is_one() {
    assert!(declares_a_dimension("const N_LAYERS: usize = 28;"));
    assert!(declares_a_dimension(
        "pub(crate) const HIDDEN: usize = 1_024;"
    ));
    assert!(!declares_a_dimension(
        "const N_LAYERS: usize = cfg.n_layers;"
    ));
    assert!(!declares_a_dimension("const TILE: usize = 64;"));
    assert!(declares_a_dimension("const N_HEADS: usize = 16;"));
    assert!(declares_a_dimension(
        "const FREQUENCY_EMBEDDING_SIZE: usize = 256;"
    ));
    assert!(!declares_a_dimension(
        "const SECOND_STREAM_AHEAD: usize = 2;"
    ));
    assert!(!declares_a_dimension("const IMAGE_MAX_DIM: usize = 8192;"));
    assert!(!declares_a_dimension(
        "const EMBED_MAX_CHARS: usize = 8192;"
    ));
    assert!(fills_a_dimension_field("        hidden_size: 4096,"));
    assert!(!fills_a_dimension_field("        hidden_size: cfg.hidden,"));
    assert!(!fills_a_dimension_field("        chunk: 64,"));
}

#[test]
fn model_dimensions_come_from_the_checkpoint() {
    let found = dimension_constants();
    let debt: std::collections::BTreeMap<String, usize> = GEOMETRY_DEBT
        .iter()
        .map(|(f, n)| (f.to_string(), *n))
        .collect();
    let mut problems = Vec::new();
    for (file, now) in &found {
        match debt.get(file) {
            None => problems.push(format!(
                "{file}: {now} - a model dimension written as a constant"
            )),
            Some(was) if now > was => problems.push(format!("{file}: {now} grew from {was}")),
            Some(was) if now < was => {
                problems.push(format!("{file}: {now} shrank from {was}; lower its debt"))
            }
            Some(_) => {}
        }
    }
    for file in debt.keys().filter(|f| !found.contains_key(*f)) {
        problems.push(format!("{file}: clean now; remove it from GEOMETRY_DEBT"));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
