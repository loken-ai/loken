//! Tiny checkpoints written by the reference library's own `save_pretrained`, one folder per
//! architecture under `tests/vectors/hf`, loaded through the entry point the server uses and
//! compared with the reference's logits.
//!
//! `scripts/hf_vectors.py` writes each folder: `config.json` and `model.safetensors` as the
//! library saves them, and `reference.json` with the logits of the last position of a fixed
//! prompt, run in float32 on the CPU, and the greedy ids of the next tokens with the gap
//! between the two top logits at each step.
//!
//! The loader rebuilds float weights in 8 bits, which moves the logits by a fraction of a
//! percent. The reference's greedy ids are fed back in turn, and the logits of the prompt and
//! of every next token must keep a cosine of at least [`MIN_COSINE`] with the reference's.
//! The argmax must be the reference's id at every step whose two top reference logits are
//! further apart than [`TIE_FACTOR`] times the largest gap between the loaded model's prompt
//! logits and the reference's, the rounding error measured on this model; a closer step is
//! within that error and its id is not compared.

use super::*;
use crate::inference::engine::llm_engine::GenericWeights;
use crate::inference::place::layer_executor::HeteroPlan;
use serde_json::Value as Json;

const MIN_COSINE: f32 = 0.998;
/// Margin over the measured rounding error under which two top logits are a tie.
const TIE_FACTOR: f32 = 2.0;

fn vectors_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/hf")
}

/// A directory of its own for each call, the tests of one folder running side by side.
fn scratch(name: &str) -> std::path::PathBuf {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-hf-reference-{name}-{}-{call}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn floats(v: &Json, key: &str) -> Vec<f32> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference.json has {key}"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, x)| {
            if *x > bv {
                (i, *x)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb)) as f32
}

/// The files of the checkpoint in the library's layout.
const COMPOSITE: &[&str] = &["config.json", "model.safetensors"];
/// The files of the same weights in the vendor's native layout.
const NATIVE: &[&str] = &["params.json", "consolidated.safetensors"];

/// The folder's checkpoint in `files`, copied to a scratch directory with an empty tokenizer,
/// built as the server builds a dense safetensors checkpoint on a host-only plan.
fn load_files(name: &str, files: &[&str]) -> GenericHeteroTransformer {
    use crate::inference::load::awq_loader as awql;
    let src = vectors_root().join(name);
    let dir = scratch(name);
    for f in files {
        std::fs::copy(src.join(f), dir.join(f)).unwrap_or_else(|e| panic!("{name}: {f}: {e}"));
    }
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let dir_str = dir.to_string_lossy().to_string();
    let built = (|| {
        if awql::find_dense_model_dir(&dir_str).as_deref() != Some(dir_str.as_str()) {
            return Err(crate::tensor::Error::msg("not read as a dense checkpoint"));
        }
        let (_, shards) = awql::checkpoint_shards(&dir).unwrap();
        let layers = awql::parse_config_json(&dir_str)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?
            .n_layers;
        GenericWeights::Awq {
            dir: &dir_str,
            shards: &shards,
        }
        .build(
            &HashMap::new(),
            &HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0),
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
            false,
            false,
        )
    })();
    let _ = std::fs::remove_dir_all(&dir);
    built.unwrap_or_else(|e| panic!("{name}: load failed: {e}"))
}

fn logits(model: &mut GenericHeteroTransformer, ids: &[u32], pos: usize) -> Vec<f32> {
    let input = Tensor::from_vec(ids.to_vec(), (1usize, ids.len()), &Device::Cpu).unwrap();
    model
        .forward(&input, pos)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn reference_of(name: &str) -> Json {
    let text = std::fs::read_to_string(vectors_root().join(name).join("reference.json"))
        .unwrap_or_else(|e| panic!("{name}: reference.json: {e}"));
    serde_json::from_str(&text).unwrap()
}

/// The reference's prompt and its greedy ids.
fn prompt_and_next(reference: &Json) -> (Vec<u32>, Vec<u32>) {
    let prompt: Vec<u32> = reference["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let next: Vec<u32> = floats(reference, "next_ids")
        .iter()
        .map(|x| *x as u32)
        .collect();
    (prompt, next)
}

/// The folder's checkpoint gives the reference's logits over the prompt and at each next
/// token, the reference's greedy ids fed back in turn.
fn check(name: &str) {
    check_files(name, COMPOSITE);
}

/// [`check`] of the folder's `files`.
fn check_files(name: &str, files: &[&str]) {
    let reference = reference_of(name);
    let (prompt, next) = prompt_and_next(&reference);
    let margins = floats(&reference, "next_margins");
    let steps: Vec<Vec<f32>> = reference["step_logits"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}: reference.json has step_logits"))
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    assert_eq!(steps.len(), next.len(), "{name}: logits for every step");

    let mut model = load_files(name, files);
    let mut tie = 0f32;
    for (i, want) in steps.iter().enumerate() {
        let got = match i {
            0 => logits(&mut model, &prompt, 0),
            _ => logits(&mut model, &[next[i - 1]], prompt.len() + i - 1),
        };
        assert_eq!(got.len(), want.len(), "{name}: logits are the vocabulary's");
        let cos = cosine(&got, want);
        assert!(
            cos >= MIN_COSINE,
            "{name}: step {i}: cosine {cos} with the reference's logits"
        );
        // The rounding error is measured on the prompt's logits; a tenth of a percent of
        // their range stands for it when the model is exact.
        if i == 0 {
            let error = got
                .iter()
                .zip(want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let (lo, hi) = want
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), x| {
                    (l.min(*x), h.max(*x))
                });
            tie = (TIE_FACTOR * error).max(1e-3 * (hi - lo));
        }
        if margins[i] <= tie {
            eprintln!(
                "{name}: step {i}: the reference's top two logits are {} apart, within the \
                 rounding error {tie}; its greedy id is not compared",
                margins[i]
            );
            continue;
        }
        assert_eq!(
            argmax(&got) as u32,
            next[i],
            "{name}: greedy id of step {i} (reference gap {})",
            margins[i]
        );
    }
}

macro_rules! reference_logits {
    ($($test:ident: $folder:literal;)*) => {
        $(
            #[test]
            fn $test() {
                check($folder);
            }
        )*

        /// The folders the tests above read.
        const FOLDERS: &[&str] = &[$($folder),*];
    };
}

reference_logits! {
    llama_matches_reference: "llama";
    mistral_matches_reference: "mistral";
    mixtral_matches_reference: "mixtral";
    qwen2_matches_reference: "qwen2";
    qwen3_matches_reference: "qwen3";
    qwen3moe_matches_reference: "qwen3moe";
    qwen3next_matches_reference: "qwen3next";
    qwen35_matches_reference: "qwen35";
    qwen35moe_matches_reference: "qwen35moe";
    smollm3_matches_reference: "smollm3";
    ernie4_5_matches_reference: "ernie4_5";
    granite_matches_reference: "granite";
    granitemoe_matches_reference: "granitemoe";
    starcoder_matches_reference: "starcoder";
    starcoder2_matches_reference: "starcoder2";
    phi2_matches_reference: "phi2";
    phi3_matches_reference: "phi3";
    glm4_matches_reference: "glm4";
    stablelm_matches_reference: "stablelm";
    olmo2_matches_reference: "olmo2";
    olmoe_matches_reference: "olmoe";
    falcon_matches_reference: "falcon";
    dbrx_matches_reference: "dbrx";
    mpt_matches_reference: "mpt";
    gemma_matches_reference: "gemma";
    gemma2_matches_reference: "gemma2";
    gemma3_matches_reference: "gemma3";
    gemma4_matches_reference: "gemma4";
    gptoss_matches_reference: "gptoss";
    lfm2_matches_reference: "lfm2";
    lfm2moe_matches_reference: "lfm2moe";
    nemotron_h_moe_matches_reference: "nemotron_h_moe";
    gemma4_keqv_matches_reference: "gemma4_keqv";
    gemma4_moe_matches_reference: "gemma4_moe";
    mistral3_matches_reference: "mistral3";
    voxtral_matches_reference: "voxtral";
}

/// The checkpoint whose attention adds an ALiBi bias is read as one only the host runs; one
/// with rotary positions is not.
#[test]
fn an_alibi_checkpoint_runs_on_the_host_only() {
    use crate::inference::load::awq_loader as awql;
    for (name, host_only) in [("mpt", true), ("llama", false)] {
        let dir = vectors_root().join(name);
        let cfg = awql::parse_config_json(&dir.to_string_lossy()).unwrap();
        let content = awql::synth_gguf_content(&cfg, awql::AwqTensorPresence::default());
        assert_eq!(
            host_only_reason(&content, &cfg.gguf_arch).is_some(),
            host_only,
            "{name}"
        );
    }
}

/// The native layout of the folder's weights gives the composite's logits over the prompt and
/// at each next token: the loader reads the two layouts as one model.
fn native_matches_composite(name: &str) {
    let reference = reference_of(name);
    let (prompt, next) = prompt_and_next(&reference);
    let mut composite = load_files(name, COMPOSITE);
    let mut native = load_files(name, NATIVE);
    for i in 0..next.len() {
        let (ids, pos) = match i {
            0 => (prompt.clone(), 0),
            _ => (vec![next[i - 1]], prompt.len() + i - 1),
        };
        let want = logits(&mut composite, &ids, pos);
        let got = logits(&mut native, &ids, pos);
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let error = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            error <= 1e-4 * scale,
            "{name}: step {i}: the native layout's logits are {error} from the composite's"
        );
    }
}

/// The two layouts of the folder's weights give the same rotary and query-scale header.
#[test]
fn mistral3_layouts_give_one_header() {
    use crate::inference::load::awq_loader as awql;
    let header = |files: &[&str]| {
        let dir = scratch("mistral3-header");
        for f in files {
            std::fs::copy(vectors_root().join("mistral3").join(f), dir.join(f)).unwrap();
        }
        let cfg = awql::parse_config_json(&dir.to_string_lossy()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let content = awql::synth_gguf_content(&cfg, awql::AwqTensorPresence::default());
        let mut keys: Vec<(String, String)> = content
            .metadata
            .iter()
            .filter(|(k, _)| k.contains(".rope.") || k.contains(".attention.temperature"))
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect();
        keys.sort();
        keys
    };
    let composite = header(&["config.json"]);
    assert!(
        composite
            .iter()
            .any(|(k, _)| k.ends_with("attention.temperature_length")),
        "{composite:?}"
    );
    assert!(
        composite
            .iter()
            .any(|(k, _)| k.ends_with("rope.scaling.attn_factor")),
        "{composite:?}"
    );
    assert_eq!(header(&["params.json"]), composite);
}

#[test]
fn mistral3_native_matches_reference() {
    check_files("mistral3", NATIVE);
}

#[test]
fn mistral3_native_matches_composite() {
    native_matches_composite("mistral3");
}

#[test]
fn voxtral_native_matches_reference() {
    check_files("voxtral", NATIVE);
}

#[test]
fn voxtral_native_matches_composite() {
    native_matches_composite("voxtral");
}

/// Every folder under `tests/vectors/hf` has a test above, and every test its folder.
#[test]
fn every_reference_folder_has_a_test() {
    let mut found: Vec<String> = std::fs::read_dir(vectors_root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    found.sort();
    let mut listed: Vec<String> = FOLDERS.iter().map(|s| s.to_string()).collect();
    listed.sort();
    assert_eq!(found, listed, "the folders and the tests that read them");
}
