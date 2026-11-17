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
    load_files_on(name, files, &HashMap::new())
}

/// [`load_files`] with every layer on card 0 of `cards`, on the host when `cards` is empty.
fn load_files_on(
    name: &str,
    files: &[&str],
    cards: &HashMap<usize, Device>,
) -> GenericHeteroTransformer {
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
        let plan = match cards.is_empty() {
            true => HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0),
            false => HeteroPlan::forced_gpu(layers, layers, 0),
        };
        GenericWeights::Awq {
            dir: &dir_str,
            shards: &shards,
        }
        .build(
            cards,
            &plan,
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
    glm_moe_dsa_matches_reference: "glm_moe_dsa";
    glm_moe_dsa_dense_matches_reference: "glm_moe_dsa_dense";
    glm_moe_dsa_sparse_matches_reference: "glm_moe_dsa_sparse";
    glm_moe_dsa_fp8_matches_reference: "glm_moe_dsa_fp8";
}

/// The positions a row of the reference's selection keeps that the row may see.
fn visible_picks(row: &Json, t: usize) -> Vec<u32> {
    let mut kept: Vec<u32> = row
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j.as_u64().unwrap() as u32)
        .filter(|j| *j as usize <= t)
        .collect();
    kept.sort_unstable();
    kept
}

/// Each decoder layer of the folder's checkpoint, run on the output of the layer before it, gives
/// the reference's output over the prompt with the cosine the logits keep, and a layer of a
/// sparse model keeps the tokens the reference's keeps.
fn check_layers(name: &str) {
    use super::mixer::Mixer;
    let reference = reference_of(name);
    let (prompt, _) = prompt_and_next(&reference);
    let layers = reference["layers"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}: reference.json has layers"));
    let mut model = load_files(name, COMPOSITE);
    assert_eq!(model.layers.len(), layers.len(), "{name}: the layers");
    let ids = Tensor::from_vec(prompt.clone(), (1usize, prompt.len()), &Device::Cpu).unwrap();
    let (mut hidden, _) = model.embed_stage(&ids, 0).unwrap();
    for (i, want) in layers.iter().enumerate() {
        hidden = model
            .run_layers_in(hidden, None, 0, prompt.len(), i, i + 1)
            .unwrap()
            .0;
        let got = hidden
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cos = cosine(&got, &floats(want, "hidden"));
        assert!(
            cos >= MIN_COSINE,
            "{name}: layer {i}: cosine {cos} with the reference's output"
        );
        let Some(rows) = want["selected"].as_array() else {
            continue;
        };
        let Mixer::Mla(mla) = &model.layers[i].mixer else {
            continue;
        };
        let Some(cell) = mla.selection_cell() else {
            continue;
        };
        let kept = cell.get(0, prompt.len()).unwrap();
        for (t, row) in rows.iter().enumerate() {
            let ours: Vec<u32> = match &kept.rows[t] {
                Some(k) => k.clone(),
                None => (0..=t as u32).collect(),
            };
            assert_eq!(
                ours,
                visible_picks(row, t),
                "{name}: layer {i}: the tokens row {t} attends to"
            );
        }
    }
}

macro_rules! reference_layers {
    ($($test:ident: $folder:literal;)*) => {
        $(
            #[test]
            fn $test() {
                check_layers($folder);
            }
        )*
    };
}

reference_layers! {
    glm_moe_dsa_layers_match_reference: "glm_moe_dsa";
    glm_moe_dsa_dense_layers_match_reference: "glm_moe_dsa_dense";
    glm_moe_dsa_sparse_layers_match_reference: "glm_moe_dsa_sparse";
    glm_moe_dsa_fp8_layers_match_reference: "glm_moe_dsa_fp8";
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

/// The folder's checkpoint on `cards` at the `reference` level when `narrows` is false, at the
/// level that narrows values otherwise.
fn load_at_level(
    name: &str,
    cards: &HashMap<usize, Device>,
    narrows: bool,
) -> GenericHeteroTransformer {
    use crate::inference::engine::llm_engine::FloatWeights;
    use crate::inference::generic_transformer::{set_float_weights, set_narrows};
    let _level = crate::inference::generic_transformer::LEVEL_LOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
    set_float_weights(FloatWeights::AsStored);
    set_narrows(narrows);
    let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_files_on(name, COMPOSITE, cards)
    }));
    set_narrows(true);
    set_float_weights(FloatWeights::ByPlan);
    model.unwrap_or_else(|e| std::panic::resume_unwind(e))
}

/// The dtypes of the caches the card layers of the folder's model hold after its prompt and 1
/// decoded token.
#[cfg(feature = "gpu")]
fn card_cache_dtypes(name: &str, cards: &HashMap<usize, Device>, narrows: bool) -> Vec<DType> {
    let reference = reference_of(name);
    let (prompt, next) = prompt_and_next(&reference);
    let mut model = load_at_level(name, cards, narrows);
    logits(&mut model, &prompt, 0);
    logits(&mut model, &next[..1], prompt.len());
    model
        .layers
        .iter()
        .filter(|l| l.attn_output.device().is_cuda())
        .filter_map(|l| l.kv_cache.current_kv().map(|(k, _)| k.dtype()))
        .collect()
}

/// At `reference` a card holds the cache of a layer with attention sinks, and of an architecture
/// whose quantised cache degrades, at the width the layer computes in.
/// At the other levels those caches are F16 on a card.
#[cfg(feature = "gpu")]
#[test]
fn a_card_at_reference_keeps_the_cache_at_full_width() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    for name in ["gptoss", "gemma4"] {
        let narrowed = card_cache_dtypes(name, &cards, true);
        let full = card_cache_dtypes(name, &cards, false);
        eprintln!("{name}: card caches {narrowed:?} when narrowed, {full:?} at reference");
        assert!(
            !narrowed.is_empty() && narrowed.iter().all(|d| *d == DType::F16),
            "{name}: the card caches when narrowed are {narrowed:?}"
        );
        assert!(
            full.len() == narrowed.len() && full.iter().all(|d| *d == DType::F32),
            "{name}: the card caches at reference are {full:?}"
        );
    }
}

/// The largest distance of the model's logits from the reference's over the prompt and every
/// next token, the reference's greedy ids fed back in turn, and the count of steps whose
/// argmax is the reference's id.
fn distance_and_hits(name: &str, model: &mut GenericHeteroTransformer) -> (f32, usize) {
    let reference = reference_of(name);
    let (prompt, next) = prompt_and_next(&reference);
    let mut distance = 0f32;
    let mut hits = 0;
    for (i, want) in reference["step_logits"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let want: Vec<f32> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let got = match i {
            0 => logits(model, &prompt, 0),
            _ => logits(model, &[next[i - 1]], prompt.len() + i - 1),
        };
        distance = got
            .iter()
            .zip(&want)
            .fold(distance, |m, (a, b)| m.max((a - b).abs()));
        hits += usize::from(argmax(&got) as u32 == next[i]);
    }
    (distance, hits)
}

/// How far 2 runs of the folder's model may sit apart when they sum the same products in
/// another order.
///
/// A float sum of `n` products moves by at most `n` times the unit roundoff times the sum of
/// their magnitudes when its order changes.
/// `n` is the hidden width, the length of the head's dot products.
/// The sum of the magnitudes is taken as the largest reference logit.
fn order_of(name: &str, width: usize) -> f32 {
    let largest = reference_of(name)["step_logits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s.as_array().unwrap().iter())
        .fold(0f32, |a, v| a.max(v.as_f64().unwrap().abs() as f32));
    width as f32 * f32::EPSILON * largest
}

/// The conversion a card load at `reference` may still declare: a float expert stack, which no
/// card expert kernel reads, rebuilt in Q8_0.
/// The docs list it among the trades no level makes yet.
fn open_at_reference(c: &crate::tensor::conversion::Conversion) -> bool {
    c.class == "experts"
        && ["f32", "f16", "bf16"].contains(&c.from.as_str())
        && c.to == "q8_0"
        && c.reason == "no expert kernel reads the stored format on this device"
}

/// At `reference` the host keeps the cache of a layer with attention sinks, and of an
/// architecture whose quantised cache degrades, at the width it computes in: no F16 store.
/// Its logits are no farther from the reference's than with the F16 store, and it finds as
/// many of its greedy ids.
#[test]
fn the_host_at_reference_keeps_no_f16_store() {
    for name in ["gptoss", "gemma4"] {
        let mut narrowed = load_at_level(name, &HashMap::new(), true);
        let (narrowed_distance, narrowed_hits) = distance_and_hits(name, &mut narrowed);
        let stores = narrowed
            .layers
            .iter()
            .filter(|l| l.cpu_f16_kv.is_some())
            .count();
        drop(narrowed);
        let mut model = load_at_level(name, &HashMap::new(), false);
        let (distance, hits) = distance_and_hits(name, &mut model);
        let width = model.config.embedding_length;
        let caches: Vec<DType> = model
            .layers
            .iter()
            .filter_map(|l| l.kv_cache.current_kv().map(|(k, _)| k.dtype()))
            .collect();
        eprintln!(
            "{name}: host off the reference by {distance} at reference, {narrowed_distance} \
             with {stores} F16 stores; reference ids found {hits}, {narrowed_hits}; caches \
             {caches:?}"
        );
        assert!(stores > 0, "{name}: the narrowed host keeps no F16 store");
        assert!(
            model.layers.iter().all(|l| l.cpu_f16_kv.is_none()),
            "{name}: the host at reference keeps an F16 store"
        );
        assert!(
            !caches.is_empty() && caches.iter().all(|d| *d == DType::F32),
            "{name}: the host caches at reference are {caches:?}"
        );
        assert!(
            distance <= narrowed_distance + order_of(name, width),
            "{name}: the host at reference is off by {distance}, with the store by \
             {narrowed_distance}"
        );
        assert!(
            hits >= narrowed_hits,
            "{name}: {hits} ids, {narrowed_hits} with the store"
        );
    }
}

/// At `reference` a card holds a float output head in its stored format, the load declares no
/// conversion of the head, and the card is no farther from the reference's logits than the
/// host at `reference` and finds at least as many of its greedy ids.
///
/// The card sums in another order than the host: equal is equal within [`order_of`].
/// A card load that declares a conversion fails, unless it is one [`open_at_reference`] names.
/// Then the card holds a weight narrower than stored, and the distance is printed only.
#[cfg(feature = "gpu")]
#[test]
fn a_card_at_reference_is_no_farther_from_the_reference_than_the_host() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    for name in ["gptoss", "gemma4", "qwen3"] {
        let mut host = load_at_level(name, &HashMap::new(), false);
        let (host_distance, host_hits) = distance_and_hits(name, &mut host);
        let width = host.config.embedding_length;
        drop(host);
        let mut narrowed = load_at_level(name, &cards, true);
        let (narrowed_distance, narrowed_hits) = distance_and_hits(name, &mut narrowed);
        drop(narrowed);
        crate::tensor::conversion::open();
        let mut model = load_at_level(name, &cards, false);
        let converted = crate::tensor::conversion::entries();
        let head: Vec<_> = converted.iter().filter(|e| e.class == "output").collect();
        assert!(head.is_empty(), "{name}: the head is converted: {head:?}");
        assert!(
            model.output_proj_cuda.is_none()
                && model
                    .output_proj_cuda_qmm
                    .as_ref()
                    .is_some_and(|q| q.float_format().is_some()),
            "{name}: the card does not hold the float head as stored"
        );
        let (distance, hits) = distance_and_hits(name, &mut model);
        eprintln!(
            "{name}: off the reference by {distance} on the card at reference, \
             {narrowed_distance} narrowed, {host_distance} on the host; reference ids found \
             {hits}, {narrowed_hits}, {host_hits}"
        );
        let unlisted: Vec<_> = converted.iter().filter(|c| !open_at_reference(c)).collect();
        assert!(
            unlisted.is_empty(),
            "{name}: the card load at reference declares {unlisted:?}"
        );
        if !converted.is_empty() {
            eprintln!("{name}: the card load declares {converted:?}");
            continue;
        }
        assert!(
            distance <= host_distance + order_of(name, width),
            "{name}: the card is off the reference by {distance}, the host by {host_distance}"
        );
        assert!(
            hits >= host_hits,
            "{name}: the card finds {hits} reference ids, the host {host_hits}"
        );
    }
}

/// The rows of `t`, `[1, rows, vocab]`, as host vectors.
fn logit_rows(t: &Tensor) -> Vec<Vec<f32>> {
    let t = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let vocab = t.dims().last().copied().unwrap();
    let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    flat.chunks(vocab).map(<[f32]>::to_vec).collect()
}

/// On the host, a verify of the gemma4 checkpoint gives each row the logits of the decode step
/// it stands for, bit for bit, whatever its width: its shared layers read the donor's host store
/// as a step reads it. A cut after a verify decodes on as the steps do.
#[test]
fn a_gemma4_verify_gives_its_decode_steps_on_the_host() {
    gemma4_verify_gives_its_decode_steps(&HashMap::new());
}

/// [`a_gemma4_verify_gives_its_decode_steps_on_the_host`] with every layer on a card: its
/// shared layers attend the donor's rows up to their own, as a step attends them.
#[cfg(feature = "gpu")]
#[test]
fn a_gemma4_verify_gives_its_decode_steps_on_a_card() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    gemma4_verify_gives_its_decode_steps(&[(0usize, card)].into_iter().collect());
}

/// Each row of a verify of every width on `cards` against the decode step it stands for, and
/// the steps after a cut verify against the steps of a run that never verified.
fn gemma4_verify_gives_its_decode_steps(cards: &HashMap<usize, Device>) {
    use crate::inference::engine::decode_step::VERIFY_ROWS;
    let name = "gemma4";
    let reference = reference_of(name);
    let (prompt, _) = prompt_and_next(&reference);
    let next: Vec<u32> = (0..VERIFY_ROWS as u32)
        .map(|i| (i * 37 + 5) % 128)
        .collect();
    let ids = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1usize, t.len()), &Device::Cpu).unwrap();
    let mut stepped = load_files_on(name, COMPOSITE, cards);
    assert!(
        stepped.config.shared_kv_layers > 0,
        "{name}: no shared layer"
    );
    stepped.forward(&ids(&prompt), 0).unwrap();
    let want: Vec<Vec<f32>> = next
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let l = stepped.forward(&ids(&[t]), prompt.len() + i).unwrap();
            logit_rows(&l).remove(0)
        })
        .collect();
    drop(stepped);
    let mut verified = load_files_on(name, COMPOSITE, cards);
    verified.forward(&ids(&prompt), 0).unwrap();
    for width in 2..=VERIFY_ROWS {
        let got = logit_rows(
            &verified
                .forward_all(&ids(&next[..width]), prompt.len())
                .unwrap(),
        );
        for (r, row) in got.iter().enumerate() {
            let gap = row
                .iter()
                .zip(&want[r])
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                row == &want[r],
                "{name}: row {r} of a {width}-row verify is {gap:e} off its decode step"
            );
        }
        verified.trim_kv(prompt.len());
    }
    verified
        .forward_all(&ids(&next[..5]), prompt.len())
        .unwrap();
    verified.trim_kv(prompt.len() + 2);
    for (i, &t) in next.iter().enumerate().skip(2) {
        let got = logit_rows(&verified.forward(&ids(&[t]), prompt.len() + i).unwrap()).remove(0);
        assert!(
            got == want[i],
            "{name}: step {i} after a cut verify differs"
        );
    }
}
