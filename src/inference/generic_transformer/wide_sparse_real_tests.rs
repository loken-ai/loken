//! The generic route against the V4.1 path, on the released V4.1 Flash file in the store.
//!
//! Ignored by default: it maps a file of hundreds of GB and runs on the host.
//! Run with
//!   cargo test --release --no-default-features --features cpu --lib -- --ignored wide_sparse_real --nocapture
//!
//! The file is read through mapped views on both routes, never copied whole into memory.
//! Each route prefills the same prompt and decodes a few greedy tokens.
//! The generic route runs first and is dropped before the V4.1 path loads.

use std::time::Instant;

use super::*;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::tensor::quantized::gguf_source::SplitGguf;

/// The store tag of the file the test reads.
const TAG: &str = "deepseek-v4.1-flash:iq2-q6attn";

/// The text whose tokens both routes prefill.
const PROMPT: &str = "The capital of France is";

/// Greedy tokens each route decodes after the prompt.
const STEPS: usize = 4;

/// Logits listed per route.
const TOP: usize = 10;

/// The path of the blob a store tag names.
fn blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, ver) = tag.split_once(':').unwrap_or((tag, "latest"));
    let manifest = std::fs::read_to_string(
        store
            .join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(ver),
    )
    .expect("manifest in the store");
    let v: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    let digest = v["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .expect("a model layer")
        .replace(':', "-");
    store.join("blobs").join(digest)
}

/// The tokens of the prompt: the file's own tokenizer, after the begin token it names.
fn prompt_ids(content: &crate::tensor::quantized::gguf_file::Content) -> Vec<u32> {
    use crate::inference::engine::llm_engine::{build_tokenizer_from_gguf, EMBEDDED_TOKENIZER_KEY};
    let text = |k: &str| {
        content
            .metadata
            .get(k)
            .and_then(|v| v.to_string().ok().cloned())
    };
    let tokenizer = match text(EMBEDDED_TOKENIZER_KEY) {
        Some(json) => tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap(),
        None => build_tokenizer_from_gguf(content).unwrap(),
    };
    let mut ids: Vec<u32> = content
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok())
        .into_iter()
        .collect();
    ids.extend_from_slice(tokenizer.encode(PROMPT, false).unwrap().get_ids());
    ids
}

/// The last row of logits of any rank.
fn row(t: &Tensor) -> Vec<f32> {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cols = *t.dims().last().unwrap();
    v[v.len() - cols..].to_vec()
}

fn argmax(r: &[f32]) -> u32 {
    r.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn top(r: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = r.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(TOP);
    v
}

/// The resident set of this process in MB.
fn resident_mb() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|x| x.parse::<u64>().ok())
        })
        .map_or(0, |pages| pages * 4096 / (1 << 20))
}

/// What one route gave: the prompt's last position logits and the greedy tokens.
struct Run {
    last: Vec<f32>,
    greedy: Vec<u32>,
}

fn report(name: &str, run: &Run) {
    println!("{name} top {TOP}: {:?}", top(&run.last));
    println!("{name} greedy: {:?}", run.greedy);
}

fn generic_run(split: &SplitGguf, ids: &[u32]) -> Run {
    let content = split.merged_content();
    let arch = content.metadata["general.architecture"]
        .to_string()
        .unwrap()
        .clone();
    let layers = content
        .metadata
        .get(&format!("{arch}.block_count"))
        .and_then(|v| v.to_u32().ok())
        .expect("a block count") as usize;
    let plan = HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0);
    let t = Instant::now();
    let mut model = GenericHeteroTransformer::from_split_gguf(
        split,
        &HashMap::new(),
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    )
    .expect("the generic loader accepts the file");
    println!(
        "generic: loaded {layers} layers in {:.1}s, resident {} MB",
        t.elapsed().as_secs_f32(),
        resident_mb()
    );
    let tensor = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let t = Instant::now();
    let mut last = row(&model.forward(&tensor(ids), 0).unwrap());
    let prefill = t.elapsed().as_secs_f32();
    println!(
        "generic: prefill of {} tokens in {prefill:.1}s ({:.2}s per layer), resident {} MB",
        ids.len(),
        prefill / layers as f32,
        resident_mb()
    );
    let first = last.clone();
    let mut greedy = Vec::new();
    for i in 0..STEPS {
        let id = argmax(&last);
        greedy.push(id);
        let t = Instant::now();
        last = row(&model.forward(&tensor(&[id]), ids.len() + i).unwrap());
        println!(
            "generic: decode step {i} in {:.1}s, resident {} MB",
            t.elapsed().as_secs_f32(),
            resident_mb()
        );
    }
    Run {
        last: first,
        greedy,
    }
}

fn v41_run(split: &SplitGguf, ids: &[u32]) -> Run {
    let t = Instant::now();
    let cfg = DeepseekV41Config::from_meta(split).expect("the V4.1 config");
    let model = DeepseekV41Model::load(split, &cfg).expect("the V4.1 loader");
    println!(
        "v41: loaded in {:.1}s, resident {} MB",
        t.elapsed().as_secs_f32(),
        resident_mb()
    );
    let mut state = model.new_decode_state();
    let t = Instant::now();
    let mut last = row(&model.prefill_into(ids, &mut state).unwrap());
    println!(
        "v41: prefill in {:.1}s, resident {} MB",
        t.elapsed().as_secs_f32(),
        resident_mb()
    );
    let first = last.clone();
    let mut greedy = Vec::new();
    for _ in 0..STEPS {
        let id = argmax(&last);
        greedy.push(id);
        let t = Instant::now();
        last = row(&model.forward_decode(id, &mut state).unwrap());
        println!(
            "v41: decode step in {:.1}s, resident {} MB",
            t.elapsed().as_secs_f32(),
            resident_mb()
        );
    }
    Run {
        last: first,
        greedy,
    }
}

/// The generic route and the V4.1 path on the released file: the last position's logits and
/// the greedy tokens.
#[test]
#[ignore]
fn wide_sparse_real_generic_against_v41() {
    let path = blob(TAG);
    let split = SplitGguf::open(&path).expect("the file opens");
    let ids = prompt_ids(&split.merged_content());
    println!("prompt ids: {ids:?}");
    let generic = generic_run(&split, &ids);
    report("generic", &generic);
    let v41 = v41_run(&split, &ids);
    report("v41", &v41);

    let (mut dot, mut na, mut nb, mut num) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in generic.last.iter().zip(&v41.last) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        num += (a - b) * (a - b);
    }
    println!(
        "cosine {:.6}, relative L2 {:.6}, top-1 generic {} v41 {}, greedy equal {}",
        dot / (na.sqrt() * nb.sqrt()),
        (num / nb).sqrt(),
        argmax(&generic.last),
        argmax(&v41.last),
        generic.greedy == v41.greedy
    );
}
