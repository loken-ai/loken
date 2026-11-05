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
pub(super) const TAG: &str = "deepseek-v4.1-flash:iq2-q6attn";

/// The text whose tokens both routes prefill.
pub(super) const PROMPT: &str = "The capital of France is";

/// Greedy tokens each route decodes after the prompt.
pub(super) const STEPS: usize = 4;

/// Logits listed per route.
const TOP: usize = 10;

/// The path of the blob a store tag names.
pub(super) fn blob(tag: &str) -> std::path::PathBuf {
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
pub(super) fn prompt_ids(content: &crate::tensor::quantized::gguf_file::Content) -> Vec<u32> {
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
pub(super) fn row(t: &Tensor) -> Vec<f32> {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cols = *t.dims().last().unwrap();
    v[v.len() - cols..].to_vec()
}

pub(super) fn argmax(r: &[f32]) -> u32 {
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
pub(super) fn resident_mb() -> u64 {
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

/// Drops the file's pages from the page cache, so a run that follows starts cold.
///
/// The mappings of the file must be gone: a mapped page stays.
pub(super) fn evict(path: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(path).expect("the file opens");
    // SAFETY: the descriptor is open for the whole call.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    assert_eq!(rc, 0, "posix_fadvise refused");
}

/// Seconds a closure takes, with its value.
pub(super) fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

/// Seconds per named stage, summed over the calls.
#[derive(Default)]
pub(super) struct Stages(std::collections::BTreeMap<&'static str, (f64, u64)>);

impl Stages {
    pub(super) fn time<R>(&mut self, name: &'static str, f: impl FnOnce() -> R) -> R {
        let (r, s) = timed(f);
        let e = self.0.entry(name).or_default();
        e.0 += s;
        e.1 += 1;
        r
    }

    pub(super) fn print(&self, who: &str) {
        for (name, (s, n)) in &self.0 {
            println!("{who} stage {name:<22} {s:8.3}s over {n} calls");
        }
    }
}

/// A forward of the generic model over `ids` from `pos`, stage by stage, as `forward_sinkhorn`
/// runs it. The last row of logits comes back. A layer with an engram runs whole.
pub(super) fn staged_forward(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    pos: usize,
    st: &mut Stages,
    layer_secs: &mut Vec<f64>,
) -> Vec<f32> {
    use crate::tensor::ops::rms_norm;
    let tensor = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu).unwrap();
    model.spec_clear();
    let (mut hidden, _) = st
        .time("embed", || model.embed_stage(&tensor, pos))
        .unwrap();
    let seq = ids.len();
    layer_secs.clear();
    for il in 0..model.layers.len() {
        let started = Instant::now();
        let layer = &mut model.layers[il];
        let block = layer.sinkhorn.as_ref().expect("a Sinkhorn layer");
        if block.engram.is_some() {
            hidden = st
                .time("engram layer (whole)", || {
                    layer.forward_sinkhorn(&hidden, pos)
                })
                .unwrap();
            layer_secs.push(started.elapsed().as_secs_f64());
            continue;
        }
        let r = hidden
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .contiguous()
            .unwrap();
        let pre = block.carry.get(pos, seq).unwrap();
        let mixes = st.time("hc mixes", || block.attn_mix.mixes(&r)).unwrap();
        let input = st
            .time("hc pre and norm", || {
                rms_norm(
                    &block.attn_mix.pre(&r, &pre).unwrap(),
                    &block.attn_norm,
                    block.eps,
                )
            })
            .unwrap();
        let kind = match &layer.mixer {
            super::mixer::Mixer::LatentMqa(_) => "mixer latent",
            super::mixer::Mixer::CompressedBand(_) => "mixer band",
            _ => "attention mixer",
        };
        let mixed = st.time(kind, || layer.mixer.forward(&input, pos)).unwrap();
        let (b, s, width) = mixed.dims3().unwrap();
        let out = st
            .time("attention output", || {
                block
                    .output
                    .apply_here(&mixed.reshape((b * s, width)).unwrap())
                    .unwrap()
                    .reshape((b, s, block.attn_mix.dim()))
            })
            .unwrap();
        let mid = st
            .time("hc post", || {
                block.attn_mix.post(&out, &r, &mixes.post, &mixes.comb)
            })
            .unwrap();
        let fmixes = st.time("hc mixes", || block.ffn_mix.mixes(&mid)).unwrap();
        let finput = st
            .time("hc pre and norm", || {
                rms_norm(
                    &block.ffn_mix.pre(&mid, &mixes.pre).unwrap(),
                    &block.ffn_norm,
                    block.eps,
                )
            })
            .unwrap();
        let fout = st.time("moe", || block.ffn.forward(&finput)).unwrap();
        // The router and the shared expert again, warm, to split the block's time.
        let x2 = finput.reshape((seq, finput.elem_count() / seq)).unwrap();
        let _ = st.time("moe router (again)", || block.ffn.router.apply_here(&x2));
        if let Some(shared) = &block.ffn.shared {
            let _ = st.time("moe shared (again)", || {
                shared.forward(&x2, block.ffn.limit)
            });
        }
        hidden = st
            .time("hc post", || {
                block.ffn_mix.post(&fout, &mid, &fmixes.post, &fmixes.comb)
            })
            .unwrap();
        block.carry.put(pos, seq, fmixes.pre);
        layer_secs.push(started.elapsed().as_secs_f64());
    }
    let logits = st
        .time("head", || model.output_stage(hidden, false, seq, false))
        .unwrap();
    row(&logits)
}

/// Prefill twice and decode `STEPS` tokens after each, on the generic route.
///
/// The first prefill starts from a cold page cache, the second from the state the first left.
/// A third pass runs stage by stage.
fn generic_cold_warm(path: &std::path::Path, ids: &[u32]) -> Vec<f32> {
    evict(path);
    let split = SplitGguf::open(path).expect("the file opens");
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
    let (model, s) = timed(|| {
        GenericHeteroTransformer::from_split_gguf(
            &split,
            &HashMap::new(),
            &plan,
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
            false,
            false,
        )
    });
    let mut model = model.expect("the generic loader accepts the file");
    println!("generic: loaded in {s:.1}s, resident {} MB", resident_mb());
    let tensor = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let mut first: Vec<f32> = Vec::new();
    for pass in ["cold", "warm"] {
        let (logits, s) = timed(|| row(&model.forward(&tensor(ids), 0).unwrap()));
        println!(
            "generic {pass}: prefill {} tokens {s:.2}s, resident {} MB",
            ids.len(),
            resident_mb()
        );
        if first.is_empty() {
            first = logits.clone();
        } else {
            assert_eq!(first, logits, "the warm prefill changed the logits");
        }
        let mut last = logits;
        let mut total = 0.0;
        for i in 0..STEPS {
            let id = argmax(&last);
            let (l, s) = timed(|| row(&model.forward(&tensor(&[id]), ids.len() + i).unwrap()));
            total += s;
            last = l;
        }
        println!(
            "generic {pass}: {STEPS} decode steps {total:.2}s ({:.3}s each)",
            total / STEPS as f64
        );
    }
    let mut prefill = Stages::default();
    let mut decode = Stages::default();
    let mut per_layer = Vec::new();
    let (logits, s) = timed(|| staged_forward(&mut model, ids, 0, &mut prefill, &mut per_layer));
    assert_eq!(first, logits, "the staged prefill changed the logits");
    println!("generic staged: prefill {s:.2}s");
    println!(
        "generic staged: prefill seconds per layer {:?}",
        per_layer
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    prefill.print("generic prefill");
    let mut last = logits;
    let mut total = 0.0;
    for i in 0..STEPS {
        let id = argmax(&last);
        let (l, s) = timed(|| {
            staged_forward(
                &mut model,
                &[id],
                ids.len() + i,
                &mut decode,
                &mut per_layer,
            )
        });
        total += s;
        last = l;
    }
    println!("generic staged: {STEPS} decode steps {total:.2}s");
    decode.print("generic decode");
    first
}

/// The same on the V4.1 path, its own stage profile on for the third pass.
fn v41_cold_warm(path: &std::path::Path, ids: &[u32]) -> Vec<f32> {
    evict(path);
    let split = SplitGguf::open(path).expect("the file opens");
    let (model, s) = timed(|| {
        let cfg = DeepseekV41Config::from_meta(&split).expect("the V4.1 config");
        DeepseekV41Model::load(&split, &cfg).expect("the V4.1 loader")
    });
    println!("v41: loaded in {s:.1}s, resident {} MB", resident_mb());
    let mut first: Vec<f32> = Vec::new();
    for pass in ["cold", "warm", "profiled"] {
        let profiled = pass == "profiled";
        if profiled {
            crate::inference::offload::stage_prof_enable(true);
        }
        let mut state = model.new_decode_state();
        let (logits, s) = timed(|| row(&model.prefill_into(ids, &mut state).unwrap()));
        println!(
            "v41 {pass}: prefill {} tokens {s:.2}s, resident {} MB",
            ids.len(),
            resident_mb()
        );
        if first.is_empty() {
            first = logits.clone();
        } else {
            assert_eq!(first, logits, "the repeated prefill changed the logits");
        }
        let after_prefill = crate::inference::offload::stage_prof_snapshot();
        let mut last = logits;
        let mut total = 0.0;
        for _ in 0..STEPS {
            let id = argmax(&last);
            let (l, s) = timed(|| row(&model.forward_decode(id, &mut state).unwrap()));
            total += s;
            last = l;
        }
        println!(
            "v41 {pass}: {STEPS} decode steps {total:.2}s ({:.3}s each)",
            total / STEPS as f64
        );
        if profiled {
            for (name, ns, n) in &after_prefill {
                println!(
                    "v41 prefill stage {name:<22} {:8.3}s over {n} calls",
                    *ns as f64 / 1e9
                );
            }
            for (name, ns, n) in crate::inference::offload::stage_prof_snapshot() {
                let (ns0, n0) = after_prefill
                    .iter()
                    .find(|(k, _, _)| *k == name)
                    .map_or((0, 0), |(_, a, b)| (*a, *b));
                println!(
                    "v41 decode stage {name:<22} {:8.3}s over {} calls",
                    (ns - ns0) as f64 / 1e9,
                    n - n0
                );
            }
            crate::inference::offload::stage_prof_enable(false);
        }
    }
    first
}

/// Each route cold, warm and stage by stage, the page cache emptied before each load.
#[test]
#[ignore]
fn wide_sparse_real_cold_and_warm() {
    let path = blob(TAG);
    let ids = {
        let split = SplitGguf::open(&path).expect("the file opens");
        prompt_ids(&split.merged_content())
    };
    println!("prompt ids: {ids:?}");
    let generic = generic_cold_warm(&path, &ids);
    let v41 = v41_cold_warm(&path, &ids);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in generic.iter().zip(&v41) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    println!(
        "cosine {:.6}, bits equal {}",
        dot / (na.sqrt() * nb.sqrt()),
        generic == v41
    );
}
