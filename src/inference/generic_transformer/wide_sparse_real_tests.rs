//! The generic route against the records of the V4.1 path, on the released V4.1 Flash file in
//! the store.
//!
//! Ignored by default: it maps a file of hundreds of GB and runs on the host.
//! Run with
//!   cargo test --release --no-default-features --features cpu --lib -- --ignored wide_sparse_real --nocapture
//!
//! The file is read through mapped views, never copied whole into memory.
//! The generic route prefills the prompt and decodes a few greedy tokens.
//! The records hold the last position's logits and the greedy tokens of the V4.1 path on the
//! same prompt.

use std::time::Instant;

use super::wide_sparse_frozen as frozen;
use super::*;
use crate::tensor::quantized::gguf_source::SplitGguf;

/// The store tag of the file the test reads.
pub(crate) const TAG: &str = "deepseek-v4.1-flash:iq2-q6attn";

/// The text whose tokens both routes prefill.
pub(crate) const PROMPT: &str = "The capital of France is";

/// Greedy tokens each route decodes after the prompt.
pub(super) const STEPS: usize = 4;

/// Logits listed per route.
const TOP: usize = 10;

/// The path of the blob a store tag names.
pub(crate) fn blob(tag: &str) -> std::path::PathBuf {
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
pub(crate) fn prompt_ids(content: &crate::tensor::quantized::gguf_file::Content) -> Vec<u32> {
    ids_of(content, PROMPT)
}

/// The tokens of `prompt`: the file's own tokenizer, after the begin token it names.
pub(crate) fn ids_of(
    content: &crate::tensor::quantized::gguf_file::Content,
    prompt: &str,
) -> Vec<u32> {
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
    ids.extend_from_slice(tokenizer.encode(prompt, false).unwrap().get_ids());
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

/// The last position's logits of the V4.1 path on the prompt, as recorded.
fn v41_last() -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/deepseek_v41_frozen/real_last_logits.f32");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let last: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    frozen::expect_f32("real", "last logits", &last);
    last
}

/// What the V4.1 path gave on the prompt `ids`, as recorded.
fn v41_run(ids: &[u32]) -> Run {
    frozen::expect_debug("real", "prompt ids", &ids);
    let greedy = frozen::recorded("real", "greedy");
    let greedy = greedy
        .trim_matches(|c| c == '[' || c == ']')
        .split(", ")
        .map(|id| id.parse().unwrap())
        .collect();
    Run {
        last: v41_last(),
        greedy,
    }
}

/// The generic route on the released file against the records of the V4.1 path: the last
/// position's logits and the greedy tokens.
#[test]
#[ignore]
fn wide_sparse_real_generic_against_v41() {
    let path = blob(TAG);
    let split = SplitGguf::open(&path).expect("the file opens");
    let ids = prompt_ids(&split.merged_content());
    println!("prompt ids: {ids:?}");
    let generic = generic_run(&split, &ids);
    report("generic", &generic);
    let v41 = v41_run(&ids);
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

/// The mixer output and the hidden state after each layer, last row, of the staged forwards that
/// run while it holds `Some`.
pub(super) static TRACE: std::sync::Mutex<Option<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>>> =
    std::sync::Mutex::new(None);

/// Records one layer of the trace when it is on.
fn trace_layer(mixed: Option<&Tensor>, twin: Option<&Tensor>, hidden: &Tensor) {
    let mut t = TRACE.lock().unwrap();
    if let Some(v) = t.as_mut() {
        let flat = |x: &Tensor| {
            row(&x
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap())
        };
        v.push((
            mixed.map(flat).unwrap_or_default(),
            twin.map(flat).unwrap_or_default(),
            flat(hidden),
        ));
    }
}

/// The mixer's output for `input` with its card set aside, `None` for a mixer of another kind.
#[cfg(not(feature = "gpu"))]
fn host_twin(_: &mut super::mixer::Mixer, _: &Tensor, _: usize) -> Option<Tensor> {
    None
}

/// The mixer's output for `input` with its card set aside, `None` for a mixer of another kind.
#[cfg(feature = "gpu")]
fn host_twin(mixer: &mut super::mixer::Mixer, input: &Tensor, pos: usize) -> Option<Tensor> {
    use super::mixer::Mixer;
    match mixer {
        Mixer::LatentMqa(m) => {
            let card = m.card.take();
            let y = m.forward(input, pos).ok();
            m.card = card;
            y
        }
        Mixer::CompressedBand(b) => {
            let card = b.base.card.take();
            let y = b.forward(input, pos).ok();
            b.base.card = card;
            y
        }
        _ => None,
    }
}

/// What one layer of a staged forward computed, every row, as the stage that made it names it.
pub(super) struct LayerRec {
    pub(super) stages: Vec<(&'static str, Tensor)>,
    /// Per row: the experts chosen, the biased score of the last one kept and of the first one
    /// dropped.
    pub(super) picks: Vec<(Vec<usize>, f32, f32)>,
    /// The engram table rows each position reads, for a layer with an engram.
    pub(super) hashes: Vec<Vec<u64>>,
    /// The pre mix the layer leaves for the next one.
    pub(super) pre: Vec<Vec<f32>>,
}

impl LayerRec {
    pub(super) fn stage(&self, name: &str) -> Option<&Tensor> {
        self.stages.iter().find(|s| s.0 == name).map(|s| &s.1)
    }
}

/// The layers recorded by the staged forwards that run while it holds `Some`.
pub(super) static RECORD: std::sync::Mutex<Option<Vec<LayerRec>>> = std::sync::Mutex::new(None);

/// A record whose hidden states and pre mixes the staged forwards that run while it holds
/// `Some` take as the input of each layer.
pub(super) static FORCED: std::sync::Mutex<Option<Vec<LayerRec>>> = std::sync::Mutex::new(None);

/// While `Some(eps)`, the staged forwards scale every value of each mixer input by a factor in
/// `1 +- eps`, from a fixed pseudo-random sequence.
pub(super) static PERTURB: std::sync::Mutex<Option<f32>> = std::sync::Mutex::new(None);

/// `t` with each value scaled by a factor in `1 +- eps`.
fn perturbed(t: &Tensor, eps: f32) -> Tensor {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for x in v.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
        *x *= 1.0 + eps * u;
    }
    Tensor::from_vec(v, t.dims().to_vec(), &Device::Cpu).unwrap()
}

/// Keeps `t` as stage `name` of the layer being recorded, when a record runs.
fn note(rec: &mut Option<LayerRec>, name: &'static str, t: &Tensor) {
    if let Some(r) = rec.as_mut() {
        let t = t
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .contiguous()
            .unwrap();
        r.stages.push((name, t));
    }
}

/// The host's window attention over key rows `keys` that are the values too, from position 0:
/// the heads' outputs `[seq, heads * hd]` and, per row and head, the largest logit minus the
/// sink's.
fn attention_replica(
    q: &[f32],
    keys: &[f32],
    sink: &[f32],
    (heads, hd): (usize, usize),
    window: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0f32; q.len()];
    let mut gaps = vec![0f32; q.len() / hd];
    for pair in 0..q.len() / hd {
        let (i, h) = (pair / heads, pair % heads);
        let lo = i.saturating_sub(window - 1);
        let qv = &q[pair * hd..(pair + 1) * hd];
        let seen = &keys[lo * hd..(i + 1) * hd];
        let mut m = sink[h];
        let logits: Vec<f32> = seen
            .chunks_exact(hd)
            .map(|key| {
                let mut dot = 0f32;
                for (a, k) in qv.iter().zip(key) {
                    dot += a * k;
                }
                let l = dot * scale;
                if l > m {
                    m = l;
                }
                l
            })
            .collect();
        gaps[pair] = m - sink[h];
        let o = &mut out[pair * hd..(pair + 1) * hd];
        let mut denom = (sink[h] - m).exp();
        for (l, key) in logits.iter().zip(seen.chunks_exact(hd)) {
            let w = (l - m).exp();
            denom += w;
            for (acc, &v) in o.iter_mut().zip(key) {
                *acc += w * v;
            }
        }
        for v in o.iter_mut() {
            *v /= denom;
        }
    }
    (out, gaps)
}

/// The Q8_K codes the host quantiser gives the heads' outputs `o` turned back by the phases of
/// their rows, as floats `[1, seq, heads * hd]`.
fn turned_back_codes(
    o: &[f32],
    d: &super::latent_mqa::LatentMqaDims,
    cos: &[f32],
    sin: &[f32],
    seq: usize,
) -> Tensor {
    use crate::tensor::quant_cpu::{BlockFormat, BlockQ8K};
    let mut o = o.to_vec();
    let back: Vec<f32> = sin.iter().map(|&s| 0.0 - s).collect();
    super::latent_mqa::rotate_tails(&mut o, d.heads, d.head_dim, d.rope_dim, cos, &back);
    let mut blocks = vec![BlockQ8K::zeros(); o.len() / 256];
    BlockQ8K::quantize(&o, &mut blocks);
    let codes: Vec<f32> = blocks
        .iter()
        .flat_map(|b| b.qs.iter().map(|&q| q as f32))
        .collect();
    Tensor::from_vec(codes, (1, seq, d.heads * d.head_dim), &Device::Cpu).unwrap()
}

/// The query factor, the queries and the key rows the mixer makes of `input`, as the mixer makes
/// them: on its card when it has one, else on the host.
#[cfg(not(feature = "gpu"))]
fn mixer_front(_: &super::mixer::Mixer, _: &Tensor, _: usize) -> Vec<(&'static str, Tensor)> {
    Vec::new()
}

/// The query factor, the queries and the key rows the mixer makes of `input`, as the mixer makes
/// them: on its card when it has one, else on the host.
#[cfg(feature = "gpu")]
fn mixer_front(
    mixer: &super::mixer::Mixer,
    input: &Tensor,
    pos: usize,
) -> Vec<(&'static str, Tensor)> {
    use super::mixer::Mixer;
    let base = match mixer {
        Mixer::LatentMqa(m) => m,
        Mixer::CompressedBand(b) => &b.base,
        _ => return Vec::new(),
    };
    let (_, seq, hidden) = input.dims3().unwrap();
    let x2 = super::latent_mqa::host_rows(input, seq, hidden).unwrap();
    let (cos, sin) = super::latent_mqa::phases(&base.inv_freq, pos..pos + seq);
    let d = base.dims;
    let host_qr = base.query_factor(&x2).unwrap();
    let q_rank = host_qr.dims()[1];
    let wide =
        |v: Vec<f32>, cols: usize| Tensor::from_vec(v, (1, seq, cols), &Device::Cpu).unwrap();
    match &base.card {
        Some(card) => {
            let rows = super::card_mixers::FrontRows::Host(&x2);
            let f = card.front_dev(base, rows, &cos, &sin).unwrap();
            let dev = card.dev();
            let dl = |s: &crate::tensor::gpu_runtime::driver::CudaSlice<f32>, n: usize| {
                super::card_mixers::download(dev, s, n).unwrap()
            };
            let (qd, kd) = (
                dl(&f.q, seq * d.heads * d.head_dim),
                dl(&f.kv, seq * d.head_dim),
            );
            let mut stages = vec![
                ("qr", wide(dl(&f.qr, seq * q_rank), q_rank)),
                ("q", wide(qd.clone(), d.heads * d.head_dim)),
                ("kv", wide(kd.clone(), d.head_dim)),
            ];
            if pos == 0 {
                use crate::tensor::cuda::{slots_attention, upload_f32, upload_i32};
                let dev: &crate::tensor::cuda::CudaDevice = dev;
                let keys = upload_f32(dev, "keys", &kd).unwrap();
                let sink = upload_f32(dev, "sink", &base.w.sink).unwrap();
                let slots = upload_i32(
                    dev,
                    "slots",
                    &super::card_mixers::window_slots(0, seq, d.window),
                )
                .unwrap();
                let o = slots_attention(
                    dev,
                    &f.q,
                    &keys,
                    &sink,
                    &slots,
                    (seq, d.heads, d.head_dim, d.window),
                    base.scale(),
                )
                .unwrap();
                let n = seq * d.heads * d.head_dim;
                let (ro, gaps) = attention_replica(
                    &qd,
                    &kd,
                    &base.w.sink,
                    (d.heads, d.head_dim),
                    d.window,
                    base.scale(),
                );
                let ok = dl(&o, n);
                stages.push(("attn_o_q8", turned_back_codes(&ok, &d, &cos, &sin, seq)));
                stages.push(("attn_o", wide(ok, d.heads * d.head_dim)));
                stages.push(("attn_o_replica", wide(ro, d.heads * d.head_dim)));
                stages.push(("attn_gap", wide(gaps, d.heads)));
            }
            stages
        }
        None => {
            let q = base.queries(&host_qr, &cos, &sin).unwrap();
            let kv = base.key_rows(&x2, &cos, &sin).unwrap();
            let mut stages = vec![
                (
                    "qr",
                    wide(host_qr.flatten_all().unwrap().to_vec1().unwrap(), q_rank),
                ),
                ("q", wide(q.clone(), d.heads * d.head_dim)),
                ("kv", wide(kv.clone(), d.head_dim)),
            ];
            if pos == 0 {
                let (ro, gaps) = attention_replica(
                    &q,
                    &kv,
                    &base.w.sink,
                    (d.heads, d.head_dim),
                    d.window,
                    base.scale(),
                );
                stages.push(("attn_o_q8", turned_back_codes(&ro, &d, &cos, &sin, seq)));
                stages.push(("attn_o", wide(ro.clone(), d.heads * d.head_dim)));
                stages.push(("attn_o_replica", wide(ro, d.heads * d.head_dim)));
                stages.push(("attn_gap", wide(gaps, d.heads)));
            }
            stages
        }
    }
}

/// The experts of each row of router `logits`, the biased score of the last kept expert and of
/// the first dropped one, as `sqrt_softplus_choice` scores them.
fn picks_of(
    ffn: &super::wide_sparse_layer::RoutedFfn,
    logits: &Tensor,
) -> Vec<(Vec<usize>, f32, f32)> {
    let soft = |z: f32| if z > 20.0 { z } else { z.exp().ln_1p() };
    logits
        .to_vec2::<f32>()
        .unwrap()
        .iter()
        .map(|row| {
            let biased: Vec<f32> = row
                .iter()
                .zip(&ffn.bias)
                .map(|(&z, &b)| soft(z / ffn.temperature).sqrt() + b)
                .collect();
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|&a, &b| {
                biased[b]
                    .partial_cmp(&biased[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let k = ffn.used;
            (order[..k].to_vec(), biased[order[k - 1]], biased[order[k]])
        })
        .collect()
}

/// A forward of the generic model over `ids` from `pos`, stage by stage, as `forward_sinkhorn`
/// runs it. The last row of logits comes back.
///
/// While `RECORD` holds `Some`, every layer's stages are recorded.
/// While `FORCED` holds `Some`, every layer starts from the hidden state and the pre mix the
/// forced record holds for the layer before.
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
        if il > 0 {
            if let Some(f) = FORCED.lock().unwrap().as_ref() {
                hidden = f[il - 1].stage("hidden").unwrap().clone();
                block.carry.put(pos, seq, f[il - 1].pre.clone());
            }
        }
        let mut rec = RECORD.lock().unwrap().is_some().then(|| LayerRec {
            stages: Vec::new(),
            picks: Vec::new(),
            hashes: Vec::new(),
            pre: Vec::new(),
        });
        let r = hidden
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .contiguous()
            .unwrap();
        // The engram, hashed from the model's token log as `with_engram` does.
        let r = match block.engram.as_ref() {
            Some(e) => {
                use super::wide_sparse_layer::TokenLog;
                let window = e.tokens.window(&e.meta, pos, seq).unwrap();
                let start = TokenLog::window_start(&e.meta, pos);
                let hashes: Vec<Vec<u64>> = (pos..pos + seq)
                    .map(|p| {
                        let from = TokenLog::window_start(&e.meta, p) - start;
                        e.meta.hash(e.which, &window[from..=p - start])
                    })
                    .collect();
                let with = st.time("engram", || e.block.forward(&r, &hashes)).unwrap();
                if let Some(rc) = rec.as_mut() {
                    rc.hashes = hashes;
                }
                note(&mut rec, "engram_add", &with.broadcast_sub(&r).unwrap());
                with
            }
            None => r,
        };
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
        note(&mut rec, "input", &input);
        if rec.is_some() {
            for (name, t) in mixer_front(&layer.mixer, &input, pos) {
                note(&mut rec, name, &t);
            }
        }
        let input = match *PERTURB.lock().unwrap() {
            Some(eps) => perturbed(&input, eps),
            None => input,
        };
        let kind = match &layer.mixer {
            super::mixer::Mixer::LatentMqa(_) => "mixer latent",
            super::mixer::Mixer::CompressedBand(_) => "mixer band",
            _ => "attention mixer",
        };
        let mixed = st.time(kind, || layer.mixer.forward(&input, pos)).unwrap();
        note(&mut rec, "mixed", &mixed);
        // While a trace runs: the same input through the mixer without its card.
        let twin = TRACE
            .lock()
            .unwrap()
            .is_some()
            .then(|| host_twin(&mut layer.mixer, &input, pos))
            .flatten();
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
        note(&mut rec, "out", &out);
        let mid = st
            .time("hc post", || {
                block.attn_mix.post(&out, &r, &mixes.post, &mixes.comb)
            })
            .unwrap();
        note(&mut rec, "mid", &mid);
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
        note(&mut rec, "finput", &finput);
        let fout = st.time("moe", || block.ffn.forward(&finput)).unwrap();
        note(&mut rec, "fout", &fout);
        // The router and the shared expert again, warm, to split the block's time.
        let x2 = finput.reshape((seq, finput.elem_count() / seq)).unwrap();
        let logits = st.time("moe router (again)", || block.ffn.router.apply_here(&x2));
        if let (Some(rc), Ok(l)) = (rec.as_mut(), logits.as_ref()) {
            rc.picks = picks_of(&block.ffn, l);
        }
        if let Ok(l) = logits.as_ref() {
            note(&mut rec, "logits", l);
        }
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
        note(&mut rec, "hidden", &hidden);
        if let Some(mut rc) = rec.take() {
            rc.pre = fmixes.pre.clone();
            if let Some(all) = RECORD.lock().unwrap().as_mut() {
                all.push(rc);
            }
        }
        block.carry.put(pos, seq, fmixes.pre);
        trace_layer(Some(&mixed), twin.as_ref(), &hidden);
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

/// The generic route cold, warm and stage by stage, the page cache emptied before the load, and
/// its logits against the recorded logits of the V4.1 path.
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
    let v41 = v41_run(&ids).last;
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
