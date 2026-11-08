//! The host and each card path of the V4.1 token mixers against an exact reference, on the
//! released V4.1 Flash file in the store.
//!
//! Ignored by default: it maps a file of hundreds of GB and needs a CUDA card.
//! Run with
//!   cargo test --release --features cuda --lib -- --ignored wide_sparse_real_exact --nocapture --test-threads=1
//!
//! The exact reference is the host model with every mixer projection read as dense F32 values
//! and multiplied with the F32 row, so no activation is quantised in a mixer.
//! The router, the experts, the engram and the layers' own projections stay as in the host.
//!
//! The paths compared to the reference:
//! - the host,
//! - the card with the product it measures and chooses per unit, see `card_packed.rs`,
//! - the card with the host's steps (rows in q8_K blocks),
//! - the card with its quantised matmul (rows in q8_1 blocks),
//! - the card with the blocks expanded to F32.
//!
//! For each prompt the logits of the last position and the greedy tokens are compared with the
//! reference's, and each path is timed on a first and a second run.

use super::card_mixer_tests::{closeness, dequantise_mixers};
use super::card_packed::route::{self, Route};
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_real_card_tests::{card_plan, first_card, layers_of, mixer_bytes, Facts};
use super::wide_sparse_real_tests::{argmax, blob, evict, ids_of, resident_mb, row, timed, TAG};
use super::*;
use crate::tensor::quantized::gguf_source::SplitGguf;

/// Greedy tokens each run decodes after the prompt.
const STEPS: usize = 8;

/// The prompts, as `(kind, text)`: short and long, of different kinds.
const PROMPTS: [(&str, &str); 5] = [
    ("fact", "The capital of France is"),
    (
        "code",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(",
    ),
    (
        "prose",
        "The harbour town had been quiet for most of the winter, and the fishermen spent their mornings mending nets on the stone quay while the gulls argued over the scraps. When the first warm wind arrived from the south, the whole village seemed to wake at once. Shutters were thrown open, boats were scraped and painted, and the baker, who had been sleeping badly for weeks, began to sing while he kneaded the dough. Children ran along the seawall counting the sails that appeared on the horizon, and the old schoolteacher, who remembered every spring since the war, sat on her porch with a cup of tea and said that this one smelled exactly like the first. By the evening the lamps along the quay were lit, a violin started somewhere near the church, and the harbour master wrote in his ledger that the season had begun. Nobody in the town could say when the habit had started, but every year, on the first night of the season, the families carried their tables down to the water and shared whatever they had cooked, the fishermen telling the same stories about storms and the ones that got away, and the youngest guests asking for them again as if they had never heard them before. The next morning",
    ),
    (
        "list",
        "Shopping list:\n1. milk\n2. eggs\n3. bread\n4. butter\n5. apples\n6. coffee\n7.",
    ),
    (
        "french",
        "La tour Eiffel est un monument situé à Paris, construit pour l'Exposition universelle de",
    ),
];

/// One prefill and the greedy steps after it.
struct Run {
    /// The logits of the prompt's last position.
    last: Vec<f32>,
    /// The tokens the steps chose.
    greedy: Vec<u32>,
    prefill_secs: f64,
    decode_secs: f64,
}

/// Prefills `ids` at position 0 and decodes `STEPS` greedy tokens.
fn run(model: &mut GenericHeteroTransformer, ids: &[u32]) -> Run {
    let tensor = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let (first, prefill_secs) = timed(|| row(&model.forward(&tensor(ids), 0).unwrap()));
    let mut last = first.clone();
    let mut greedy = Vec::new();
    let mut decode_secs = 0.0;
    for i in 0..STEPS {
        let id = argmax(&last);
        greedy.push(id);
        let (l, s) = timed(|| row(&model.forward(&tensor(&[id]), ids.len() + i).unwrap()));
        decode_secs += s;
        last = l;
    }
    Run {
        last: first,
        greedy,
        prefill_secs,
        decode_secs,
    }
}

/// A model of the file, with the plan `plan_of` gives, and the ordinal of its first card.
fn load(
    path: &std::path::Path,
    plan_of: &dyn Fn(&SplitGguf, usize) -> HeteroPlan,
) -> (GenericHeteroTransformer, usize) {
    let split = SplitGguf::open(path).expect("the file opens");
    let layers = layers_of(&split);
    let plan = plan_of(&split, layers);
    let ordinal = first_card(&plan);
    let model = GenericHeteroTransformer::from_split_gguf(
        &split,
        &HashMap::new(),
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    )
    .expect("the generic loader accepts the file");
    (model, ordinal)
}

/// The host model of the file.
fn load_host(path: &std::path::Path) -> GenericHeteroTransformer {
    load(path, &|_, layers| {
        HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0)
    })
    .0
}

/// A path's runs: for each prompt, the first and the second run.
type Runs = Vec<(Run, Run)>;

/// Runs every prompt twice on `model`.
fn both_runs(model: &mut GenericHeteroTransformer, prompts: &[Vec<u32>]) -> Runs {
    prompts
        .iter()
        .map(|ids| {
            let first = run(model, ids);
            let second = run(model, ids);
            (first, second)
        })
        .collect()
}

/// The middle value of `v`.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

/// The line per prompt of a path and its distances to the reference.
///
/// Returns the relative L2 distances, the cosines and whether every greedy run equals the
/// reference's.
fn report(name: &str, reference: &[Run], runs: &Runs, lens: &[usize]) -> (Vec<f64>, bool) {
    let mut l2s = Vec::new();
    let mut tokens_equal = true;
    for (p, ((first, second), want)) in runs.iter().zip(reference).enumerate() {
        let (cos, l2) = closeness(&want.last, &second.last);
        let equal = second.greedy == want.greedy && first.greedy == want.greedy;
        tokens_equal &= equal;
        l2s.push(l2);
        println!(
            "TABLE {name:<10} {:<6} len {:>4} cosine {cos:.6} l2 {l2:.6} tokens equal {equal} prefill {:.2}s / {:.2}s decode {:.3}s / {:.3}s per step",
            PROMPTS[p].0,
            lens[p],
            first.prefill_secs,
            second.prefill_secs,
            first.decode_secs / STEPS as f64,
            second.decode_secs / STEPS as f64,
        );
    }
    (l2s, tokens_equal)
}

/// The seconds a path spends on the second runs: prefill and decode, over every prompt.
fn warm_totals(runs: &Runs) -> (f64, f64) {
    runs.iter().fold((0.0, 0.0), |(p, d), (_, s)| {
        (p + s.prefill_secs, d + s.decode_secs / STEPS as f64)
    })
}

/// The file the reference's answers are kept in, in the build directory of the crate.
///
/// The reference holds the mixers as F32, so it runs alone and the paths run after it.
fn reference_file() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/wide_sparse_exact_reference.bin")
}

/// The tokens of every prompt, and the model's meta, read from the file.
fn prompts_of(path: &std::path::Path) -> (WideSparseMeta, Vec<Vec<u32>>) {
    let split = SplitGguf::open(path).expect("the file opens");
    let content = split.merged_content();
    let meta = WideSparseMeta::from_file(&content).expect("the meta");
    let ids = PROMPTS.iter().map(|(_, t)| ids_of(&content, t)).collect();
    (meta, ids)
}

/// Keeps the reference's runs: per prompt, its greedy tokens and its last logits.
fn save_reference(runs: &[Run]) {
    let mut bytes = Vec::new();
    bytes.extend((runs.len() as u64).to_le_bytes());
    for r in runs {
        bytes.extend((r.last.len() as u64).to_le_bytes());
        bytes.extend(r.greedy.iter().flat_map(|t| t.to_le_bytes()));
        bytes.extend(r.last.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(reference_file(), bytes).expect("the reference is written");
}

/// The reference's runs kept by `save_reference`, as `(greedy, last)` per prompt.
fn load_reference() -> Vec<(Vec<u32>, Vec<f32>)> {
    let bytes = std::fs::read(reference_file()).expect("run the reference test first");
    let mut at = 0;
    let mut u64_at = |at: &mut usize| {
        let v = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
        *at += 8;
        v as usize
    };
    let count = u64_at(&mut at);
    (0..count)
        .map(|_| {
            let n = u64_at(&mut at);
            let greedy = (0..STEPS)
                .map(|i| u32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                .collect();
            at += 4 * STEPS;
            let last = (0..n)
                .map(|i| f32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                .collect();
            at += 4 * n;
            (greedy, last)
        })
        .collect()
}

/// The lines of a path against the reference, and its summary line.
///
/// A path passes when its tokens equal the reference's on every prompt and its distances to the
/// reference are no larger than the host's (the host's are `host`, empty for the host itself).
fn table(name: &str, runs: &Runs, lens: &[usize], host: &[f64]) -> Vec<f64> {
    let reference = load_reference();
    let mut l2s = Vec::new();
    let mut tokens_equal = true;
    for (p, ((first, second), (greedy, last))) in runs.iter().zip(&reference).enumerate() {
        let (cos, l2) = closeness(last, &second.last);
        let equal = &second.greedy == greedy && &first.greedy == greedy;
        tokens_equal &= equal;
        l2s.push(l2);
        println!(
            "TABLE {name:<10} {:<6} len {:>4} cosine {cos:.6} l2 {l2:.6} tokens equal {equal} prefill {:.2}s / {:.2}s decode {:.3}s / {:.3}s per step",
            PROMPTS[p].0,
            lens[p],
            first.prefill_secs,
            second.prefill_secs,
            first.decode_secs / STEPS as f64,
            second.decode_secs / STEPS as f64,
        );
    }
    let max = l2s.iter().cloned().fold(0.0, f64::max);
    let mean = l2s.iter().sum::<f64>() / l2s.len() as f64;
    let (p, d) = runs.iter().fold((0.0, 0.0), |(p, d), (_, s)| {
        (p + s.prefill_secs, d + s.decode_secs / STEPS as f64)
    });
    let nearer = l2s.iter().zip(host).filter(|(c, h)| c <= h).count();
    println!(
        "VERDICT {name}: median l2 {:.6}, mean {mean:.6}, max {max:.6}, nearer than host on {nearer} of {} prompts, tokens equal {tokens_equal}, warm prefill {p:.2}s, warm decode {d:.3}s per step (sum over prompts)",
        median(&l2s),
        l2s.len(),
    );
    l2s
}

/// The exact reference on every prompt: keeps its answers in `reference_file`.
#[test]
#[ignore]
fn wide_sparse_real_exact_reference() {
    let path = blob(TAG);
    let (_, prompts) = prompts_of(&path);
    let mut model = load_host(&path);
    let (_, s) = timed(|| dequantise_mixers(&mut model));
    println!(
        "reference: mixers dequantised in {s:.1}s, resident {} MB",
        resident_mb()
    );
    let runs: Vec<Run> = prompts
        .iter()
        .enumerate()
        .map(|(p, ids)| {
            let r = run(&mut model, ids);
            println!(
                "reference {:<6}: prefill {:.2}s, decode {:.3}s per step, greedy {:?}, resident {} MB",
                PROMPTS[p].0,
                r.prefill_secs,
                r.decode_secs / STEPS as f64,
                r.greedy,
                resident_mb()
            );
            r
        })
        .collect();
    save_reference(&runs);
}

/// The host model against the reference kept by `wide_sparse_real_exact_reference`.
#[test]
#[ignore]
fn wide_sparse_real_exact_host() {
    let path = blob(TAG);
    let (_, prompts) = prompts_of(&path);
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let mut host = load_host(&path);
    let runs = both_runs(&mut host, &prompts);
    table("host", &runs, &lens, &[]);
}

/// Each card path against the reference kept by `wide_sparse_real_exact_reference`.
///
/// The host distances come from the log of `wide_sparse_real_exact_host`: the summary line of
/// a card path counts the prompts where it is nearer than the host from `HOST_L2`, which this
/// test recomputes by running the host once more before the card is loaded.
#[test]
#[ignore]
fn wide_sparse_real_exact_card() {
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test measures the mixers on a card"
    );
    let path = blob(TAG);
    let (meta, prompts) = prompts_of(&path);
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let longest = *lens.iter().max().unwrap();
    let mut host = load_host(&path);
    let facts = Facts {
        mixer_bytes: mixer_bytes(&host).iter().map(|b| b.0 as u64).collect(),
        width: host.config.embedding_length,
    };
    let host_l2 = {
        let runs = both_runs(&mut host, &prompts);
        table("host", &runs, &lens, &[])
    };
    drop(host);
    evict(&path);
    let (mut card, ordinal) = load(&path, &|_, _| card_plan(&facts, &meta, longest + STEPS));
    println!("card: resident {} MB, ordinal {ordinal}", resident_mb());
    // The measured product of each unit is logged at its first product.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    route::clear();
    let runs = both_runs(&mut card, &prompts);
    table("card_chosen", &runs, &lens, &host_l2);
    for (name, r) in [
        ("card_q8k", Route::HostSteps),
        ("card_q8_1", Route::Q81),
        ("card_f32", Route::F32),
    ] {
        route::set(r);
        let runs = both_runs(&mut card, &prompts);
        table(name, &runs, &lens, &host_l2);
    }
    route::clear();
}
