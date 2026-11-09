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
use super::wide_sparse_real_card_tests::{
    card_demand, card_plan, first_card, layers_of, mixer_bytes, Facts,
};
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
    // The load's split puts the weights every token reads on the card, by measurement.
    let demand = card_demand(&facts, &meta, longest + STEPS);
    let plan = card_plan(&facts, &meta, longest + STEPS);
    card.note_wide_card_rooms(
        &plan,
        &[],
        demand.kv_bytes_per_layer,
        demand.runtime_reserve,
    );
    card.place_always_read(crate::inference::place::expert_variants::VariantChoice::Plan);
    println!(
        "always-read: residency {:?}, rooms left {:?}",
        card.wide_card_residency(),
        card.wide_card_rooms()
    );
    let runs = both_runs(&mut card, &prompts);
    table("card_always", &runs, &lens, &host_l2);
    route::set(Route::HostSteps);
    let runs = both_runs(&mut card, &prompts);
    table("always_q8k", &runs, &lens, &host_l2);
    route::clear();
}

/// Greedy steps a free decode of the batched path runs to find where 2 paths part.
const FREE_STEPS: usize = 48;

/// The logits of a prompt through the batched decode: the prefill's last row, then 1 single-row
/// step per token.
///
/// `feed` gives the token of each step, else the step takes the greedy token of the last
/// logits.
fn batched(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    steps: usize,
    feed: Option<&[u32]>,
) -> Vec<Vec<f32>> {
    let mut seq = model.sequence_mixers().expect("a sequence");
    let mut out = vec![row(&model
        .prefill_sequence(ids, &mut seq)
        .expect("the prefill"))];
    for i in 0..steps {
        let t = match feed {
            Some(f) => f[i],
            None => argmax(out.last().unwrap()),
        };
        let l = model
            .decode_sequences(&[t], &[ids.len() + i], &mut [&mut seq])
            .expect("a decode step");
        out.push(row(&l));
    }
    out
}

/// The gap between the 2 largest logits of `v`.
fn margin(v: &[f32]) -> f32 {
    let top = argmax(v) as usize;
    let second = (0..v.len())
        .filter(|&k| k != top)
        .map(|k| v[k])
        .fold(f32::MIN, f32::max);
    v[top] - second
}

/// What one path gives on one prompt against the reference.
struct GatePath {
    /// The relative L2 distance of the prefill's last logits to the reference's.
    l2: f64,
    /// Per step fed the reference's tokens, whether the greedy token is the reference's next.
    forced: Vec<bool>,
    /// The greedy tokens of a free decode, the prefill's first.
    free: Vec<u32>,
}

/// `path` on `ids`: the prefill's distance, the steps fed the reference's tokens `greedy`, and a
/// free decode of `free_steps` steps.
fn gate_path(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    (greedy, last): (&[u32], &[f32]),
    free_steps: usize,
) -> (GatePath, Vec<Vec<f32>>) {
    let forced_logits = batched(model, ids, STEPS - 1, Some(greedy));
    let forced = forced_logits[1..]
        .iter()
        .zip(&greedy[1..])
        .map(|(l, &want)| argmax(l) == want)
        .collect();
    let l2 = closeness(last, &forced_logits[0]).1;
    let free_logits = batched(model, ids, free_steps, None);
    let free = free_logits.iter().map(|l| argmax(l)).collect();
    (GatePath { l2, forced, free }, free_logits)
}

/// The index of the first token where `a` and `b` part, their length when they never do.
fn parting(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The single decoded row over the card stream against the host's batched decode, both against
/// the reference kept by `wide_sparse_real_exact_reference`, in 1 process.
///
/// The model loads its mixers on the card by the plan and places the weights every token reads
/// as the load does.
/// Every layer whose mixer is on a card then keeps its Sinkhorn mixers there.
/// The host path runs with the Sinkhorn mixers taken off the layers, the card stream with them
/// back, unmeasured, so each layer is admitted or refused at its first decoded row.
///
/// The card stream passes when:
/// - on every prompt its prefill is no farther from the reference than the host path's,
/// - fed the reference's tokens, it gives at least as many of the reference's next tokens as the
///   host path, counted over every prompt.
///
/// The tokens of the free decodes that match the reference's are counted beside them.
/// Where the free decodes of the 2 paths part, the card stream is fed the host path's tokens and
/// the steps where its greedy token differs are printed with both margins.
#[test]
#[ignore]
fn wide_sparse_real_exact_card_stream() {
    use crate::inference::place::expert_variants::VariantChoice;
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test runs the card stream"
    );
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let path = blob(TAG);
    let (meta, prompts) = prompts_of(&path);
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let longest = *lens.iter().max().unwrap();
    let content = SplitGguf::open(&path)
        .expect("the file opens")
        .merged_content();
    let facts = Facts {
        mixer_bytes: super::wide_sparse_plan::card_mixer_bytes(&content)
            .expect("a wide residual")
            .per_layer,
        width: meta.d_model,
    };
    let ids = longest + FREE_STEPS;
    let demand = card_demand(&facts, &meta, ids);
    let plan = card_plan(&facts, &meta, ids);
    let (mut model, ordinal) = load(&path, &|_, _| plan.clone());
    model.note_wide_card_rooms(
        &plan,
        &[],
        demand.kv_bytes_per_layer,
        demand.runtime_reserve,
    );
    model.place_always_read(VariantChoice::Plan);
    let choices = super::card_packed::Choices::default();
    for il in 0..model.layers.len() {
        let held = model.layers[il]
            .sinkhorn
            .as_ref()
            .is_some_and(|b| b.card_mix.ordinal().is_some());
        if let (false, Some(o)) = (held, model.layers[il].mixer.card_ordinal()) {
            model
                .keep_always_read(
                    o,
                    super::wide_always_read::AlwaysRead::Sinkhorn,
                    il,
                    &choices,
                )
                .expect("the Sinkhorn mixers fit on the card");
        }
    }
    let on_card = model
        .layers
        .iter()
        .filter(|l| {
            l.sinkhorn
                .as_ref()
                .is_some_and(|b| b.card_mix.ordinal().is_some())
        })
        .count();
    println!(
        "card stream: ordinal {ordinal}, {on_card} of {} layers keep their Sinkhorn mixers on a card, resident {} MB",
        model.layers.len(),
        resident_mb()
    );
    let reference = load_reference();

    // The host path: every layer's Sinkhorn mixers off the card.
    let mut kept: Vec<_> = model
        .layers
        .iter_mut()
        .map(|l| l.sinkhorn.as_mut().map(|b| std::mem::take(&mut b.card_mix)))
        .collect();
    let host: Vec<(GatePath, Vec<Vec<f32>>)> = prompts
        .iter()
        .zip(&reference)
        .map(|(ids, (g, l))| gate_path(&mut model, ids, (g, l), FREE_STEPS))
        .collect();

    // The card stream: the mixers back, each layer measured at its first decoded row.
    for (l, k) in model.layers.iter_mut().zip(kept.iter_mut()) {
        if let (Some(b), Some(k)) = (l.sinkhorn.as_mut(), k.take()) {
            b.card_mix = k;
        }
    }
    let mut prefill_ok = true;
    // Reference tokens found over every prompt, host then card: fed, then free.
    let (mut forced, mut free, mut steps) = ([0usize; 2], [0usize; 2], [0usize; 2]);
    for (p, ids) in prompts.iter().enumerate() {
        let (greedy, last) = (&reference[p].0, &reference[p].1);
        let (card, _) = gate_path(&mut model, ids, (greedy, last), STEPS - 1);
        let h = &host[p].0;
        let host_part = parting(&h.free, greedy);
        let card_part = parting(&card.free, greedy);
        let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
        forced[0] += count(&h.forced);
        forced[1] += count(&card.forced);
        free[0] += host_part;
        free[1] += card_part;
        steps[0] += h.forced.len();
        steps[1] += greedy.len();
        prefill_ok &= card.l2 <= h.l2;
        println!(
            "GATE {:<6} len {:>4}: prefill l2 host {:.6} card {:.6}, fed the reference's tokens host {:?} card {:?}, free decode parts from the reference at host {host_part} card {card_part} of {STEPS}",
            PROMPTS[p].0,
            lens[p],
            h.l2,
            card.l2,
            h.forced.iter().map(|&b| b as u8).collect::<Vec<_>>(),
            card.forced.iter().map(|&b| b as u8).collect::<Vec<_>>(),
        );
        // The card stream over the host path's free decode.
        let fed = &h.free;
        let card_logits = batched(&mut model, ids, FREE_STEPS, Some(fed));
        let host_logits = &host[p].1;
        for (k, (c, hl)) in card_logits.iter().zip(host_logits).enumerate() {
            if argmax(c) != argmax(hl) {
                println!(
                    "PART {:<6} step {k}: host {} margin {:.4}, card {} margin {:.4}, card's logit of the host's token {:.4}, logits apart {:.3e}",
                    PROMPTS[p].0,
                    argmax(hl),
                    margin(hl),
                    argmax(c),
                    margin(c),
                    c[argmax(hl) as usize],
                    closeness(hl, c).1
                );
            }
        }
    }
    let admitted = model
        .layers
        .iter()
        .filter(|l| {
            l.sinkhorn
                .as_ref()
                .is_some_and(|b| b.card_mix.verdict() == Some(true))
        })
        .count();
    let pass = prefill_ok && forced[1] >= forced[0];
    println!(
        "VERDICT card_stream: {admitted} layers admitted, reference tokens fed host {} card {} of {}, free host {} card {} of {}, prefill no farther {prefill_ok}: {}",
        forced[0],
        forced[1],
        steps[0],
        free[0],
        free[1],
        steps[1],
        if pass { "pass" } else { "FAIL" }
    );
    assert!(
        pass,
        "the card stream finds fewer reference tokens than the host path"
    );
}

/// The card model with and without the routed experts the load gives the card, against the
/// reference kept by `wide_sparse_real_exact_reference`.
///
/// 1 model in the process: the mixers go to the card by the plan, the prompts run without card
/// experts, then the load's measure places the card tier and the prompts run again.
/// The page cache of the file is dropped before each set of runs.
/// The plan's facts come from the file's header, as the engine reads them.
#[test]
#[ignore]
fn wide_sparse_real_exact_card_tier() {
    use crate::inference::place::expert_variants::VariantChoice;
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test measures the routed experts on a card"
    );
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let path = blob(TAG);
    let (meta, prompts) = prompts_of(&path);
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let longest = *lens.iter().max().unwrap();
    let content = SplitGguf::open(&path)
        .expect("the file opens")
        .merged_content();
    let facts = Facts {
        mixer_bytes: super::wide_sparse_plan::card_mixer_bytes(&content)
            .expect("a wide residual")
            .per_layer,
        width: meta.d_model,
    };
    let demand = card_demand(&facts, &meta, longest + STEPS);
    let plan = card_plan(&facts, &meta, longest + STEPS);
    evict(&path);
    let (mut card, ordinal) = load(&path, &|_, _| plan.clone());
    println!("card: resident {} MB, ordinal {ordinal}", resident_mb());
    let bare = both_runs(&mut card, &prompts);
    let bare_l2 = table("card_bare", &bare, &lens, &[]);
    card.note_wide_card_rooms(
        &plan,
        &[],
        demand.kv_bytes_per_layer,
        demand.runtime_reserve,
    );
    println!("rooms for routed experts: {:?}", card.wide_card_rooms());
    let (_, secs) = timed(|| card.place_wide_experts(VariantChoice::Plan));
    let kept = card.wide_card_expert_count();
    println!(
        "card tier: {kept} routed experts kept, rooms {:?}, residency {:?}, placed in {secs:.1}s",
        card.wide_card_rooms(),
        card.wide_card_residency()
    );
    evict(&path);
    let tier = both_runs(&mut card, &prompts);
    table("card_tier", &tier, &lens, &bare_l2);
    let pairs: u64 = card
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .map(|b| b.ffn.card.pairs_run())
        .sum();
    let cold = |runs: &Runs| runs.iter().map(|(f, _)| f.prefill_secs).sum::<f64>();
    let cold_decode = |runs: &Runs| {
        runs.iter()
            .map(|(f, _)| f.decode_secs / STEPS as f64)
            .sum::<f64>()
    };
    println!(
        "COLD card_bare prefill {:.2}s decode {:.3}s per step, card_tier prefill {:.2}s decode {:.3}s per step (sums over prompts), card expert pairs run {pairs}",
        cold(&bare),
        cold_decode(&bare),
        cold(&tier),
        cold_decode(&tier)
    );
}
