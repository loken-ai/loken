//! The V4.1 token mixers on a card, on the released V4.1 Flash file in the store.
//!
//! Ignored by default: it maps a file of hundreds of GB and needs a CUDA card.
//! Run with
//!   cargo test --release --features cuda --lib -- --ignored wide_sparse_real_card --nocapture --test-threads=1
//!
//! The host model prefills the prompt and decodes greedy tokens: that run is the reference.
//! The host model is dropped, then the model with its mixers on the card runs the same steps
//! twice, from a cold page cache and warm.
//! The comparison follows the card versus host policy of `card_mixer_tests.rs`.
//!
//! The harness takes closures that build the models, so the toy model of the card tests
//! exercises it on any machine with a card.

use super::card_mixer_tests::{closeness, generic_on_card, MAX_REL_L2, MIN_COSINE};
use super::mixer::Mixer;
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::wide_sparse_real_tests::{
    argmax, blob, evict, prompt_ids, resident_mb, row, staged_forward, timed, Stages, STEPS, TAG,
};
use super::*;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::place::llm::offer;
use crate::inference::place::runtime_demand::llm_work_reserve;
use crate::inference::place::{ladder::Demand, layer_executor::DeviceKind};
use crate::tensor::quantized::gguf_source::SplitGguf;

/// What the host model tells the card plan: the bytes each layer's token mixer holds, and the
/// width of the widest activation.
pub(super) struct Facts {
    pub(super) mixer_bytes: Vec<u64>,
    pub(super) width: usize,
}

/// One prefill and the greedy steps after it.
struct Pass {
    /// The logits of the prompt's last position.
    last: Vec<f32>,
    /// The tokens the steps chose.
    greedy: Vec<u32>,
    prefill_secs: f64,
    decode_secs: f64,
}

/// Prefills `ids` at position 0 and decodes `STEPS` greedy tokens.
fn pass(model: &mut GenericHeteroTransformer, ids: &[u32]) -> Pass {
    let tensor = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let (last, prefill_secs) = timed(|| row(&model.forward(&tensor(ids), 0).unwrap()));
    let first = last.clone();
    let mut last = last;
    let mut greedy = Vec::new();
    let mut decode_secs = 0.0;
    for i in 0..STEPS {
        let id = argmax(&last);
        greedy.push(id);
        let (l, s) = timed(|| row(&model.forward(&tensor(&[id]), ids.len() + i).unwrap()));
        decode_secs += s;
        last = l;
    }
    Pass {
        last: first,
        greedy,
        prefill_secs,
        decode_secs,
    }
}

/// The bytes each layer's token mixer holds: as the host form, and as the card form.
fn mixer_bytes(model: &GenericHeteroTransformer) -> Vec<(usize, usize)> {
    model
        .layers
        .iter()
        .map(|l| match &l.mixer {
            Mixer::LatentMqa(m) => (m.resident_bytes(), m.card_bytes()),
            Mixer::CompressedBand(m) => (m.resident_bytes(), m.card_bytes()),
            _ => (0, 0),
        })
        .collect()
}

/// The free and total bytes of card `ordinal`.
fn card_memory(ordinal: usize) -> Option<(u64, u64)> {
    let card = Device::new_cuda(ordinal).ok()?;
    let (free, total) = card
        .as_cuda_device()
        .ok()?
        .cuda_stream()
        .context()
        .mem_get_info()
        .ok()?;
    Some((free as u64, total as u64))
}

/// The line that says what sits on the card, for the phase `phase`.
fn card_line(phase: &str, model: &GenericHeteroTransformer, ordinal: usize) -> String {
    let bytes = mixer_bytes(model);
    let on_card = bytes.iter().filter(|b| b.1 > 0).count();
    let held: usize = bytes.iter().map(|b| b.1).sum();
    let free = card_memory(ordinal).map_or(0, |(f, _)| f);
    format!(
        "{phase}: {on_card} of {} layers hold mixer weights on card {ordinal}, {held} bytes on the card, {free} bytes free, resident {} MB",
        bytes.len(),
        resident_mb()
    )
}

/// The ordinal of the first card a plan names, else 0.
fn first_card(plan: &HeteroPlan) -> usize {
    plan.segments
        .iter()
        .find_map(|s| match s.kind {
            DeviceKind::Cuda(o) => Some(o),
            _ => None,
        })
        .unwrap_or(0)
}

/// The plan that puts the layers whose mixers fit on the cards offered, and the cards.
///
/// The cards, their free bytes and the plan are the loader's own: `offer` and `Demand::plan`.
pub(super) fn card_plan(facts: &Facts, meta: &WideSparseMeta, ids: usize) -> HeteroPlan {
    let layers = facts.mixer_bytes.len();
    let demand = Demand {
        layers,
        block_bytes: facts.mixer_bytes.iter().sum(),
        head_bytes: 0,
        kv_bytes_per_layer: meta.kv_bytes_per_layer(ids + STEPS),
        runtime_reserve: llm_work_reserve(facts.width, ids),
    };
    let cards = offer(None, demand.whole(), 0);
    assert!(
        !cards.cards.is_empty(),
        "no CUDA card offers free memory: this test needs a card"
    );
    let plan = demand.plan(&cards.cards);
    println!("cards offered (ordinal, free bytes): {:?}", cards.cards);
    println!(
        "demand: {} bytes of mixers over {layers} layers, {} bytes of cache a layer, {} bytes reserved",
        demand.block_bytes, demand.kv_bytes_per_layer, demand.runtime_reserve
    );
    for s in &plan.segments {
        println!(
            "plan: layers {}..{} on {:?}",
            s.layer_start, s.layer_end, s.kind
        );
    }
    plan
}

/// The host model against the same model with its mixers on a card, within the card policy.
///
/// `build_host` makes the reference model, run once to fault its file in and once timed.
/// The host model is dropped, then `before_card` runs, then `build_card` makes the card model
/// from what the host model reported.
/// The card model runs twice and then once stage by stage.
pub(super) fn host_against_card(
    ids: &[u32],
    build_host: impl FnOnce() -> GenericHeteroTransformer,
    before_card: impl FnOnce(),
    build_card: impl FnOnce(&Facts) -> (GenericHeteroTransformer, usize),
) {
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test measures the mixers on a card"
    );
    let (host, s) = timed(build_host);
    let mut host = host;
    println!("host: loaded in {s:.1}s, resident {} MB", resident_mb());
    let facts = Facts {
        mixer_bytes: mixer_bytes(&host).iter().map(|b| b.0 as u64).collect(),
        width: host.config.embedding_length,
    };
    let _ = pass(&mut host, ids);
    let want = pass(&mut host, ids);
    println!(
        "host warm: prefill {} tokens {:.2}s, {STEPS} decode steps {:.2}s ({:.3}s each), greedy {:?}",
        ids.len(),
        want.prefill_secs,
        want.decode_secs,
        want.decode_secs / STEPS as f64,
        want.greedy
    );
    drop(host);
    before_card();

    let ((mut card, ordinal), s) = timed(|| build_card(&facts));
    println!("card: loaded in {s:.1}s");
    println!("{}", card_line("card loaded", &card, ordinal));
    let mut passes = Vec::new();
    for phase in ["cold", "warm"] {
        let got = pass(&mut card, ids);
        println!(
            "card {phase}: prefill {} tokens {:.2}s, {STEPS} decode steps {:.2}s ({:.3}s each), greedy {:?}",
            ids.len(),
            got.prefill_secs,
            got.decode_secs,
            got.decode_secs / STEPS as f64,
            got.greedy
        );
        println!("{}", card_line(&format!("card {phase}"), &card, ordinal));
        passes.push(got);
    }

    let mut prefill = Stages::default();
    let mut decode = Stages::default();
    let mut per_layer = Vec::new();
    let (logits, s) = timed(|| staged_forward(&mut card, ids, 0, &mut prefill, &mut per_layer));
    println!("card staged: prefill {s:.2}s");
    prefill.print("card prefill");
    let mut last = logits;
    let mut total = 0.0;
    for i in 0..STEPS {
        let id = argmax(&last);
        let (l, s) =
            timed(|| staged_forward(&mut card, &[id], ids.len() + i, &mut decode, &mut per_layer));
        total += s;
        last = l;
    }
    println!("card staged: {STEPS} decode steps {total:.2}s");
    decode.print("card decode");

    let mut failed = Vec::new();
    for (phase, got) in ["cold", "warm"].iter().zip(&passes) {
        let (cos, l2) = closeness(&want.last, &got.last);
        println!(
            "card {phase} against host: cosine {cos:.6}, relative L2 {l2:.6}, greedy equal {}",
            got.greedy == want.greedy
        );
        if cos.is_nan() || cos < MIN_COSINE {
            failed.push(format!("{phase}: cosine {cos} is not above {MIN_COSINE}"));
        }
        if l2.is_nan() || l2 > MAX_REL_L2 {
            failed.push(format!(
                "{phase}: relative L2 {l2} is not below {MAX_REL_L2}"
            ));
        }
        if got.greedy != want.greedy {
            failed.push(format!(
                "{phase}: greedy tokens {:?} != host {:?}",
                got.greedy, want.greedy
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "the card mixers leave the policy:\n{}",
        failed.join("\n")
    );
}

/// The mixers of the released V4.1 Flash file on a card against the host, cold and warm.
#[test]
#[ignore]
fn wide_sparse_real_card() {
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test measures the mixers on a card"
    );
    let path = blob(TAG);
    let ids = {
        let split = SplitGguf::open(&path).expect("the file opens");
        prompt_ids(&split.merged_content())
    };
    println!("prompt ids: {ids:?}");
    // A model of the file, its mixers on the host or on the cards of `plan_of`.
    let load = |plan_of: &dyn Fn(&SplitGguf, usize) -> HeteroPlan| {
        let split = SplitGguf::open(&path).expect("the file opens");
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
    };
    host_against_card(
        &ids,
        || load(&|_, layers| HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0)).0,
        || evict(&path),
        |facts| {
            load(&|split, _| {
                let meta = WideSparseMeta::from_file(&split.merged_content()).expect("the meta");
                card_plan(facts, &meta, ids.len())
            })
        },
    );
}

/// The layer count a file declares.
fn layers_of(split: &SplitGguf) -> usize {
    let content = split.merged_content();
    let arch = content.metadata["general.architecture"]
        .to_string()
        .unwrap()
        .clone();
    content
        .metadata
        .get(&format!("{arch}.block_count"))
        .and_then(|v| v.to_u32().ok())
        .expect("a block count") as usize
}

/// The harness runs on the synthetic model of the card tests, in each band layout.
#[test]
fn the_real_card_harness_runs_on_the_toy() {
    if Device::new_cuda(0).is_err() {
        return;
    }
    for (ratios, kv_src, index_src) in LAYOUTS {
        let (_, tokens) = synthetic(ratios, kv_src, index_src, 0);
        host_against_card(
            &tokens,
            || generic(ratios, kv_src, index_src),
            || {},
            |facts| {
                assert_eq!(facts.mixer_bytes.len(), ratios.len());
                assert!(facts.mixer_bytes.iter().all(|&b| b > 0));
                (generic_on_card(ratios, kv_src, index_src), 0)
            },
        );
    }
}
