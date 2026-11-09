//! The V4.1 token mixers on a card, on the released V4.1 Flash file in the store.
//!
//! Ignored by default: it maps a file of hundreds of GB and needs a CUDA card.
//! Run with
//!   cargo test --release --features cuda --lib -- --ignored wide_sparse_real_card --nocapture --test-threads=1
//!
//! The exact reference prefills the prompt and decodes greedy tokens, then the host model does.
//! The host model is dropped, then the model with its mixers on the card runs the same steps
//! twice, from a cold page cache and warm.
//! The card passes when it is no farther from the reference than the host (`no_farther` of
//! `card_mixer_tests.rs`) and chooses the reference's tokens.
//!
//! The harness takes closures that build the models, so the toy model of the card tests
//! exercises it on any machine with a card.

use super::card_mixer_tests::{
    closeness, dequantise_mixers, distances, exact, generic_on_card, no_farther,
};
use super::mixer::Mixer;
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::wide_sparse_real_tests::{
    argmax, blob, evict, prompt_ids, resident_mb, row, staged_forward, timed, LayerRec, Stages,
    FORCED, PERTURB, RECORD, STEPS, TAG, TRACE,
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
pub(super) fn mixer_bytes(model: &GenericHeteroTransformer) -> Vec<(usize, usize)> {
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
pub(super) fn first_card(plan: &HeteroPlan) -> usize {
    plan.segments
        .iter()
        .find_map(|s| match s.kind {
            DeviceKind::Cuda(o) => Some(o),
            _ => None,
        })
        .unwrap_or(0)
}

/// What the card plan asks of the cards for prompts of `ids` tokens and their steps.
pub(super) fn card_demand(facts: &Facts, meta: &WideSparseMeta, ids: usize) -> Demand {
    Demand {
        layers: facts.mixer_bytes.len(),
        block_bytes: facts.mixer_bytes.iter().sum(),
        head_bytes: 0,
        kv_bytes_per_layer: meta.kv_bytes_per_layer(ids + STEPS),
        runtime_reserve: llm_work_reserve(facts.width, ids),
    }
}

/// The plan that puts the layers whose mixers fit on the cards offered, and the cards.
///
/// The cards, their free bytes and the plan are the loader's own: `offer` and `Demand::plan`.
pub(super) fn card_plan(facts: &Facts, meta: &WideSparseMeta, ids: usize) -> HeteroPlan {
    let layers = facts.mixer_bytes.len();
    let demand = card_demand(facts, meta, ids);
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

/// The host model and the same model with its mixers on a card, against the exact reference.
///
/// `build_reference` makes the exact reference, run once and dropped.
/// `build_host` makes the host model, run once to fault its file in and once timed.
/// The host model is dropped, then `before_card` runs, then `build_card` makes the card model
/// from what the host model reported.
/// The card model runs twice and then once stage by stage.
pub(super) fn host_against_card(
    ids: &[u32],
    build_reference: impl FnOnce() -> GenericHeteroTransformer,
    build_host: impl FnOnce() -> GenericHeteroTransformer,
    before_card: impl FnOnce(),
    build_card: impl FnOnce(&Facts) -> (GenericHeteroTransformer, usize),
) {
    assert!(
        Device::new_cuda(0).is_ok(),
        "no CUDA device: this test measures the mixers on a card"
    );
    let mut reference = build_reference();
    let truth = pass(&mut reference, ids);
    println!("reference: greedy {:?}", truth.greedy);
    drop(reference);
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
    // The host's stage by stage trace, to find the layer where the card departs.
    let host_rec;
    let host_trace = {
        *TRACE.lock().unwrap() = Some(Vec::new());
        let (mut a, mut b) = (Stages::default(), Vec::new());
        *RECORD.lock().unwrap() = Some(Vec::new());
        let want_staged = staged_forward(&mut host, ids, 0, &mut a, &mut b);
        host_rec = RECORD.lock().unwrap().take().unwrap();
        // The host against itself with the mixer inputs moved by a fraction of a rounding step.
        for eps in [f32::EPSILON, f32::EPSILON / 16.0] {
            *PERTURB.lock().unwrap() = Some(eps);
            *RECORD.lock().unwrap() = Some(Vec::new());
            let got = staged_forward(&mut host, ids, 0, &mut a, &mut b);
            let rec = RECORD.lock().unwrap().take().unwrap();
            *PERTURB.lock().unwrap() = None;
            let (cos, l2) = closeness(&want_staged, &got);
            println!("== host against itself, mixer inputs scaled by 1 +- {eps:e}: logits cosine {cos:.6}, relative L2 {l2:.6} ==");
            compare_records(&host_rec, &rec);
        }
        TRACE.lock().unwrap().take().unwrap()
    };
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
    *TRACE.lock().unwrap() = Some(Vec::new());
    *RECORD.lock().unwrap() = Some(Vec::new());
    let (logits, s) = timed(|| staged_forward(&mut card, ids, 0, &mut prefill, &mut per_layer));
    let card_trace = TRACE.lock().unwrap().take().unwrap();
    let card_rec = RECORD.lock().unwrap().take().unwrap();
    println!("== free run: card against host, every stage ==");
    compare_records(&host_rec, &card_rec);
    // The card model again, each layer started from the host's hidden state.
    *FORCED.lock().unwrap() = Some(host_rec);
    *RECORD.lock().unwrap() = Some(Vec::new());
    let (mut fa, mut fb) = (Stages::default(), Vec::new());
    let _ = staged_forward(&mut card, ids, 0, &mut fa, &mut fb);
    let forced_rec = RECORD.lock().unwrap().take().unwrap();
    let host_rec = FORCED.lock().unwrap().take().unwrap();
    println!("== forced run: each layer starts from the host's state ==");
    compare_records(&host_rec, &forced_rec);
    for (il, (h, c)) in host_trace.iter().zip(&card_trace).enumerate() {
        let mixer = if h.0.is_empty() {
            String::from("-")
        } else {
            let (cos, l2) = closeness(&h.0, &c.0);
            format!("{cos:.6} {l2:.6}")
        };
        // The card's mixer against the same mixer on the host, on the same input.
        let twin = if c.1.is_empty() {
            String::from("-")
        } else {
            let (cos, l2) = closeness(&c.1, &c.0);
            format!("{cos:.6} {l2:.6}")
        };
        let (cos, l2) = closeness(&h.2, &c.2);
        println!(
            "layer {il}: mixer card against host {mixer}; same input {twin}; hidden {cos:.6} {l2:.6}"
        );
    }
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

    let terms = card.config.embedding_length;
    let mut failed = Vec::new();
    for (phase, got) in ["cold", "warm"].iter().zip(&passes) {
        let (host_l2, card_l2) = distances(&truth.last, &want.last, &got.last);
        println!(
            "card {phase}: relative L2 from the reference {card_l2:.6}, host {host_l2:.6}, greedy equal to the reference {}",
            got.greedy == truth.greedy
        );
        if card_l2.is_nan() || !no_farther(host_l2, card_l2, terms) {
            failed.push(format!(
                "{phase}: relative L2 {card_l2} from the reference, host {host_l2}"
            ));
        }
        if got.greedy != truth.greedy {
            failed.push(format!(
                "{phase}: greedy tokens {:?} != reference {:?}",
                got.greedy, truth.greedy
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "the card mixers are farther from the reference than the host:\n{}",
        failed.join("\n")
    );
}

/// The relative L2 of `b` against `a`, or the largest absolute gap when `a` is all zeros.
fn gap(a: &Tensor, b: &Tensor) -> f64 {
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (a, b) = (flat(a), flat(b));
    let num: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// One line per layer: how far each stage of `card` is from the same stage of `host`, the
/// experts each row chose and their margins, and whether the engram read the same rows.
fn compare_records(host: &[LayerRec], card: &[LayerRec]) {
    let mut first_stage = std::collections::BTreeMap::new();
    for (il, (h, c)) in host.iter().zip(card).enumerate() {
        let mut line = format!("layer {il:2}:");
        for (name, ht) in &h.stages {
            if let Some(ct) = c.stage(name) {
                let g = gap(ht, ct);
                if g > 0.0 {
                    first_stage.entry(*name).or_insert((il, g));
                }
                line += &format!(" {name}={g:.2e}");
            }
        }
        let flips = h
            .picks
            .iter()
            .zip(&c.picks)
            .filter(|(a, b)| {
                let (mut x, mut y) = (a.0.clone(), b.0.clone());
                x.sort();
                y.sort();
                x != y
            })
            .count();
        let margin = |r: &LayerRec| {
            r.picks
                .iter()
                .map(|p| p.1 - p.2)
                .fold(f32::INFINITY, f32::min)
        };
        line += &format!(
            " | flips {flips} of {} margin host {:.2e} card {:.2e}",
            h.picks.len(),
            margin(h),
            margin(c)
        );
        if !h.hashes.is_empty() {
            line += &format!(" | engram rows equal {}", h.hashes == c.hashes);
        }
        println!("{line}");
        // Row by row, for the stages a token mixer makes.
        for name in [
            "input",
            "qr",
            "q",
            "kv",
            "attn_o",
            "attn_o_replica",
            "attn_o_q8",
            "attn_gap",
            "mixed",
            "out",
            "mid",
        ] {
            if let (Some(ht), Some(ct)) = (h.stage(name), c.stage(name)) {
                let rows = ht.dims()[1];
                let per: Vec<String> = (0..rows)
                    .map(|i| {
                        let a = ht.narrow(1, i, 1).unwrap();
                        let b = ct.narrow(1, i, 1).unwrap();
                        format!("{:.1e}", gap(&a, &b))
                    })
                    .collect();
                println!("    {name} by row: {}", per.join(" "));
            }
        }
        if let (Some(ht), Some(ct)) = (h.stage("attn_o_q8"), c.stage("attn_o_q8")) {
            let rows = ht.dims()[1];
            let per: Vec<String> = (0..rows)
                .map(|i| {
                    let a = ht
                        .narrow(1, i, 1)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let b = ct
                        .narrow(1, i, 1)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    format!("{}", a.iter().zip(&b).filter(|(x, y)| x != y).count())
                })
                .collect();
            println!(
                "    codes of the output projection input that differ, by row: {}",
                per.join(" ")
            );
        }
        if let (Some(k), Some(r)) = (c.stage("attn_o"), c.stage("attn_o_replica")) {
            let rows = k.dims()[1];
            let per: Vec<String> = (0..rows)
                .map(|i| {
                    let a = k.narrow(1, i, 1).unwrap();
                    let b = r.narrow(1, i, 1).unwrap();
                    format!("{:.1e}", gap(&b, &a))
                })
                .collect();
            println!(
                "    card attention kernel against the host loop on the same queries, by row: {}",
                per.join(" ")
            );
        }
        if let Some(g) = h.stage("attn_gap") {
            let v = g
                .narrow(1, 0, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("    row 0 host: largest logit minus sink, over heads: {lo:.3} to {hi:.3}");
        }
    }
    for (name, (il, g)) in first_stage {
        println!("first difference of stage {name}: layer {il}, {g:.3e}");
    }
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
    let host =
        || load(&|_, layers| HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0)).0;
    host_against_card(
        &ids,
        || {
            let mut model = host();
            dequantise_mixers(&mut model);
            model
        },
        host,
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
pub(super) fn layers_of(split: &SplitGguf) -> usize {
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
            || exact(ratios, kv_src, index_src),
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
