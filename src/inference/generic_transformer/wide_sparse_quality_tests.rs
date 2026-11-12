//! The routing truncation each quality level chooses on the generic route, against the records
//! of the V4.1 calibration, on the synthetic model of the frozen V4.1 oracle.
//!
//! The generic calibration takes the tokens, the experts in memory and the pair of runs the V4.1
//! calibration took when it was recorded: a prefill of every token, and the same tokens decoded
//! 1 at a time.
//! The tests compare decisions: the logits under each candidate, the candidates, the reads each
//! saves and the truncation each level chooses.

use std::collections::HashSet;

use super::routing_calibration as calibration;
use super::wide_sparse_frozen as frozen;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::engine::llm_engine::QualityLevel;
use crate::inference::generic_transformer::wide_sparse_toy::prompt;

/// The record file of these tests.
const FROZEN: &str = "quality";

const LEVELS: [QualityLevel; 3] = [
    QualityLevel::Exact,
    QualityLevel::Balanced,
    QualityLevel::Fast,
];

/// Every position's logits of a generic prefill of `ids` from position 0.
fn generic_prefill(m: &mut GenericHeteroTransformer, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    let x = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu)?;
    let all = m.forward_all(&x, 0)?.flatten_all()?.to_vec1::<f32>()?;
    let vocab = all.len() / ids.len().max(1);
    Ok(all.chunks(vocab.max(1)).map(<[f32]>::to_vec).collect())
}

/// Every position's logits of `ids` decoded 1 at a time by the generic model from position 0.
fn generic_decode(m: &mut GenericHeteroTransformer, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    ids.iter()
        .enumerate()
        .map(|(pos, &t)| {
            let x = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            m.forward(&x, pos)?.flatten_all()?.to_vec1::<f32>()
        })
        .collect()
}

/// Per layer, the even experts of its routed feed-forward: the experts counted in memory.
fn even_experts(m: &GenericHeteroTransformer) -> Vec<HashSet<usize>> {
    m.layers
        .iter()
        .map(|l| {
            l.sinkhorn.as_ref().map_or_else(HashSet::new, |b| {
                (0..b.ffn.experts.len()).step_by(2).collect()
            })
        })
        .collect()
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|r| r.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// The truncation a level chooses from `table` for a model of `noise`, as `apply_budget`
/// applies it: none for a budget of 0.
fn generic_choice(
    table: &[calibration::Measured],
    level: QualityLevel,
    noise: f64,
) -> Option<(usize, u32)> {
    let budget = calibration::budget_for(level, Some(noise)).filter(|b| *b > 0.0)?;
    calibration::pick(table, budget).map(|m| (m.keep, m.miss_below.to_bits()))
}

/// A recorded number held as the bits of an `f64`.
fn recorded_f64(key: &str) -> f64 {
    f64::from_bits(frozen::recorded(FROZEN, key).parse().unwrap())
}

/// Under every candidate truncation, with the even experts in memory, the generic prefill gives
/// the recorded bits of the V4.1 prefill, and each saves the recorded reads.
#[test]
fn every_candidate_routes_the_generic_model_as_the_v41_model() {
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let tokens = prompt();
        let mut model = generic(ratios, kv_src, index_src);
        let resident = even_experts(&model);
        frozen::expect(
            FROZEN,
            &format!("l{n} activated"),
            &format!("{}", model.routed_per_token()),
        );
        let candidates = calibration::candidates(model.routed_per_token());
        for (k, t) in candidates.into_iter().enumerate() {
            model.set_resident_override(Some(resident.as_slice()));
            model.set_truncation(t);
            let _ = model.take_saved();
            let got = generic_prefill(&mut model, &tokens).unwrap();
            let got_saved = model.take_saved();
            frozen::expect_f32(FROZEN, &format!("l{n} c{k} prefill"), &got.concat());
            frozen::expect(
                FROZEN,
                &format!("l{n} c{k} saved"),
                &format!("{got_saved:?}"),
            );
        }
    }
}

/// The generic calibration on the recorded tokens, experts in memory and runs chooses the
/// truncation the V4.1 calibration chose at every level.
///
/// The table holds the recorded candidates and reads saved.
/// Each level then chooses from the noise the generic side measures, against the V4.1 choice from
/// the noise the V4.1 side measured.
/// It also chooses from each candidate's own divergence taken as the noise, which puts a budget
/// on every candidate in turn.
#[test]
fn both_calibrations_choose_the_same_truncation_per_level() {
    let mut chose = false;
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let tokens = prompt();
        let mut model = generic(ratios, kv_src, index_src);
        // Only the first expert of each layer is in memory, so that dropping the others saves reads.
        let resident: Vec<HashSet<usize>> = even_experts(&model)
            .iter()
            .map(|set| set.iter().copied().filter(|&e| e == 0).collect())
            .collect();

        let got =
            calibration::measure(&mut model, &tokens, &resident, &mut generic_prefill).unwrap();
        let decided = |keep: usize, miss: f32, saved: f64| (keep, miss.to_bits(), saved.to_bits());
        frozen::expect_debug(
            FROZEN,
            &format!("l{n} table"),
            &got.iter()
                .map(|m| decided(m.keep, m.miss_below, m.saved_per_token))
                .collect::<Vec<_>>(),
        );
        let want_kl: Vec<f64> = (0..got.len())
            .map(|j| recorded_f64(&format!("l{n} kl{j}")))
            .collect();

        let want_noise = recorded_f64(&format!("l{n} noise"));
        let got_noise = calibration::noise(
            &mut model,
            &tokens,
            &mut generic_prefill,
            &mut generic_decode,
        )
        .unwrap();
        let mut noises = vec![(want_noise, got_noise)];
        // A divergence of rounding size is not a measurement: the two quality instruments differ
        // there in the last bits of a sum, and a candidate that changes nothing reads as 0 on one
        // side and as a residue on the other.
        noises.extend(
            want_kl
                .iter()
                .zip(&got)
                .filter(|(w, g)| w.max(g.kl) >= 1e-12)
                .map(|(w, g)| (*w, g.kl)),
        );
        // A noise floor that admits every candidate, so that a pick is made and compared.
        noises.push((1.0e3, 1.0e3));
        for (i, (want_noise, got_noise)) in noises.into_iter().enumerate() {
            for level in LEVELS {
                let g = generic_choice(&got, level, got_noise);
                let key = format!("l{n} n{i} {level:?}");
                assert_eq!(
                    format!("{g:?}"),
                    frozen::recorded(FROZEN, &key),
                    "{ratios:?} {level:?} noise {want_noise}"
                );
                chose |= g.is_some();
            }
        }
    }
    assert!(
        chose,
        "no level chose a truncation: the comparison proves nothing"
    );
}

/// A zero budget runs nothing and leaves the logits bit for bit. Measuring tries every
/// candidate and leaves no setting on.
#[test]
fn a_zero_budget_runs_nothing_and_measuring_leaves_routing_exact() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let tokens = prompt();
    let mut model = generic(ratios, kv_src, index_src);
    let resident = even_experts(&model);
    let plain = generic_prefill(&mut model, &tokens).unwrap();
    let mut nothing = |_: &mut GenericHeteroTransformer, _: &[u32]| -> Result<Vec<Vec<f32>>> {
        panic!("a zero budget must not run the model")
    };
    let chosen =
        calibration::apply_budget(&mut model, 0.0, "k", &tokens, &resident, &mut nothing).unwrap();
    assert!(chosen.is_none());
    assert_eq!(
        bits(&generic_prefill(&mut model, &tokens).unwrap()),
        bits(&plain),
        "a zero budget changed the logits"
    );

    let table = calibration::measure(&mut model, &tokens, &resident, &mut generic_prefill).unwrap();
    assert_eq!(
        table.len(),
        calibration::candidates(model.routed_per_token()).len()
    );
    assert_eq!(
        bits(&generic_prefill(&mut model, &tokens).unwrap()),
        bits(&plain),
        "measuring left a setting on"
    );
    assert_eq!(model.take_saved(), 0);
}

/// The routed feed-forward of the first layer of `model` that has one.
fn first_ffn(model: &GenericHeteroTransformer) -> &super::wide_sparse_layer::RoutedFfn {
    model
        .layers
        .iter()
        .find_map(|l| l.sinkhorn.as_ref().map(|b| &b.ffn))
        .unwrap()
}

/// Router logits of `experts` experts, spread so that the weights of a choice differ.
fn spread_logits(experts: usize) -> Vec<f32> {
    (0..experts)
        .map(|e| ((e * 7 + 3) % experts) as f32 / experts as f32 * 4.0 - 2.0)
        .collect()
}

/// With a noise and experts in memory, `fast` chooses a candidate that saves reads.
/// The exact level chooses none.
/// Under the choice, the toy's answers stay within the budget against the exact run.
#[test]
fn fast_chooses_a_trade_within_its_budget_and_exact_chooses_none() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let tokens = prompt();
    let mut model = generic(ratios, kv_src, index_src);
    let resident: Vec<HashSet<usize>> = even_experts(&model)
        .iter()
        .map(|set| set.iter().copied().filter(|&e| e == 0).collect())
        .collect();
    let table = calibration::measure(&mut model, &tokens, &resident, &mut generic_prefill).unwrap();
    // The noise of a model whose smallest saving divergence is 2 times its noise.
    let noise = table
        .iter()
        .filter(|m| m.saved_per_token > 0.0 && m.kl > 0.0)
        .map(|m| m.kl)
        .fold(f64::INFINITY, f64::min)
        / 2.0;
    assert!(noise.is_finite(), "no candidate saves a read on the toy");
    assert!(calibration::budget_for(QualityLevel::Exact, Some(noise)).is_none());
    let budget = calibration::budget_for(QualityLevel::Fast, Some(noise)).unwrap();
    let chosen = calibration::pick(&table, budget).cloned().unwrap();
    assert!(chosen.saved_per_token > 0.0);

    // The chosen truncation run against the exact run, as the calibration scores a noise.
    let truncation = chosen.truncation();
    let mut under = |m: &mut GenericHeteroTransformer, ids: &[u32]| -> Result<Vec<Vec<f32>>> {
        m.set_resident_override(Some(resident.as_slice()));
        m.set_truncation(truncation);
        let rows = generic_prefill(m, ids);
        m.set_truncation(Truncation::default());
        m.set_resident_override(None);
        rows
    };
    let _ = model.take_saved();
    let kl = calibration::noise(&mut model, &tokens, &mut generic_prefill, &mut under).unwrap();
    assert!(model.take_saved() > 0, "the choice saved no read");
    assert!(kl <= budget, "KL {kl} over the budget {budget}");
}

/// A light expert that is not in memory gives its slot to the best expert in memory beyond
/// the choice, and the slot counts as a read saved.
#[test]
fn a_light_missed_expert_gives_its_slot_to_one_in_memory() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    let ffn = first_ffn(&model);
    let row = spread_logits(ffn.experts.len());
    let exact = ffn.route(std::slice::from_ref(&row)).remove(0).0;
    let k = ffn.used;
    let outside = (0..ffn.experts.len()).find(|e| !exact.contains(e)).unwrap();
    let layers = model.layers.len();
    model.set_resident_override(Some(&vec![HashSet::from([outside]); layers]));
    model.set_truncation(Truncation {
        miss_below: 1.0 / k as f32,
        ..Default::default()
    });
    let _ = model.take_saved();
    let ffn = first_ffn(&model);
    let got = ffn.route(std::slice::from_ref(&row)).remove(0).0;
    assert_eq!(got.len(), exact.len(), "a replacement keeps every slot");
    assert!(
        got.contains(&outside),
        "{exact:?} gave no slot to {outside}: {got:?}"
    );
    assert_eq!(model.take_saved(), 1);
}

/// At the exact level the router routes as the plain choice, whatever is in memory, and saves
/// nothing.
#[test]
fn the_exact_level_neither_replaces_nor_skips() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    let row = spread_logits(first_ffn(&model).experts.len());
    let plain = first_ffn(&model).route(std::slice::from_ref(&row));
    let layers = model.layers.len();
    let all: HashSet<usize> = (0..first_ffn(&model).experts.len()).collect();
    for resident in [None, Some(vec![all; layers])] {
        model.set_resident_override(resident.as_deref());
        model.set_truncation(Truncation::default());
        let _ = model.take_saved();
        assert_eq!(first_ffn(&model).route(std::slice::from_ref(&row)), plain);
        assert_eq!(model.take_saved(), 0);
    }
}

/// Without an override, the experts in memory are those the tiers keep, and the toy's tiers
/// keep none: no light expert finds one to give its slot to.
#[test]
fn without_an_override_the_tiers_say_what_is_in_memory() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    assert!(model.held_experts().iter().all(HashSet::is_empty));
    let ffn = first_ffn(&model);
    let row = spread_logits(ffn.experts.len());
    let exact = ffn.route(std::slice::from_ref(&row)).remove(0).0;
    let k = ffn.used;
    model.set_resident_override(None);
    model.set_truncation(Truncation {
        miss_below: 1.0 / k as f32,
        ..Default::default()
    });
    let _ = model.take_saved();
    let got = first_ffn(&model)
        .route(std::slice::from_ref(&row))
        .remove(0)
        .0;
    assert_eq!(got, exact);
    assert_eq!(model.take_saved(), 0);
}

/// A directory of its own under the system's temporary directory, empty.
fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("loken-routing-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// The calibration run between requests measures the noise and the table the blocking
/// calibration measures, bit for bit, and installs the same pick.
/// Between 2 steps, a prompt gets the exact logits.
/// The decoded run gives way at once and keeps nothing.
/// A new load resumes after the candidates measured, without running the earlier runs again.
#[test]
fn the_calibration_between_requests_resumes_and_picks_as_the_blocking_one() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let tokens = prompt();
    let probe: Vec<u32> = tokens.iter().rev().copied().collect();
    let resident: Vec<HashSet<usize>> = even_experts(&generic(ratios, kv_src, index_src))
        .iter()
        .map(|set| set.iter().copied().filter(|&e| e == 0).collect())
        .collect();

    let mut blocking = generic(ratios, kv_src, index_src);
    let exact = generic_prefill(&mut blocking, &probe).unwrap();
    let noise = calibration::noise(
        &mut blocking,
        &tokens,
        &mut generic_prefill,
        &mut generic_decode,
    )
    .unwrap();
    let table =
        calibration::measure(&mut blocking, &tokens, &resident, &mut generic_prefill).unwrap();
    let budget = calibration::budget_for(QualityLevel::Fast, Some(noise)).unwrap();
    let want = calibration::pick_held(&table, budget).cloned();

    let dir = scratch("steps");
    let level = QualityLevel::Fast;
    let mut model = generic(ratios, kv_src, index_src);
    let at_load = calibration::prepare(
        &mut model,
        level,
        "toy",
        &tokens,
        &resident,
        Some(&dir),
        true,
    )
    .unwrap();
    assert!(at_load.is_none(), "the load waited for the calibration");
    let pending = model
        .routing_pending()
        .cloned()
        .expect("a calibration is left");
    let no = || false;
    let yes = || true;
    let mut decode = |m: &mut GenericHeteroTransformer, ids: &[u32]| {
        calibration::decode_run_yielding(m, ids, &no)
    };
    let mut give_way = |m: &mut GenericHeteroTransformer, ids: &[u32]| {
        calibration::decode_run_yielding(m, ids, &yes)
    };
    let answered_exactly = |model: &mut GenericHeteroTransformer| {
        assert_eq!(
            bits(&generic_prefill(model, &probe).unwrap()),
            bits(&exact),
            "a prompt between 2 steps was not answered exactly"
        );
    };
    let steps = [
        calibration::Step::Measured,
        calibration::Step::Yielded,
        calibration::Step::Measured,
        calibration::Step::Measured,
    ];
    for (i, want_step) in steps.into_iter().enumerate() {
        let stepped = match i {
            1 => pending.step(
                &mut model,
                Some(&dir),
                &mut generic_prefill,
                &mut give_way,
                &yes,
            ),
            _ => pending.step(
                &mut model,
                Some(&dir),
                &mut generic_prefill,
                &mut decode,
                &no,
            ),
        };
        assert_eq!(stepped.unwrap(), want_step, "step {i}");
        answered_exactly(&mut model);
    }

    // A new load of the same file resumes after the first candidate.
    let mut model = generic(ratios, kv_src, index_src);
    calibration::prepare(
        &mut model,
        level,
        "toy",
        &tokens,
        &resident,
        Some(&dir),
        true,
    )
    .unwrap();
    let pending = model
        .routing_pending()
        .cloned()
        .expect("the calibration resumes");
    let mut runs = 0;
    let mut counted = |m: &mut GenericHeteroTransformer, ids: &[u32]| {
        runs += 1;
        generic_prefill(m, ids)
    };
    let mut never = |_: &mut GenericHeteroTransformer, _: &[u32]| -> Result<Vec<Vec<f32>>> {
        panic!("the noise was measured again")
    };
    let done = loop {
        match pending
            .step(&mut model, Some(&dir), &mut counted, &mut never, &no)
            .unwrap()
        {
            calibration::Step::Done(pick) => break pick,
            calibration::Step::Measured => {}
            calibration::Step::Yielded => panic!("nothing asked the calibration to give way"),
        }
    };
    assert_eq!(
        runs,
        table.len() - 1,
        "the resumed calibration ran earlier runs again"
    );
    assert_eq!(
        done, want,
        "the pick differs from the blocking calibration's"
    );

    let kept = |file: &str| std::fs::read_to_string(dir.join(file)).unwrap();
    let tables: std::collections::HashMap<String, Vec<calibration::Measured>> =
        serde_json::from_str(&kept("quality-calibration.json")).unwrap();
    let noises: std::collections::HashMap<String, f64> =
        serde_json::from_str(&kept("quality-noise.json")).unwrap();
    assert_eq!(tables.values().collect::<Vec<_>>(), [&table]);
    assert_eq!(
        noises.values().map(|n| n.to_bits()).collect::<Vec<_>>(),
        [noise.to_bits()]
    );

    // A third load finds the measure whole and sets the pick at once.
    let mut model = generic(ratios, kv_src, index_src);
    let at_load = calibration::prepare(
        &mut model,
        level,
        "toy",
        &tokens,
        &resident,
        Some(&dir),
        true,
    )
    .unwrap();
    assert!(model.routing_pending().is_none());
    assert_eq!(at_load, want);
    let _ = std::fs::remove_dir_all(&dir);
}
