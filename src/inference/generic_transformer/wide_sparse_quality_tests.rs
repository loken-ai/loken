//! The routing truncation each quality level chooses on the generic route, against the V4.1
//! calibration, on the synthetic model of the frozen V4.1 oracle.
//!
//! Both calibrations take the same tokens, the same experts in memory and the same pair of
//! runs: a prefill of every token, and the same tokens decoded 1 at a time.
//! The tests compare decisions: the logits under each candidate, the candidates, the reads each
//! saves and the truncation each level chooses.

use std::collections::HashSet;

use super::routing_calibration as calibration;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::engine::llm_engine::QualityLevel;
use crate::inference::generic_transformer::wide_sparse_toy::prompt;
use crate::inference::model::deepseek_v41::calibrate;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::moe::Truncation as V41Truncation;

const LEVELS: [QualityLevel; 3] = [
    QualityLevel::Exact,
    QualityLevel::Balanced,
    QualityLevel::Fast,
];

/// Every position's logits of a V4.1 prefill of `ids`.
fn v41_prefill(m: &DeepseekV41Model, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    m.forward_prefill(ids)?.to_vec2::<f32>()
}

/// Every position's logits of `ids` decoded 1 at a time by the V4.1 model from a fresh state.
fn v41_decode(m: &DeepseekV41Model, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut state = m.new_decode_state();
    ids.iter()
        .map(|&t| {
            m.forward_decode(t, &mut state)?
                .flatten_all()?
                .to_vec1::<f32>()
        })
        .collect()
}

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

fn v41_truncation(t: super::moe_weights::Truncation) -> V41Truncation {
    V41Truncation {
        keep: t.keep,
        drop_mass: t.drop_mass,
        miss_below: t.miss_below,
        miss_drops: t.miss_drops,
    }
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|r| r.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// The truncation a level chooses from `table` for a model of `noise`, as `apply_budget`
/// applies it: none for a budget of 0.
fn v41_choice(
    table: &[calibrate::Measured],
    level: QualityLevel,
    noise: f64,
) -> Option<(usize, u32)> {
    let budget = calibrate::budget_for(level, Some(noise)).filter(|b| *b > 0.0)?;
    calibrate::pick(table, budget).map(|m| (m.keep, m.miss_below.to_bits()))
}

/// [`v41_choice`] on the generic side.
fn generic_choice(
    table: &[calibration::Measured],
    level: QualityLevel,
    noise: f64,
) -> Option<(usize, u32)> {
    let budget = calibration::budget_for(level, Some(noise)).filter(|b| *b > 0.0)?;
    calibration::pick(table, budget).map(|m| (m.keep, m.miss_below.to_bits()))
}

/// Under every candidate truncation, with the even experts in memory, the generic prefill gives
/// the bits of the V4.1 prefill, and each saves the same reads.
#[test]
fn every_candidate_routes_the_generic_model_as_the_v41_model() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let (v41, tokens) = synthetic(ratios, kv_src, index_src, 0);
        let mut model = generic(ratios, kv_src, index_src);
        let resident = even_experts(&model);
        assert_eq!(model.routed_per_token(), v41.n_activated());
        for t in calibration::candidates(model.routed_per_token()) {
            v41.set_resident_override(Some(resident.as_slice()));
            v41.set_truncation(v41_truncation(t));
            let _ = v41.take_saved();
            let want = v41_prefill(&v41, &tokens).unwrap();
            let want_saved = v41.take_saved();
            model.set_resident_override(Some(resident.as_slice()));
            model.set_truncation(t);
            let _ = model.take_saved();
            let got = generic_prefill(&mut model, &tokens).unwrap();
            let got_saved = model.take_saved();
            assert_eq!(bits(&got), bits(&want), "{ratios:?} {t:?}");
            assert_eq!(got_saved, want_saved, "{ratios:?} {t:?}");
        }
    }
}

/// Both calibrations on the same tokens, experts in memory and runs choose the same truncation
/// at every level.
///
/// The tables hold the same candidates and the same reads saved.
/// Each level then chooses from the noise each side measured, and from each candidate's own
/// divergence taken as the noise, which puts a budget on every candidate in turn.
#[test]
fn both_calibrations_choose_the_same_truncation_per_level() {
    let mut chose = false;
    for (ratios, kv_src, index_src) in LAYOUTS {
        let (v41, tokens) = synthetic(ratios, kv_src, index_src, 0);
        let mut model = generic(ratios, kv_src, index_src);
        // Only the first expert of each layer is in memory, so that dropping the others saves reads.
        let resident: Vec<HashSet<usize>> = even_experts(&model)
            .iter()
            .map(|set| set.iter().copied().filter(|&e| e == 0).collect())
            .collect();

        let want = calibrate::measure(&v41, &tokens, &resident, &v41_prefill).unwrap();
        let got =
            calibration::measure(&mut model, &tokens, &resident, &mut generic_prefill).unwrap();
        let decided = |keep: usize, miss: f32, saved: f64| (keep, miss.to_bits(), saved.to_bits());
        assert_eq!(
            got.iter()
                .map(|m| decided(m.keep, m.miss_below, m.saved_per_token))
                .collect::<Vec<_>>(),
            want.iter()
                .map(|m| decided(m.keep, m.miss_below, m.saved_per_token))
                .collect::<Vec<_>>(),
            "{ratios:?}: candidates or reads saved"
        );

        let want_noise = calibrate::noise(&v41, &tokens, &v41_prefill, &v41_decode).unwrap();
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
            want.iter()
                .zip(&got)
                .filter(|(w, g)| w.kl.max(g.kl) >= 1e-12)
                .map(|(w, g)| (w.kl, g.kl)),
        );
        // A noise floor that admits every candidate, so that a pick is made and compared.
        noises.push((1.0e3, 1.0e3));
        for (want_noise, got_noise) in noises {
            for level in LEVELS {
                let w = v41_choice(&want, level, want_noise);
                let g = generic_choice(&got, level, got_noise);
                assert_eq!(g, w, "{ratios:?} {level:?} noise {want_noise}");
                chose |= w.is_some();
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
