//! The routing heat of a wide-residual model on the generic route against the V4.1 path, on the
//! small file of the frozen V4.1 oracle.
//!
//! The file lies in a directory of its own, beside a companion file that declares a routing
//! prior.
//! A routing profile is a file as a run writes it.
//! The V4.1 side follows the steps its engine arm took: the prior beside the file when the file
//! carries none, then the profile seeded into the model, then its ranks.

use super::block_draft_tests::generic_of;
use super::expert_profile::ExpertProfile;
use super::wide_sparse_sibling as sibling;
use crate::inference::generic_transformer::wide_sparse_toy::{
    real_entries_for, real_md_layout, VOCAB,
};
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::offload::streamed::StreamedModel;
use crate::inference::place::expert_variants::{ranks_from_counts, VariantChoice};
use crate::tensor::gguf_write::write_gguf_with_metadata;
use crate::tensor::quantized::gguf_file::{open_mapped, Value};
use crate::tensor::quantized::gguf_source::SplitGguf;
use std::path::{Path, PathBuf};

/// The oracle's first band layout: compress ratios, kv source layers, index source layers.
const LAYOUT: (&[u32], &[u32], &[u32]) = (&[0, 2, 0], &[1], &[1]);

/// Routed experts of a layer of the small file.
pub(crate) const ROUTED: usize = 8;

/// Layers of the small file.
pub(crate) fn layers() -> usize {
    LAYOUT.0.len()
}

/// A directory of its own holding the small file, carrying `own` as its prior when given.
///
/// Returns the model file's path.
/// The caller removes its directory.
pub(crate) fn toy_dir(tag: &str, own: Option<&[Vec<u32>]>) -> PathBuf {
    let (ratios, kv_src, index_src) = LAYOUT;
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut md = real_md_layout(ratios, kv_src, index_src);
    let arch = md
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| v.to_string().ok().cloned())
        .unwrap();
    // A token no request produces ends nothing early.
    md.push((
        "tokenizer.ggml.eos_token_id".to_string(),
        Value::U32(VOCAB as u32 + 1),
    ));
    for (l, list) in own.into_iter().flatten().enumerate() {
        md.push((
            format!("{arch}.hot_experts.{l}"),
            Value::Array(list.iter().map(|&e| Value::U32(e)).collect()),
        ));
    }
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-wide-routing-{tag}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    write_gguf_with_metadata(&model, &md, &real_entries_for(&bands)).unwrap();
    model
}

/// Per layer, every expert id below `experts` in an order that turns with the layer.
pub(crate) fn turning(layers: usize, experts: u32) -> Vec<Vec<u32>> {
    (0..layers)
        .map(|l| {
            (0..experts)
                .rev()
                .map(|e| (e + 3 * l as u32) % experts)
                .collect()
        })
        .collect()
}

fn as_ids(lists: &[Vec<u32>]) -> Vec<Vec<usize>> {
    lists
        .iter()
        .map(|l| l.iter().map(|&e| e as usize).collect())
        .collect()
}

/// A profile of `layers` layers of `experts` experts with `counts`, written beside the model.
pub(crate) fn profile_file(
    model: &Path,
    layers: usize,
    experts: usize,
    counts: Vec<u32>,
) -> PathBuf {
    let path = model.with_file_name("routing.profile");
    ExpertProfile {
        layers,
        experts,
        counts,
    }
    .write(&path)
    .unwrap();
    path
}

/// The heat of the V4.1 path, by the steps of its engine arm.
pub(crate) fn v41_heat(model: &Path, profile: Option<&Path>) -> Vec<Vec<usize>> {
    let source = SplitGguf::open(model).unwrap();
    let mut cfg = DeepseekV41Config::from_meta(&source).unwrap();
    if cfg.hot_experts.iter().all(Vec::is_empty) {
        if let Some(prior) = sibling::prior_beside(model, cfg.n_layers, cfg.n_routed_experts) {
            cfg.hot_experts = prior;
        }
    }
    let v41 = DeepseekV41Model::load(&source, &cfg).unwrap();
    let ranks = profile.and_then(|path| {
        let profile = ExpertProfile::read_if_present(path).ok()??;
        v41.seed_routing_profile(&profile).ok()?;
        ranks_from_counts(
            &profile.counts,
            profile.layers,
            profile.experts,
            &cfg.hot_experts,
        )
    });
    v41.set_variant_policy(VariantChoice::Plan, ranks);
    StreamedModel::routing_prior(&v41).to_vec()
}

/// The heat of the generic route, by the steps of the generic engine arm, and the refusal of the
/// profile when there is one.
fn generic_heat(model: &Path, profile: Option<&Path>) -> (Vec<Vec<usize>>, Option<String>) {
    let mut generic = generic_of(open_mapped(model).unwrap());
    generic.locate_routing_prior(model);
    let refused = profile
        .and_then(|path| ExpertProfile::read_if_present(path).unwrap())
        .and_then(|p| generic.prime_routing_heat(&p).err())
        .map(|e| e.to_string());
    (generic.routing_heat().to_vec(), refused)
}

fn remove(model: &Path) {
    let _ = std::fs::remove_dir_all(model.parent().unwrap());
}

/// A file that carries no prior takes the one a companion beside it declares, on both routes.
#[test]
fn a_companion_prior_ranks_the_experts_of_both_routes() {
    let model = toy_dir("companion", None);
    let prior = turning(layers(), ROUTED as u32);
    sibling::write_prior(&model.with_file_name("side"), &prior).unwrap();
    let want = v41_heat(&model, None);
    assert_eq!(want, as_ids(&prior), "the V4.1 path took the companion");
    let (got, refused) = generic_heat(&model, None);
    assert_eq!(got, want);
    assert_eq!(refused, None);
    remove(&model);
}

/// A profile's counts rank the experts, the companion's prior ordering the uncounted, and a
/// layer the profile never counted keeps the prior, on both routes.
#[test]
fn a_profile_ranks_over_the_companion_prior_on_both_routes() {
    let model = toy_dir("profile", None);
    let prior = turning(layers(), ROUTED as u32);
    sibling::write_prior(&model.with_file_name("side"), &prior).unwrap();
    // Layer 0 counts 4 experts with a tie, layer 1 one expert, layer 2 none.
    let mut counts = vec![0u32; layers() * ROUTED];
    counts[2] = 9;
    counts[5] = 4;
    counts[6] = 4;
    counts[7] = 1;
    counts[ROUTED + 3] = 2;
    let profile = profile_file(&model, layers(), ROUTED, counts);
    let want = v41_heat(&model, Some(&profile));
    assert_eq!(want[0][..4], [2, 5, 6, 7], "counted first, ties by id");
    assert_eq!(
        want[2],
        as_ids(&prior)[2],
        "an uncounted layer keeps the prior"
    );
    let (got, refused) = generic_heat(&model, Some(&profile));
    assert_eq!(got, want);
    assert_eq!(refused, None);
    remove(&model);
}

/// A profile of another shape and a profile that counted nothing leave the prior on both
/// routes; the generic route says why it refused the first.
#[test]
fn a_profile_that_does_not_fit_or_counted_nothing_leaves_the_prior() {
    let model = toy_dir("unfit", None);
    let prior = turning(layers(), ROUTED as u32);
    sibling::write_prior(&model.with_file_name("side"), &prior).unwrap();
    let wide = profile_file(
        &model,
        layers() + 1,
        ROUTED,
        vec![1; (layers() + 1) * ROUTED],
    );
    let want = v41_heat(&model, Some(&wide));
    assert_eq!(want, as_ids(&prior));
    let (got, refused) = generic_heat(&model, Some(&wide));
    assert_eq!(got, want);
    assert!(
        refused.is_some_and(|e| e.contains("the profile counts 4 layers of 8 experts")),
        "the shapes are named"
    );
    let zero = profile_file(&model, layers(), ROUTED, vec![0; layers() * ROUTED]);
    let want = v41_heat(&model, Some(&zero));
    assert_eq!(want, as_ids(&prior));
    assert_eq!(generic_heat(&model, Some(&zero)), (want, None));
    remove(&model);
}

/// The prior the file carries wins over a companion's, and a companion whose prior names an
/// expert the model does not have is passed over, on both routes.
#[test]
fn the_files_own_prior_wins_and_an_unfit_companion_is_passed_over() {
    let own: Vec<Vec<u32>> = (0..layers() as u32).map(|l| vec![l, 7 - l]).collect();
    let model = toy_dir("own", Some(&own));
    sibling::write_prior(
        &model.with_file_name("side"),
        &turning(layers(), ROUTED as u32),
    )
    .unwrap();
    let want = v41_heat(&model, None);
    assert_eq!(want, as_ids(&own));
    assert_eq!(generic_heat(&model, None).0, want);
    remove(&model);

    let model = toy_dir("unfit-side", None);
    sibling::write_prior(
        &model.with_file_name("side"),
        &turning(layers(), ROUTED as u32 + 2),
    )
    .unwrap();
    let want = v41_heat(&model, None);
    assert!(want.iter().all(Vec::is_empty), "no prior on the V4.1 path");
    assert_eq!(generic_heat(&model, None).0, want);
    remove(&model);
}
