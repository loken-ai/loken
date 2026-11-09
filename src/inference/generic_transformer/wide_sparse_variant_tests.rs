//! The routed experts of a wide-residual model read from 2 files of its checkpoint, on the
//! generic route against the V4.1 path, on the small file of the frozen V4.1 oracle.
//!
//! The model's own file stores its experts as F32 stacks.
//! The variant file stores the same experts as Q4_K, 1 tensor per expert, and every other tensor
//! as the model's file does.
//! The V4.1 side follows the steps its engine arm took: the check, the load through its variant
//! source, the policy, the host tier, and the plan with no card.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::block_draft_tests::{generic_of, prompt};
use super::wide_sparse_oracle_tests::{bit_difference, record_generic, record_v41, RECORDS};
use super::wide_sparse_variants::open_checked;
use super::GenericHeteroTransformer;
use crate::inference::generic_transformer::wide_sparse_toy::{
    real_entries_for, real_md_layout, VOCAB,
};
use crate::inference::load::variant_files::check_variant;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::source::VariantSource;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::place::expert_variants::{VariantChoice, VariantPlan};
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, Value};
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::quantized::GgmlDType;

/// The oracle's first band layout: compress ratios, kv source layers, index source layers.
pub(crate) const LAYOUT: (&[u32], &[u32], &[u32]) = (&[0, 2, 0], &[1], &[1]);

/// Every policy.
pub(crate) const CHOICES: [VariantChoice; 3] = [
    VariantChoice::Largest,
    VariantChoice::Smallest,
    VariantChoice::Plan,
];

/// What a file of the checkpoint changes from the model's own file.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    /// The experts as Q4_K, every other tensor the same.
    Q4,
    /// The experts as Q4_K, and 1 norm with other values: a file of another checkpoint.
    OtherNorm,
    /// The experts as Q4_K, and 1 expert fewer in layer 0.
    FewerExperts,
}

/// Whether `name` is a stack of routed experts.
fn is_stack(name: &str) -> bool {
    ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .any(|s| name.ends_with(s))
}

/// The tensors of the small file, in the order the model's file writes them.
///
/// Every tensor but the norms holds small values of its own.
/// The routed experts then weigh in the logits.
fn entries() -> Vec<GgufEntry> {
    let (ratios, kv_src, index_src) = LAYOUT;
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut entries = real_entries_for(&bands);
    for (t, entry) in entries
        .iter_mut()
        .enumerate()
        .filter(|(_, e)| !e.name.contains("norm"))
    {
        let n: usize = entry.dims.iter().product();
        entry.data = (0..n)
            .flat_map(|i| (((i * 7 + t * 131) as f32 * 0.37).sin() * 0.05).to_le_bytes())
            .collect();
    }
    entries
}

/// The tensors of a variant file: every expert stack of the model's file stored as Q4_K, 1
/// tensor per expert, and the change `kind` makes.
fn variant_entries(kind: Variant) -> Vec<GgufEntry> {
    let mut out = Vec::new();
    for entry in entries() {
        if !is_stack(&entry.name) {
            let mut entry = entry;
            if kind == Variant::OtherNorm && entry.name == "blk.0.attn_norm" {
                entry.data[0] ^= 1;
            }
            out.push(entry);
            continue;
        }
        let values: Vec<f32> = entry
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (experts, rows, cols) = (entry.dims[0], entry.dims[1], entry.dims[2]);
        let kept = match (kind, entry.name.starts_with("blk.0.")) {
            (Variant::FewerExperts, true) => experts - 1,
            _ => experts,
        };
        for e in 0..kept {
            let one = &values[e * rows * cols..(e + 1) * rows * cols];
            out.push(GgufEntry {
                name: format!("{}.{e}", entry.name),
                dims: vec![rows, cols],
                dtype: GgmlDType::Q4K,
                data: crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q4K, one).unwrap(),
            });
        }
    }
    out
}

/// The metadata of the small file: its description and the end token no request produces.
pub(crate) fn metadata() -> Vec<(String, Value)> {
    let (ratios, kv_src, index_src) = LAYOUT;
    let mut md = real_md_layout(ratios, kv_src, index_src);
    md.push((
        "tokenizer.ggml.eos_token_id".to_string(),
        Value::U32(VOCAB as u32 + 1),
    ));
    md
}

/// The model's file at `model` and a variant file of `kind` at `variant`.
pub(crate) fn write_pair(model: &Path, variant: &Path, kind: Variant) {
    write_gguf_with_metadata(model, &metadata(), &entries()).unwrap();
    write_gguf_with_metadata(variant, &metadata(), &variant_entries(kind)).unwrap();
}

/// A directory of its own holding the model's file and a variant file of `kind`.
///
/// Returns both paths.
/// The caller removes their directory.
fn pair_dir(tag: &str, kind: Variant) -> (PathBuf, PathBuf) {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-wide-variants-{tag}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (model, variant) = (dir.join("model.gguf"), dir.join("variant.gguf"));
    write_pair(&model, &variant, kind);
    (model, variant)
}

/// The refusal the V4.1 arm gave a variant file, for the model named `model_id`.
pub(crate) fn v41_refusal(model: &Path, variant: &Path, model_id: &str) -> Option<String> {
    let primary = SplitGguf::open(model).unwrap();
    let other = SplitGguf::open(variant).unwrap();
    check_variant(&primary, &other, &|name| name.contains("engram_embd"))
        .err()
        .map(|e| format!("{} cannot serve {model_id}: {e}", variant.display()))
}

/// The V4.1 model of the 2 files under `choice`, by the steps of its engine arm, and its plan.
pub(crate) fn v41_with_variant(
    model: &Path,
    variant: &Path,
    choice: VariantChoice,
) -> (DeepseekV41Model, VariantPlan) {
    assert_eq!(v41_refusal(model, variant, "toy"), None);
    let primary = SplitGguf::open(model).unwrap();
    let variants = [SplitGguf::open(variant).unwrap()];
    let cfg = DeepseekV41Config::from_meta(&primary).unwrap();
    let v41 = DeepseekV41Model::load(
        &VariantSource {
            primary: &primary,
            variants: &variants,
        },
        &cfg,
    )
    .unwrap();
    v41.set_variant_policy(choice, None);
    v41.size_expert_cache_to_memory(&cfg).unwrap();
    // No card took experts: the placement plans with none.
    let plan = v41.place_variants(0).expect("2 variants");
    (v41, plan)
}

/// The generic model of the 2 files under `choice`, by the steps of the generic engine arm, and
/// its plan.
fn generic_with_variant(
    model: &Path,
    variant: &Path,
    choice: VariantChoice,
) -> (GenericHeteroTransformer, VariantPlan) {
    let files = open_checked(model, &[variant.to_path_buf()], "toy").unwrap();
    let mut generic = generic_of(open_mapped(model).unwrap());
    generic.locate_routing_prior(model);
    generic.attach_expert_variants(&files).unwrap();
    let plan = generic.place_expert_variants(choice).expect("2 variants");
    (generic, plan)
}

fn remove(model: &Path) {
    let _ = std::fs::remove_dir_all(model.parent().unwrap());
}

/// Under every policy, the generic route reads each expert from the file the V4.1 path reads it
/// from: the F32 stacks for the largest, the Q4_K file otherwise, with no card to keep experts.
#[test]
fn every_policy_picks_the_v41_variant_of_each_expert() {
    let (model, variant) = pair_dir("pick", Variant::Q4);
    for choice in CHOICES {
        let (_, want) = v41_with_variant(&model, &variant, choice);
        let (generic, got) = generic_with_variant(&model, &variant, choice);
        assert_eq!(got.chosen, want.chosen, "{choice:?}");
        assert_eq!(generic.expert_variant_choice(), want.chosen, "{choice:?}");
        let expected = match choice {
            VariantChoice::Largest => 0,
            _ => 1,
        };
        assert!(
            want.chosen.iter().flatten().all(|&v| v == expected),
            "{choice:?}: the V4.1 path read variant {expected}"
        );
        assert_eq!(generic.expert_variants(), 2);
    }
    remove(&model);
}

/// Under every policy, the generic route gives the bits of the V4.1 path on every record of the
/// oracle: the 2 read the same variant bytes.
/// The largest variant gives the bits of the model's own file, the smallest those of the Q4_K
/// file alone, and the 2 differ.
#[test]
fn every_policy_gives_the_bits_of_the_v41_path() {
    let (model, variant) = pair_dir("bits", Variant::Q4);
    let tokens = prompt();
    let mut by_choice = Vec::new();
    for choice in CHOICES {
        let (v41, _) = v41_with_variant(&model, &variant, choice);
        let (mut generic, _) = generic_with_variant(&model, &variant, choice);
        let (mut want, mut got) = (BTreeMap::new(), BTreeMap::new());
        record_v41(&v41, &tokens, 0, &mut want);
        record_generic(&mut generic, &tokens, 0, &mut got);
        for record in RECORDS {
            let name = format!("l0_{record}");
            assert_eq!(
                bit_difference(&want[&name], &got[&name]),
                None,
                "{choice:?} {name}"
            );
        }
        by_choice.push(got);
    }
    let alone = |file: &Path| {
        let mut out = BTreeMap::new();
        record_generic(
            &mut generic_of(open_mapped(file).unwrap()),
            &tokens,
            0,
            &mut out,
        );
        out
    };
    let (own, q4) = (alone(&model), alone(&variant));
    for record in RECORDS {
        let name = format!("l0_{record}");
        assert_eq!(
            bit_difference(&own[&name], &by_choice[0][&name]),
            None,
            "{name}"
        );
        assert_eq!(
            bit_difference(&q4[&name], &by_choice[1][&name]),
            None,
            "{name}"
        );
    }
    assert!(
        bit_difference(&own["l0_prefill"], &q4["l0_prefill"]).is_some(),
        "the 2 files compute the same experts"
    );
    remove(&model);
}

/// A variant file of another checkpoint, or with an expert fewer, is refused with the words of
/// the V4.1 arm, and a model of 1 file has no variants to place.
#[test]
fn a_broken_variant_is_refused_as_the_v41_arm_refused_it() {
    for (kind, why) in [
        (Variant::OtherNorm, "not of one checkpoint"),
        (
            Variant::FewerExperts,
            "in the first file and None in the other",
        ),
    ] {
        let (model, variant) = pair_dir("broken", kind);
        let want = v41_refusal(&model, &variant, "toy").expect("the V4.1 arm refused");
        assert!(want.contains(why), "{want}");
        let got = open_checked(&model, std::slice::from_ref(&variant), "toy")
            .err()
            .expect("the generic route refused")
            .to_string();
        assert_eq!(got, want);
        remove(&model);
    }
    let (model, _) = pair_dir("single", Variant::Q4);
    let mut generic = generic_of(open_mapped(&model).unwrap());
    assert_eq!(generic.expert_variants(), 1);
    assert!(generic.place_expert_variants(VariantChoice::Plan).is_none());
    remove(&model);
}
