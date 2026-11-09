//! The card tier of the routed experts of a wide-residual model, against the V4.1 path.
//!
//! The small files are those of the frozen V4.1 oracle.
//! The card tests store the routed experts in a format the card kernels read: gate and up in
//! `IQ2_XXS`, down in `Q2_K`, 1 tensor per expert.

use super::block_draft_tests::{generic_of, prompt, v41_of};
use super::expert_profile::ExpertProfile;
use super::wide_sparse_oracle_tests::{bit_difference, record_generic, RECORDS};
use super::wide_sparse_variant_tests::{metadata, write_pair, Variant, CHOICES, LAYOUT};
use super::wide_sparse_variants::open_checked;
use super::GenericHeteroTransformer;
use crate::inference::generic_transformer::wide_sparse_toy::real_entries_for;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::source::VariantSource;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::place::expert_variants::{ranks_from_counts, VariantChoice};
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, Value};
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::quantized::GgmlDType;
use std::path::{Path, PathBuf};

/// A directory of its own for the files of 1 test.
///
/// The caller removes it.
fn scratch(tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-wide-card-tier-{tag}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Routing counts of `layers` layers of `experts` experts that rank each layer apart.
///
/// Expert `e` of layer `l` is counted `1 + (e * (2 * l + 3)) % experts` times.
fn counts(layers: usize, experts: usize) -> ExpertProfile {
    let counts = (0..layers)
        .flat_map(|l| (0..experts).map(move |e| 1 + ((e * (2 * l + 3)) % experts) as u32))
        .collect();
    ExpertProfile {
        layers,
        experts,
        counts,
    }
}

/// Per layer, a prior of `2 + l` experts that differs by layer.
fn prior(layers: usize, experts: usize) -> Vec<Vec<u32>> {
    (0..layers)
        .map(|l| {
            (0..2 + l)
                .map(|r| ((r * 3 + l * 5 + 1) % experts) as u32)
                .collect()
        })
        .collect()
}

/// The routed experts of layer `0` of `model`: their count and the bytes of variant `v` of
/// expert 0.
fn expert_facts(model: &GenericHeteroTransformer, v: usize) -> (usize, u64) {
    let ffn = &model.layers[0].sinkhorn.as_ref().unwrap().ffn;
    (ffn.experts.len(), ffn.variant_bytes(0, v).unwrap())
}

/// The V4.1 model of the 2 files, its policy `choice` over the heat `profile` ranks.
fn v41_ranked(
    model: &Path,
    variant: &Path,
    choice: VariantChoice,
    profile: &ExpertProfile,
) -> DeepseekV41Model {
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
    let ranks = ranks_from_counts(
        &profile.counts,
        profile.layers,
        profile.experts,
        &cfg.hot_experts,
    );
    v41.set_variant_policy(choice, ranks);
    v41.size_expert_cache_to_memory(&cfg).unwrap();
    v41
}

/// The generic model of the 2 files, its heat from `profile`.
fn generic_ranked(
    model: &Path,
    variant: &Path,
    profile: &ExpertProfile,
) -> GenericHeteroTransformer {
    let files = open_checked(model, &[variant.to_path_buf()], "toy").unwrap();
    let mut generic = generic_of(open_mapped(model).unwrap());
    generic.locate_routing_prior(model);
    generic.prime_routing_heat(profile).unwrap();
    generic.attach_expert_variants(&files).unwrap();
    generic
}

/// Under every policy and every room, a card gives each expert the variant the V4.1 placement
/// gives it, and keeps as many experts per layer, under the same heat.
#[test]
fn a_card_room_gives_each_expert_the_v41_variant() {
    let dir = scratch("variants");
    let (model, variant) = (dir.join("model.gguf"), dir.join("variant.gguf"));
    write_pair(&model, &variant, Variant::Q4);
    let layers = LAYOUT.0.len();
    for choice in CHOICES {
        let mut generic = generic_ranked(&model, &variant, &counts(layers, 8));
        let (experts, big) = expert_facts(&generic, 0);
        let (_, small) = expert_facts(&generic, 1);
        let profile = counts(layers, experts);
        let v41 = v41_ranked(&model, &variant, choice, &profile);
        let all = (layers * experts) as u64;
        for room in [
            0,
            big,
            3 * big + small,
            5 * big - 1,
            all * big,
            all * big + small,
        ] {
            let want = v41.place_variants(room).expect("2 variants");
            generic.set_wide_card_rooms(vec![(0, room)]);
            let got = generic.place_expert_variants(choice).expect("2 variants");
            assert_eq!(got.chosen, want.chosen, "{choice:?} room {room}");
            let card = got.kept.first().cloned().unwrap_or(vec![0; layers]);
            assert_eq!(card, want.kept[0], "{choice:?} room {room}");
            assert_eq!(generic.expert_variant_choice(), want.chosen);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Over 2 cards, the first card keeps the hottest experts rank by rank across the layers, the
/// second the next ones, and each layer's card experts are a prefix of its heat.
#[test]
fn the_cards_keep_the_hottest_experts_rank_by_rank() {
    let file = super::wide_sparse_oracle_tests::toy_file(LAYOUT.0, LAYOUT.1, LAYOUT.2);
    let mut generic = generic_of(file);
    let layers = LAYOUT.0.len();
    let (experts, bytes) = expert_facts(&generic, 0);
    generic
        .prime_routing_heat(&counts(layers, experts))
        .unwrap();
    let caps = [4usize, 3];
    generic.set_wide_card_rooms(vec![
        (0, caps[0] as u64 * bytes),
        (1, caps[1] as u64 * bytes),
    ]);
    let plan = generic
        .plan_wide_experts(VariantChoice::Plan)
        .expect("a wide model");
    let heat = generic.held_heat();
    // Rank by rank across the layers, each card until it holds its cap.
    let mut want: Vec<Vec<(usize, Vec<usize>)>> = vec![Vec::new(); layers];
    let (mut card, mut held) = (0usize, 0usize);
    'fill: for rank in 0..experts {
        for (l, ids) in heat.iter().enumerate() {
            if held == caps[card] {
                card += 1;
                held = 0;
                if card == caps.len() {
                    break 'fill;
                }
            }
            match want[l].iter_mut().find(|(o, _)| *o == card) {
                Some((_, v)) => v.push(ids[rank]),
                None => want[l].push((card, vec![ids[rank]])),
            }
            held += 1;
        }
    }
    assert_eq!(generic.wide_card_experts_of(&plan), want);
    for (l, cards) in want.iter().enumerate() {
        let ids: Vec<usize> = cards.iter().flat_map(|(_, v)| v.clone()).collect();
        assert_eq!(ids, heat[l][..ids.len()].to_vec(), "layer {l}");
    }
}

/// A build without cards measures no card: the rooms are dropped and no expert is kept.
#[cfg(not(feature = "gpu"))]
#[test]
fn a_build_without_cards_keeps_no_routed_expert() {
    let file = super::wide_sparse_oracle_tests::toy_file(LAYOUT.0, LAYOUT.1, LAYOUT.2);
    let mut generic = generic_of(file);
    let (_, bytes) = expert_facts(&generic, 0);
    generic.set_wide_card_rooms(vec![(0, 64 * bytes)]);
    let plan = generic
        .place_wide_experts(VariantChoice::Plan)
        .expect("a wide model");
    assert!(generic.wide_card_rooms().is_empty());
    assert!(plan.kept.is_empty());
    assert_eq!(generic.wide_card_expert_count(), 0);
    assert_eq!(generic.wide_card_residency(), None);
}

/// The small file with its routed experts in a format the card kernels read, 1 tensor per
/// expert, and a prior in its metadata, written at `path`.
///
/// Every tensor but the norms holds small values of its own, so the routed experts weigh in
/// the logits.
#[cfg(feature = "gpu")]
fn card_format_file(path: &Path) {
    let (ratios, kv_src, index_src) = LAYOUT;
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut out = Vec::new();
    for (t, entry) in real_entries_for(&bands).into_iter().enumerate() {
        let n: usize = entry.dims.iter().product();
        let values: Vec<f32> = match entry.name.contains("norm") {
            true => entry
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            false => (0..n)
                .map(|i| (((i * 7 + t * 131) as f32) * 0.37).sin() * 0.05)
                .collect(),
        };
        let name = entry.name.clone();
        let dtype = if name.ends_with("ffn_gate_exps") || name.ends_with("ffn_up_exps") {
            GgmlDType::Iq2Xxs
        } else if name.ends_with("ffn_down_exps") {
            GgmlDType::Q2K
        } else {
            out.push(GgufEntry {
                data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ..entry
            });
            continue;
        };
        let (experts, rows, cols) = (entry.dims[0], entry.dims[1], entry.dims[2]);
        for e in 0..experts {
            let one = &values[e * rows * cols..(e + 1) * rows * cols];
            out.push(GgufEntry {
                name: format!("{}.{e}", entry.name),
                dims: vec![rows, cols],
                dtype,
                data: crate::tensor::quant_cpu::from_float_bytes(dtype, one).unwrap(),
            });
        }
    }
    let mut md = metadata();
    let layers = ratios.len();
    for (l, list) in prior(layers, 8).into_iter().enumerate() {
        md.push((
            format!("deepseek4.hot_experts.{l}"),
            Value::Array(list.into_iter().map(Value::U32).collect()),
        ));
    }
    write_gguf_with_metadata(path, &md, &out).unwrap();
}

/// The generic model of the card-format file with the cards keeping the experts a room of
/// `experts` experts gives, on card 0.
#[cfg(feature = "gpu")]
fn with_card_experts(path: &Path, experts: u64) -> GenericHeteroTransformer {
    let mut model = generic_of(open_mapped(path).unwrap());
    let (_, bytes) = expert_facts(&model, 0);
    model.set_wide_card_rooms(vec![(0, experts * bytes)]);
    let plan = model
        .plan_wide_experts(VariantChoice::Plan)
        .expect("a wide model");
    model.keep_wide_card_experts(&plan);
    model
}

/// The (expert, row) pairs the cards of `model` ran.
#[cfg(feature = "gpu")]
fn pairs_run(model: &GenericHeteroTransformer) -> u64 {
    model
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .map(|b| b.ffn.card.pairs_run())
        .sum()
}

/// With some and with all of its prior's experts on the card, the model gives the host's bits on
/// every record of the oracle: the prefill, the decode, the verifies and the greedy tokens.
#[cfg(feature = "gpu")]
#[test]
fn card_experts_give_the_hosts_bits_on_every_record() {
    use std::collections::BTreeMap;
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("records");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let tokens = prompt();
    let mut want = BTreeMap::new();
    record_generic(
        &mut generic_of(open_mapped(&path).unwrap()),
        &tokens,
        0,
        &mut want,
    );
    let prior_len: usize = prior(LAYOUT.0.len(), 8).iter().map(Vec::len).sum();
    for room in [3u64, 64] {
        let mut model = with_card_experts(&path, room);
        let kept = model.wide_card_expert_count();
        assert_eq!(
            kept,
            (room as usize).min(prior_len),
            "room of {room} experts"
        );
        let mut got = BTreeMap::new();
        record_generic(&mut model, &tokens, 0, &mut got);
        for record in RECORDS {
            let name = format!("l0_{record}");
            assert_eq!(
                bit_difference(&want[&name], &got[&name]),
                None,
                "room of {room} experts: {name}"
            );
        }
        assert!(pairs_run(&model) > 0, "the card ran no expert");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The relative L2 distance of `got` to `exact`.
#[cfg(feature = "gpu")]
fn relative_l2(exact: &[f64], got: &[f32]) -> f64 {
    let (mut dd, mut ww) = (0f64, 0f64);
    for (&w, &g) in exact.iter().zip(got) {
        dd += (g as f64 - w) * (g as f64 - w);
        ww += w * w;
    }
    (dd / ww.max(f64::MIN_POSITIVE)).sqrt()
}

/// `w` `[out, inp]` decoded to F32, times the F32 rows `x` `[n, inp]`, summed in F64.
#[cfg(feature = "gpu")]
fn exact_product(
    w: &crate::tensor::blockscaled::projection::Projection,
    x: &[f64],
    n: usize,
) -> Vec<f64> {
    use crate::tensor::blockscaled::projection::Projection;
    let Projection::Quant(q) = w else {
        panic!("a GGUF expert")
    };
    let dense: Vec<f32> = q
        .dequantize(&crate::tensor::Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let (out, inp) = (w.dims()[0], w.dims()[1]);
    (0..n)
        .flat_map(|t| {
            let (dense, x) = (&dense, &x[t * inp..(t + 1) * inp]);
            (0..out).map(move |o| {
                dense[o * inp..(o + 1) * inp]
                    .iter()
                    .zip(x)
                    .map(|(&a, &b)| a as f64 * b)
                    .sum()
            })
        })
        .collect()
}

/// The products of the experts the card keeps are the host's bits, and so no farther from the
/// exact product (blocks decoded to F32, rows in F32, sums in F64) than the host's, up to the
/// rounding of an F32 sum.
#[cfg(feature = "gpu")]
#[test]
fn card_expert_products_are_no_farther_from_the_exact_product_than_the_hosts() {
    use super::card_packed::no_farther;
    use crate::tensor::ops::bounded_swiglu::swiglu;
    use crate::tensor::{Device, Tensor};
    if Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("exact");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let model = with_card_experts(&path, 64);
    let ffn = &model.layers[0].sinkhorn.as_ref().unwrap().ffn;
    let ids: Vec<usize> = (0..ffn.experts.len())
        .filter(|&e| ffn.card.holds(e))
        .collect();
    assert!(!ids.is_empty(), "the card keeps experts of layer 0");
    let dim = ffn.experts[0].gate.dims()[1];
    let n = 5usize;
    // Rows wide enough for the bound to bite, 1 row of zeros.
    let mut x: Vec<f32> = (0..n * dim)
        .map(|i| ((i as f32 * 0.61).sin() + (i as f32 * 0.013).cos()) * 3.0)
        .collect();
    x[2 * dim..3 * dim].fill(0.0);
    let rows: Vec<Vec<usize>> = ids.iter().map(|_| (0..n).collect()).collect();
    let card = ffn
        .card
        .run(&ids, &rows, &x, dim, ffn.limit)
        .expect("the card runs its experts");
    let xt = Tensor::from_vec(x.clone(), (n, dim), &Device::Cpu).unwrap();
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    for (i, &e) in ids.iter().enumerate() {
        let expert = &ffn.experts[e];
        let host: Vec<f32> = expert
            .forward(&xt, ffn.limit)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(bits(&card[i]), bits(&host), "expert {e}");
        let inter = expert.gate.dims()[0];
        let gate = exact_product(&expert.gate, &x64, n);
        let up = exact_product(&expert.up, &x64, n);
        let h: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| swiglu(g as f32, u as f32, ffn.limit) as f64)
            .collect();
        let exact = exact_product(&expert.down, &h, n);
        let (host_l2, card_l2) = (relative_l2(&exact, &host), relative_l2(&exact, &card[i]));
        println!("expert {e}: host {host_l2:.3e}, card {card_l2:.3e} from the exact product");
        assert!(
            no_farther(host_l2, card_l2, inter.max(dim)),
            "expert {e}: card {card_l2} against host {host_l2}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// With every prior expert on the card, the card residency, the cost shape and the placement
/// inputs equal those of the V4.1 backend, whose card warms the same experts from the prior.
#[cfg(feature = "gpu")]
#[test]
fn the_card_residency_and_the_planner_inputs_are_the_v41_backends() {
    use crate::inference::engine::model_backend::{
        DeepseekV41Backend, GenericBackend, ModelBackend,
    };
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("residency");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let v41 = DeepseekV41Backend::new(v41_of(&open_mapped(&path).unwrap()), 0, 128);
    let want = (
        v41.card_residency(),
        v41.expert_cost_shape(),
        v41.placement_model_inputs(),
    );
    drop(v41);
    let generic = GenericBackend(with_card_experts(&path, 64));
    let got = (
        generic.card_residency(),
        generic.expert_cost_shape(),
        generic.placement_model_inputs(),
    );
    println!("V4.1 {want:?}");
    println!("generic {got:?}");
    assert!(
        got.0.as_ref().is_some_and(|c| c[0].1 > 0),
        "experts on the card"
    );
    assert_eq!(got, want);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The load measures the card on a decode's routed set: the card gives the host's bits, and the
/// tier keeps experts exactly when the card was measured faster.
#[cfg(feature = "gpu")]
#[test]
fn the_load_keeps_experts_on_a_card_only_when_it_measures_faster() {
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("measure");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let mut model = generic_of(open_mapped(&path).unwrap());
    let measured = model
        .measure_wide_card(0, VariantChoice::Plan)
        .expect("the card gives the host's bits");
    println!("{measured:?}");
    let (_, bytes) = expert_facts(&model, 0);
    model.set_wide_card_rooms(vec![(0, 64 * bytes)]);
    model.place_wide_experts(VariantChoice::Plan);
    let kept = model.wide_card_expert_count();
    println!("kept {kept}, rooms {:?}", model.wide_card_rooms());
    assert_eq!(kept > 0, !model.wide_card_rooms().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
