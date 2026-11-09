//! The card tier of the routed experts of a wide-residual model, against the records of the
//! V4.1 path.
//!
//! The small files are those of the frozen V4.1 oracle.
//! The card tests store the routed experts in a format the card kernels read: gate and up in
//! `IQ2_XXS`, down in `Q2_K`, 1 tensor per expert.

use super::block_draft_tests::{generic_of, prompt};
use super::expert_profile::ExpertProfile;
use super::wide_sparse_frozen as frozen;
use super::wide_sparse_oracle_tests::{record_digest, record_generic, RECORDS};
use super::wide_sparse_variant_tests::{metadata, write_pair, Variant, CHOICES, LAYOUT};
use super::wide_sparse_variants::open_checked;
use super::GenericHeteroTransformer;
use crate::inference::generic_transformer::wide_sparse_toy::real_entries_for;
use crate::inference::place::expert_variants::VariantChoice;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, Value};
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
/// gave it, and keeps as many experts per layer, under the same heat.
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
        let all = (layers * experts) as u64;
        let rooms = [
            0,
            big,
            3 * big + small,
            5 * big - 1,
            all * big,
            all * big + small,
        ];
        for (k, room) in rooms.into_iter().enumerate() {
            generic.set_wide_card_rooms(vec![(0, room)]);
            let got = generic.place_expert_variants(choice).expect("2 variants");
            let key = |part: &str| format!("{choice:?} room{k} {part}");
            frozen::expect_debug("card_tier", &key("chosen"), &got.chosen);
            let card = got.kept.first().cloned().unwrap_or(vec![0; layers]);
            frozen::expect_debug("card_tier", &key("kept"), &card);
            assert_eq!(generic.expert_variant_choice(), got.chosen);
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
                record_digest(&got[&name]),
                record_digest(&want[&name]),
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
/// inputs equal those of the V4.1 backend, whose card warmed the same experts from the prior.
#[cfg(feature = "gpu")]
#[test]
fn the_card_residency_and_the_planner_inputs_are_the_v41_backends() {
    use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("residency");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let generic = GenericBackend(with_card_experts(&path, 64));
    let got = (
        generic.card_residency(),
        generic.expert_cost_shape(),
        generic.placement_model_inputs(),
    );
    println!("generic {got:?}");
    assert!(
        got.0.as_ref().is_some_and(|c| c[0].1 > 0),
        "experts on the card"
    );
    frozen::expect_debug("card_tier", "residency", &got);
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

/// A lent room's need follows the tokens up to the reservation, and its shortfall is what the
/// need takes beyond the part the experts do not hold.
#[test]
fn a_lent_room_falls_short_by_what_the_tokens_need_beyond_the_unlent_part() {
    use super::wide_sparse_card_tier::LentRoom;
    let room = LentRoom {
        ordinal: 0,
        layers: 4,
        reserved: 1000,
        lent: 0,
        work: 0,
    };
    let growth = room.need(1) / 4;
    assert!(growth >= 1);
    assert_eq!(room.need(10), 40 * growth);
    assert_eq!(room.need(1000), 1000, "the need stops at the reservation");
    assert_eq!(
        room.shortfall(1000),
        0,
        "nothing lent, nothing to take back"
    );
    let lent = LentRoom { lent: 900, ..room };
    assert_eq!(lent.shortfall(10), (40 * growth).saturating_sub(100));
    assert_eq!(lent.shortfall(1000), 900);
    assert_eq!(lent.shortfall(0), 0);
}

/// A card that lends its cache room to routed experts drops its coldest experts when a forward
/// needs the room, from the layers that keep the most, the fewest that cover the shortfall.
/// The model then gives the host's bits on every record, and a forward within the room drops
/// nothing more.
#[cfg(feature = "gpu")]
#[test]
fn a_card_takes_back_its_lent_room_from_the_coldest_experts_and_keeps_the_hosts_bits() {
    use super::wide_sparse_card_tier::LentRoom;
    use std::collections::BTreeMap;
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("lent");
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
    let mut model = with_card_experts(&path, 64);
    let per_card = |m: &GenericHeteroTransformer| -> Vec<usize> {
        m.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| b.ffn.card.count_on(0))
            .collect()
    };
    let before = per_card(&model);
    let held: usize = before.iter().sum();
    assert_eq!(held, model.wide_card_expert_count());
    let routed_bytes: u64 = model
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .flat_map(|b| b.ffn.card.bytes_by_card())
        .map(|(_, b)| b)
        .sum();
    let each = routed_bytes / held as u64;
    // A shortfall of 2 experts and a half drops 3 at a forward of `at` tokens.
    let at = 4096usize;
    let probe = LentRoom {
        ordinal: 0,
        layers: model.layers.len() as u64,
        reserved: u64::MAX / 4,
        lent: 0,
        work: 0,
    };
    let per_layer = model
        .wide_sparse
        .as_ref()
        .unwrap()
        .meta
        .kv_bytes_per_layer(at);
    let lent = probe.reserved - probe.need(per_layer) + 3 * each - each / 2;
    model.wide_sparse.as_mut().unwrap().lent = vec![LentRoom { lent, ..probe }];
    model.take_back_card_room(at);
    let after = per_card(&model);
    assert_eq!(
        after.iter().sum::<usize>(),
        held - 3,
        "{before:?} to {after:?}"
    );
    // The drops come from the layers that kept the most.
    let most = before.iter().copied().max().unwrap();
    for (l, (&b, &a)) in before.iter().zip(&after).enumerate() {
        assert!(
            a == b || b + 1 >= most,
            "layer {l}: {b} to {a}, the fullest held {most}"
        );
    }
    // Each layer keeps a prefix of its heat.
    for (l, ids) in model.held_heat().iter().enumerate() {
        let card = &model.layers[l].sinkhorn.as_ref().unwrap().ffn.card;
        let n = after[l];
        assert!(ids[..n].iter().all(|&e| card.holds(e)), "layer {l}");
        assert!(ids[n..].iter().all(|&e| !card.holds(e)), "layer {l}");
    }
    let left = model.wide_sparse.as_ref().unwrap().lent[0].lent;
    assert!(
        left < lent && left + 3 * each >= lent,
        "{lent} lent, {left} left"
    );
    let mut got = BTreeMap::new();
    record_generic(&mut model, &tokens, 0, &mut got);
    for record in RECORDS {
        let name = format!("l0_{record}");
        assert_eq!(
            record_digest(&got[&name]),
            record_digest(&want[&name]),
            "{name}"
        );
    }
    assert_eq!(
        model.wide_card_expert_count(),
        held - 3,
        "a forward within the room drops nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The cache's take-back ranks the weights every token reads with the routed experts by the
/// load's saving per byte: the cheapest weight goes first, then the coldest experts, and a
/// weight that saves more than every expert stays.
/// The model then gives the host's bits on every record.
#[cfg(feature = "gpu")]
#[test]
fn a_take_back_drops_weights_and_experts_by_their_saving_per_byte() {
    use super::wide_always_read::{AlwaysRead, CardSaving};
    use super::wide_sparse_card_tier::LentRoom;
    use std::collections::BTreeMap;
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("lent_weights");
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
    let mut model = with_card_experts(&path, 64);
    let held = model.wide_card_expert_count();
    let routed_bytes: u64 = model
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .flat_map(|b| b.ffn.card.bytes_by_card())
        .map(|(_, b)| b)
        .sum();
    let each = routed_bytes / held as u64;
    let layers = model.layers.len();
    let choices = super::card_packed::Choices::default();
    for il in 0..layers {
        model
            .keep_always_read(0, AlwaysRead::Router, il, &choices)
            .unwrap();
    }
    let head = model
        .keep_always_read(0, AlwaysRead::Head, layers - 1, &choices)
        .unwrap();
    model.wide_sparse.as_mut().unwrap().savings = vec![CardSaving {
        ordinal: 0,
        kinds: vec![(AlwaysRead::Head, 0.0), (AlwaysRead::Router, 1.0)],
        per_expert: 1e-3,
    }];
    // A shortfall of the head and 1 expert and a half drops the head and 2 experts.
    let at = 4096usize;
    let probe = LentRoom {
        ordinal: 0,
        layers: layers as u64,
        reserved: u64::MAX / 4,
        lent: 0,
        work: 0,
    };
    let per_layer = model
        .wide_sparse
        .as_ref()
        .unwrap()
        .meta
        .kv_bytes_per_layer(at);
    let lent = probe.reserved - probe.need(per_layer) + head + 2 * each - each / 2;
    model.wide_sparse.as_mut().unwrap().lent = vec![LentRoom { lent, ..probe }];
    model.take_back_card_room(at);
    let left = model.always_read_on(0);
    assert_eq!(model.wide_card_expert_count(), held - 2);
    assert!(left.iter().all(|(k, _)| *k == AlwaysRead::Router));
    assert_eq!(left.len(), layers, "every router stays");
    let mut got = BTreeMap::new();
    record_generic(&mut model, &tokens, 0, &mut got);
    for record in RECORDS {
        let name = format!("l0_{record}");
        assert_eq!(
            record_digest(&got[&name]),
            record_digest(&want[&name]),
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A prompt that finds the card short of free memory for the request's work gives back the
/// lent room first: the card drops its coldest experts, at most what it borrows, and the model
/// keeps the host's bits.
#[cfg(feature = "gpu")]
#[test]
fn a_prompt_short_of_card_memory_gives_the_lent_room_back_first() {
    use super::wide_sparse_card_tier::LentRoom;
    use std::collections::BTreeMap;
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let dir = scratch("lent_prompt");
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
    let mut model = with_card_experts(&path, 64);
    let held = model.wide_card_expert_count();
    let routed_bytes: u64 = model
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .flat_map(|b| b.ffn.card.bytes_by_card())
        .map(|(_, b)| b)
        .sum();
    let each = routed_bytes / held as u64;
    // A work larger than any card leaves every lent byte to give back: 3 experts' worth.
    let lent = 3 * each - each / 2;
    model.wide_sparse.as_mut().unwrap().lent = vec![LentRoom {
        ordinal: 0,
        layers: model.layers.len() as u64,
        reserved: u64::MAX / 4,
        lent,
        work: u64::MAX / 8,
    }];
    let mut got = BTreeMap::new();
    record_generic(&mut model, &tokens, 0, &mut got);
    assert_eq!(model.wide_card_expert_count(), held - 3);
    assert_eq!(model.wide_sparse.as_ref().unwrap().lent[0].lent, 0);
    for record in RECORDS {
        let name = format!("l0_{record}");
        assert_eq!(
            record_digest(&got[&name]),
            record_digest(&want[&name]),
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Whether the machine shows a second card.
#[cfg(feature = "gpu")]
fn second_card() -> bool {
    crate::tensor::Device::new_cuda(1).is_ok()
}

/// Experts card `ordinal` keeps over every layer of `model`.
#[cfg(feature = "gpu")]
fn kept_on(model: &GenericHeteroTransformer, ordinal: usize) -> usize {
    model
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .map(|b| b.ffn.card.count_on(ordinal))
        .sum()
}

/// The generic model of the card-format file with the cards keeping the experts `rooms` give,
/// each a card ordinal and a room in experts.
#[cfg(feature = "gpu")]
fn with_rooms(path: &Path, rooms: &[(usize, u64)]) -> GenericHeteroTransformer {
    let mut model = generic_of(open_mapped(path).unwrap());
    let (_, bytes) = expert_facts(&model, 0);
    model.set_wide_card_rooms(rooms.iter().map(|&(o, n)| (o, n * bytes)).collect());
    let plan = model
        .plan_wide_experts(VariantChoice::Plan)
        .expect("a wide model");
    model.keep_wide_card_experts(&plan);
    model
}

/// The routed feed-forward of layer 0 of `model` over 3 rows: the bits of its outputs.
#[cfg(feature = "gpu")]
fn first_layer_bits(model: &GenericHeteroTransformer) -> Vec<u32> {
    let block = model.layers[0].sinkhorn.as_ref().unwrap();
    let dim = block.ffn_norm.elem_count();
    let rows = 3;
    let x: Vec<f32> = (0..rows * dim)
        .map(|i| ((i * 13 + 5) as f32 * 0.21).sin())
        .collect();
    let x = crate::tensor::Tensor::from_vec(x, (rows, dim), &crate::tensor::Device::Cpu).unwrap();
    let y = block.ffn.forward(&x).unwrap();
    y.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A layer's routed experts give the same bits on the host, on the first card, on the second
/// card and over both: its feed-forward over 3 rows and every record of the oracle, the
/// prefill, the decode, the verifies and the greedy tokens.
#[cfg(feature = "gpu")]
#[test]
fn the_experts_give_the_same_bits_on_the_host_and_on_either_card() {
    use std::collections::BTreeMap;
    if !second_card() {
        eprintln!("no card 1: skipped");
        return;
    }
    let dir = scratch("two-cards");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let tokens = prompt();
    let mut host = generic_of(open_mapped(&path).unwrap());
    let layer = first_layer_bits(&host);
    let mut want = BTreeMap::new();
    record_generic(&mut host, &tokens, 0, &mut want);
    for rooms in [vec![(0, 64)], vec![(1, 64)], vec![(0, 3), (1, 64)]] {
        let mut model = with_rooms(&path, &rooms);
        for &(ordinal, _) in &rooms {
            assert!(kept_on(&model, ordinal) > 0, "{rooms:?}: card {ordinal}");
        }
        assert_eq!(first_layer_bits(&model), layer, "{rooms:?}: layer 0");
        let mut got = BTreeMap::new();
        record_generic(&mut model, &tokens, 0, &mut got);
        for record in RECORDS {
            let name = format!("l0_{record}");
            assert_eq!(
                record_digest(&got[&name]),
                record_digest(&want[&name]),
                "{rooms:?}: {name}"
            );
        }
        assert!(pairs_run(&model) > 0, "{rooms:?}: the cards ran no expert");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A card the load was offered outside the plan gets a room for routed experts and lends
/// nothing: only the plan's card keeps a cache.
#[cfg(feature = "gpu")]
#[test]
fn a_card_outside_the_plan_gets_a_room_and_lends_nothing() {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
    if !second_card() {
        eprintln!("no card 1: skipped");
        return;
    }
    let dir = scratch("outside");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let mut model = generic_of(open_mapped(&path).unwrap());
    let layers = model.layers.len();
    let plan = HeteroPlan {
        segments: vec![HeteroSegment {
            kind: DeviceKind::Cuda(0),
            layer_start: 0,
            layer_end: layers,
            free_memory_bytes: 1 << 30,
        }],
        tp_cards: None,
        total_layers: layers,
        host_experts_from: None,
    };
    let cache = model
        .wide_sparse
        .as_ref()
        .unwrap()
        .meta
        .kv_bytes_per_layer(8 * crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
    model.note_wide_card_rooms(&plan, &[0, 1], cache, 0);
    let ordinals: Vec<usize> = model.wide_card_rooms().iter().map(|r| r.0).collect();
    assert_eq!(ordinals, vec![0, 1]);
    let lent = |model: &GenericHeteroTransformer| -> Vec<usize> {
        let ws = model.wide_sparse.as_ref().unwrap();
        ws.lent.iter().map(|l| l.ordinal).collect()
    };
    assert_eq!(lent(&model), vec![0]);
    model.lend_card_room();
    assert_eq!(lent(&model), vec![0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The load measures the second card as it measures the first, and each card keeps experts
/// exactly when it was measured faster than the host.
#[cfg(feature = "gpu")]
#[test]
fn the_load_measures_each_card_and_keeps_experts_where_it_pays() {
    if !second_card() {
        eprintln!("no card 1: skipped");
        return;
    }
    let dir = scratch("measure-two");
    let path = dir.join("model.gguf");
    card_format_file(&path);
    let mut model = generic_of(open_mapped(&path).unwrap());
    for o in [0, 1] {
        let m = model
            .measure_wide_card(o, VariantChoice::Plan)
            .expect("the card gives the host's bits");
        println!("card {o}: {m:?}");
    }
    let (_, bytes) = expert_facts(&model, 0);
    model.set_wide_card_rooms(vec![(0, 3 * bytes), (1, 64 * bytes)]);
    model.place_wide_experts(VariantChoice::Plan);
    let rooms: Vec<usize> = model.wide_card_rooms().iter().map(|r| r.0).collect();
    println!("rooms {rooms:?}");
    for o in [0, 1] {
        assert_eq!(kept_on(&model, o) > 0, rooms.contains(&o), "card {o}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
