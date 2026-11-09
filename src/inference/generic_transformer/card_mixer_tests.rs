//! The latent attention and compressed band mixers on a card, against the same model on the host.
//!
//! The model is the synthetic one of the frozen V4.1 oracle, in its 2 band layouts.
//! Every layer's token mixer runs on the card in one run and on the host in the other.
//! The records are the oracle's: prefill, decode, verify, the rewound verify and decode, and the
//! greedy run.
//!
//! The host quantises the rows it multiplies and a card sums in another order, so neither is the
//! truth. The truth is the exact reference: the host model with every mixer projection read as
//! dense F32 values. A card passes when:
//! - the relative L2 distance of a row to the reference is no larger than the host's, up to the
//!   rounding of F32 (`no_farther`),
//! - it chooses the reference's greedy tokens.

use std::collections::BTreeMap;

use super::mixer::Mixer;
use super::wide_sparse_engram_tests::engram_file;
use super::wide_sparse_oracle_tests::{generic, record_generic, toy_file, Record, LAYOUTS};
use super::*;
use crate::inference::generic_transformer::wide_sparse_toy::prompt;
use crate::inference::place::layer_executor::HeteroSegment;
use crate::tensor::quantized::gguf_file::MappedGguf;

/// The bar the card measures its products by is the bar of these tests.
pub(super) use super::card_packed::no_farther;

/// The relative L2 distances of `host` and of `card` to `reference`.
pub(super) fn distances(reference: &[f32], host: &[f32], card: &[f32]) -> (f64, f64) {
    (closeness(reference, host).1, closeness(reference, card).1)
}

/// Every token mixer of `model` reads its projections as dense F32 values: the exact reference.
pub(super) fn dequantise_mixers(model: &mut GenericHeteroTransformer) {
    for layer in model.layers.iter_mut() {
        match &mut layer.mixer {
            Mixer::LatentMqa(m) => m.dequantise_projections().unwrap(),
            Mixer::CompressedBand(m) => m.dequantise_projections().unwrap(),
            _ => {}
        }
    }
}

/// The host model of the synthetic file with its mixers made exact.
pub(super) fn exact(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> GenericHeteroTransformer {
    let mut model = generic(ratios, kv_src, index_src);
    dequantise_mixers(&mut model);
    model
}

/// Whether card 0 opens.
fn card_present() -> bool {
    Device::new_cuda(0).is_ok()
}

/// The generic model of the synthetic file, every layer on card 0.
pub(crate) fn generic_on_card(
    ratios: &[u32],
    kv_src: &[u32],
    index_src: &[u32],
) -> GenericHeteroTransformer {
    on_card(toy_file(ratios, kv_src, index_src), ratios.len())
}

/// The generic model of the file `g` of `layers` layers, every layer on card 0.
fn on_card(g: MappedGguf, layers: usize) -> GenericHeteroTransformer {
    let MappedGguf { content, mmap, .. } = g;
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
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// The generic model of the synthetic file with engram layers, every layer on the host or on
/// card 0.
fn engram_model(
    ratios: &[u32],
    kv_src: &[u32],
    index_src: &[u32],
    card: bool,
) -> GenericHeteroTransformer {
    let g = engram_file(ratios, kv_src, index_src);
    if card {
        return on_card(g, ratios.len());
    }
    let MappedGguf { content, mmap, .. } = g;
    let plan = HeteroPlan::calculate_with_kv(ratios.len(), 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// The cosine and the relative L2 distance of `got` to `want`.
pub(super) fn closeness(want: &[f32], got: &[f32]) -> (f64, f64) {
    let (mut dot, mut ww, mut gg, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (&w, &g) in want.iter().zip(got) {
        let (w, g) = (w as f64, g as f64);
        dot += w * g;
        ww += w * w;
        gg += g * g;
        dd += (w - g) * (w - g);
    }
    (dot / (ww.sqrt() * gg.sqrt()), (dd / ww).sqrt())
}

/// The worst relative L2 distance of the host and of the card to the reference over the rows of
/// every record, and the records whose greedy tokens differ from the reference's or whose rows
/// the card holds farther from the reference than the host, up to the rounding of a sum of
/// `terms` terms.
fn compare(
    reference: &BTreeMap<String, Record>,
    host: &BTreeMap<String, Record>,
    card: &BTreeMap<String, Record>,
    terms: usize,
) -> (f64, f64, Vec<String>) {
    let (mut worst_host, mut worst_card) = (0f64, 0f64);
    let mut failed = Vec::new();
    for (name, w) in reference {
        let (Some(h), Some(c)) = (host.get(name), card.get(name)) else {
            failed.push(format!("{name}: missing"));
            continue;
        };
        match (w, h, c) {
            (Record::Ids(a), _, Record::Ids(b)) => {
                if a != b {
                    failed.push(format!("{name}: tokens {b:?} != {a:?}"));
                }
            }
            (Record::Logits(wr, wc, a), Record::Logits(_, _, hb), Record::Logits(gr, gc, b)) => {
                if (wr, wc) != (gr, gc) {
                    failed.push(format!("{name}: shape [{gr}, {gc}] != [{wr}, {wc}]"));
                    continue;
                }
                let rows = a
                    .chunks_exact(*wc)
                    .zip(hb.chunks_exact(*wc))
                    .zip(b.chunks_exact(*wc));
                for (row, ((x, y), z)) in rows.enumerate() {
                    let (host_l2, card_l2) = distances(x, y, z);
                    worst_host = worst_host.max(host_l2);
                    worst_card = worst_card.max(card_l2);
                    if card_l2.is_nan() || !no_farther(host_l2, card_l2, terms) {
                        failed.push(format!(
                            "{name} row {row}: card {card_l2} from the reference, host {host_l2}"
                        ));
                    }
                }
            }
            _ => failed.push(format!("{name}: kind changed")),
        }
    }
    (worst_host, worst_card, failed)
}

/// Every layer's token mixer holds weights on the card, and none is left on the host.
#[test]
fn every_token_mixer_of_a_card_plan_holds_weights_on_the_card() {
    if !card_present() {
        return;
    }
    for (ratios, kv_src, index_src) in LAYOUTS {
        let model = generic_on_card(ratios, kv_src, index_src);
        for (il, layer) in model.layers.iter().enumerate() {
            let bytes = match &layer.mixer {
                Mixer::LatentMqa(m) => m.card_bytes(),
                Mixer::CompressedBand(m) => m.card_bytes(),
                _ => 0,
            };
            assert!(bytes > 0, "layer {il} holds nothing on the card");
        }
        let host = generic(ratios, kv_src, index_src);
        for layer in &host.layers {
            let bytes = match &layer.mixer {
                Mixer::LatentMqa(m) => m.card_bytes(),
                Mixer::CompressedBand(m) => m.card_bytes(),
                _ => 1,
            };
            assert_eq!(bytes, 0, "a host plan put weights on a card");
        }
    }
}

/// The model with its token mixers on the card gives records no farther from the exact
/// reference's than the host model's, and the reference's greedy tokens.
#[test]
fn the_card_mixers_stay_as_close_to_the_exact_reference_as_the_host() {
    if !card_present() {
        return;
    }
    let (mut worst_host, mut worst_card) = (0f64, 0f64);
    let mut failed = Vec::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let tokens = prompt();
        let (mut truth, mut want, mut got) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        let mut reference = exact(ratios, kv_src, index_src);
        let terms = reference.config.embedding_length;
        record_generic(&mut reference, &tokens, n, &mut truth);
        let mut host = generic(ratios, kv_src, index_src);
        record_generic(&mut host, &tokens, n, &mut want);
        let mut card = generic_on_card(ratios, kv_src, index_src);
        record_generic(&mut card, &tokens, n, &mut got);
        // The last records decode on the card: each ring stands for every token the layer saw.
        for (il, layer) in card.layers.iter().enumerate() {
            let (seen, ring) = match &layer.mixer {
                Mixer::LatentMqa(m) => (m.len(), m.card_ring_tokens()),
                Mixer::CompressedBand(m) => (m.len(), m.card_ring_tokens()),
                _ => (0, 0),
            };
            assert!(seen > 0, "layer {il} saw no token");
            assert_eq!(
                ring, seen,
                "layer {il}: the card ring does not stand for the state"
            );
        }
        let (host_l2, card_l2, bad) = compare(&truth, &want, &got, terms);
        eprintln!("layout {n}: from the reference, host {host_l2}, card {card_l2}");
        worst_host = worst_host.max(host_l2);
        worst_card = worst_card.max(card_l2);
        failed.extend(bad);
    }
    eprintln!("worst relative L2 from the reference: host {worst_host}, card {worst_card}");
    assert!(
        failed.is_empty(),
        "the card mixers are farther from the reference than the host:\n{}",
        failed.join("\n")
    );
}

/// The engram model with its token mixers on the card gives records no farther from the exact
/// reference's than the same model on the host, and the reference's greedy tokens.
/// The engram, the residual mixing and the feed-forward stay on the host.
#[test]
fn the_card_mixers_with_engram_layers_stay_as_close_to_the_exact_reference_as_the_host() {
    if !card_present() {
        return;
    }
    let mut failed = Vec::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let tokens = prompt();
        let (mut truth, mut want, mut got) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        let mut reference = engram_model(ratios, kv_src, index_src, false);
        dequantise_mixers(&mut reference);
        let terms = reference.config.embedding_length;
        record_generic(&mut reference, &tokens, n, &mut truth);
        let mut host = engram_model(ratios, kv_src, index_src, false);
        record_generic(&mut host, &tokens, n, &mut want);
        let mut card = engram_model(ratios, kv_src, index_src, true);
        record_generic(&mut card, &tokens, n, &mut got);
        let (host_l2, card_l2, bad) = compare(&truth, &want, &got, terms);
        eprintln!(
            "layout {n} with engram layers: from the reference, host {host_l2}, card {card_l2}"
        );
        failed.extend(bad);
    }
    assert!(
        failed.is_empty(),
        "the card mixers with engram layers are farther from the reference than the host:\n{}",
        failed.join("\n")
    );
}

/// Rows of a test input: deterministic values of order 1.
fn rows(seed: u64, n: usize, width: usize) -> Tensor {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let v: Vec<f32> = (0..n * width)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    Tensor::from_vec(v, (1, n, width), &Device::Cpu).unwrap()
}

/// Every layer's mixer output, before the layer's output projection, is no farther from the exact
/// reference's on the card than on the host, on the same rows: a prompt, a decode past the
/// window, and a continued prompt.
#[test]
fn the_mixer_outputs_of_the_card_stay_as_close_to_the_exact_ones_as_the_host() {
    if !card_present() {
        return;
    }
    let (mut worst_host, mut worst_card) = (0f64, 0f64);
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut reference = exact(ratios, kv_src, index_src);
        let mut host = generic(ratios, kv_src, index_src);
        let mut card = generic_on_card(ratios, kv_src, index_src);
        let hidden = host.config.embedding_length;
        // A prompt, 20 decoded rows (the window is shorter), then 5 rows continuing the state.
        let steps: Vec<(usize, usize)> = std::iter::once((0, 9))
            .chain((9..29).map(|p| (p, 1)))
            .chain(std::iter::once((29, 5)))
            .collect();
        for (step, &(pos, seq)) in steps.iter().enumerate() {
            for model in [&reference, &host, &card] {
                if let Some(head) = &model.wide_sparse {
                    head.cell.open(pos, seq);
                }
            }
            for il in 0..ratios.len() {
                let x = rows((step * 31 + il) as u64, seq, hidden);
                let truth = reference.layers[il].mixer.forward(&x, pos).unwrap();
                let want = host.layers[il].mixer.forward(&x, pos).unwrap();
                let got = card.layers[il].mixer.forward(&x, pos).unwrap();
                let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let (host_l2, card_l2) = distances(&flat(&truth), &flat(&want), &flat(&got));
                worst_host = worst_host.max(host_l2);
                worst_card = worst_card.max(card_l2);
                assert!(
                    no_farther(host_l2, card_l2, hidden),
                    "layer {il}, rows {pos}..{}: card {card_l2} from the reference, host {host_l2}",
                    pos + seq
                );
            }
        }
    }
    eprintln!(
        "mixer outputs, worst relative L2 from the reference: host {worst_host}, card {worst_card}"
    );
}

/// Mean milliseconds of one decode step through every layer's token mixer, on the card and on
/// the host, over the steps after the first few.
///
/// The step ends with the output on the host, so the host clock around it spans the card's work.
/// The figures are for comparing two builds on one machine.
#[test]
fn a_decode_step_through_the_token_mixers_is_timed() {
    if !card_present() {
        return;
    }
    const WARM: usize = 6;
    const STEPS: usize = 40;
    let mut total = (0f64, 0f64);
    let mut count = 0usize;
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut host = generic(ratios, kv_src, index_src);
        let mut card = generic_on_card(ratios, kv_src, index_src);
        let hidden = host.config.embedding_length;
        let layers = ratios.len();
        let mut spent = (0f64, 0f64);
        for (step, pos) in (0..STEPS).map(|s| (s, if s == 0 { 0 } else { 8 + s })) {
            let seq = if step == 0 { 9 } else { 1 };
            for model in [&host, &card] {
                if let Some(head) = &model.wide_sparse {
                    head.cell.open(pos, seq);
                }
            }
            let mut took = (0f64, 0f64);
            for il in 0..layers {
                let x = rows((step * 31 + il) as u64, seq, hidden);
                let t = std::time::Instant::now();
                host.layers[il].mixer.forward(&x, pos).unwrap();
                took.0 += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                card.layers[il].mixer.forward(&x, pos).unwrap();
                took.1 += t.elapsed().as_secs_f64();
            }
            if step >= WARM {
                spent.0 += took.0;
                spent.1 += took.1;
            }
        }
        let n = (STEPS - WARM) as f64;
        eprintln!(
            "layout {ratios:?}: {layers} layers, host {:.3} ms, card {:.3} ms per decode step",
            spent.0 / n * 1e3,
            spent.1 / n * 1e3
        );
        total.0 += spent.0 / n;
        total.1 += spent.1 / n;
        count += 1;
    }
    eprintln!(
        "decode step through the token mixers: host {:.3} ms, card {:.3} ms (mean over {count} layouts)",
        total.0 / count as f64 * 1e3,
        total.1 / count as f64 * 1e3
    );
}
