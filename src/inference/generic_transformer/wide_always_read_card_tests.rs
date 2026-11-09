//! The weights every token reads, on the card that runs the token mixers of a small V4.1 file.
//!
//! The file stores the output projections, the shared experts and the head in Q6_K, so the card
//! keeps them in their packed form and their q8_K route takes the host's steps.

use super::card_packed::route::{self, Route};
use super::wide_always_read::AlwaysRead;
use super::wide_expert_card::gpu::RouterCard;
use super::*;
use crate::inference::generic_transformer::wide_sparse_toy::{
    prompt, real_entries_for, real_md_layout,
};
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::open_mapped;
use crate::tensor::quantized::GgmlDType;

/// The band layouts of the oracle: window layers around sources and readers.
const LAYOUT: (&[u32], &[u32], &[u32]) = super::wide_sparse_oracle_tests::LAYOUTS[1];

pub(super) fn card_present() -> bool {
    Device::new_cuda(0).is_ok()
}

/// Whether `name` is a weight every token reads that a card keeps in its packed form.
fn packed(name: &str) -> bool {
    name.ends_with("attn_output_b") || name.ends_with("_shexp") || name == "output"
}

/// The small file of `LAYOUT`, its packed weights in Q6_K, every other weight F32.
///
/// Every tensor but the norms holds small values of its own.
fn q6_file() -> crate::tensor::quantized::gguf_file::MappedGguf {
    let (ratios, kv_src, index_src) = LAYOUT;
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let entries: Vec<GgufEntry> = real_entries_for(&bands)
        .into_iter()
        .enumerate()
        .map(|(t, entry)| {
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
            match packed(&entry.name) {
                true => GgufEntry {
                    dtype: GgmlDType::Q6K,
                    data: crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q6K, &values)
                        .unwrap(),
                    ..entry
                },
                false => GgufEntry {
                    data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ..entry
                },
            }
        })
        .collect();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "loken-always-read-{}-{seq}.gguf",
        std::process::id()
    ));
    write_gguf_with_metadata(&path, &real_md_layout(ratios, kv_src, index_src), &entries).unwrap();
    let g = open_mapped(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    g
}

/// The model of `q6_file` with every token mixer on card 0.
pub(super) fn on_card() -> GenericHeteroTransformer {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
    let layers = LAYOUT.0.len();
    let crate::tensor::quantized::gguf_file::MappedGguf { content, mmap, .. } = q6_file();
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

/// The logits of the prompt, then of each greedy step after it, every value's bits.
///
/// With `rows_at_once` the prompt is 1 forward of all its rows, else 1 forward a token.
fn greedy_bits(
    model: &mut GenericHeteroTransformer,
    steps: usize,
    rows_at_once: bool,
) -> Vec<Vec<u32>> {
    let tokens = prompt();
    let ids = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let row = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut last = Vec::new();
    match rows_at_once {
        true => last = row(&model.forward(&ids(&tokens), 0).unwrap()),
        false => {
            for (i, &t) in tokens.iter().enumerate() {
                last = row(&model.forward(&ids(&[t]), i).unwrap());
                out.push(last.iter().map(|v| v.to_bits()).collect());
            }
        }
    }
    out.push(last.iter().map(|v| v.to_bits()).collect());
    for i in 0..steps {
        let next = (0..last.len())
            .max_by(|&a, &b| last[a].total_cmp(&last[b]))
            .unwrap() as u32;
        last = row(&model.forward(&ids(&[next]), tokens.len() + i).unwrap());
        out.push(last.iter().map(|v| v.to_bits()).collect());
    }
    out
}

/// Keeps every weight every token reads of `model` on card 0.
fn keep_all(model: &mut GenericHeteroTransformer) {
    let choices = super::card_packed::Choices::default();
    let layers = model.layers.len();
    for il in 0..layers {
        for kind in [AlwaysRead::Output, AlwaysRead::Shared, AlwaysRead::Router] {
            model.keep_always_read(0, kind, il, &choices).unwrap();
        }
    }
    model
        .keep_always_read(0, AlwaysRead::Head, layers - 1, &choices)
        .unwrap();
}

/// The card's router gives the host's bits for a single row, whatever the input width.
#[test]
fn the_card_router_gives_the_hosts_bits() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    for (out, inp) in [(37usize, 1000usize), (384, 5120), (8, 31)] {
        let values: Vec<f32> = (0..out * inp)
            .map(|i| ((i as f32 * 0.731).sin() * (1.0 + (i % 17) as f32)) * 0.01)
            .collect();
        let router = Projection::Dense(Tensor::from_vec(values, (out, inp), &Device::Cpu).unwrap());
        let card = RouterCard::new(0, &router).unwrap();
        for seed in 0..4 {
            let x: Vec<f32> = (0..inp)
                .map(|i| ((i * 13 + seed * 7) as f32 * 0.29).cos() * (1 + seed) as f32)
                .collect();
            let xt = Tensor::from_vec(x.clone(), (1, inp), &Device::Cpu).unwrap();
            let host: Vec<u32> = router
                .apply_here(&xt)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let got: Vec<u32> = card
                .logits(&x)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(got, host, "[{out}, {inp}] row {seed}");
        }
    }
}

/// The greedy tokens of `q6_file` with its weights every token reads on the host and on the
/// card, by the q8_K route, and how far apart their logits are.
///
/// The q8_K route quantises the rows as the host does and folds the same integer sums.
/// Its float steps may still differ from the host's kernel in the last bits, so the logits are
/// compared by their greedy token, and their distance is printed.
fn same_greedy(rows_at_once: bool) {
    route::set(Route::HostSteps);
    let mut host_products = on_card();
    let want = greedy_bits(&mut host_products, 4, rows_at_once);
    let mut card = on_card();
    keep_all(&mut card);
    assert!(card.layers.iter().all(|l| l.mixer.projects_output()));
    let got = greedy_bits(&mut card, 4, rows_at_once);
    route::clear();
    let argmax = |b: &[u32]| {
        (0..b.len())
            .max_by(|&x, &y| f32::from_bits(b[x]).total_cmp(&f32::from_bits(b[y])))
            .unwrap()
    };
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        let ulps = g
            .iter()
            .zip(w)
            .map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs())
            .max()
            .unwrap_or(0);
        println!("logits {k}: at most {ulps} units in the last place apart");
        assert_eq!(argmax(g), argmax(w), "logits {k}");
    }
}

/// A prompt fed 1 token at a time over the card weights chooses the greedy tokens the host's
/// products choose.
#[test]
fn a_prompt_row_by_row_over_the_card_weights_keeps_the_greedy_tokens() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    same_greedy(false);
}

/// A prompt of several rows over the card weights chooses the greedy tokens the host's products
/// choose.
#[test]
fn a_prompt_over_the_card_weights_keeps_the_greedy_tokens() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    same_greedy(true);
}

/// The batched decode of 2 sequences over the card weights gives each sequence the bits of its
/// serial decode.
#[test]
fn a_batched_decode_over_the_card_weights_gives_each_sequence_its_serial_bits() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let mut model = on_card();
    keep_all(&mut model);
    let tokens = prompt();
    let (a, b) = (&tokens[..9], &tokens[3..15]);
    let serial = |model: &mut GenericHeteroTransformer, p: &[u32], next: u32| -> Vec<u32> {
        let ids = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
        model.forward(&ids(p), 0).unwrap();
        let row = model.forward(&ids(&[next]), p.len()).unwrap();
        row.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let want_a = serial(&mut model, a, 5);
    let want_b = serial(&mut model, b, 7);
    let mut sa = model.sequence_mixers().unwrap();
    let mut sb = model.sequence_mixers().unwrap();
    model.prefill_sequence(a, &mut sa).unwrap();
    model.prefill_sequence(b, &mut sb).unwrap();
    let logits = model
        .decode_sequences(&[5, 7], &[a.len(), b.len()], &mut [&mut sa, &mut sb])
        .unwrap();
    let rows = logits.to_vec2::<f32>().unwrap();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&rows[0]), want_a);
    assert_eq!(bits(&rows[1]), want_b);
}

/// The load's split keeps on the card only what it measured to save, takes it from the room,
/// and the card residency counts it.
#[test]
fn the_split_takes_from_the_room_what_the_card_keeps() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let mut model = on_card();
    let before: u64 = model
        .wide_card_residency()
        .unwrap_or_default()
        .iter()
        .map(|c| c.1)
        .sum();
    let room = 1u64 << 30;
    model.set_wide_card_rooms(vec![(0, room)]);
    model.place_always_read(crate::inference::place::expert_variants::VariantChoice::Plan);
    let after: u64 = model
        .wide_card_residency()
        .unwrap_or_default()
        .iter()
        .map(|c| c.1)
        .sum();
    let left = model.wide_card_rooms().first().map_or(0, |r| r.1);
    println!("residency {before} -> {after}, room {room} -> {left}");
    assert_eq!(after - before, room - left);
    for kind in [AlwaysRead::Output, AlwaysRead::Shared, AlwaysRead::Router] {
        let m = model.measure_always_read(0, kind, 0).unwrap();
        println!("{m:?}");
    }
}

/// Keeps on card 0 every weight every token reads that `keep_all` keeps, and every layer's
/// Sinkhorn mixers.
fn keep_all_and_mixers(model: &mut GenericHeteroTransformer) {
    keep_all(model);
    let choices = super::card_packed::Choices::default();
    for il in 0..model.layers.len() {
        model
            .keep_always_read(0, AlwaysRead::Sinkhorn, il, &choices)
            .unwrap();
    }
}

/// The greedy tokens of the prompt and `steps` steps after it.
///
/// With `release_at`, the cache takes back more room than the card holds before that step:
/// the card drops every weight every token reads.
fn greedy_tokens(
    model: &mut GenericHeteroTransformer,
    steps: usize,
    release_at: Option<usize>,
) -> Vec<u32> {
    use super::wide_sparse_card_tier::LentRoom;
    let tokens = prompt();
    let ids = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let argmax = |t: &Tensor| {
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32
    };
    let mut next = argmax(&model.forward(&ids(&tokens), 0).unwrap());
    let mut out = vec![next];
    for i in 0..steps {
        if release_at == Some(i) {
            let all = u64::MAX / 4;
            model.wide_sparse.as_mut().unwrap().lent = vec![LentRoom {
                ordinal: 0,
                layers: all,
                reserved: all,
                lent: all,
                work: 0,
            }];
        }
        next = argmax(&model.forward(&ids(&[next]), tokens.len() + i).unwrap());
        out.push(next);
    }
    out
}

/// A cache that takes its room back in the middle of a decode drops the weights every token
/// reads from the card, and the host's paths give the host's greedy tokens from then on.
#[test]
fn a_weight_the_cache_takes_back_in_a_decode_runs_on_the_host_from_then_on() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    route::set(Route::HostSteps);
    let mut host = on_card();
    let want = greedy_tokens(&mut host, 4, None);
    let mut card = on_card();
    keep_all_and_mixers(&mut card);
    let held = card.always_read_on(0);
    assert!(held.iter().any(|(k, _)| *k == AlwaysRead::Sinkhorn));
    let got = greedy_tokens(&mut card, 4, Some(2));
    route::clear();
    assert_eq!(got, want);
    assert!(card.always_read_on(0).is_empty(), "every weight dropped");
    assert!(card.layers.iter().all(|l| !l.mixer.projects_output()));
    assert_eq!(card.wide_sparse.as_ref().unwrap().lent[0].lent, 0);
}

/// A cache reserved for the working window lends nothing, and a decode drops no weight every
/// token reads from the card.
#[test]
fn nothing_is_lent_or_dropped_at_the_working_window() {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let mut model = on_card();
    keep_all_and_mixers(&mut model);
    let held = model.always_read_on(0);
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
    let per_layer = |model: &GenericHeteroTransformer, tokens: usize| {
        model
            .wide_sparse
            .as_ref()
            .unwrap()
            .meta
            .kv_bytes_per_layer(tokens)
    };
    let window = crate::inference::cache::KV_WORKING_WINDOW_TOKENS;
    let longer = per_layer(&model, 8 * window);
    model.note_wide_card_rooms(&plan, &[], longer, 0);
    model.lend_card_room();
    let lent = model.wide_sparse.as_ref().unwrap().lent[0].lent;
    assert!(lent > 0, "a longer window lends");
    let at_window = per_layer(&model, window);
    model.note_wide_card_rooms(&plan, &[], at_window, 0);
    model.lend_card_room();
    assert!(model
        .wide_sparse
        .as_ref()
        .unwrap()
        .lent
        .iter()
        .all(|l| l.lent == 0));
    greedy_tokens(&mut model, 4, None);
    assert_eq!(model.always_read_on(0), held);
}
