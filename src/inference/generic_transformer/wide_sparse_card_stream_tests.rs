//! A decoded row's residual on the card, on the small V4.1 file of the always-read tests.
//!
//! The card's sums are not the host's running sums.
//! A decoded row's distance is printed beside the bound the layers' admissions derive.
//! A greedy token is compared only where the host's margin exceeds the distance of the 2 rows.
//! The rows the host path gives, the prefills and a step of 2 rows, keep the host's bits.

use super::wide_always_read::AlwaysRead;
use super::wide_always_read_card_tests::{card_present, on_card};
use super::wide_sparse_layer::SequenceMixers;
use super::wide_sparse_toy::prompt;
use super::*;

/// Decoded steps of each sequence: past the toy's window of 32 after a prompt of 24.
const STEPS: usize = 12;

/// The model of the always-read tests with every layer's Sinkhorn mixers on card 0.
fn with_card_mixers() -> GenericHeteroTransformer {
    let mut model = on_card();
    let choices = super::card_packed::Choices::default();
    for il in 0..model.layers.len() {
        model
            .keep_always_read(0, AlwaysRead::Sinkhorn, il, &choices)
            .unwrap();
    }
    model
}

fn row(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32
}

/// The relative L2 distance of `got` to `want`.
fn apart(want: &[f32], got: &[f32]) -> f64 {
    let (mut dd, mut ww) = (0f64, 0f64);
    for (&w, &g) in want.iter().zip(got) {
        dd += (w as f64 - g as f64).powi(2);
        ww += (w as f64).powi(2);
    }
    (dd / ww).sqrt()
}

/// The logits of each step of a script over `model`, the tokens fed being `feed`'s.
///
/// The script:
/// - sequence A takes the whole prompt, sequence B a part of it,
/// - their single-row steps alternate, past the window,
/// - sequence A is prefilled again from a shorter prompt and decodes on,
/// - the last step decodes both sequences in 1 step of 2 rows.
fn script(model: &mut GenericHeteroTransformer, feed: Option<&[u32]>) -> Vec<Vec<f32>> {
    let tokens = prompt();
    let (pa, pb) = (tokens.clone(), tokens[3..15].to_vec());
    let mut out: Vec<Vec<f32>> = Vec::new();
    let next = |out: &Vec<Vec<f32>>| -> u32 {
        match feed {
            Some(f) => f[out.len() - 1],
            None => argmax(out.last().unwrap()),
        }
    };
    let mut sa: SequenceMixers = model.sequence_mixers().unwrap();
    let mut sb: SequenceMixers = model.sequence_mixers().unwrap();
    out.push(row(&model.prefill_sequence(&pa, &mut sa).unwrap()));
    let mut ta = next(&out);
    out.push(row(&model.prefill_sequence(&pb, &mut sb).unwrap()));
    let mut tb = next(&out);
    for i in 0..STEPS {
        let la = model
            .decode_sequences(&[ta], &[pa.len() + i], &mut [&mut sa])
            .unwrap();
        out.push(row(&la));
        ta = next(&out);
        let lb = model
            .decode_sequences(&[tb], &[pb.len() + i], &mut [&mut sb])
            .unwrap();
        out.push(row(&lb));
        tb = next(&out);
    }
    // A rewind: sequence A starts again from a shorter prompt.
    let short = &tokens[..10];
    out.push(row(&model.prefill_sequence(short, &mut sa).unwrap()));
    ta = next(&out);
    for i in 0..3 {
        let la = model
            .decode_sequences(&[ta], &[short.len() + i], &mut [&mut sa])
            .unwrap();
        out.push(row(&la));
        ta = next(&out);
    }
    let both = model
        .decode_sequences(
            &[ta, tb],
            &[short.len() + 3, pb.len() + STEPS],
            &mut [&mut sa, &mut sb],
        )
        .unwrap();
    out.extend(both.to_vec2::<f32>().unwrap());
    out
}

/// `model` with every layer's output projection and router, and the head, on card 0.
fn with_card_outputs(mut model: GenericHeteroTransformer) -> GenericHeteroTransformer {
    let choices = super::card_packed::Choices::default();
    for il in 0..model.layers.len() {
        for kind in [AlwaysRead::Output, AlwaysRead::Router] {
            model.keep_always_read(0, kind, il, &choices).unwrap();
        }
    }
    let last = model.layers.len() - 1;
    model
        .keep_always_read(0, AlwaysRead::Head, last, &choices)
        .unwrap();
    model
}

/// Loads of the host and the card model each test compares.
const LOADS: usize = 5;

/// Forces the `q8_1` route on every card product while it lives, so that 2 loads read the same
/// products whatever their timings.
struct SameRoute;

impl SameRoute {
    fn new() -> Self {
        super::card_packed::route::set(super::card_packed::route::Route::Q81);
        Self
    }
}

impl Drop for SameRoute {
    fn drop(&mut self) {
        super::card_packed::route::clear();
    }
}

/// The indices of the logits of `script` that the host path gives on both models: the 2
/// prefills before any decoded row and the rewind's prefill.
fn host_rows() -> [usize; 3] {
    [0, 1, 2 + 2 * STEPS]
}

/// The bound of the relative L2 distance of a decoded row's logits between the card stream of
/// `card` and the host path.
///
/// Each layer's admission measured every product the stream moves to the card, the host's and
/// the card's, against the exact one.
/// By the triangle inequality the card's product is no farther from the host's than the sum of
/// the 2 distances (`CardSinkhorn::apart`).
/// A row passes every layer once, then the final collapse when the card keeps it.
/// To first order the relative differences a row picks up add over the layers it passes.
/// The cached rows a step reads carry the same bound, and the token mixer weighs them by
/// weights that sum to 1.
fn derived_bound(card: &GenericHeteroTransformer) -> f64 {
    let layers: f64 = card
        .layers
        .iter()
        .filter_map(|l| l.sinkhorn.as_ref())
        .map(|b| b.card_mix.apart())
        .sum();
    layers + card.wide_sparse.as_ref().unwrap().head_collapse.apart()
}

/// The host's top-1 margin of `logits`: the largest value less the second largest.
fn margin(logits: &[f32]) -> f64 {
    let top = argmax(logits) as usize;
    let second = (0..logits.len())
        .filter(|&i| i != top)
        .map(|i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    logits[top] as f64 - second as f64
}

/// The L2 distance of `got` to `want`.
fn l2(want: &[f32], got: &[f32]) -> f64 {
    want.iter()
        .zip(got)
        .map(|(&w, &g)| (w as f64 - g as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Every layer admits its card mixers on the toy's values, the prefills give the host's bits,
/// and the decoded rows over the card stream choose the host's greedy token wherever the host's
/// margin exceeds their distance, on a script with 2 interleaved sequences, a rewind and a step
/// of 2 rows, over 5 loads.
#[test]
fn a_decoded_row_on_the_card_stream_keeps_the_hosts_tokens_beyond_a_near_tie() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let _route = SameRoute::new();
    for load in 0..LOADS {
        println!("load {load}");
        stream_keeps_the_hosts_tokens_beyond_a_near_tie(on_card(), with_card_mixers());
    }
}

/// A token mixer that runs its output projection on the stream's card reads its input and
/// leaves its output there, a router there reads the feed-forward's input there, and the final
/// collapse is measured on the card that keeps the head.
/// The prefills give the host's bits and the decoded rows choose the host's greedy token
/// wherever the host's margin exceeds their distance, over 5 loads.
#[test]
fn a_token_mixer_on_the_stream_card_keeps_the_hosts_tokens_beyond_a_near_tie() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let _route = SameRoute::new();
    for load in 0..LOADS {
        println!("load {load}");
        let card = with_card_outputs(with_card_mixers());
        for (il, l) in card.layers.iter().enumerate() {
            assert!(l.mixer.decodes_on(0), "layer {il}");
            let ffn = &l.sinkhorn.as_ref().unwrap().ffn;
            assert!(ffn.card.router_on(0).is_some(), "layer {il}");
        }
        let card =
            stream_keeps_the_hosts_tokens_beyond_a_near_tie(with_card_outputs(on_card()), card);
        let verdict = card.wide_sparse.as_ref().unwrap().head_collapse.verdict();
        println!("final collapse on the card: {verdict:?}");
        assert_eq!(verdict, Some(true), "the final collapse was not admitted");
    }
}

/// `card` admits its card mixers on every layer, fed `host`'s tokens on the script:
/// - the rows the host path gives on both models have the host's bits,
/// - every other row's distance is printed beside the bound the admissions derive
///   (`derived_bound`), which holds while the 2 paths make the same discrete choices,
/// - a row chooses the host's greedy token where the host's top-1 margin exceeds the L2
///   distance of the 2 rows, and a row where it does not is printed as a near tie.
fn stream_keeps_the_hosts_tokens_beyond_a_near_tie(
    mut host: GenericHeteroTransformer,
    mut card: GenericHeteroTransformer,
) -> GenericHeteroTransformer {
    let want = script(&mut host, None);
    let feed: Vec<u32> = want.iter().map(|l| argmax(l)).collect();
    let got = script(&mut card, Some(&feed));
    for (il, l) in card.layers.iter().enumerate() {
        let block = l.sinkhorn.as_ref().unwrap();
        assert_eq!(block.card_mix.verdict(), Some(true), "layer {il}");
    }
    assert_eq!(got.len(), want.len());
    let bound = derived_bound(&card);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    for k in host_rows() {
        assert_eq!(bits(&got[k]), bits(&want[k]), "logits {k}: the host path");
    }
    let (mut ties, mut past) = (0usize, 0usize);
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        let (rel, dist, m) = (apart(w, g), l2(w, g), margin(w));
        println!("logits {k}: {rel:.3e} apart, bound {bound:.3e}, margin {m:.3e}, L2 {dist:.3e}");
        if rel > bound {
            past += 1;
            println!("logits {k}: past the bound, a discrete choice of the row parted");
        }
        if m > dist {
            assert_eq!(argmax(g), argmax(w), "logits {k}");
        } else {
            ties += 1;
            println!("logits {k}: near tie, margin {m:.3e} within the L2 distance {dist:.3e}");
        }
    }
    println!(
        "{ties} near ties and {past} rows past the bound of {} rows",
        got.len()
    );
    card
}

/// A step of 2 rows over a model whose layers keep their mixers on the card runs on the host and
/// gives each row the host's bits.
#[test]
fn a_step_of_2_rows_keeps_the_hosts_bits() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let _route = SameRoute::new();
    let tokens = prompt();
    let (a, b) = (&tokens[..9], &tokens[3..15]);
    let run = |model: &mut GenericHeteroTransformer| -> Vec<u32> {
        let mut sa = model.sequence_mixers().unwrap();
        let mut sb = model.sequence_mixers().unwrap();
        model.prefill_sequence(a, &mut sa).unwrap();
        model.prefill_sequence(b, &mut sb).unwrap();
        let logits = model
            .decode_sequences(&[5, 7], &[a.len(), b.len()], &mut [&mut sa, &mut sb])
            .unwrap();
        row(&logits).iter().map(|v| v.to_bits()).collect()
    };
    let want = run(&mut on_card());
    let got = run(&mut with_card_mixers());
    assert_eq!(got, want);
    let want = run(&mut with_card_outputs(on_card()));
    let got = run(&mut with_card_outputs(with_card_mixers()));
    assert_eq!(got, want, "with the outputs, routers and head on the card");
}

/// The load times the card mixers of a layer against the host's and counts their bytes.
#[test]
fn the_load_measures_the_card_mixers() {
    if !card_present() {
        eprintln!("no card 0: skipped");
        return;
    }
    let model = on_card();
    let m = model
        .measure_always_read(0, AlwaysRead::Sinkhorn, 0)
        .unwrap();
    println!("{m:?}");
    assert!(m.bytes > 0 && m.host_secs > 0.0 && m.card_secs > 0.0);
}
