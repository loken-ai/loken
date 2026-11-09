//! The card's copies of the window and band rows kept across decode steps.
//!
//! The model is the synthetic one of the frozen V4.1 oracle, in its 2 band layouts, every layer
//! on card 0.
//! A run that keeps its copies gives the bits of the same run that rebuilds them before every
//! step, on the same model.
//! A decode step after a decode step of the same state rebuilds nothing.
//! A step after a prefill, a rewind, a state read back or another sequence's step rebuilds.

use super::card_mixer_tests::{generic_on_card, rows};
use super::mixer::{Mixer, MixerState};
use super::wide_sparse_batch_tests::{batched, bits, greedy, prompts, serial, Run, STEPS};
use super::wide_sparse_oracle_tests::LAYOUTS;
use super::*;

/// Whether card 0 opens.
fn card_present() -> bool {
    Device::new_cuda(0).is_ok()
}

/// A step of a script over every layer's token mixer.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// The rows `pos..pos + seq` through every layer, inputs drawn from `seed`.
    Rows { pos: usize, seq: usize, seed: u64 },
    /// Every layer's state copied as the mark.
    Mark,
    /// Every layer's state made a copy of the mark: a rewind or a draft trimmed.
    Back,
    /// Every layer's state made a copy of the mark under the name 0: a state read back.
    BackRead,
}

/// What a run does to a layer's card copies before each of its forwards.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Copies {
    /// Keeps them.
    Kept,
    /// Drops their names.
    Forgotten,
    /// Drops the buffers.
    Dropped,
}

/// Drops the names of the card copies of `m`.
fn forget(m: &Mixer) {
    match m {
        Mixer::LatentMqa(x) => x.forget_card_copies(),
        Mixer::CompressedBand(x) => x.forget_card_copies(),
        _ => {}
    }
}

/// Times the card copies of every layer of `model` were rebuilt.
fn rebuilds(model: &GenericHeteroTransformer) -> u64 {
    let each = model.layers.iter().map(|l| match &l.mixer {
        Mixer::LatentMqa(x) => x.card_rebuilds(),
        Mixer::CompressedBand(x) => x.card_rebuilds(),
        _ => 0,
    });
    each.sum()
}

/// `state` under the name 0.
fn unnamed(state: &MixerState) -> MixerState {
    let mut s = state.clone();
    match &mut s {
        MixerState::LatentMqa(Some(x)) => x.stamp = 0,
        MixerState::CompressedBand(Some(x)) => x.window.stamp = 0,
        _ => {}
    }
    s
}

/// The script: a prompt, decode steps past the window, a continued prompt, decode steps, a
/// mark, decode steps, a rewind to the mark, other decode steps, a state read back, decode
/// steps.
fn script() -> Vec<Op> {
    let decode = |from: usize, to: usize, seed: u64| {
        (from..to).map(move |pos| Op::Rows {
            pos,
            seq: 1,
            seed: seed + pos as u64,
        })
    };
    let mut ops = vec![Op::Rows {
        pos: 0,
        seq: 9,
        seed: 1,
    }];
    ops.extend(decode(9, 29, 100));
    ops.push(Op::Rows {
        pos: 29,
        seq: 5,
        seed: 2,
    });
    ops.extend(decode(34, 38, 200));
    ops.push(Op::Mark);
    ops.extend(decode(38, 42, 300));
    ops.push(Op::Back);
    ops.extend(decode(38, 41, 400));
    ops.push(Op::BackRead);
    ops.extend(decode(38, 40, 500));
    ops
}

/// The output bits of every layer at every step of `ops` on `model`, and the rebuilds each op
/// caused.
///
/// `copies` says what happens to a layer's card copies before each of its forwards.
fn run(
    model: &mut GenericHeteroTransformer,
    ops: &[Op],
    copies: Copies,
) -> (Vec<Vec<u32>>, Vec<u64>) {
    let hidden = model.config.embedding_length;
    let mut mark: Vec<MixerState> = Vec::new();
    let (mut outs, mut caused) = (Vec::new(), Vec::new());
    for op in ops {
        let before = rebuilds(model);
        match *op {
            Op::Rows { pos, seq, seed } => {
                if let Some(head) = &model.wide_sparse {
                    head.cell.open(pos, seq);
                }
                for il in 0..model.layers.len() {
                    let mixer = &mut model.layers[il].mixer;
                    match copies {
                        Copies::Kept => {}
                        Copies::Forgotten => forget(mixer),
                        Copies::Dropped => mixer.drop_card_copies(),
                    }
                    let x = rows(seed * 31 + il as u64, seq, hidden);
                    let out = mixer.forward(&x, pos).unwrap();
                    let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    outs.push(flat.iter().map(|v| v.to_bits()).collect());
                }
            }
            Op::Mark => {
                mark = model
                    .layers
                    .iter()
                    .map(|l| l.mixer.snapshot().unwrap().unwrap())
                    .collect();
            }
            Op::Back | Op::BackRead => {
                for (l, m) in model.layers.iter_mut().zip(&mark) {
                    let m = match op {
                        Op::BackRead => unnamed(m),
                        _ => m.clone(),
                    };
                    l.mixer.restore(&m).unwrap();
                }
            }
        }
        caused.push(rebuilds(model).saturating_sub(before));
    }
    (outs, caused)
}

/// Every layer keeps its card copies across decode steps: the outputs are the bits of copies
/// rebuilt before every step, through a continued prompt, a rewind and a state read back.
#[test]
fn kept_card_copies_give_the_bits_of_copies_rebuilt_every_step() {
    if !card_present() {
        return;
    }
    let ops = script();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut card = generic_on_card(ratios, kv_src, index_src);
        let (kept, caused) = run(&mut card, &ops, Copies::Kept);
        let (rebuilt, _) = run(&mut card, &ops, Copies::Forgotten);
        let (dropped, _) = run(&mut card, &ops, Copies::Dropped);
        assert_eq!(kept.len(), rebuilt.len());
        assert_eq!(kept.len(), dropped.len());
        for (i, ((a, b), c)) in kept.iter().zip(&rebuilt).zip(&dropped).enumerate() {
            assert_eq!(
                a, b,
                "layout {ratios:?}: output {i} differs from the rebuilt copies'"
            );
            assert_eq!(
                a, c,
                "layout {ratios:?}: output {i} differs from the dropped copies'"
            );
        }
        // A decode step after a decode step rebuilds nothing.
        // The first decode step after a prompt, a rewind or a state read back rebuilds.
        let decode = |op: &Op| matches!(op, Op::Rows { seq: 1, pos, .. } if *pos > 0);
        let mut last = None;
        for (op, &n) in ops.iter().zip(&caused) {
            if decode(op) {
                match last {
                    Some(Op::Rows { seq: 1, .. }) => {
                        assert_eq!(
                            n, 0,
                            "layout {ratios:?}: {op:?} after a decode step rebuilt"
                        )
                    }
                    _ => assert!(
                        n > 0,
                        "layout {ratios:?}: {op:?} after {last:?} rebuilt nothing"
                    ),
                }
            }
            if !matches!(op, Op::Mark) {
                last = Some(*op);
            }
        }
    }
}

/// One sequence decoded by batched steps keeps its card copies from its second step on, and
/// gives the bits of its serial run.
#[test]
fn one_sequence_of_batched_steps_keeps_its_card_copies_and_its_serial_bits() {
    if !card_present() {
        return;
    }
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut card = generic_on_card(ratios, kv_src, index_src);
        let prompt = prompts()[0].clone();
        let want = serial(&mut card, std::slice::from_ref(&prompt));
        let mut seq = card.sequence_mixers().unwrap();
        let logits = card.prefill_sequence(&prompt, &mut seq).unwrap();
        let mut got: Run = (vec![greedy(&logits)], vec![bits(&logits)]);
        for i in 0..STEPS {
            let before = rebuilds(&card);
            let token = *got.0.last().unwrap();
            let logits = card
                .decode_sequences(&[token], &[prompt.len() + i], &mut [&mut seq])
                .unwrap();
            got.0.push(greedy(&logits));
            got.1.push(bits(&logits));
            let n = rebuilds(&card) - before;
            match i {
                0 => assert!(n > 0, "layout {ratios:?}: the first step rebuilt nothing"),
                _ => assert_eq!(n, 0, "layout {ratios:?}: step {i} rebuilt {n} copies"),
            }
        }
        assert_eq!(
            got, want[0],
            "layout {ratios:?}: batched bits differ from the serial run"
        );
    }
}

/// Sequences interleaved in batched steps on the card give the bits of their serial runs on the
/// same card: a copy kept for one sequence never serves another.
#[test]
fn interleaved_sequences_on_the_card_give_the_bits_of_their_serial_runs() {
    if !card_present() {
        return;
    }
    let p = prompts();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut card = generic_on_card(ratios, kv_src, index_src);
        let want = serial(&mut card, &p);
        let got = batched(&mut card, &p);
        for (s, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g, w,
                "layout {ratios:?}: sequence {s} differs from its serial run"
            );
        }
    }
}
