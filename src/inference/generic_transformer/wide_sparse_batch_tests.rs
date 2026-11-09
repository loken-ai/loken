//! Sequences batched across on the Sinkhorn wide-residual model give the bits of the same
//! sequences run serially, on the synthetic model of the frozen V4.1 oracle, in both its band
//! layouts.
//!
//! The serial run is each prompt on 1 model, one after the other: its prefill, then its greedy
//! tokens decoded 1 at a time.
//! The batched run keeps 1 state per sequence and decodes the live sequences together.
//! The 4th sequence joins after the first 3 have decoded a few tokens.

use std::collections::HashMap;

use super::wide_sparse_layer::SequenceMixers;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::generic_transformer::wide_sparse_toy::VOCAB;
use crate::inference::serve::continuous_batch::{ContinuousBatchEngine, GenReq, SamplingParams};
use crate::inference::serve::continuous_serve::SequenceBatched;

/// Tokens each sequence decodes after the one its prompt gives.
const STEPS: usize = 8;

/// Steps the first 3 sequences take before the 4th joins.
const LATE: usize = 2;

/// 4 prompts of other tokens: the 1st and the 3rd stand at the same positions.
pub(super) fn prompts() -> Vec<Vec<u32>> {
    [(12usize, 7u32, 3u32), (9, 5, 1), (12, 3, 2), (7, 11, 4)]
        .iter()
        .map(|&(len, step, start)| {
            (0..len as u32)
                .map(|i| (i * step + start) % VOCAB as u32)
                .collect()
        })
        .collect()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

/// The greedy choice of the continuous-batch server's sampler, on logits `[1, vocab]`.
fn greedy(logits: &Tensor) -> u32 {
    logits
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .argmax(crate::tensor::D::Minus1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap()
}

fn bits(logits: &Tensor) -> Vec<u32> {
    logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// What a sequence gives: its greedy tokens, and the bits of the logits each was chosen from.
pub(super) type Run = (Vec<u32>, Vec<Vec<u32>>);

/// Each prompt run alone on `model`, one after the other.
pub(super) fn serial(model: &mut GenericHeteroTransformer, prompts: &[Vec<u32>]) -> Vec<Run> {
    prompts
        .iter()
        .map(|p| {
            let mut logits = model.forward(&ids(p), 0).unwrap();
            let (mut tokens, mut rows) = (Vec::new(), Vec::new());
            for i in 0..=STEPS {
                let t = greedy(&logits);
                tokens.push(t);
                rows.push(bits(&logits));
                if i < STEPS {
                    logits = model.forward(&ids(&[t]), p.len() + i).unwrap();
                }
            }
            (tokens, rows)
        })
        .collect()
}

/// The prompts batched on `model`: each prefilled over a state of its own, the live sequences
/// decoded together, the last prompt prefilled after [`LATE`] steps.
pub(super) fn batched(model: &mut GenericHeteroTransformer, prompts: &[Vec<u32>]) -> Vec<Run> {
    let n = prompts.len();
    let mut seqs: Vec<SequenceMixers> = (0..n).map(|_| model.sequence_mixers().unwrap()).collect();
    let mut out: Vec<Run> = vec![(Vec::new(), Vec::new()); n];
    let prefill = |model: &mut GenericHeteroTransformer,
                   seq: &mut SequenceMixers,
                   prompt: &[u32],
                   run: &mut Run| {
        let logits = model.prefill_sequence(prompt, seq).unwrap();
        run.0.push(greedy(&logits));
        run.1.push(bits(&logits));
    };
    let first = seqs.iter_mut().zip(prompts).zip(out.iter_mut()).take(n - 1);
    for ((seq, prompt), run) in first {
        prefill(&mut *model, seq, prompt.as_slice(), run);
    }
    let mut step = 0;
    loop {
        if step == LATE {
            prefill(
                &mut *model,
                &mut seqs[n - 1],
                prompts[n - 1].as_slice(),
                &mut out[n - 1],
            );
        }
        let live: Vec<usize> = (0..n)
            .filter(|&s| !out[s].0.is_empty() && out[s].0.len() <= STEPS)
            .collect();
        if live.is_empty() {
            break;
        }
        let tokens: Vec<u32> = live.iter().map(|&s| *out[s].0.last().unwrap()).collect();
        let positions: Vec<usize> = live
            .iter()
            .map(|&s| prompts[s].len() + out[s].0.len() - 1)
            .collect();
        let mut refs: Vec<&mut SequenceMixers> = seqs
            .iter_mut()
            .enumerate()
            .filter(|(s, _)| live.contains(s))
            .map(|(_, m)| m)
            .collect();
        let logits = model
            .decode_sequences(&tokens, &positions, &mut refs)
            .unwrap();
        for (j, &s) in live.iter().enumerate() {
            let row = logits.narrow(0, j, 1).unwrap();
            out[s].0.push(greedy(&row));
            out[s].1.push(bits(&row));
        }
        step += 1;
    }
    out
}

/// The prompts served by the continuous-batch engine over the batched model, the last one
/// submitted after [`LATE`] steps: each sequence's tokens by its id.
fn served(model: GenericHeteroTransformer, prompts: &[Vec<u32>]) -> HashMap<u64, Vec<u32>> {
    let adapter = SequenceBatched::new(model).unwrap();
    let mut engine = ContinuousBatchEngine::new(adapter, Vec::new(), 64, 16, 8, 4096).unwrap();
    let submit = |engine: &mut ContinuousBatchEngine<SequenceBatched>, s: usize| {
        engine.submit(GenReq {
            id: s as u64,
            prompt: prompts[s].clone(),
            max_new: STEPS + 1,
            sampling: SamplingParams::greedy(),
        })
    };
    let n = prompts.len();
    for s in 0..n - 1 {
        submit(&mut engine, s);
    }
    let mut done = Vec::new();
    for _ in 0..LATE {
        done.extend(engine.step().unwrap());
    }
    submit(&mut engine, n - 1);
    done.extend(engine.run_to_idle().unwrap());
    done.into_iter().map(|d| (d.id, d.tokens)).collect()
}

/// For both layouts, 4 sequences batched give the tokens and the logits bits of the same 4
/// sequences run serially, step for step.
#[test]
fn four_sequences_batched_give_the_bits_of_the_same_sequences_run_serially() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let prompts = prompts();
        let want = serial(&mut generic(ratios, kv_src, index_src), &prompts);
        let got = batched(&mut generic(ratios, kv_src, index_src), &prompts);
        for (s, (w, g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(
                w.0, g.0,
                "layout {ratios:?}: sequence {s} chose other tokens"
            );
            for (k, (wr, gr)) in w.1.iter().zip(&g.1).enumerate() {
                assert!(
                    wr == gr,
                    "layout {ratios:?}: sequence {s} left the serial bits at step {k}"
                );
            }
        }
    }
}

/// For both layouts, the continuous-batch engine serves each of the 4 sequences the tokens of
/// its serial run, with the 4th admitted in a step that decodes the others.
#[test]
fn the_engine_serves_four_sequences_the_tokens_of_their_serial_runs() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let prompts = prompts();
        let want = serial(&mut generic(ratios, kv_src, index_src), &prompts);
        let got = served(generic(ratios, kv_src, index_src), &prompts);
        assert_eq!(got.len(), prompts.len());
        for (s, w) in want.iter().enumerate() {
            assert_eq!(
                got.get(&(s as u64)),
                Some(&w.0),
                "layout {ratios:?}: sequence {s} was served other tokens"
            );
        }
    }
}

/// The prompt `oracle_gate::synthetic` gives the V4.1 batch tests: 24 tokens.
fn oracle_tokens() -> Vec<u32> {
    (0..24).map(|i| (i * 7 + 3) % VOCAB as u32).collect()
}

/// Port of the V4.1 `forward_decode_batch_matches_separate_decodes`, same sequences.
///
/// 3 sequences of distinct streams and lengths are each decoded alone up to their last token.
/// 1 batched step then runs over their last tokens.
/// Each row gives the bits of its sequence's last logits decoded alone, token by token.
#[test]
fn a_decode_batch_gives_the_bits_of_separate_decodes() {
    let tokens = oracle_tokens();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let seqs: Vec<Vec<u32>> = (0..3)
            .map(|j| {
                let len = tokens.len() - 2 * j - 1;
                (0..len)
                    .map(|i| tokens[(i + 5 * j) % tokens.len()])
                    .collect()
            })
            .collect();
        let mut alone = generic(ratios, kv_src, index_src);
        let want: Vec<Vec<u32>> = seqs
            .iter()
            .map(|s| {
                let mut last = Vec::new();
                for (i, &t) in s.iter().enumerate() {
                    last = bits(&alone.forward(&ids(&[t]), i).unwrap());
                }
                last
            })
            .collect();
        let mut model = generic(ratios, kv_src, index_src);
        let mut states: Vec<SequenceMixers> = seqs
            .iter()
            .map(|s| {
                let mut st = model.sequence_mixers().unwrap();
                for (i, &t) in s[..s.len() - 1].iter().enumerate() {
                    model.decode_sequences(&[t], &[i], &mut [&mut st]).unwrap();
                }
                st
            })
            .collect();
        let last: Vec<u32> = seqs.iter().map(|s| *s.last().unwrap()).collect();
        let positions: Vec<usize> = seqs.iter().map(|s| s.len() - 1).collect();
        let mut refs: Vec<&mut SequenceMixers> = states.iter_mut().collect();
        let logits = model
            .decode_sequences(&last, &positions, &mut refs)
            .unwrap();
        for (j, w) in want.iter().enumerate() {
            assert!(
                bits(&logits.narrow(0, j, 1).unwrap()) == *w,
                "layout {ratios:?}: sequence {j} left the bits of its lone decode"
            );
        }
    }
}

/// Port of the V4.1 `generate_batch_matches_separate_generations` and
/// `batch_server_streams_the_batched_tokens`, same prompts and budgets.
///
/// 3 prompts of distinct streams and lengths, with budgets of 6, 3 and 5 tokens, go through the
/// continuous-batch engine together, so the batch narrows as the shorter ones finish.
/// Each prompt gets the greedy tokens it gets generated alone, chosen by the server's sampler.
#[test]
fn a_narrowing_batch_generates_the_tokens_of_separate_generations() {
    let tokens = oracle_tokens();
    let max_new = [6usize, 3, 5];
    for (ratios, kv_src, index_src) in LAYOUTS {
        let prompts: Vec<Vec<u32>> = (0..3)
            .map(|j| {
                let len = 8 + 2 * j;
                (0..len)
                    .map(|i| tokens[(i + 5 * j) % tokens.len()])
                    .collect()
            })
            .collect();
        let mut alone = generic(ratios, kv_src, index_src);
        let want: Vec<Vec<u32>> = prompts
            .iter()
            .zip(max_new)
            .map(|(p, m)| {
                let mut logits = alone.forward(&ids(p), 0).unwrap();
                let mut out = Vec::new();
                while out.len() < m {
                    let t = greedy(&logits);
                    out.push(t);
                    logits = alone.forward(&ids(&[t]), p.len() + out.len() - 1).unwrap();
                }
                out
            })
            .collect();
        let adapter = SequenceBatched::new(generic(ratios, kv_src, index_src)).unwrap();
        let mut engine = ContinuousBatchEngine::new(adapter, Vec::new(), 64, 16, 8, 4096).unwrap();
        for (s, (p, m)) in prompts.iter().zip(max_new).enumerate() {
            engine.submit(GenReq {
                id: s as u64,
                prompt: p.clone(),
                max_new: m,
                sampling: SamplingParams::greedy(),
            });
        }
        let got: HashMap<u64, Vec<u32>> = engine
            .run_to_idle()
            .unwrap()
            .into_iter()
            .map(|d| (d.id, d.tokens))
            .collect();
        for (s, w) in want.iter().enumerate() {
            assert_eq!(
                got.get(&(s as u64)),
                Some(w),
                "layout {ratios:?}: sequence {s} was served other tokens"
            );
        }
    }
}

/// A model that keeps no latent or band state is refused a batch of sequences.
#[test]
fn a_model_without_latent_layers_keeps_no_sequence_state() {
    let path = super::tiny_model::write("wide-sparse-batch-dense", 2);
    let model = super::tiny_model::load(&path, 2);
    assert!(model.sequence_mixers().is_err());
    assert!(SequenceBatched::new(model).is_err());
    super::tiny_model::remove(&path);
}
