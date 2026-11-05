//! The continuous-batch paths on the host for a mixture, a stretched rotation and a partial
//! rotation.
//!
//! A sequence decoded inside a batch gives the tokens it gives decoded alone.
//! The batched paths give the logits the serial forward gives for the same tokens.

use super::tiny_model;
use super::*;
use crate::inference::cache::paged_kv::PagedKvStore;
use crate::inference::serve::continuous_serve::{CbToken, ContinuousServer, CB_BLOCK_SIZE};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::DecodeOrder;

const LAYERS: usize = 2;

/// Tokens per block of the paged stores the tests build.
const BLOCK: usize = 4;

/// Blocks each sequence owns in those stores.
const BLOCKS_PER_SEQ: usize = 8;

/// Decode steps after each prompt.
const STEPS: usize = 6;

/// The largest logit difference, as a share of the logits' range, that the dense batch
/// invariance test of `dense.rs` grants.
const SHARE: f32 = 1e-4;

/// 4 prompts of different lengths and tokens: 2 of them stand at the same positions.
fn prompts() -> Vec<Vec<u32>> {
    let vocab = tiny_model::VOCAB as u32;
    [(9usize, 1u32, 3u32), (7, 2, 5), (9, 4, 1), (8, 3, 7)]
        .iter()
        .map(|&(len, step, start)| {
            (0..len as u32)
                .map(|i| (i * step + start) % vocab)
                .collect()
        })
        .collect()
}

/// The first of the largest.
fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, &v)| if v > m.1 { (i, v) } else { m },
        )
        .0 as u32
}

fn rows_of(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

/// The largest difference between `want` and `got`, as a share of `want`'s range.
fn share(want: &[f32], got: &[f32]) -> f32 {
    let range = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let worst = want
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    worst / range.max(f32::MIN_POSITIVE)
}

/// What a sequence gives: its greedy tokens, and the logits each of them was chosen from.
type Run = (Vec<u32>, Vec<Vec<f32>>);

/// The sequences `which` of `prompts` run together through the batched paths: their prompts
/// prefilled in one forward, then `STEPS` decode steps of all of them at once.
///
/// Sequence `s` owns the same blocks whichever sequences run beside it.
fn run_batched(
    model: &GenericHeteroTransformer,
    prompts: &[Vec<u32>],
    which: &[usize],
) -> Vec<Run> {
    let (n_kv, hd, n_layers, _) = model.paged_geometry();
    let device = model.compute_device();
    let mut stores: Vec<PagedKvStore> = (0..n_layers)
        .map(|_| {
            PagedKvStore::new(
                prompts.len() * BLOCKS_PER_SEQ,
                BLOCK,
                n_kv,
                hd,
                DType::F32,
                &device,
            )
            .unwrap()
        })
        .collect();
    let table = |s: usize| -> Vec<u32> {
        (0..BLOCKS_PER_SEQ)
            .map(|b| (s * BLOCKS_PER_SEQ + b) as u32)
            .collect()
    };
    let slot = |s: usize, p: usize| table(s)[p / BLOCK] as usize * BLOCK + p % BLOCK;
    let slots: Vec<Vec<usize>> = which
        .iter()
        .map(|&s| (0..prompts[s].len()).map(|p| slot(s, p)).collect())
        .collect();
    let seqs: Vec<(&[u32], &[usize])> = which
        .iter()
        .zip(&slots)
        .map(|(&s, sl)| (prompts[s].as_slice(), sl.as_slice()))
        .collect();
    let first = rows_of(&model.paged_prefill_batch(&seqs, &mut stores).unwrap());
    let mut out: Vec<Run> = first
        .into_iter()
        .map(|row| (vec![argmax(&row)], vec![row]))
        .collect();
    for step in 0..STEPS {
        let pos: Vec<usize> = which.iter().map(|&s| prompts[s].len() + step).collect();
        let next: Vec<u32> = out.iter().map(|(t, _)| *t.last().unwrap()).collect();
        let at: Vec<usize> = which.iter().zip(&pos).map(|(&s, &p)| slot(s, p)).collect();
        let tables: Vec<Vec<u32>> = which.iter().map(|&s| table(s)).collect();
        let ctx: Vec<usize> = pos.iter().map(|p| p + 1).collect();
        let logits = rows_of(
            &model
                .batched_paged_decode(&next, &pos, &at, &tables, &ctx, &mut stores)
                .unwrap(),
        );
        for (o, row) in out.iter_mut().zip(logits) {
            o.0.push(argmax(&row));
            o.1.push(row);
        }
    }
    out
}

/// Each sequence run alone gives the tokens it gives beside the 3 others, and logits within
/// [`SHARE`] of the range of its logits alone.
fn alone_and_batched_agree(model: &GenericHeteroTransformer) {
    assert!(model.cb_eligible(), "{:?}", model.cb_unsupported_reason());
    let prompts = prompts();
    let together = run_batched(model, &prompts, &[0, 1, 2, 3]);
    let mut worst = 0f32;
    for (s, beside) in together.iter().enumerate() {
        let alone = run_batched(model, &prompts, &[s]);
        assert_eq!(
            alone[0].0, beside.0,
            "sequence {s} decoded other tokens inside the batch"
        );
        for (a, b) in alone[0].1.iter().zip(&beside.1) {
            worst = worst.max(share(a, b));
        }
    }
    eprintln!("alone against batched: largest logit difference {worst:.3e} of the range");
    assert!(
        worst <= SHARE,
        "a sequence's logits moved by {worst} of their range in a batch"
    );
}

/// The batched paths give, for every prompt and every token they chose, logits within
/// [`SHARE`] of the range of the serial forward's on a model of its own.
fn batched_follows_serial(path: &std::path::Path) {
    let batched = tiny_model::load(path, LAYERS);
    let prompts = prompts();
    let together = run_batched(&batched, &prompts, &[0, 1, 2, 3]);
    let mut worst = 0f32;
    for (prompt, (tokens, logits)) in prompts.iter().zip(&together) {
        let mut serial = tiny_model::load(path, LAYERS);
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut want = vec![flat(serial.forward(&ids(prompt), 0).unwrap())];
        for (i, &t) in tokens[..STEPS].iter().enumerate() {
            want.push(flat(serial.forward(&ids(&[t]), prompt.len() + i).unwrap()));
        }
        for (w, g) in want.iter().zip(logits) {
            worst = worst.max(share(w, g));
        }
    }
    eprintln!("serial against batched: largest logit difference {worst:.3e} of the range");
    assert!(
        worst <= SHARE,
        "the batched logits moved by {worst} of their range"
    );
}

/// The tiny dense model with `extra` header keys, written under `tag`.
fn dense_with(tag: &str, extra: Vec<(String, Value)>) -> std::path::PathBuf {
    tiny_model::write_wide_with(tag, LAYERS, false, extra)
}

/// Under the decode order the mixture gives every row of a batch the bits it gives the row
/// alone: each row routes with its own choice and sums its experts as a decoded token does.
#[test]
fn a_mixture_routes_each_row_of_a_batch_as_it_routes_the_row_alone() {
    let path = tiny_model::write_moe("cb-moe-rows", LAYERS);
    let model = tiny_model::load(&path, LAYERS);
    let layer = &model.layers[0];
    let moe = layer
        .moe
        .as_ref()
        .expect("the tiny mixture routes its rows");
    let width = model.hidden_size();
    let rows = 5;
    let x: Vec<f32> = (0..rows * width)
        .map(|i| (i * 37 % 101) as f32 / 101.0 - 0.5)
        .collect();
    let x = Tensor::from_vec(x, (rows, width), &Device::Cpu).unwrap();
    let normed = layer
        .ffn_norm
        .as_ref()
        .unwrap()
        .forward(&x)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let x = x.unsqueeze(0).unwrap();
    let bits = |t: &Tensor| -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let batch = {
        let _order = DecodeOrder::enter();
        moe.forward_rows(
            &normed,
            &x,
            layer.embedding_length_for_moe,
            layer.ffn_norm_eps,
        )
        .unwrap()
    };
    for r in 0..rows {
        let row = |t: &Tensor| t.narrow(1, r, 1).unwrap().contiguous().unwrap();
        let alone = moe
            .forward_rows(
                &row(&normed),
                &row(&x),
                layer.embedding_length_for_moe,
                layer.ffn_norm_eps,
            )
            .unwrap();
        assert_eq!(
            bits(&row(&batch)),
            bits(&alone),
            "row {r} of the batch left the bits it has alone"
        );
    }
    tiny_model::remove(&path);
}

/// A mixture is served by the batched paths on the host: a sequence keeps its tokens inside a
/// batch, and the batch follows the serial forward.
#[test]
fn a_mixture_sequence_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let path = tiny_model::write_moe("cb-moe-batch", LAYERS);
    alone_and_batched_agree(&tiny_model::load(&path, LAYERS));
    batched_follows_serial(&path);
    tiny_model::remove(&path);
}

/// The continuous-batch server on the host gives a mixture's sequence the tokens it gives
/// that sequence served alone, whatever steps the 3 others share with it.
#[test]
fn the_server_gives_a_mixture_sequence_its_tokens_alone_and_beside_others() {
    let path = tiny_model::write_moe("cb-moe-server", LAYERS);
    let prompts = prompts();
    let serve = |which: &[usize]| -> Vec<Vec<u32>> {
        let model = Arc::new(tiny_model::load(&path, LAYERS));
        let server =
            ContinuousServer::spawn(model, Vec::new(), 64, CB_BLOCK_SIZE, 8, 4096).unwrap();
        let receivers: Vec<_> = which
            .iter()
            .map(|&s| server.submit(prompts[s].clone(), STEPS))
            .collect();
        let out = receivers
            .into_iter()
            .map(|rx| {
                let mut tokens = Vec::new();
                for token in rx.iter() {
                    match token {
                        CbToken::Tok(t) => tokens.push(t),
                        CbToken::Done(_) => break,
                        CbToken::Admitted { .. } => {}
                    }
                }
                tokens
            })
            .collect();
        let _ = server.stop();
        out
    };
    let together = serve(&[0, 1, 2, 3]);
    for (s, beside) in together.iter().enumerate() {
        assert_eq!(beside.len(), STEPS, "sequence {s} ended early");
        assert_eq!(
            &serve(&[s])[0],
            beside,
            "sequence {s} was served other tokens beside the others"
        );
    }
    tiny_model::remove(&path);
}

/// A stretched rotation is served by the batched paths on the host, from each layer's tables.
#[test]
fn a_stretched_rotation_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let path = dense_with(
        "cb-yarn",
        vec![
            (
                "llama.rope.scaling.type".into(),
                Value::String("yarn".into()),
            ),
            ("llama.rope.scaling.factor".into(), Value::F32(4.0)),
            (
                "llama.rope.scaling.original_context_length".into(),
                Value::U32(8),
            ),
        ],
    );
    let model = tiny_model::load(&path, LAYERS);
    assert!(
        model.config.yarn.is_some(),
        "the file stretches its rotation"
    );
    alone_and_batched_agree(&model);
    batched_follows_serial(&path);
    tiny_model::remove(&path);
}

/// A rotation over part of each head is served by the batched paths on the host, from each
/// layer's tables.
#[test]
fn a_partial_rotation_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let rotated = tiny_model::MOE_HEAD_DIM / 2;
    let path = dense_with(
        "cb-partial-rope",
        vec![(
            "llama.rope.dimension_count".into(),
            Value::U32(rotated as u32),
        )],
    );
    let model = tiny_model::load(&path, LAYERS);
    assert_eq!(model.config.rope_dim, Some(rotated));
    alone_and_batched_agree(&model);
    batched_follows_serial(&path);
    tiny_model::remove(&path);
}
