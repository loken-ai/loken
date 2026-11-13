//! The continuous-batch paths on the host for a mixture, a stretched rotation, a partial
//! rotation, attention sinks and a windowed layer.
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
    // The stores the server gives the device.
    let held = match device.is_cpu() {
        true => DType::F32,
        false => DType::F16,
    };
    let mut stores: Vec<PagedKvStore> = (0..n_layers)
        .map(|_| {
            PagedKvStore::new(
                prompts.len() * BLOCKS_PER_SEQ,
                BLOCK,
                n_kv,
                hd,
                held,
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
    served_alone_and_beside_agree(&path);
    tiny_model::remove(&path);
}

/// The continuous-batch server on the host gives each sequence of the model at `path` the
/// tokens it gives that sequence served alone, whatever steps the 3 others share with it.
fn served_alone_and_beside_agree(path: &std::path::Path) {
    served_alone_and_beside_agree_on(&|| tiny_model::load(path, LAYERS));
}

/// [`served_alone_and_beside_agree`] for the model `load` gives.
fn served_alone_and_beside_agree_on(load: &dyn Fn() -> GenericHeteroTransformer) {
    let prompts = prompts();
    let serve = |which: &[usize]| -> Vec<Vec<u32>> {
        let model = Arc::new(load());
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

/// The header keys that window the first layer to its last 4 keys and leave the second global.
fn first_layer_windowed() -> Vec<(String, Value)> {
    vec![
        ("llama.attention.sliding_window".into(), Value::U32(4)),
        (
            "llama.attention.sliding_window_pattern".into(),
            Value::Array(vec![Value::U32(1), Value::U32(0)]),
        ),
    ]
}

/// Attention sinks are served by the batched paths on the host: each head's sink stands in
/// its softmax denominator, for every sequence of the batch.
#[test]
fn a_sink_sequence_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let path = tiny_model::write_wide_sinks("cb-sinks", LAYERS, false, Vec::new());
    let model = tiny_model::load(&path, LAYERS);
    assert!(model.layers.iter().all(|l| l.attn_sinks.is_some()));
    alone_and_batched_agree(&model);
    batched_follows_serial(&path);
    served_alone_and_beside_agree(&path);
    tiny_model::remove(&path);
}

/// A windowed layer reads its last keys in the batched paths on the host, as the serial
/// forward does, and its global neighbour reads them all.
#[test]
fn a_windowed_layer_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let path = tiny_model::write_wide_with("cb-window", LAYERS, false, first_layer_windowed());
    let model = tiny_model::load(&path, LAYERS);
    assert_eq!(model.layers[0].sliding_window, Some(4));
    assert_eq!(model.layers[1].sliding_window, None);
    alone_and_batched_agree(&model);
    batched_follows_serial(&path);
    tiny_model::remove(&path);
}

/// A mixture with sinks, a windowed layer and a stretched rotation is served by the batched
/// paths and by the server on the host.
#[test]
fn a_mixture_with_sinks_window_and_stretched_rotation_batches_on_the_host() {
    let extra = first_layer_windowed()
        .into_iter()
        .chain([
            (
                "llama.rope.scaling.type".into(),
                Value::String("yarn".into()),
            ),
            ("llama.rope.scaling.factor".into(), Value::F32(4.0)),
            (
                "llama.rope.scaling.original_context_length".into(),
                Value::U32(8),
            ),
        ])
        .collect();
    let path = tiny_model::write_wide_sinks("cb-sinks-moe", LAYERS, true, extra);
    let model = tiny_model::load(&path, LAYERS);
    assert!(model.config.yarn.is_some());
    assert!(model
        .layers
        .iter()
        .all(|l| l.moe.is_some() && l.attn_sinks.is_some()));
    alone_and_batched_agree(&model);
    batched_follows_serial(&path);
    served_alone_and_beside_agree(&path);
    tiny_model::remove(&path);
}

/// The file of `tag` in the configured model store.
fn store_blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, tag) = tag.split_once(':').unwrap_or((tag, "latest"));
    let manager = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = manager
        .read_manifest_json(name, tag)
        .expect("manifest in the store");
    let digest = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .expect("a model layer")
        .replace(':', "-");
    store.join("blobs").join(digest)
}

/// A real mixture with sinks, windowed layers, biases and a stretched rotation, whole on the
/// host: a sequence keeps its tokens inside a batch, and the batch follows the serial forward.
///
/// Ignored by default: it needs the model in the store and its weights in host memory.
#[test]
#[ignore]
fn a_real_sink_mixture_decodes_to_the_same_tokens_alone_and_inside_a_batch() {
    let path = store_blob("gpt-oss:20b");
    let layers = {
        let g = crate::tensor::quantized::gguf_file::open_mapped(&path).unwrap();
        let arch = g.content.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        GenericTransformerConfig::from_gguf(&g.content, &arch)
            .unwrap()
            .n_layers
    };
    let model = tiny_model::load(&path, layers);
    assert!(model.layers.iter().all(|l| l.attn_sinks.is_some()));
    alone_and_batched_agree(&model);
    let prompts = prompts();
    let together = run_batched(&model, &prompts, &[0, 1, 2, 3]);
    drop(model);
    // The serial forward chooses the tokens the batch chose, step by step.
    for (s, (prompt, (tokens, logits))) in prompts.iter().zip(&together).enumerate() {
        let mut serial = tiny_model::load(&path, layers);
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut want = vec![flat(serial.forward(&ids(prompt), 0).unwrap())];
        for (i, &t) in tokens[..STEPS].iter().enumerate() {
            want.push(flat(serial.forward(&ids(&[t]), prompt.len() + i).unwrap()));
        }
        let shares: Vec<String> = want
            .iter()
            .zip(logits)
            .map(|(w, g)| format!("{:.2e}", share(w, g)))
            .collect();
        eprintln!("sequence {s}: serial against batched by step {shares:?}");
        let chosen: Vec<u32> = want.iter().map(|w| argmax(w)).collect();
        assert_eq!(
            &chosen, tokens,
            "sequence {s}: the serial forward chose other tokens"
        );
    }
}

/// A model moved into the continuous-batch server still names the format its projections run
/// in, and names the same one once moved back out.
#[test]
fn a_batched_model_names_the_format_it_runs_in() {
    use crate::inference::engine::model_backend::{
        BoxedModelBackend, ContinuousBackend, ModelBackend,
    };
    let path = tiny_model::write("cb-format", LAYERS);
    let model = tiny_model::load(&path, LAYERS);
    let format = model.running_format();
    assert!(format.is_some(), "the serial model names no format");
    let server =
        ContinuousServer::spawn(Arc::new(model), Vec::new(), 64, CB_BLOCK_SIZE, 8, 4096).unwrap();
    let batched: BoxedModelBackend = Box::new(ContinuousBackend(Arc::new(server), format.clone()));
    assert_eq!(batched.running_format(), format);
    let Ok(serial) = batched.into_serial() else {
        panic!("the model did not leave its worker");
    };
    assert_eq!(serial.running_format(), format);
    tiny_model::remove(&path);
}

/// The variants a card batches: each sequence decodes to the tokens it gives alone in the
/// batched paths and in the server, every layer on card 0.
#[cfg(feature = "gpu")]
mod on_a_card {
    use super::*;

    /// Whether card 0 opens. A run without a card says so on its error stream.
    fn card() -> bool {
        let present = Device::new_cuda(0).is_ok();
        if !present {
            eprintln!("no card: the card variants checked nothing");
        }
        present
    }

    /// Each sequence decodes to the tokens it gives alone, in the batched paths and in the
    /// server, with the model at `path` on card 0.
    fn alone_and_batched_agree_on_a_card(path: &std::path::Path) {
        let model = tiny_model::load_on_card(path, LAYERS);
        assert!(!model.compute_device().is_cpu());
        assert!(model.cb_eligible(), "{:?}", model.cb_unsupported_reason());
        let prompts = prompts();
        let together = run_batched(&model, &prompts, &[0, 1, 2, 3]);
        let mut worst = 0f32;
        for (s, beside) in together.iter().enumerate() {
            let alone = run_batched(&model, &prompts, &[s]);
            assert_eq!(
                alone[0].0, beside.0,
                "sequence {s} decoded other tokens inside the batch"
            );
            for (a, b) in alone[0].1.iter().zip(&beside.1) {
                worst = worst.max(share(a, b));
            }
        }
        eprintln!(
            "alone against batched on a card: largest logit difference {worst:.3e} of the range"
        );
        drop(model);
        served_alone_and_beside_agree_on(&|| tiny_model::load_on_card(path, LAYERS));
    }

    /// The quantised toy written under `tag` with `extras`.
    fn quant(tag: &str, extras: tiny_model::QuantExtras) -> std::path::PathBuf {
        tiny_model::write_quant_with(tag, LAYERS, extras)
    }

    /// The header keys of a stretched rotation of the quantised toy.
    fn stretched() -> Vec<(String, Value)> {
        vec![
            (
                "qwen3.rope.scaling.type".into(),
                Value::String("yarn".into()),
            ),
            ("qwen3.rope.scaling.factor".into(), Value::F32(4.0)),
            (
                "qwen3.rope.scaling.original_context_length".into(),
                Value::U32(8),
            ),
        ]
    }

    /// Keys a windowed layer of the quantised toy reads.
    const WINDOW: usize = 4;

    /// The header keys that window the first layer of the quantised toy to its last
    /// [`WINDOW`] keys.
    fn windowed() -> Vec<(String, Value)> {
        vec![
            (
                "qwen3.attention.sliding_window".into(),
                Value::U32(WINDOW as u32),
            ),
            (
                "qwen3.attention.sliding_window_pattern".into(),
                Value::Array(vec![Value::U32(1), Value::U32(0)]),
            ),
        ]
    }

    /// The header keys that window every layer of the quantised toy to its last [`WINDOW`]
    /// keys.
    fn windowed_everywhere() -> Vec<(String, Value)> {
        vec![
            (
                "qwen3.attention.sliding_window".into(),
                Value::U32(WINDOW as u32),
            ),
            (
                "qwen3.attention.sliding_window_pattern".into(),
                Value::Array(vec![Value::U32(1); LAYERS]),
            ),
        ]
    }

    #[test]
    fn a_stretched_rotation_batches_on_a_card() {
        if !card() {
            return;
        }
        let keys = stretched();
        let path = quant(
            "cb-card-yarn",
            tiny_model::QuantExtras {
                keys,
                ..Default::default()
            },
        );
        alone_and_batched_agree_on_a_card(&path);
        tiny_model::remove(&path);
    }

    #[test]
    fn a_partial_rotation_batches_on_a_card() {
        if !card() {
            return;
        }
        let rotated = tiny_model::QUANT_EMBD / tiny_model::QUANT_HEADS / 2;
        let keys = vec![(
            "qwen3.rope.dimension_count".into(),
            Value::U32(rotated as u32),
        )];
        let path = quant(
            "cb-card-partial",
            tiny_model::QuantExtras {
                keys,
                ..Default::default()
            },
        );
        alone_and_batched_agree_on_a_card(&path);
        tiny_model::remove(&path);
    }

    #[test]
    fn attention_sinks_and_a_window_batch_on_a_card() {
        if !card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            keys: windowed(),
            sinks: true,
            ..Default::default()
        };
        let path = quant("cb-card-sinks", extras);
        alone_and_batched_agree_on_a_card(&path);
        tiny_model::remove(&path);
    }

    /// A layer without sinks windowed to its last [`WINDOW`] keys reads those keys alone in
    /// the batched paths and in the server on a card, as the serial forward does.
    ///
    /// Both layers are windowed, so a row reads the tokens of its last `reach` positions.
    /// 2 prompts that differ only before those give the same logits from their last row on.
    #[test]
    fn a_window_without_sinks_batches_on_a_card() {
        if !card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            keys: windowed_everywhere(),
            ..Default::default()
        };
        let path = quant("cb-card-window", extras);
        let model = tiny_model::load_on_card(&path, LAYERS);
        assert!(model
            .layers
            .iter()
            .all(|l| l.sliding_window == Some(WINDOW) && l.attn_sinks.is_none()));
        assert!(model.cb_eligible(), "{:?}", model.cb_unsupported_reason());
        let reach = LAYERS * (WINDOW - 1) + 1;
        let vocab = tiny_model::QUANT_VOCAB as u32;
        let tail: Vec<u32> = (0..reach as u32).map(|i| (i * 5 + 2) % vocab).collect();
        let prompts: Vec<Vec<u32>> = [1u32, 9]
            .iter()
            .map(|&s| {
                (0..WINDOW as u32)
                    .map(|i| (i * 7 + s) % vocab)
                    .chain(tail.clone())
                    .collect()
            })
            .collect();
        assert_ne!(prompts[0], prompts[1]);
        let flat = |t: Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let serial: Vec<Vec<f32>> = prompts
            .iter()
            .map(|p| {
                flat(
                    tiny_model::load_on_card(&path, LAYERS)
                        .forward(&ids(p), 0)
                        .unwrap(),
                )
            })
            .collect();
        let serial_gap = share(&serial[0], &serial[1]);
        let runs = run_batched(&model, &prompts, &[0, 1]);
        drop(model);
        let batched_gap = runs[0]
            .1
            .iter()
            .zip(&runs[1].1)
            .map(|(a, b)| share(a, b))
            .fold(0f32, f32::max);
        eprintln!(
            "prompts differing before the reach: serial gap {serial_gap:.3e}, batched gap \
             {batched_gap:.3e}, batched against serial {:.3e} of the range",
            share(&serial[0], &runs[0].1[0])
        );
        assert!(
            serial_gap <= SHARE,
            "the serial forward read past its window"
        );
        assert_eq!(
            runs[0].0, runs[1].0,
            "the batched paths read past the window"
        );
        assert!(
            batched_gap <= SHARE,
            "the batched logits moved by {batched_gap} of their range past the window"
        );
        let model = Arc::new(tiny_model::load_on_card(&path, LAYERS));
        let server =
            ContinuousServer::spawn(model, Vec::new(), 64, CB_BLOCK_SIZE, 8, 4096).unwrap();
        let receivers: Vec<_> = prompts
            .iter()
            .map(|p| server.submit(p.clone(), STEPS))
            .collect();
        let served: Vec<Vec<u32>> = receivers
            .into_iter()
            .map(|rx| {
                rx.iter()
                    .take_while(|t| !matches!(t, CbToken::Done(_)))
                    .filter_map(|t| match t {
                        CbToken::Tok(t) => Some(t),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let _ = server.stop();
        assert_eq!(served[0].len(), STEPS);
        assert_eq!(served[0], served[1], "the server read past the window");
        tiny_model::remove(&path);
    }

    /// A prompt of a model with sinks whose attention scores take more than the card has free
    /// answers, and chooses the token it chooses on a card with room.
    ///
    /// The tight card runs the attention in smaller query tiles.
    /// Its logits differ from the roomy card's by the rounding of the half-precision products.
    #[test]
    fn a_sink_prompt_whose_scores_exceed_the_free_card_memory_answers() {
        use crate::inference::place::room::free_now;
        if !card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            sinks: true,
            ..Default::default()
        };
        let path = quant("cb-card-sinks-room", extras);
        let model = tiny_model::load_on_card(&path, LAYERS);
        let device = model.compute_device();
        let (n_kv, hd, n_layers, _) = model.paged_geometry();
        let len = 4096usize;
        let vocab = tiny_model::QUANT_VOCAB as u32;
        let prompt: Vec<u32> = (0..len as u32).map(|i| (i * 7 + 3) % vocab).collect();
        let slots: Vec<usize> = (0..len).collect();
        let prefill = || {
            let mut stores: Vec<PagedKvStore> = (0..n_layers)
                .map(|_| {
                    PagedKvStore::new(len.div_ceil(BLOCK), BLOCK, n_kv, hd, DType::F16, &device)
                        .unwrap()
                })
                .collect();
            model.paged_prefill_batch(&[(prompt.as_slice(), slots.as_slice())], &mut stores)
        };
        let want = rows_of(&prefill().unwrap());
        // The card keeps free half the scores the whole prompt holds at once.
        let scores = model.prompt_attention_bytes(len, len);
        crate::inference::engine::llm_engine::release_cuda_pools();
        let free = free_now(&device).unwrap();
        let filler = Tensor::zeros_on(
            (free.saturating_sub(scores / 2) as usize,),
            DType::U8,
            &device,
        )
        .unwrap();
        let left = free_now(&device).unwrap();
        assert!(
            left < scores,
            "the card kept {left} bytes free for scores of {scores}"
        );
        let got = prefill().expect("the prompt ran out of card memory");
        drop(filler);
        let got = rows_of(&got);
        let worst = share(&want[0], &got[0]);
        eprintln!("tight against roomy card: largest logit difference {worst:.3e} of the range");
        assert_eq!(
            argmax(&want[0]),
            argmax(&got[0]),
            "the tight card chose another token"
        );
        tiny_model::remove(&path);
    }

    #[test]
    fn a_mixture_batches_on_a_card() {
        if !card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            routed: true,
            ..Default::default()
        };
        let path = quant("cb-card-moe", extras);
        alone_and_batched_agree_on_a_card(&path);
        tiny_model::remove(&path);
    }

    /// A mixture with sinks, a windowed layer and a stretched rotation batches on a card.
    #[test]
    fn a_mixture_with_sinks_window_and_stretched_rotation_batches_on_a_card() {
        if !card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            keys: windowed().into_iter().chain(stretched()).collect(),
            sinks: true,
            routed: true,
        };
        let path = quant("cb-card-all", extras);
        alone_and_batched_agree_on_a_card(&path);
        tiny_model::remove(&path);
    }

    /// Whether card 1 opens. A run without it says so on its error stream.
    fn second_card() -> bool {
        let present = Device::new_cuda(1).is_ok();
        if !present {
            eprintln!("no second card: the split variant checked nothing");
        }
        present
    }

    /// A mixture with sinks whose first layer sits on card 1 and second layer on card 0 stays
    /// out of the batch worker.
    /// Served serially, each prompt gives the tokens it gives with every layer on card 0.
    #[test]
    fn a_model_whose_first_layer_sits_on_the_second_card_serves_serially() {
        use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
        if !card() || !second_card() {
            return;
        }
        let extras = tiny_model::QuantExtras {
            sinks: true,
            routed: true,
            ..Default::default()
        };
        let path = quant("cb-two-cards", extras);
        let split = tiny_model::load_on_cards(&path, LAYERS, &[(1, 0), (0, 1)]);
        let on = |l: usize| split.layers()[l].attn_norm.weight().device().location();
        assert_eq!(on(0), crate::tensor::DeviceLocation::Cuda { gpu_id: 1 });
        assert_eq!(on(1), crate::tensor::DeviceLocation::Cuda { gpu_id: 0 });
        let split = GenericBackend(split);
        assert!(
            !split.cb_eligible_gpu(),
            "the batch worker took a model whose layers sit on 2 cards"
        );
        let mut split = split.0;
        let mut whole = tiny_model::load_on_card(&path, LAYERS);
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut worst = 0f32;
        for prompt in prompts() {
            let mut want = flat(whole.forward(&ids(&prompt), 0).unwrap());
            let mut got = flat(split.forward(&ids(&prompt), 0).unwrap());
            for step in 0..STEPS {
                worst = worst.max(share(&want, &got));
                let (a, b) = (argmax(&want), argmax(&got));
                assert_eq!(a, b, "step {step} of the split model chose another token");
                let at = prompt.len() + step;
                want = flat(whole.forward(&ids(&[a]), at).unwrap());
                got = flat(split.forward(&ids(&[b]), at).unwrap());
            }
        }
        eprintln!("split against one card: largest logit difference {worst:.3e} of the range");
        assert!(
            worst <= SHARE,
            "the split logits moved by {worst} of their range"
        );
        tiny_model::remove(&path);
    }
}
