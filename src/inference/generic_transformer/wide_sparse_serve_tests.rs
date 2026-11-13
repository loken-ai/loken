//! The continuous-batch server over a Sinkhorn wide-residual model, on the synthetic model of the
//! frozen V4.1 oracle in both its band layouts, and on the released file.
//!
//! Each sequence served beside others gets the greedy tokens it gets run alone through the same
//! generic model.
//! Half the sequences start together, in pairs of the same length.
//! The other half join once the first sequence has streamed a few tokens.
//! The budgets differ, so sequences leave while others run.

use std::sync::mpsc::Receiver;

use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::cache::state_disk::StateStore;
use crate::inference::generic_transformer::wide_sparse_toy::VOCAB;
use crate::inference::serve::continuous_serve::{CbToken, ContinuousServer};

/// Sequences that decode together in the tests.
const MAX_RUNNING: usize = 8;

/// Prompt tokens 1 step prefills in the tests.
const MAX_PREFILL: usize = 4096;

/// Tokens the first sequence streams before the other half joins.
const JOIN_AFTER: usize = 2;

/// The numbers of sequences served at once.
const WIDTHS: [usize; 3] = [1, 4, 16];

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

/// `n` prompts of tokens below `vocab` and their budgets.
///
/// Prompts `2k` and `2k + 1` have the same length and other tokens.
fn workload(n: usize, vocab: u32) -> (Vec<Vec<u32>>, Vec<usize>) {
    let prompts = (0..n)
        .map(|s| {
            let len = 4 + (s / 2 * 5) % 9;
            let (step, start) = (2 * (s as u32 % 5) + 1, 3 * s as u32 + 1);
            (0..len as u32)
                .map(|i| (i * step + start) % vocab)
                .collect()
        })
        .collect();
    let budgets = (0..n).map(|s| 3 + (s * 5) % 7).collect();
    (prompts, budgets)
}

/// The `max_new` greedy tokens of `prompt` run alone on `model`.
fn alone(model: &mut GenericHeteroTransformer, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let mut logits = model.forward(&ids(prompt), 0).unwrap();
    let mut out = Vec::new();
    while out.len() < max_new {
        let t = greedy(&logits);
        out.push(t);
        if out.len() < max_new {
            logits = model
                .forward(&ids(&[t]), prompt.len() + out.len() - 1)
                .unwrap();
        }
    }
    out
}

/// What a stream gave: its tokens, the prompt tokens served from a cache, and whether it ended.
#[derive(Default)]
struct Stream {
    tokens: Vec<u32>,
    cached: usize,
    ended: bool,
}

/// Reads `rx` into `got` until it holds `until` tokens or the stream ends.
fn read(rx: &Receiver<CbToken>, got: &mut Stream, until: usize) {
    while !got.ended && got.tokens.len() < until {
        match rx.recv() {
            Ok(CbToken::Tok(t)) => got.tokens.push(t),
            Ok(CbToken::Admitted {
                cached_prompt_tokens,
                ..
            }) => got.cached = cached_prompt_tokens,
            Ok(CbToken::Done(_)) | Err(_) => got.ended = true,
        }
    }
}

/// The streams of the prompts served by 1 server over `model`, which is handed back.
///
/// The first half is submitted at once.
/// The second half is submitted once the first stream holds [`JOIN_AFTER`] tokens.
fn served(
    model: GenericHeteroTransformer,
    prompts: &[Vec<u32>],
    budgets: &[usize],
) -> (Vec<Stream>, GenericHeteroTransformer) {
    let context = prompts
        .iter()
        .zip(budgets)
        .map(|(p, m)| p.len() + m)
        .max()
        .unwrap();
    let server =
        ContinuousServer::spawn_sequences(model, Vec::new(), context, MAX_RUNNING, MAX_PREFILL)
            .map_err(|(_, e)| e)
            .unwrap();
    let n = prompts.len();
    let early = n.div_ceil(2);
    let submit = |s: usize| server.submit(prompts[s].clone(), budgets[s]);
    let mut rxs: Vec<Receiver<CbToken>> = (0..early).map(submit).collect();
    let mut streams: Vec<Stream> = (0..n).map(|_| Stream::default()).collect();
    read(&rxs[0], &mut streams[0], JOIN_AFTER);
    rxs.extend((early..n).map(submit));
    for (rx, stream) in rxs.iter().zip(streams.iter_mut()) {
        read(rx, stream, usize::MAX);
    }
    let model = server.stop().unwrap();
    (streams, model)
}

/// For each width, every sequence served gets the tokens it gets alone, on `model`.
fn every_width_serves_the_lone_tokens(mut model: GenericHeteroTransformer, what: &str) {
    for n in WIDTHS {
        let (prompts, budgets) = workload(n, VOCAB as u32);
        let want: Vec<Vec<u32>> = prompts
            .iter()
            .zip(&budgets)
            .map(|(p, &m)| alone(&mut model, p, m))
            .collect();
        let (got, back) = served(model, &prompts, &budgets);
        model = back;
        for (s, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(g.ended, "{what}: {n} sequences: sequence {s} did not end");
            assert_eq!(
                &g.tokens, w,
                "{what}: {n} sequences: sequence {s} was served other tokens"
            );
        }
    }
}

/// For both layouts on the host, 1, 4 and 16 sequences served together each get their lone
/// tokens, and the server hands the model back after each width.
#[test]
fn the_server_serves_each_sequence_its_lone_tokens() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let what = format!("host layout {ratios:?}");
        every_width_serves_the_lone_tokens(generic(ratios, kv_src, index_src), &what);
    }
}

/// For both layouts with every token mixer on card 0, 1, 4 and 16 sequences served together
/// each get the tokens they get alone on the same card model.
///
/// The pairs of the same length decode in the same steps, each over its own state.
#[cfg(feature = "gpu")]
#[test]
fn the_server_serves_each_sequence_its_lone_tokens_with_the_mixers_on_a_card() {
    if Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    for (ratios, kv_src, index_src) in LAYOUTS {
        let model = super::card_mixer_tests::generic_on_card(ratios, kv_src, index_src);
        assert!(model.mixers_on_card(), "layout {ratios:?}");
        let what = format!("card layout {ratios:?}");
        every_width_serves_the_lone_tokens(model, &what);
    }
}

/// For both layouts with every token mixer on card 0, 3 sequences batched give the tokens and
/// the logits bits of the same sequences run serially on the card model.
///
/// The first 2 stand at the same positions.
/// They decode alone together until the 3rd joins, so each step starts on the buffers the other
/// left.
/// Each finds the card's buffers built from its own state.
#[cfg(feature = "gpu")]
#[test]
fn sequences_batched_with_the_mixers_on_a_card_give_the_bits_of_their_serial_runs() {
    use super::wide_sparse_batch_tests::{batched, prompts, serial};
    if Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    for (ratios, kv_src, index_src) in LAYOUTS {
        let card = || super::card_mixer_tests::generic_on_card(ratios, kv_src, index_src);
        let p = prompts();
        let prompts = vec![p[0].clone(), p[2].clone(), p[1].clone()];
        assert_eq!(prompts[0].len(), prompts[1].len());
        let want = serial(&mut card(), &prompts);
        let got = batched(&mut card(), &prompts);
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

/// A model without latent or band layers is handed back unserved.
#[test]
fn a_model_without_sequence_state_is_handed_back() {
    let path = super::tiny_model::write("wide-sparse-serve-dense", 2);
    let model = super::tiny_model::load(&path, 2);
    let refused = ContinuousServer::spawn_sequences(model, Vec::new(), 64, 2, MAX_PREFILL);
    match refused {
        Ok(_) => panic!("a dense model was served per sequence"),
        Err((back, _)) => assert!(back.sequence_mixers().is_err()),
    }
    super::tiny_model::remove(&path);
}

/// Prompt rows the scheduler of the worker admits a step in the cut test.
const BUDGET: usize = 4;

/// Tokens of the prompt of the cut test: 2 budgets and a part.
const LONG_PROMPT: usize = 2 * BUDGET + BUDGET / 2;

/// Tokens the answer to the long prompt holds.
const ANSWER: usize = 4;

/// Milliseconds the prefill pauses after each report short of the whole prompt, and after each
/// layer of a forward short of its last.
const PAUSE_MS: u64 = 20;

/// The readings of a prompt of `total` rows read in 1 forward through `layers` layers.
///
/// The reading starts at 0.
/// Each layer short of the last adds its share of the rows.
/// The reading ends with the whole prompt.
fn one_forward(total: u64, layers: usize) -> Vec<(u64, u64)> {
    let mut readings = vec![(0, total)];
    for done in (1..layers).map(|l| total * l as u64 / layers as u64) {
        if readings.last() != Some(&(done, total)) {
            readings.push((done, total));
        }
    }
    readings.push((total, total));
    readings
}

/// Sets the pauses of the prefill and of the layers of a forward to `ms`.
fn pause(ms: u64) {
    use std::sync::atomic::Ordering;
    crate::request_metrics::PREFILL_PAUSE_MS.store(ms, Ordering::Relaxed);
    crate::request_metrics::LAYER_PAUSE_MS.store(ms, Ordering::Relaxed);
}

/// The tokens a server over `model` whose scheduler admits `budget` prompt rows a step gives
/// `prompt`, and each reading of the request's probe seen until its first token.
fn served_watched(
    model: GenericHeteroTransformer,
    prompt: &[u32],
    budget: usize,
) -> (Vec<u32>, Vec<(u64, u64)>) {
    use crate::inference::serve::continuous_batch::SamplingParams;
    use crate::request_metrics::RequestProbe;
    use std::sync::mpsc::RecvTimeoutError;
    let server =
        ContinuousServer::spawn_sequences(model, Vec::new(), prompt.len() + ANSWER, 2, budget)
            .map_err(|(_, e)| e)
            .unwrap();
    let probe = std::sync::Arc::new(RequestProbe::default());
    let rx = server.submit_watched(
        prompt.to_vec(),
        ANSWER,
        SamplingParams::greedy(),
        Some(std::sync::Arc::clone(&probe)),
    );
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let mut got = Stream::default();
    while got.tokens.is_empty() && !got.ended {
        match rx.recv_timeout(std::time::Duration::from_millis(1)) {
            Ok(CbToken::Tok(t)) => got.tokens.push(t),
            Ok(CbToken::Admitted { .. }) | Err(RecvTimeoutError::Timeout) => {}
            Ok(CbToken::Done(_)) | Err(RecvTimeoutError::Disconnected) => got.ended = true,
        }
        let reading = probe.prefill();
        if seen.last() != Some(&reading) {
            seen.push(reading);
        }
    }
    read(&rx, &mut got, usize::MAX);
    let _ = server.stop();
    (got.tokens, seen)
}

/// The model of layout `layout` for the cut test, with a fresh state cache when `cached`.
fn cut_model(layout: usize, cached: bool) -> GenericHeteroTransformer {
    let (ratios, kv_src, index_src) = LAYOUTS[layout];
    let mut model = generic(ratios, kv_src, index_src);
    if cached {
        let store = Arc::new(StateStore::in_memory(1 << 26, BLOCK));
        let identity = crate::inference::cache::state_disk::identity(&[b"cut states"]);
        assert!(model.attach_state_cache(store, identity), "{ratios:?}");
    }
    model
}

/// For both layouts on the host, with and without a state cache, a prompt longer than the
/// worker's prefill budget is read in the forwards the model's own prefill cuts it into.
///
/// The cut does not follow the budget.
/// The prompt gives the tokens it gives under a budget that holds it whole.
/// The request's probe hears that the prompt starts, then each layer of the forward, then that
/// the prompt is read whole.
#[test]
fn a_long_prompt_is_cut_by_the_model_and_not_by_the_budget() {
    let paused = crate::request_metrics::PREFILL_PAUSE_LOCK.blocking_lock();
    pause(PAUSE_MS);
    let prompt: Vec<u32> = (0..LONG_PROMPT as u32)
        .map(|i| (i * 3 + 1) % VOCAB as u32)
        .collect();
    let total = LONG_PROMPT as u64;
    let mut runs = Vec::new();
    for layout in 0..LAYOUTS.len() {
        for cached in [false, true] {
            let layers = cut_model(layout, cached).layers.len();
            let whole = served_watched(cut_model(layout, cached), &prompt, LONG_PROMPT);
            let budgeted = served_watched(cut_model(layout, cached), &prompt, BUDGET);
            runs.push((layout, cached, layers, whole, budgeted));
        }
    }
    pause(0);
    drop(paused);
    for (layout, cached, layers, whole, budgeted) in runs {
        let model_cut = one_forward(total, layers);
        let what = format!("layout {layout}, state cache {cached}");
        assert_eq!(whole.0.len(), ANSWER, "{what}: the whole read ended early");
        assert_eq!(
            budgeted.0, whole.0,
            "{what}: the budgeted read gave other tokens"
        );
        let of_prompt = |seen: &[(u64, u64)]| -> Vec<(u64, u64)> {
            seen.iter().copied().filter(|&(_, t)| t == total).collect()
        };
        assert_eq!(of_prompt(&whole.1), model_cut, "{what}: {:?}", whole.1);
        assert_eq!(
            of_prompt(&budgeted.1),
            model_cut,
            "{what}: {:?}",
            budgeted.1
        );
    }
}

/// For both layouts on the host, a prompt read in 1 forward on the serial path reports each
/// layer of the forward.
///
/// A reading strictly between 0 and the whole prompt is seen before the prompt is read whole.
/// The readings never go back.
#[test]
fn a_slowed_serial_forward_reports_its_layers() {
    use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
    use crate::request_metrics::RequestProbe;
    use std::sync::atomic::{AtomicBool, Ordering};
    let paused = crate::request_metrics::PREFILL_PAUSE_LOCK.blocking_lock();
    pause(PAUSE_MS);
    let prompt: Vec<u32> = (0..LONG_PROMPT as u32)
        .map(|i| (i * 3 + 1) % VOCAB as u32)
        .collect();
    let total = LONG_PROMPT as u64;
    let mut runs = Vec::new();
    for layout in 0..LAYOUTS.len() {
        let mut backend = GenericBackend(cut_model(layout, false));
        let probe = Arc::new(RequestProbe::default());
        let read = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (probe, read) = (Arc::clone(&probe), Arc::clone(&read));
            std::thread::spawn(move || {
                let mut seen: Vec<(u64, u64)> = Vec::new();
                while !read.load(Ordering::Relaxed) {
                    let reading = probe.prefill();
                    if seen.last() != Some(&reading) {
                        seen.push(reading);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                seen
            })
        };
        {
            let _bound = crate::request_metrics::bind_prefill(Some(Arc::clone(&probe)));
            backend.forward_prefill_chunked(&ids(&prompt), 0).unwrap();
        }
        read.store(true, Ordering::Relaxed);
        runs.push((layout, watcher.join().unwrap(), probe.prefill()));
    }
    pause(0);
    drop(paused);
    for (layout, seen, last) in runs {
        assert_eq!(last, (total, total), "layout {layout}");
        assert!(
            seen.iter()
                .any(|&(done, t)| t == total && done > 0 && done < total),
            "layout {layout}: {seen:?}"
        );
        assert!(
            seen.windows(2).all(|w| w[0].0 <= w[1].0),
            "layout {layout}: {seen:?}"
        );
    }
}

/// Tokens per block of the stored states, as the V4.1 batch worker test cuts them.
const BLOCK: usize = 8;

/// Polls `store` until it holds a state of `len` tokens.
fn wait_for_len(store: &StateStore, len: usize) {
    for _ in 0..600 {
        if store.lengths().contains(&len) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("no {len}-token state was stored: {:?}", store.lengths());
}

/// Port of the V4.1 `the_batch_worker_restores_a_stored_prefix_and_streams_the_same_tokens`,
/// same prompts and budget.
///
/// A server over a model with a state cache serves a first prompt of 24 tokens.
/// It keeps the prompt's end state and the state the finished sequence stands in.
/// The engine feeds a sequence all its tokens but the last, so that state stands after 27.
/// A second prompt of 40 tokens that extends the first is restored at 24.
/// Both stream the tokens a server without a cache streams.
#[test]
fn the_server_restores_a_stored_prefix_and_streams_the_same_tokens() {
    let tokens: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let (first, longer) = (tokens[..24].to_vec(), tokens.clone());
    let budget = 4;
    for (ratios, kv_src, index_src) in LAYOUTS {
        let store = Arc::new(StateStore::in_memory(1 << 26, BLOCK));
        let identity = crate::inference::cache::state_disk::identity(&[b"served states"]);
        let mut model = generic(ratios, kv_src, index_src);
        assert!(
            model.attach_state_cache(store.clone(), identity),
            "{ratios:?}"
        );

        let (got, model) = served(model, std::slice::from_ref(&first), &[budget]);
        let first_run = &got[0];
        assert_eq!(first_run.cached, 0, "{ratios:?}: nothing was stored yet");
        wait_for_len(&store, first.len());
        wait_for_len(&store, first.len() + budget - 1);
        let (got, _) = served(model, std::slice::from_ref(&longer), &[budget]);
        let longer_run = &got[0];
        assert_eq!(longer_run.cached, first.len(), "{ratios:?}: the restore");

        let plain = generic(ratios, kv_src, index_src);
        let (want, plain) = served(plain, std::slice::from_ref(&first), &[budget]);
        assert_eq!(
            first_run.tokens, want[0].tokens,
            "{ratios:?}: the first prompt"
        );
        let (want, _) = served(plain, std::slice::from_ref(&longer), &[budget]);
        assert_eq!(
            longer_run.tokens, want[0].tokens,
            "{ratios:?}: the restored sequence streams the same tokens"
        );
    }
}

/// Tokens of the prompt the serial path reads in the serial cut tests.
///
/// It is longer than the chunk the serial path takes for a model whose forwards read no routed
/// expert from the host, when no card is measured.
const SERIAL_PROMPT: usize = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * BLOCK;

/// The greedy token the serial path gives `prompt` on `model`, and each reading of the probe
/// seen until the prompt is read.
///
/// A model with a state cache reads the prompt through the restore of a stored prefix.
/// A model without one reads it through the chunked prefill.
fn serial_watched(model: GenericHeteroTransformer, prompt: &[u32]) -> (u32, Vec<(u64, u64)>) {
    use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
    use crate::request_metrics::RequestProbe;
    use std::sync::atomic::{AtomicBool, Ordering};
    let cached = model.state_cache().is_some();
    let mut backend = GenericBackend(model);
    let probe = Arc::new(RequestProbe::default());
    let read = Arc::new(AtomicBool::new(false));
    let watcher = {
        let (probe, read) = (Arc::clone(&probe), Arc::clone(&read));
        std::thread::spawn(move || {
            let mut seen: Vec<(u64, u64)> = Vec::new();
            while !read.load(Ordering::Relaxed) {
                let reading = probe.prefill();
                if seen.last() != Some(&reading) {
                    seen.push(reading);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            seen
        })
    };
    let logits = {
        let _bound = crate::request_metrics::bind_prefill(Some(Arc::clone(&probe)));
        match cached {
            true => backend.try_restore_prefix(prompt).unwrap(),
            false => Some(backend.forward_prefill_chunked(&ids(prompt), 0).unwrap()),
        }
    };
    read.store(true, Ordering::Relaxed);
    let logits = logits.expect("a model with a state cache reads the prompt");
    // The reading of the whole prompt comes without a pause, and may come after the watcher.
    let mut seen = watcher.join().unwrap();
    let last = probe.prefill();
    if seen.last() != Some(&last) {
        seen.push(last);
    }
    (greedy(&logits), seen)
}

/// For both layouts on the host, with and without a state cache, the serial path cuts a long
/// prompt where the batch worker cuts it, for a model whose forwards read routed experts that
/// no card keeps.
///
/// The probe sees the same readings on both paths.
/// Both read the prompt in 1 forward and give the same first token.
#[test]
fn the_serial_path_cuts_a_long_prompt_as_the_worker_does() {
    let paused = crate::request_metrics::PREFILL_PAUSE_LOCK.blocking_lock();
    crate::inference::engine::llm_engine::reset_adaptive_prefill_chunk();
    pause(PAUSE_MS);
    let prompt: Vec<u32> = (0..SERIAL_PROMPT as u32)
        .map(|i| (i * 5 + 2) % VOCAB as u32)
        .collect();
    let total = SERIAL_PROMPT as u64;
    let mut runs = Vec::new();
    for layout in 0..LAYOUTS.len() {
        for cached in [false, true] {
            let model = cut_model(layout, cached);
            assert!(model.wide_forwards_read_experts(), "layout {layout}");
            let layers = model.layers.len();
            let serial = serial_watched(model, &prompt);
            let worker = served_watched(cut_model(layout, cached), &prompt, BUDGET);
            runs.push((layout, cached, layers, serial, worker));
        }
    }
    pause(0);
    drop(paused);
    let of_prompt = |seen: &[(u64, u64)]| -> Vec<(u64, u64)> {
        seen.iter().copied().filter(|&(_, t)| t == total).collect()
    };
    for (layout, cached, layers, serial, worker) in runs {
        let what = format!("layout {layout}, state cache {cached}");
        assert_eq!(
            of_prompt(&serial.1),
            of_prompt(&worker.1),
            "{what}: the serial cut"
        );
        assert_eq!(of_prompt(&serial.1), one_forward(total, layers), "{what}");
        assert_eq!(Some(&serial.0), worker.0.first(), "{what}");
    }
}

/// For both layouts on the host, with and without a state cache, a serial read whose forward
/// does not fit the card is read again in forwards of half the rows.
///
/// It gives the first token of a read that fits.
#[test]
fn a_serial_forward_that_does_not_fit_is_read_in_halves() {
    use crate::inference::engine::llm_engine::ADAPTIVE_PREFILL_CHUNK;
    use crate::inference::engine::model_backend::FORCED_CHUNK_ROWS;
    use std::sync::atomic::Ordering;
    let paused = crate::request_metrics::PREFILL_PAUSE_LOCK.blocking_lock();
    let prompt: Vec<u32> = (0..SERIAL_PROMPT as u32)
        .map(|i| (i * 5 + 2) % VOCAB as u32)
        .collect();
    for (layout, cached) in (0..LAYOUTS.len()).flat_map(|l| [(l, false), (l, true)]) {
        let what = format!("layout {layout}, state cache {cached}");
        crate::inference::engine::llm_engine::reset_adaptive_prefill_chunk();
        let (fits, _) = serial_watched(cut_model(layout, cached), &prompt);
        FORCED_CHUNK_ROWS.with(|c| c.set(SERIAL_PROMPT - 1));
        let (halved, _) = serial_watched(cut_model(layout, cached), &prompt);
        FORCED_CHUNK_ROWS.with(|c| c.set(usize::MAX));
        let used = ADAPTIVE_PREFILL_CHUNK.load(Ordering::Relaxed);
        crate::inference::engine::llm_engine::reset_adaptive_prefill_chunk();
        assert_eq!(used, SERIAL_PROMPT / 2, "{what}: the rung");
        assert_eq!(halved, fits, "{what}");
    }
    drop(paused);
}

/// On the released file on the host, 1, 4 and 16 sequences served together each get the
/// greedy tokens they get alone.
///
/// The prompts are the test prompt's tokens turned by the sequence's index, of other lengths.
/// Run with
///   cargo test --release --no-default-features --features cpu --lib -- --ignored wide_sparse_real_served --nocapture
#[test]
#[ignore]
fn wide_sparse_real_served_sequences_get_their_lone_tokens() {
    use super::wide_sparse_real_tests::{blob, prompt_ids, resident_mb, TAG};
    use crate::tensor::quantized::gguf_source::SplitGguf;
    let path = blob(TAG);
    let split = SplitGguf::open(&path).expect("the file opens");
    let content = split.merged_content();
    let base = prompt_ids(&content);
    let arch = content.metadata["general.architecture"]
        .to_string()
        .unwrap()
        .clone();
    let layers = content
        .metadata
        .get(&format!("{arch}.block_count"))
        .and_then(|v| v.to_u32().ok())
        .expect("a block count") as usize;
    let plan = HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0);
    let mut model = GenericHeteroTransformer::from_split_gguf(
        &split,
        &HashMap::new(),
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    )
    .expect("the generic loader accepts the file");
    println!("loaded, resident {} MB", resident_mb());
    // The begin token stays first, the rest turns by the sequence's index.
    let (head, rest) = base.split_at(1);
    for n in WIDTHS {
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|s| {
                let mut p = head.to_vec();
                let len = rest.len() - s / 2 % rest.len().min(3);
                p.extend(rest.iter().cycle().skip(s).take(len));
                p
            })
            .collect();
        let budgets: Vec<usize> = (0..n).map(|s| 2 + s % 3).collect();
        let started = std::time::Instant::now();
        let want: Vec<Vec<u32>> = prompts
            .iter()
            .zip(&budgets)
            .map(|(p, &m)| alone(&mut model, p, m))
            .collect();
        println!(
            "{n} sequences alone in {:.1}s",
            started.elapsed().as_secs_f32()
        );
        let started = std::time::Instant::now();
        let (got, back) = served(model, &prompts, &budgets);
        model = back;
        println!(
            "{n} sequences served in {:.1}s, resident {} MB",
            started.elapsed().as_secs_f32(),
            resident_mb()
        );
        for (s, (w, g)) in want.iter().zip(&got).enumerate() {
            println!(
                "{n} sequences: sequence {s} alone {w:?} served {:?}",
                g.tokens
            );
            assert!(g.ended, "{n} sequences: sequence {s} did not end");
            assert_eq!(
                &g.tokens, w,
                "{n} sequences: sequence {s} was served other tokens"
            );
        }
    }
}
