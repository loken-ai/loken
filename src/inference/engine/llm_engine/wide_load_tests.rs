//! A Sinkhorn wide-residual file loaded by the engine, as the server loads it, on the small file
//! of the frozen V4.1 oracle.
//!
//! The engine finds the file through its models path, parses it, places it through the plan and
//! the ladder, and wraps the model in the generic backend.
//! The answers are compared with the generic model built from the file directly, and with the
//! records of the V4.1 backend the oracle tests proved equal to it.

use super::block_draft_tests::{engine_of, params, run, tokenizer};
use super::*;
use crate::inference::generic_transformer::block_draft_tests::{
    draft_config, draft_entries, generic_of, prompt, toy_files,
};
use crate::inference::generic_transformer::wide_sparse_frozen as frozen;
use crate::inference::generic_transformer::wide_sparse_names::ARCH;
use crate::inference::generic_transformer::wide_sparse_toy::{
    real_entries_for, real_md_layout, VOCAB,
};
use crate::tensor::gguf_write::write_gguf_with_metadata;
use crate::tensor::quantized::gguf_file::Value;

/// The oracle's band layouts: compress ratios, kv source layers, index source layers.
const LAYOUTS: [(&[u32], &[u32], &[u32]); 2] =
    [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];

/// Tokens each request generates.
const NEW: usize = 20;

/// The small file of `layout` on disk beside its tokenizer, with the draft the V4.1 writer
/// keeps when `with_draft`.
///
/// Returns the model file's path. The caller removes its directory.
fn toy_on_disk((ratios, kv_src, index_src): (&[u32], &[u32], &[u32]), with_draft: bool) -> PathBuf {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut md = real_md_layout(ratios, kv_src, index_src);
    let dim = md
        .iter()
        .find(|(k, _)| k.ends_with(".embedding_length"))
        .map(|(_, v)| v.to_u32().unwrap() as usize)
        .unwrap();
    let mut entries = real_entries_for(&bands);
    if with_draft {
        let c = draft_config(dim);
        md.extend(c.to_metadata(ARCH));
        entries.extend(draft_entries(&c));
    }
    // A token no request produces ends nothing early.
    md.push((
        "tokenizer.ggml.eos_token_id".to_string(),
        Value::U32(VOCAB as u32 + 1),
    ));
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("loken-wide-load-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    write_gguf_with_metadata(&model, &md, &entries).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    model
}

/// The prompt as the words of the small tokenizer.
fn words() -> String {
    prompt()
        .iter()
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// An engine that has loaded `model` through its models path.
async fn loaded(model: &std::path::Path, config: InferenceConfig) -> LlmEngine {
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: "toy".to_string(),
        models_dir: Some(model.to_path_buf()),
        context_length: 128,
        ..config
    });
    engine
        .load_model()
        .await
        .expect("the engine loads the file");
    engine
}

/// Whether the engine's model is the generic backend holding a wide residual, and whether that
/// model carries a block draft.
async fn is_wide_generic(engine: &LlmEngine) -> (bool, bool) {
    let mut state = engine.model_state.lock().await;
    let backend = std::mem::replace(
        &mut state.as_mut().expect("a loaded model").model,
        Box::new(crate::inference::engine::model_backend::TakenBackend),
    );
    match backend.take_generic() {
        Ok(model) => {
            let wide = model.sequence_mixers().is_ok();
            let draft = model.has_block_draft();
            state.as_mut().unwrap().model = Box::new(
                crate::inference::engine::model_backend::GenericBackend(model),
            );
            (wide, draft)
        }
        Err(other) => {
            state.as_mut().unwrap().model = other;
            (false, false)
        }
    }
}

/// An engine that loads the file answers with the tokens and the text of the generic model
/// built from the same file, and of the V4.1 backend as recorded: with and without the draft the
/// file carries, on both band layouts.
#[tokio::test(flavor = "multi_thread")]
async fn a_wide_file_loaded_by_the_engine_answers_as_the_oracle_models() {
    for layout in LAYOUTS {
        for with_draft in [false, true] {
            let label = format!("layout {:?}, draft {with_draft}", layout.0);
            let path = toy_on_disk(layout, with_draft);
            let engine = loaded(&path, InferenceConfig::default()).await;
            assert_eq!(
                is_wide_generic(&engine).await,
                (true, with_draft),
                "{label}: the generic arm took the file"
            );
            let got = run(&engine, &words(), NEW).await;

            let (direct, prompt) = engine_of(generic_of(toy_files(layout, with_draft).0)).await;
            let want = run(&direct, &prompt, NEW).await;
            assert!(got.tokens.len() > NEW, "{label}: the answer has its tokens");
            assert_eq!(
                got.tokens, want.tokens,
                "{label}: tokens of the direct model"
            );
            assert_eq!(got.text, want.text, "{label}: text of the direct model");

            frozen::expect_debug("engine", &format!("{label} tokens"), &got.tokens);
            frozen::expect_debug("engine", &format!("{label} text"), &got.text);

            // A second request and a streamed one give the same answer again.
            assert_eq!(
                run(&engine, &words(), NEW).await.tokens,
                want.tokens,
                "{label}"
            );
            let mut rx = engine.generate_stream(&words(), params(NEW)).await.unwrap();
            let mut streamed = String::new();
            while let Some(chunk) = rx.recv().await {
                streamed.push_str(&chunk.unwrap());
            }
            assert_eq!(streamed, want.text, "{label}: the streamed text");
            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }
    }
}

/// An engine that loads a wide file whose prior lies beside it gives the cluster the cost shape
/// and the placement inputs the V4.1 backend of the file gave, and says its experts run on the
/// host.
#[tokio::test(flavor = "multi_thread")]
async fn a_loaded_wide_file_gives_the_cluster_the_inputs_of_the_v41_backend() {
    use crate::inference::generic_transformer::wide_sparse_routing_tests::{turning, ROUTED};
    use crate::inference::generic_transformer::wide_sparse_sibling as sibling;
    for layout in LAYOUTS {
        let label = format!("layout {:?}", layout.0);
        let path = toy_on_disk(layout, false);
        sibling::write_prior(
            &path.with_file_name("side"),
            &turning(layout.0.len(), ROUTED as u32),
        )
        .unwrap();
        let engine = loaded(&path, InferenceConfig::default()).await;
        assert_eq!(is_wide_generic(&engine).await, (true, false), "{label}");
        let shape = engine.expert_cost_shape().await;
        assert!(
            shape.is_some(),
            "{label}: a clustered node prices a missed expert"
        );
        frozen::expect_debug("engine", &format!("{label} cost shape"), &shape);
        let inputs = engine.placement_model_inputs().await;
        assert!(
            inputs.is_some(),
            "{label}: the prior beside the file sizes the working set"
        );
        frozen::expect_debug("engine", &format!("{label} placement inputs"), &inputs);
        let (host, cards) = engine.layers_on_host_and_cards().await.unwrap();
        assert!(host, "{label}: the experts are read on the host");
        if !cfg!(feature = "gpu") {
            assert!(
                !cards,
                "{label}: a build without cards holds nothing on one"
            );
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}

/// A quality level other than exact loads and calibrates through the engine: a model on 1
/// device kind has no second run to measure its noise against, so routing stays exact and the
/// answer is the exact one.
#[tokio::test(flavor = "multi_thread")]
async fn a_quality_level_on_a_wide_file_keeps_routing_exact_without_a_second_run() {
    let layout = LAYOUTS[0];
    let path = toy_on_disk(layout, false);
    let exact = loaded(&path, InferenceConfig::default()).await;
    let want = run(&exact, &words(), NEW).await;
    for level in [QualityLevel::Balanced, QualityLevel::Fast] {
        let engine = loaded(
            &path,
            InferenceConfig {
                quality: level,
                ..InferenceConfig::default()
            },
        )
        .await;
        let got = run(&engine, &words(), NEW).await;
        assert_eq!(got.tokens, want.tokens, "{level:?}");
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// With a state cache open, the engine stores the states of a prompt under the file and a longer
/// prompt that extends it restores them: the answer is the one of an engine without a cache.
#[tokio::test(flavor = "multi_thread")]
async fn the_state_cache_of_a_loaded_wide_file_restores_a_prefix_and_keeps_the_answer() {
    let layout = LAYOUTS[0];
    let path = toy_on_disk(layout, false);
    let long = format!("{} t5 t9 t13 t17", words());
    let plain = loaded(&path, InferenceConfig::default()).await;
    let want = run(&plain, &long, NEW).await;

    let store = std::sync::Arc::new(crate::inference::cache::state_disk::StateStore::in_memory(
        1 << 26,
        8,
    ));
    let mut engine = LlmEngine::with_config(InferenceConfig {
        model_id: "toy".to_string(),
        models_dir: Some(path.clone()),
        context_length: 128,
        ..InferenceConfig::default()
    });
    engine.state_cache = Some(store.clone());
    engine
        .load_model()
        .await
        .expect("the engine loads the file");
    assert!(
        store.lengths().is_empty(),
        "nothing stored before a request"
    );
    run(&engine, &words(), NEW).await;
    assert!(
        !store.lengths().is_empty(),
        "the first prompt's states were stored"
    );
    let got = run(&engine, &long, NEW).await;
    assert_eq!(got.tokens, want.tokens, "the answer after a restore");
    assert_eq!(got.text, want.text);
    assert!(got.cached_prompt_tokens > 0 || !store.lengths().is_empty());
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// The cap on sequences is the host's budget divided by the state 1 sequence holds at the
/// window, and nothing below 2.
#[tokio::test(flavor = "multi_thread")]
async fn the_sequence_cap_follows_the_state_a_sequence_holds() {
    let model = generic_of(toy_files(LAYOUTS[1], false).0);
    let per_sequence = model.sequence_state_bytes(128).unwrap();
    assert!(per_sequence > 0);
    assert!(model.sequence_state_bytes(256).unwrap() >= per_sequence);
    assert_eq!(
        super::load::sequence_cap(&model, 128, per_sequence * 5 + 1),
        Some(5)
    );
    assert_eq!(
        super::load::sequence_cap(&model, 128, per_sequence * 2),
        Some(2)
    );
    assert_eq!(
        super::load::sequence_cap(&model, 128, per_sequence * 2 - 1),
        None
    );
    assert_eq!(super::load::sequence_cap(&model, 128, 0), None);
}

/// Every file under `dir`, at any depth.
fn files_under(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Port of `a_damaged_snapshot_is_ignored_and_the_files_hold_no_tokens` to the engine.
///
/// A store on disk holds the states of a prompt a first engine served.
/// With every file of it truncated, emptied or flipped, a second engine opening the directory
/// answers a longer prompt with the tokens of an engine that has no store: a damaged state is
/// passed over and never answers wrongly.
/// No file of the store holds a run of the prompt's ids as written.
#[tokio::test(flavor = "multi_thread")]
async fn a_damaged_state_store_is_ignored_and_its_files_hold_no_tokens() {
    let layout = LAYOUTS[0];
    let path = toy_on_disk(layout, false);
    let long = format!("{} t5 t9 t13 t17", words());
    let plain = loaded(&path, InferenceConfig::default()).await;
    let want = run(&plain, &long, NEW).await;

    let store_dir = path.parent().unwrap().join("states");
    let config = || InferenceConfig {
        state_cache_dir: Some(store_dir.to_string_lossy().into_owned()),
        state_cache_budget_gb: 0.05,
        ..InferenceConfig::default()
    };
    let first = loaded(&path, config()).await;
    run(&first, &words(), NEW).await;
    // The store writes behind the request.
    let started = std::time::Instant::now();
    while files_under(&store_dir).is_empty() && started.elapsed().as_secs() < 30 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    drop(first);
    let originals: Vec<(PathBuf, Vec<u8>)> = files_under(&store_dir)
        .into_iter()
        .map(|p| {
            let bytes = std::fs::read(&p).unwrap();
            (p, bytes)
        })
        .collect();
    assert!(
        !originals.is_empty(),
        "the first prompt's states reached the disk"
    );
    // The ids of the first prompt as 4-byte values, a run of 6.
    for run_of_ids in prompt().windows(6) {
        let needle: Vec<u8> = run_of_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        for (p, bytes) in &originals {
            assert!(
                !bytes.windows(needle.len()).any(|w| w == &needle[..]),
                "{} holds a run of the prompt's ids",
                p.display()
            );
        }
    }
    type Damage = (&'static str, fn(&[u8]) -> Vec<u8>);
    let damages: [Damage; 3] = [
        ("truncated", |b| b[..b.len() / 2].to_vec()),
        ("emptied", |_| Vec::new()),
        ("flipped", |b| {
            let mut v = b.to_vec();
            let mid = v.len() / 2;
            v[mid] ^= 0x01;
            v
        }),
    ];
    for (name, damage) in damages {
        for (p, bytes) in &originals {
            std::fs::write(p, damage(bytes)).unwrap();
        }
        let second = loaded(&path, config()).await;
        let got = run(&second, &long, NEW).await;
        assert_eq!(got.tokens, want.tokens, "{name}: answered wrongly");
        assert_eq!(got.text, want.text, "{name}");
        drop(second);
        for (p, bytes) in &originals {
            let _ = std::fs::write(p, bytes);
        }
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// A block-scaled checkpoint directory with engram layers, found in a hub snapshot layout by
/// its repository id and loaded by the engine, answers with the recorded tokens of the V4.1
/// checkpoint path loaded through the same engine.
#[tokio::test(flavor = "multi_thread")]
async fn a_wide_checkpoint_loaded_by_the_engine_answers_as_the_v41_checkpoint_path() {
    use crate::inference::generic_transformer::block_scaled_engram_tests::{
        engram_checkpoint, tokenizer as checkpoint_tokenizer,
    };
    use crate::inference::generic_transformer::block_scaled_oracle_tests::tokens;

    let made = engram_checkpoint("engine");
    let root = std::env::temp_dir().join(format!("loken-wide-hub-{}", std::process::id()));
    let snapshot = root.join("models--org--toy").join("snapshots").join("s1");
    std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
    std::fs::rename(&made, &snapshot).unwrap();
    // The checkpoint's tokenizer lists its begin and end tokens as added tokens, where the
    // loader looks their ids up.
    let listing = snapshot.join("tokenizer.json");
    let mut tokenizer_file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&listing).unwrap()).unwrap();
    tokenizer_file["added_tokens"] = serde_json::json!([1, 2].map(|id| serde_json::json!({
        "id": id, "content": format!("t{id}"), "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": true
    })));
    std::fs::write(&listing, tokenizer_file.to_string()).unwrap();
    let text = checkpoint_tokenizer().decode(&tokens(), false).unwrap();

    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: "org/toy".to_string(),
        models_dir: Some(root.clone()),
        context_length: 128,
        ..InferenceConfig::default()
    });
    engine
        .load_model()
        .await
        .expect("the engine loads the checkpoint");
    assert_eq!(
        is_wide_generic(&engine).await,
        (true, false),
        "the generic arm took the checkpoint"
    );
    let got = run(&engine, &text, NEW).await;
    assert!(got.tokens.len() > NEW, "the answer has its tokens");

    frozen::expect_debug("engine", "checkpoint tokens", &got.tokens);
    frozen::expect_debug("engine", "checkpoint text", &got.text);
    let _ = std::fs::remove_dir_all(&root);
}

/// A wide model whose mixers run on a card joins the batch server with a cap derived from its
/// state, and the server streams the text the serial decode gives.
#[cfg(feature = "gpu")]
#[tokio::test(flavor = "multi_thread")]
async fn a_wide_model_with_its_mixers_on_a_card_streams_through_the_batch_server() {
    use super::block_draft_tests::collect;
    use crate::inference::generic_transformer::card_mixer_tests::generic_on_card;

    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let (engine, prompt) = engine_of(generic_on_card(ratios, kv_src, index_src)).await;
    let want = run(&engine, &prompt, NEW).await;
    assert!(!engine.is_continuous().await, "the model starts serial");
    engine.serve_batched().await;
    assert!(
        engine.is_continuous().await,
        "the model with its mixers on a card joined the batch server"
    );
    assert_eq!(collect(engine, prompt, NEW).await, want.text);
}

/// A whole answer the batch worker writes counts the prompt and the answer the serial path
/// counts.
#[cfg(feature = "gpu")]
#[tokio::test(flavor = "multi_thread")]
async fn a_whole_answer_from_the_worker_counts_its_prompt() {
    use crate::inference::generic_transformer::card_mixer_tests::generic_on_card;

    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let (engine, prompt) = engine_of(generic_on_card(ratios, kv_src, index_src)).await;
    let serial = run(&engine, &prompt, NEW).await;
    engine.serve_batched().await;
    assert!(
        engine.is_continuous().await,
        "the model joined the batch server"
    );
    let batched = run(&engine, &prompt, NEW).await;
    assert_eq!(
        batched.prompt_eval_count + batched.cached_prompt_tokens,
        serial.prompt_eval_count + serial.cached_prompt_tokens,
        "the prompt count"
    );
    assert_eq!(batched.eval_count, serial.eval_count, "the answer count");
}

/// A lone request on the serial path moves into the batch worker when another request is
/// admitted, and its answer is the one it gives alone.
/// The other request then streams through the worker the same answer.
#[cfg(feature = "gpu")]
#[tokio::test(flavor = "multi_thread")]
async fn a_lone_serial_request_moves_into_the_worker_when_another_arrives() {
    use crate::inference::generic_transformer::card_mixer_tests::generic_on_card;

    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let (engine, prompt) = engine_of(generic_on_card(ratios, kv_src, index_src)).await;
    let want = run(&engine, &prompt, NEW).await;
    assert!(
        !engine.is_continuous().await,
        "the lone request decoded serially"
    );
    let first = engine.admit_request().await;
    let second = engine.admit_request().await;
    assert!(second.wants_batched(), "an overlap calls for the worker");
    let mut rx = engine.generate_stream(&prompt, params(NEW)).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = rx.recv().await {
        text.push_str(&chunk.unwrap());
    }
    assert!(
        engine.is_continuous().await,
        "the serial request moved into the worker"
    );
    assert_eq!(text, want.text, "the moved request kept its answer");
    drop(first);
    let mut rx = engine.generate_stream(&prompt, params(NEW)).await.unwrap();
    let mut batched = String::new();
    while let Some(chunk) = rx.recv().await {
        batched.push_str(&chunk.unwrap());
    }
    drop(second);
    assert_eq!(batched, want.text, "the worker gives the lone answer");
}
