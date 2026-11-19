//! 2 different models served by 1 server on card 0, in both orders.
//!
//! A model keeps nothing on the card in a way that pushes the other off it, keeps it serial,
//! unloads it while it serves, or moves its placement for an escalation it did not raise.
//! Every size is derived from the card, so the tests run on a small card and on a large one.
//!
//! The models fill the card: run them alone, on one thread.

use super::*;
use crate::inference::place::device_probe::probe_cuda_gpus;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest a test may run before the watchdog fails it.
const WATCHDOG: Duration = Duration::from_secs(1800);

/// Head width of the generated models: the quantised caches serve it.
const HEAD_DIM: usize = 128;

/// The size of card 0, `None` without one.
fn card_total() -> Option<u64> {
    probe_cuda_gpus()
        .into_iter()
        .find(|g| g.index == 0)
        .map(|g| g.total)
}

/// What card 0 has free now, its queued work run and its pool handed back.
fn card_free() -> u64 {
    crate::inference::place::room::settle(0);
    crate::inference::place::vram_manager::free_on(0).unwrap_or(0)
}

/// The shape of a generated llama checkpoint.
#[derive(Debug, Clone, Copy)]
struct Shape {
    layers: usize,
    hidden: usize,
    kv_heads: usize,
    ffn: usize,
    vocab: usize,
    context: usize,
}

impl Shape {
    /// A model of a few hundred kilobytes.
    fn tiny(context: usize) -> Self {
        Self {
            layers: 2,
            hidden: 128,
            kv_heads: 1,
            ffn: 256,
            vocab: 256,
            context,
        }
    }

    /// A model whose weights take about `bytes`.
    fn weighing(bytes: u64, context: usize) -> Self {
        let layers = 4;
        // F16 weights of a layer: q, k, v and o at the model width, gate, up and down at twice it.
        let per_width_squared = (4 + 3 * 2) * 2 * layers as u64;
        let hidden =
            (((bytes / per_width_squared) as f64).sqrt() as usize / HEAD_DIM).max(1) * HEAD_DIM;
        Self {
            layers,
            hidden,
            kv_heads: 1,
            ffn: 2 * hidden,
            vocab: 256,
            context,
        }
    }

    /// A narrow model whose cache takes about `bytes` a position in F16.
    fn caching(bytes: u64, context: usize) -> Self {
        let layers = 8;
        let kv_heads = (bytes / (layers as u64 * 2 * HEAD_DIM as u64 * 2)).max(1) as usize;
        Self {
            layers,
            hidden: 256,
            kv_heads,
            ffn: 512,
            vocab: 256,
            context,
        }
    }

    fn heads(&self) -> usize {
        (self.hidden / HEAD_DIM).max(self.kv_heads)
    }

    /// The tensors of the checkpoint: name, `[out, in]` dims, and whether it is a norm.
    fn tensors(&self) -> Vec<(String, Vec<usize>, bool)> {
        let (h, f, v) = (self.hidden, self.ffn, self.vocab);
        let q = self.heads() * HEAD_DIM;
        let kv = self.kv_heads * HEAD_DIM;
        let mut out = vec![
            ("model.embed_tokens.weight".to_string(), vec![v, h], false),
            ("model.norm.weight".to_string(), vec![h], true),
            ("lm_head.weight".to_string(), vec![v, h], false),
        ];
        for l in 0..self.layers {
            let p = format!("model.layers.{l}");
            out.push((format!("{p}.input_layernorm.weight"), vec![h], true));
            out.push((
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                true,
            ));
            out.push((format!("{p}.self_attn.q_proj.weight"), vec![q, h], false));
            out.push((format!("{p}.self_attn.k_proj.weight"), vec![kv, h], false));
            out.push((format!("{p}.self_attn.v_proj.weight"), vec![kv, h], false));
            out.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q], false));
            out.push((format!("{p}.mlp.gate_proj.weight"), vec![f, h], false));
            out.push((format!("{p}.mlp.up_proj.weight"), vec![f, h], false));
            out.push((format!("{p}.mlp.down_proj.weight"), vec![h, f], false));
        }
        out
    }
}

/// Writes `shape` as the hub snapshot of `id` under `root`, its weights drawn from `seed`, with a
/// word-level vocabulary of its size.
fn write_model(root: &Path, id: &str, shape: Shape, seed: u32) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    let (org, name) = id.split_once('/').unwrap();
    let snapshot = root
        .join(format!("models--{org}--{name}"))
        .join("snapshots")
        .join("main");
    std::fs::create_dir_all(&snapshot).unwrap();
    let mut state = seed.max(1);
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = shape
        .tensors()
        .into_iter()
        .map(|(name, dims, norm)| {
            let n: usize = dims.iter().product();
            let bytes = (0..n)
                .flat_map(|_| {
                    let x = if norm {
                        1.0
                    } else {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 16) as f32 / 65536.0 - 0.5) * 0.1
                    };
                    half::f16::from_f32(x).to_le_bytes()
                })
                .collect();
            (name, dims, bytes)
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = data
        .iter()
        .map(|(name, dims, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F16, dims.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &snapshot.join("model.safetensors")).unwrap();
    let config = json!({
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": false,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": shape.hidden,
        "intermediate_size": shape.ffn,
        "max_position_embeddings": shape.context,
        "model_type": "llama",
        "num_attention_heads": shape.heads(),
        "num_hidden_layers": shape.layers,
        "num_key_value_heads": shape.kv_heads,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
        "vocab_size": shape.vocab,
    });
    std::fs::write(snapshot.join("config.json"), config.to_string()).unwrap();
    let vocab: ahash::AHashMap<String, u32> = (0..shape.vocab as u32)
        .map(|i| (format!("t{i}"), i))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .save(snapshot.join("tokenizer.json"), false)
        .unwrap();
}

/// A scratch directory of its own for a test's checkpoints.
fn scratch(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("loken-mm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// A server whose hub directory is `root`, every model at the context `context` and the cache
/// format `kv`.
fn server(
    root: &Path,
    context: usize,
    kv: crate::inference::engine::llm_engine::KvQuant,
) -> APIServer {
    let mut state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        root.to_string_lossy().into_owned(),
    );
    state.default_inference_config.context_length = context;
    state.default_inference_config.kv_quant = kv;
    state
}

/// A prompt of `words` words of a vocabulary of `vocab`, starting at word `first`.
fn prompt(words: usize, vocab: usize, first: usize) -> String {
    (0..words)
        .map(|i| format!("t{}", 3 + (i * 7 + first) % (vocab - 3)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The answer of `state`'s `model` to `prompt`, at most `tokens` tokens, greedy.
async fn answer(state: &APIServer, model: &str, prompt: String, tokens: usize) -> String {
    let mut request = OllamaGenerateRequest::new(model.to_string(), prompt);
    request.stream = false;
    request.options = Some(json!({"temperature": 0, "num_predict": tokens}));
    let response = super::ollama_generate(
        axum::extract::State(state.clone()),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .unwrap_or_else(|e| panic!("{model} answers: {e:?}"));
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let whole: Value = serde_json::from_slice(&bytes).unwrap();
    whole["response"]
        .as_str()
        .unwrap_or_else(|| panic!("{model} answered {whole}"))
        .to_string()
}

/// Whether `model` is loaded in `state`.
async fn loaded(state: &APIServer, model: &str) -> bool {
    state
        .engines
        .read()
        .await
        .iter()
        .any(|e| e.model_id == model)
}

/// Unloads every model of `state` and removes `root`: the next scenario starts from an empty
/// card.
async fn finish(state: &APIServer, root: &Path) {
    let models: Vec<String> = state
        .engines
        .read()
        .await
        .iter()
        .map(|e| e.model_id.clone())
        .collect();
    for model in models {
        state.unload_model(&model).await.unwrap();
    }
    let _ = std::fs::remove_dir_all(root);
}

/// The placement of the loaded `model`.
async fn placement(state: &APIServer, model: &str) -> crate::inference::place::Placement {
    state
        .get_engine(model)
        .await
        .expect("the model is loaded")
        .placement()
        .await
        .expect("the model reports its placement")
}

/// Runs `body` under the watchdog, with card 0's size; skipped without a card.
async fn on_card<F: std::future::Future<Output = ()>>(name: &str, body: impl FnOnce(u64) -> F) {
    // The placements and the give-backs are logged with the result.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "info,loken::inference::place::reclaim=debug",
        ))
        .with_test_writer()
        .try_init();
    let Some(total) = card_total() else {
        eprintln!("{name}: no card 0, skipped");
        return;
    };
    tokio::time::timeout(WATCHDOG, body(total))
        .await
        .unwrap_or_else(|_| panic!("{name}: the watchdog fired"));
}

/// T2: a model whose batch worker serves no request gives its pool back to a model beside it
/// that batches, and stays loaded.
async fn an_idle_worker_yields(first: &str, second: &str) {
    let root = scratch("idle-worker");
    let shape = Shape::tiny(4096);
    write_model(&root, first, shape, 11);
    write_model(
        &root,
        second,
        Shape {
            hidden: 256,
            ..shape
        },
        29,
    );
    let state = server(&root, 0, Default::default());
    for (model, seed) in [(first, 1), (second, 2)] {
        state.ensure_loaded(model, 5).await.unwrap();
        state
            .get_engine(model)
            .await
            .unwrap()
            .hold_lone_requests_on(true);
        answer(&state, model, prompt(16, shape.vocab, seed), 4).await;
    }
    let engine = |m| {
        let state = state.clone();
        async move { state.get_engine(m).await.unwrap() }
    };
    assert!(
        engine(second).await.is_continuous().await,
        "{second} stayed serial beside the idle worker of {first}"
    );
    assert!(loaded(&state, first).await, "{first} was unloaded");
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn an_idle_worker_gives_its_pool_to_the_model_beside_it() {
    on_card("idle worker", |_| async {
        an_idle_worker_yields("test/mm-a", "test/mm-b").await;
        an_idle_worker_yields("test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// T3: the conversation states one model keeps give way to a model loaded beside it, which then
/// sits whole on the card, and the first model stays loaded.
async fn snapshots_yield(total: u64, holder: &str, incoming: &str) {
    let root = scratch("snapshots");
    // The cache's working window, which the states leave free, takes a sixteenth of the card.
    let window = crate::inference::cache::KV_WORKING_WINDOW_TOKENS;
    let held = Shape::caching(total / 16 / window as u64, window);
    let words = window / 2;
    write_model(&root, holder, held, 7);
    let state = server(&root, 0, Default::default());
    state.ensure_loaded(holder, 5).await.unwrap();
    state
        .get_engine(holder)
        .await
        .unwrap()
        .hold_lone_requests_on(false);
    // Distinct prompts until the states hold a third of the card.
    let start = card_free();
    for i in 0..256 {
        answer(&state, holder, prompt(words, held.vocab, 100 + i), 1).await;
        if start.saturating_sub(card_free()) >= total / 3 {
            break;
        }
    }
    assert!(
        start.saturating_sub(card_free()) >= total / 3,
        "the states of {holder} took less than a third of the card"
    );
    // The incoming model needs more than the card has left beside the states, and less than
    // the states hold beside it.
    let incoming_shape = Shape::weighing(card_free() + total / 6, 2048);
    write_model(&root, incoming, incoming_shape, 13);
    state.ensure_loaded(incoming, 5).await.unwrap();
    assert!(
        loaded(&state, holder).await,
        "{holder} was unloaded for its states' room"
    );
    let p = placement(&state, incoming).await;
    assert_eq!(
        p.host_layers, 0,
        "{incoming} spilled beside the states of {holder}: {p:?}"
    );
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn kept_states_give_way_to_a_model_loaded_beside_them() {
    on_card("kept states", |total| async move {
        snapshots_yield(total, "test/mm-a", "test/mm-b").await;
        snapshots_yield(total, "test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// T5: a batch worker sized while another component loads leaves the room that load declared,
/// and a model loaded after it still sits whole on the card.
async fn a_load_keeps_its_room(total: u64, batched: &str, incoming: &str) {
    let root = scratch("admission");
    let shape = Shape::tiny(4096);
    write_model(&root, batched, shape, 5);
    let state = server(&root, 0, Default::default());
    state.ensure_loaded(batched, 5).await.unwrap();
    state
        .get_engine(batched)
        .await
        .unwrap()
        .hold_lone_requests_on(true);
    let declared = card_free() / 2;
    {
        // A loader holds the admission and has declared what it is about to take.
        let _admission = crate::inference::place::vram_manager::load_admission().await;
        let _demand =
            crate::inference::place::vram_manager::declare_demand("test-loader", declared);
        answer(&state, batched, prompt(16, shape.vocab, 1), 4).await;
        assert!(
            card_free() >= declared,
            "the worker of {batched} took the room a load declared: {} MB free of {} MB",
            card_free() >> 20,
            declared >> 20
        );
    }
    answer(&state, batched, prompt(16, shape.vocab, 2), 4).await;
    write_model(&root, incoming, Shape::weighing(total / 4, 2048), 9);
    state.ensure_loaded(incoming, 5).await.unwrap();
    let p = placement(&state, incoming).await;
    assert_eq!(
        p.host_layers, 0,
        "{incoming} spilled beside the worker of {batched}: {p:?}"
    );
    assert!(loaded(&state, batched).await, "{batched} was unloaded");
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn a_batch_worker_leaves_the_room_a_load_declared() {
    on_card("load admission", |total| async move {
        a_load_keeps_its_room(total, "test/mm-a", "test/mm-b").await;
        a_load_keeps_its_room(total, "test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// Starts a long answer of `model` in the background, and returns once the engine runs it.
async fn busy_with(
    state: &APIServer,
    model: &str,
    prompt: String,
    tokens: usize,
) -> tokio::task::JoinHandle<String> {
    let engine = state.get_engine(model).await.unwrap();
    let task = {
        let (state, model) = (state.clone(), model.to_string());
        tokio::spawn(async move { answer(&state, &model, prompt, tokens).await })
    };
    while engine.in_flight() == 0 {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    task
}

/// T6: a model loaded beside another that decodes leaves the growth that model's cache was
/// planned for; the decode reaches its planned context without running out of memory, and the
/// model loaded beside it stays loaded.
async fn planned_growth_stays(total: u64, grower: &str, incoming: &str) {
    use crate::inference::engine::llm_engine::KvQuant;
    let root = scratch("growth");
    let window = crate::inference::cache::KV_WORKING_WINDOW_TOKENS;
    let context = 3 * window;
    // The cache grows past its window by about a quarter of the card.
    let grows = Shape::caching(total / 4 / (context - window) as u64, context);
    write_model(&root, grower, grows, 17);
    let state = server(&root, context, KvQuant::Q8);
    state.ensure_loaded(grower, 5).await.unwrap();
    state
        .get_engine(grower)
        .await
        .unwrap()
        .hold_lone_requests_on(false);
    let first = prompt(64, grows.vocab, 3);
    let tokens = context - 2 * 64;
    let alone = answer(&state, grower, first.clone(), tokens).await;
    eprintln!(
        "{grower} alone wrote {} words past a window of {window}",
        alone.split_whitespace().count()
    );
    // The grower forgets the conversation: its cache starts again from the window.
    state.unload_model(grower).await.unwrap();
    state.ensure_loaded(grower, 5).await.unwrap();
    state
        .get_engine(grower)
        .await
        .unwrap()
        .hold_lone_requests_on(false);
    // The incoming model takes what the card reads free, less a margin.
    write_model(
        &root,
        incoming,
        Shape::weighing(card_free() - total / 16, 2048),
        19,
    );
    let decoding = busy_with(&state, grower, first, tokens).await;
    state.ensure_loaded(incoming, 5).await.unwrap();
    let text = decoding.await.unwrap();
    assert_eq!(text, alone, "{grower} answered otherwise beside {incoming}");
    assert!(
        loaded(&state, incoming).await,
        "{incoming} was unloaded for {grower}'s cache"
    );
    assert!(loaded(&state, grower).await, "{grower} was unloaded");
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn a_cache_keeps_the_growth_its_load_planned() {
    on_card("planned growth", |total| async move {
        planned_growth_stays(total, "test/mm-a", "test/mm-b").await;
        planned_growth_stays(total, "test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// T7: a model that serves a request is not unloaded for a model that needs more than the card
/// has free; its answer is the one it gives alone.
async fn a_busy_model_stays(total: u64, serving: &str, incoming: &str) {
    let root = scratch("busy");
    let shape = Shape::weighing(total / 4, 4096);
    write_model(&root, serving, shape, 23);
    let state = server(&root, 0, Default::default());
    state.ensure_loaded(serving, 5).await.unwrap();
    state
        .get_engine(serving)
        .await
        .unwrap()
        .hold_lone_requests_on(false);
    let first = prompt(64, shape.vocab, 5);
    let tokens = 4096 - 2 * 64;
    let alone = answer(&state, serving, first.clone(), tokens).await;
    write_model(
        &root,
        incoming,
        // More than the card has free, whatever the width rounds to.
        Shape::weighing(card_free() + total / 4, 2048),
        31,
    );
    let decoding = busy_with(&state, serving, first, tokens).await;
    state.ensure_loaded(incoming, 5).await.unwrap();
    assert!(
        loaded(&state, serving).await,
        "{serving} was unloaded while it served"
    );
    let text = decoding.await.unwrap();
    assert_eq!(
        text, alone,
        "{serving} answered otherwise while {incoming} loaded"
    );
    assert!(loaded(&state, serving).await, "{serving} was unloaded");
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn a_model_serving_a_request_is_never_unloaded_for_another() {
    on_card("busy model", |total| async move {
        a_busy_model_stays(total, "test/mm-a", "test/mm-b").await;
        a_busy_model_stays(total, "test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// T8: the pressure one model raised after running out of memory moves its own placements, not
/// those of a model loaded meanwhile, and that load leaves the escalation as it was.
async fn pressure_stays_with_its_model(raiser: &str, incoming: &str) {
    use crate::inference::place::{ladder::PRESSURE_RUNGS, recover::Pressure, vram_manager};
    let root = scratch("pressure");
    write_model(&root, raiser, Shape::tiny(2048), 41);
    write_model(
        &root,
        incoming,
        Shape {
            hidden: 256,
            ..Shape::tiny(2048)
        },
        37,
    );
    let state = server(&root, 0, Default::default());
    state.ensure_loaded(raiser, 5).await.unwrap();
    // The raiser's request walked every rung and keeps card 0 back.
    let mut pressure = Pressure::of(raiser);
    for _ in 0..PRESSURE_RUNGS {
        pressure.raise(raiser, &"out of memory");
    }
    assert!(pressure.raise_card(raiser, 0, &"out of memory"));
    let level = vram_manager::degrade_level_for(raiser);
    state.ensure_loaded(incoming, 5).await.unwrap();
    let p = placement(&state, incoming).await;
    assert!(
        p.host_layers == 0 && p.cards.contains(&0),
        "{incoming} was placed by the pressure of {raiser}: {p:?}"
    );
    assert_eq!(
        vram_manager::degrade_level_for(raiser),
        level,
        "loading {incoming} reset the escalation of {raiser}"
    );
    drop(pressure);
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn out_of_memory_pressure_belongs_to_the_model_that_raised_it() {
    on_card("pressure", |_| async {
        pressure_stays_with_its_model("test/mm-a", "test/mm-b").await;
        pressure_stays_with_its_model("test/mm-b", "test/mm-a").await;
    })
    .await;
}

/// One model decodes alone past its cache's working window, its decode recorded: the recorded
/// step is dropped and recorded again as the cache grows, and the answer reaches its length.
async fn a_long_decode_runs(
    kv: crate::inference::engine::llm_engine::KvQuant,
    fidelity: crate::inference::engine::llm_engine::Fidelity,
) {
    let root = scratch("long-decode");
    let window = crate::inference::cache::KV_WORKING_WINDOW_TOKENS;
    let context = 3 * window;
    let shape = Shape {
        kv_heads: 16,
        ..Shape::caching(0, context)
    };
    write_model(&root, "test/mm-a", shape, 17);
    let mut state = server(&root, context, kv);
    state.default_inference_config.fidelity = fidelity;
    state.ensure_loaded("test/mm-a", 5).await.unwrap();
    state
        .get_engine("test/mm-a")
        .await
        .unwrap()
        .hold_lone_requests_on(false);
    let tokens = context / 2;
    let text = answer(&state, "test/mm-a", prompt(64, shape.vocab, 3), tokens).await;
    assert_eq!(
        text.split_whitespace().count(),
        tokens,
        "the decode stopped before its length"
    );
    finish(&state, &root).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fills card 0: run alone"]
async fn a_long_decode_grows_its_cache_past_the_window() {
    use crate::inference::engine::llm_engine::{Fidelity, KvQuant};
    on_card("long decode", |_| async {
        a_long_decode_runs(KvQuant::Q8, Fidelity::Balanced).await;
        // A full-width cache.
        a_long_decode_runs(KvQuant::Off, Fidelity::Reference).await;
    })
    .await;
}
