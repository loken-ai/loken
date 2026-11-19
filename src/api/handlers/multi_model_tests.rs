//! 2 different models served by 1 server on card 0, in both orders.
//!
//! A model keeps nothing on the card in a way that pushes the other off it, keeps it serial,
//! unloads it while it serves, or moves its placement for an escalation it did not raise.
//! Every size is derived from the card, so the tests run on a small card and on a large one.
//!
//! The models fill the card: run them alone, on one thread.

use super::*;
use crate::inference::place::device_probe::probe_cuda_gpus;
use serde_json::json;
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
