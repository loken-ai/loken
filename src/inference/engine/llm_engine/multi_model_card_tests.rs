//! 2 different small models served in 1 process on card 0.
//! What one model does on the card never changes the other's text, stalls it or fails it.
//!
//! The models are written by the test: dense llama files of 2 blocks, Q8_0, that differ in the
//! width of their feed-forward.
//! Each test runs both orders, and a watchdog fails one that stalls.

use super::*;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

const HIDDEN: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const BLOCKS: usize = 2;
const VOCAB: usize = 512;
/// The feed-forward of the narrow model.
const NARROW: usize = 512;
/// The feed-forward of the wide model: its decode scratch is larger than the narrow one's.
const WIDE: usize = 8192;
/// Tokens each answer writes.
const NEW: usize = 24;
/// The longest a test may take before the watchdog fails it.
const WATCHDOG: std::time::Duration = std::time::Duration::from_secs(900);

fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn tokenizer() -> Tokenizer {
    let vocab: ahash::AHashMap<String, u32> =
        (0..VOCAB as u32).map(|i| (format!("t{i}"), i)).collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

/// A dense llama of feed-forward `ffn` on disk beside its tokenizer, in a directory of its own.
/// Returns the directory; the caller removes it.
fn model_on_disk(ffn: usize) -> std::path::PathBuf {
    let meta = vec![
        (
            "general.architecture".to_string(),
            Value::String("llama".into()),
        ),
        (
            "llama.attention.head_count".to_string(),
            Value::U32(HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv".to_string(),
            Value::U32(KV_HEADS as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        ),
        ("llama.block_count".to_string(), Value::U32(BLOCKS as u32)),
        (
            "llama.embedding_length".to_string(),
            Value::U32(HIDDEN as u32),
        ),
        (
            "llama.feed_forward_length".to_string(),
            Value::U32(ffn as u32),
        ),
        ("llama.context_length".to_string(), Value::U32(512)),
        // A token no answer produces ends nothing early.
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(VOCAB as u32 + 1),
        ),
    ];
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("token_embd.weight".into(), vec![VOCAB, HIDDEN]),
        ("output_norm.weight".into(), vec![HIDDEN]),
        ("output.weight".into(), vec![VOCAB, HIDDEN]),
    ];
    for il in 0..BLOCKS {
        tensors.push((format!("blk.{il}.attn_norm.weight"), vec![HIDDEN]));
        tensors.push((
            format!("blk.{il}.attn_q.weight"),
            vec![HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_k.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_v.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_output.weight"),
            vec![HIDDEN, HEADS * HEAD_DIM],
        ));
        tensors.push((format!("blk.{il}.ffn_norm.weight"), vec![HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_gate.weight"), vec![ffn, HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_up.weight"), vec![ffn, HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_down.weight"), vec![HIDDEN, ffn]));
    }
    let entries: Vec<GgufEntry> = tensors
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let n: usize = dims.iter().product();
            let norm = name.ends_with("norm.weight");
            let (dtype, vals) = if norm {
                let v = values(n, i as u64, 0.1)
                    .into_iter()
                    .map(|x| 1.0 + x)
                    .collect();
                (GgmlDType::F32, v)
            } else {
                let amp = 1.5 / (*dims.last().unwrap() as f32).sqrt();
                (GgmlDType::Q8_0, values(n, i as u64 + 1, amp))
            };
            GgufEntry {
                name,
                dims,
                dtype,
                data: crate::tensor::quant_cpu::from_float_bytes(dtype, &vals).unwrap(),
            }
        })
        .collect();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("loken-two-models-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_gguf_with_metadata(&dir.join("model.gguf"), &meta, &entries).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    dir
}

/// An engine that has loaded the model in `dir`.
async fn loaded(dir: &std::path::Path, name: &str) -> LlmEngine {
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: name.to_string(),
        models_dir: Some(dir.join("model.gguf")),
        context_length: 256,
        ..InferenceConfig::default()
    });
    engine
        .load_model()
        .await
        .expect("the engine loads the file");
    engine
}

fn prompt(first: usize) -> String {
    (0..12)
        .map(|i| format!("t{}", (first + 7 * i) % VOCAB))
        .collect::<Vec<_>>()
        .join(" ")
}

fn params(new: usize) -> GenerationParams {
    GenerationParams {
        max_tokens: Some(new),
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..GenerationParams::default()
    }
}

/// The text of a greedy answer of `engine` to `prompt`, or the error it gave.
async fn answer(engine: &LlmEngine, prompt: &str, new: usize) -> Result<String, String> {
    engine
        .generate(prompt, params(new))
        .await
        .map(|r| r.text)
        .map_err(|e| e.to_string())
}

/// The texts of 2 greedy answers of `engine` read at once by its batch worker.
async fn batched_answers(engine: &LlmEngine, prompts: [&str; 2]) -> Result<Vec<String>, String> {
    engine.hold_lone_requests_on(true);
    if !engine.serve_batched().await {
        return Err("the model does not move into its batch worker".into());
    }
    let mut streams = Vec::new();
    for p in prompts {
        streams.push(
            engine
                .generate_stream(p, params(NEW))
                .await
                .map_err(|e| e.to_string())?,
        );
    }
    let mut texts = Vec::new();
    for mut rx in streams {
        let mut text = String::new();
        while let Some(chunk) = rx.recv().await {
            text.push_str(&chunk?);
        }
        texts.push(text);
    }
    Ok(texts)
}

/// The texts of `engine`'s answers to `prompts`, one after the other or read at once by its
/// batch worker.
async fn run(
    engine: &LlmEngine,
    prompts: &[String; 2],
    batched: bool,
) -> Result<Vec<String>, String> {
    match batched {
        true => batched_answers(engine, [&prompts[0], &prompts[1]]).await,
        false => Ok(vec![
            answer(engine, &prompts[0], NEW).await?,
            answer(engine, &prompts[1], NEW).await?,
        ]),
    }
}

/// T1 for the kept model of feed-forward `kept` beside the neighbour of feed-forward
/// `neighbour`, serial or batched.
async fn graphs_survive_a_neighbour(kept: usize, neighbour: usize, batched: bool) {
    let (kept_dir, neighbour_dir) = (model_on_disk(kept), model_on_disk(neighbour));
    let prompts = [prompt(3), prompt(11)];
    // The kept model alone: the second answers replay the graphs the first recorded.
    let alone = loaded(&kept_dir, "kept").await;
    let want = run(&alone, &prompts, batched)
        .await
        .expect("the model alone answers");
    assert_eq!(
        run(&alone, &prompts, batched)
            .await
            .expect("a second run alone"),
        want,
        "the model alone is not stable"
    );
    alone.unload().await.unwrap();
    drop(alone);

    let a = loaded(&kept_dir, "kept").await;
    assert_eq!(
        run(&a, &prompts, batched).await.unwrap(),
        want,
        "the model before its neighbour"
    );
    // The neighbour records its own graphs, then leaves the card.
    let b = loaded(&neighbour_dir, "neighbour").await;
    let _ = answer(&b, &prompt(5), 2)
        .await
        .expect("the neighbour answers");
    let _ = answer(&b, &prompt(7), 2)
        .await
        .expect("the neighbour answers again");
    b.unload().await.unwrap();
    drop(b);
    // An allocation of another user of the card ran out of memory once.
    if let Ok(dev) = crate::tensor::cuda::CudaDevice::new(0) {
        crate::tensor::cuda::reclaim_pool(&dev);
    }
    // Buffers that take the memory given back.
    let device = crate::tensor::Device::new_cuda(0).unwrap();
    let filler: Vec<crate::tensor::Tensor> = (0..8)
        .filter_map(|_| {
            crate::tensor::Tensor::ones((WIDE, HIDDEN), crate::tensor::DType::F32, &device).ok()
        })
        .collect();
    let after = run(&a, &prompts, batched).await;
    drop(filler);
    a.unload().await.unwrap();
    let _ = std::fs::remove_dir_all(&kept_dir);
    let _ = std::fs::remove_dir_all(&neighbour_dir);
    assert_eq!(
        after.expect("the model answers after its neighbour"),
        want,
        "the model's text changed after its neighbour (kept {kept}, neighbour {neighbour}, \
         batched {batched})"
    );
}

/// T1: a model's recorded graphs survive what a neighbour does to the card's shared scratch:
/// a wider decode that grows it, an unload and an out-of-memory reclaim that release it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a card"]
async fn graphs_survive_a_neighbours_scratch() {
    let test = async {
        for batched in [false, true] {
            graphs_survive_a_neighbour(NARROW, WIDE, batched).await;
            graphs_survive_a_neighbour(WIDE, NARROW, batched).await;
        }
    };
    tokio::time::timeout(WATCHDOG, test)
        .await
        .expect("the watchdog: the test stalled");
}

/// The card's free memory once its pools gave back their free blocks.
fn card_free(device: &crate::tensor::Device) -> u64 {
    if let Ok(cd) = device.as_cuda_device() {
        let _ = cd.0.stream().synchronize();
        crate::tensor::cuda::trim_pool(cd.0.ordinal());
    }
    crate::inference::place::room::free_now(device).unwrap_or(0)
}

/// T11 for the model of feed-forward `leaving` batched and unloaded beside the model of
/// feed-forward `staying`, which holds the card open throughout.
async fn the_card_comes_back(leaving: usize, staying: usize) {
    let (leaving_dir, staying_dir) = (model_on_disk(leaving), model_on_disk(staying));
    let device = crate::tensor::Device::new_cuda(0).unwrap();
    let stays = loaded(&staying_dir, "stays").await;
    let _ = answer(&stays, &prompt(1), 2)
        .await
        .expect("the staying model answers");
    let before = card_free(&device);
    let leaves = loaded(&leaving_dir, "leaves").await;
    let prompts = [prompt(3), prompt(11)];
    run(&leaves, &prompts, true)
        .await
        .expect("the leaving model answers in its batch worker");
    leaves.unload().await.unwrap();
    drop(leaves);
    let after = card_free(&device);
    // The recorders a card makes live as long as the card, and a free block is given back
    // a page at a time.
    let recorders = device.as_cuda_device().map_or(0, |cd| {
        cd.0.recorder_bytes() * cd.0.recorder_count() as u64 + cd.0.alloc_shape().0
    });
    stays.unload().await.unwrap();
    let _ = std::fs::remove_dir_all(&leaving_dir);
    let _ = std::fs::remove_dir_all(&staying_dir);
    assert!(
        before.saturating_sub(after) <= recorders,
        "the card kept {:.1} MB after the model left (recorders and a page {:.1} MB; leaving \
         {leaving}, staying {staying})",
        before.saturating_sub(after) as f64 / 1048576.0,
        recorders as f64 / 1048576.0
    );
}

/// T11: what a batched model held on the card comes back when it is unloaded, while another
/// model keeps the card open.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a card"]
async fn the_card_comes_back_after_a_batched_model() {
    let test = async {
        the_card_comes_back(NARROW, WIDE).await;
        the_card_comes_back(WIDE, NARROW).await;
    };
    tokio::time::timeout(WATCHDOG, test)
        .await
        .expect("the watchdog: the test stalled");
}

/// The blob an ollama tag names in the store, `None` when the store does not hold its weights.
fn ollama_blob(tag: &str) -> Option<std::path::PathBuf> {
    let dir = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, ver) = tag.split_once(':').unwrap_or((tag, "latest"));
    let text = std::fs::read_to_string(
        dir.join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(ver),
    )
    .ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let digest = v["layers"]
        .as_array()?
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")?["digest"]
        .as_str()?
        .replace(':', "-");
    let blob = dir.join("blobs").join(digest);
    blob.exists().then_some(blob)
}

/// An engine that has loaded the store's `tag`.
async fn loaded_tag(tag: &str) -> LlmEngine {
    let blob = ollama_blob(tag).unwrap();
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: tag.to_string(),
        models_dir: Some(blob),
        context_length: 2048,
        ..InferenceConfig::default()
    });
    engine
        .load_model()
        .await
        .expect("the engine loads the model");
    engine
}

/// T9 for `ranking`, whose requests end by moving the host locks of its hot experts, beside
/// `decoding`, which decodes meanwhile.
async fn registration_beside(ranking: &str, decoding: &str) {
    let text = "Write a short story about a lighthouse keeper who finds a message in a bottle.";
    let alone = loaded_tag(decoding).await;
    let started = std::time::Instant::now();
    let want = answer(&alone, text, 96)
        .await
        .expect("the model alone answers");
    let alone_secs = started.elapsed().as_secs_f64();
    alone.unload().await.unwrap();
    drop(alone);

    let a = std::sync::Arc::new(loaded_tag(ranking).await);
    let b = loaded_tag(decoding).await;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ranks = {
        let (a, stop) = (a.clone(), stop.clone());
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let mut requests = 0;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                answer(&a, &prompt(requests), 8)
                    .await
                    .expect("the ranking model answers");
                requests += 1;
            }
            (requests, started.elapsed().as_secs_f64())
        })
    };
    let started = std::time::Instant::now();
    let got = answer(&b, text, 96)
        .await
        .expect("the decoding model answers");
    let beside_secs = started.elapsed().as_secs_f64();
    stop.store(true, std::sync::atomic::Ordering::Release);
    let (requests, ranking_secs) = ranks.await.unwrap();
    b.unload().await.unwrap();
    a.unload().await.unwrap();
    assert_eq!(got, want, "{decoding} answered otherwise beside {ranking}");
    // The decode may wait for a registration in progress, never for a whole run of them.
    let per_request = ranking_secs / requests.max(1) as f64;
    assert!(
        beside_secs <= alone_secs * 2.0 + per_request,
        "{decoding} took {beside_secs:.1} s beside {ranking}'s {requests} requests, {alone_secs:.1} s alone"
    );
}

/// T9: a model that locks its hot experts in host memory at each request's end never stalls a
/// model whose card waits on its tier worker meanwhile.
/// Both models need expert tiers: they are run on a card of the 16 GB class.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a card and 2 MoE models in the store"]
async fn a_registration_never_stalls_another_models_decode() {
    const A: &str = "gpt-oss:20b";
    const B: &str = "qwen3-coder:30b";
    if ollama_blob(A).is_none() || ollama_blob(B).is_none() {
        eprintln!("{A} or {B} is not in the store: skipped");
        return;
    }
    let test = async {
        registration_beside(A, B).await;
        registration_beside(B, A).await;
    };
    tokio::time::timeout(WATCHDOG, test)
        .await
        .expect("the watchdog: the test stalled");
}
