//! A draft verify against the decode steps it stands for: the same tokens from the same
//! cache, once as one multi-row `forward_all`, once one `forward` per token. Greedy output
//! may not depend on whether the tokens were drafted, so each row's logits must equal the
//! step's.
//!
//! Ignored by default: it needs the model in the store and a CUDA card. Run with
//!   LOKEN_PARITY_MODEL=qwen2.5:0.5b cargo test --release verify_rows_match_decode \
//!     -- --ignored --nocapture

use super::*;
use crate::inference::engine::decode_step::VERIFY_ROWS;
use crate::inference::engine::llm_engine::KvQuant;

fn model_blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, tag) = tag.split_once(':').unwrap_or((tag, "latest"));
    let m = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = m
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

fn host_rows(t: &Tensor) -> Vec<Vec<f32>> {
    let t = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let vocab = t.dims().last().copied().unwrap();
    let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    flat.chunks(vocab).map(<[f32]>::to_vec).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// The model named by `LOKEN_PARITY_MODEL`, wholly on the card `LOKEN_PARITY_CARD` names
/// (0 by default) with the Q8 cache a served
/// model gets, and the card. `None` without a card.
fn load(ctx: usize) -> Option<(String, GenericHeteroTransformer, Device)> {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "qwen2.5:0.5b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let n_layers = {
        let c = read();
        let arch = c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        GenericTransformerConfig::from_gguf(&c, &arch)
            .unwrap()
            .n_layers
    };
    let ordinal: usize = std::env::var("LOKEN_PARITY_CARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let Ok(card) = Device::new_cuda(ordinal) else {
        eprintln!("no CUDA device");
        return None;
    };
    let cards: HashMap<usize, Device> = [(ordinal, card.clone())].into_iter().collect();
    let m = GenericHeteroTransformer::from_gguf_with_kv_quant(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, n_layers, ordinal),
        KvQuant::Q8,
        Some(ctx),
    )
    .unwrap();
    Some((tag, m, card))
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = host_rows(a).concat();
    let b = host_rows(b).concat();
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Each matmul of layer 0 and the head over a batch of rows, against the same rows one at a
/// time: plainly, and under `DecodeOrder`.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn matmul_rows_match_single_rows() {
    use crate::tensor::quantized::DecodeOrder;
    let Some((tag, m, card)) = load(256) else {
        return;
    };
    let l = &m.layers[0];
    type Mm<'a> = Box<dyn Fn(&Tensor) -> Tensor + 'a>;
    let mut named: Vec<(&str, usize, Mm)> = Vec::new();
    for (name, w) in [
        ("attn_qkv", l.attn_qkv.as_ref()),
        ("attn_q", l.attn_q.as_ref()),
        ("attn_k", l.attn_k.as_ref()),
        ("attn_v", l.attn_v.as_ref()),
        ("attn_output", Some(&l.attn_output)),
        ("ffn_gate", l.ffn_gate.as_ref()),
        ("ffn_up", Some(&l.ffn_up)),
        ("ffn_down", Some(&l.ffn_down)),
    ] {
        if let Some(k) = w.and_then(|w| w.qtensor()).map(|q| q.shape().dims()[1]) {
            let w = w.unwrap();
            named.push((name, k, Box::new(move |x: &Tensor| w.forward(x).unwrap())));
        }
    }
    if let Some(crate::tensor::quantized::QMatMul::QTensor(q)) = m.output_proj_cuda_qmm.as_ref() {
        let head = m.output_proj_cuda_qmm.as_ref().unwrap();
        named.push((
            "head",
            q.shape().dims()[1],
            Box::new(move |x: &Tensor| head.forward(x).unwrap()),
        ));
    }
    let rows = VERIFY_ROWS;
    for (name, k, w) in named {
        let x = Tensor::randn(0f32, 1f32, (1, rows, k), &card).unwrap();
        let single: Vec<Tensor> = (0..rows).map(|r| w(&x.narrow(1, r, 1).unwrap())).collect();
        let mut plain = Vec::new();
        let mut exact = Vec::new();
        for b in 2..=rows {
            let want = Tensor::cat(&single[..b].iter().collect::<Vec<_>>(), 1).unwrap();
            let xb = x.narrow(1, 0, b).unwrap();
            plain.push(format!("{:.1e}", max_diff(&w(&xb), &want)));
            let _order = DecodeOrder::enter();
            exact.push(format!("{:.1e}", max_diff(&w(&xb), &want)));
        }
        println!("{tag} {name:>12} k={k}: plain rows 2..: {plain:?}");
        println!("{tag} {name:>12} k={k}: decode order rows 2..: {exact:?}");
    }

    // The decode head fuses the final norm into the head's activation quantisation.
    if let (Some(crate::tensor::quantized::QMatMul::QTensor(qt)), Some(norm_w)) = (
        m.output_proj_cuda_qmm.as_ref(),
        m.output_norm_cuda_weight.as_ref(),
    ) {
        let h = m.config.embedding_length;
        let x = Tensor::randn(0f32, 1f32, (1, h), &card).unwrap();
        let eps = m.config.rms_norm_eps as f32;
        if let Ok(fused) =
            crate::inference::moe_cuda::rms_norm_then_qmatmul(&x, norm_w, qt.as_ref(), eps)
        {
            let normed = crate::tensor::ops::rms_norm(&x, norm_w, eps).unwrap();
            let unfused = m
                .output_proj_cuda_qmm
                .as_ref()
                .unwrap()
                .forward(&normed)
                .unwrap();
            println!(
                "{tag} head: fused norm+matmul vs norm then matmul: {:.1e}",
                max_diff(&fused.reshape(unfused.shape()).unwrap(), &unfused)
            );
        }
    }
}

#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn verify_rows_match_decode() {
    let steps: usize = std::env::var("LOKEN_PARITY_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let Some((tag, mut m, card)) = load(4 * (steps + 64)) else {
        return;
    };

    let prompt: Vec<u32> = (0..24u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let ids = |t: &[u32]| Tensor::new(t, &card).unwrap().unsqueeze(0).unwrap();
    let first = host_rows(&m.forward(&ids(&prompt), 0).unwrap());
    let base = prompt.len();

    // The decode steps: one token each, greedy.
    let mut tokens = vec![argmax(&first[0]) as u32];
    let mut decoded = Vec::with_capacity(steps);
    for i in 0..steps {
        let row = host_rows(&m.forward(&ids(&tokens[i..=i]), base + i).unwrap()).remove(0);
        tokens.push(argmax(&row) as u32);
        decoded.push(row);
    }

    // The same tokens again, verified in windows of every size a draft can take.
    let mut worst = 0f32;
    let mut differing_rows = 0usize;
    let mut flipped = 0usize;
    let mut at = 0usize;
    let mut width = 1usize;
    while at < steps {
        let rows = width.min(steps - at);
        m.trim_kv(base + at);
        let verified = host_rows(
            &m.forward_all(&ids(&tokens[at..at + rows]), base + at)
                .unwrap(),
        );
        for (r, row) in verified.iter().enumerate() {
            let step = &decoded[at + r];
            let diff = row
                .iter()
                .zip(step)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if diff > 0.0 {
                differing_rows += 1;
            }
            if argmax(row) != argmax(step) {
                flipped += 1;
            }
            worst = worst.max(diff);
        }
        println!(
            "{tag}: rows {:>3}..{:>3} (width {rows}): max logit diff {:.6}",
            at,
            at + rows,
            verified
                .iter()
                .zip(&decoded[at..])
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0f32, f32::max)
        );
        at += rows;
        width = width % VERIFY_ROWS + 1;
    }
    println!(
        "{tag}: {steps} steps, {differing_rows} rows differ, {flipped} argmax flips, \
         worst logit diff {worst:.6}"
    );

    // The same tokens through the body a captured decode graph records, run uncaptured.
    m.trim_kv(base);
    let mut graph_worst = 0f32;
    let mut graph_differing = 0usize;
    for i in 0..steps {
        m.update_graph_state(base + i).unwrap();
        m.embed_for_graph(
            &Tensor::new(&tokens[i..=i], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        )
        .unwrap();
        let row = match m.forward_from_hidden() {
            Ok(logits) => host_rows(&logits).remove(0),
            Err(e) => {
                println!("{tag}: no graph body for this placement ({e})");
                break;
            }
        };
        m.sync_kv_len_for_graph(base + i + 1);
        let diff = row
            .iter()
            .zip(&decoded[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if diff > 0.0 {
            graph_differing += 1;
        }
        graph_worst = graph_worst.max(diff);
    }
    m.invalidate_graph_state();
    println!(
        "{tag}: graph body: {graph_differing} of {steps} steps differ from eager decode, \
         worst logit diff {graph_worst:.6}"
    );

    assert_eq!(
        graph_differing, 0,
        "a graph decode step differs from the eager one"
    );
    assert_eq!(
        differing_rows, 0,
        "a verified row differs from its decode step"
    );
}

/// What a verify costs: `k+1` rows as decode steps (this forward_all), the same rows
/// batched as a prefill batches them, and `k+1` single-token forwards.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn verify_cost_against_decode_steps() {
    let Some((tag, mut m, card)) = load(1024) else {
        return;
    };
    let prompt: Vec<u32> = (0..64u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let ids = |t: &[u32]| Tensor::new(t, &card).unwrap().unsqueeze(0).unwrap();
    m.forward(&ids(&prompt), 0).unwrap();
    let base = prompt.len();
    let toks: Vec<u32> = (0..VERIFY_ROWS as u32).map(|i| 2000 + i * 13).collect();
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let reps = 9;
    for rows in [2usize, 3, 5, VERIFY_ROWS] {
        let t = &toks[..rows];
        let mut verify = Vec::new();
        let mut batched = Vec::new();
        let mut steps = Vec::new();
        for _ in 0..reps {
            m.trim_kv(base);
            let s = std::time::Instant::now();
            host_rows(&m.forward_all(&ids(t), base).unwrap());
            verify.push(s.elapsed().as_secs_f64());

            m.trim_kv(base);
            let s = std::time::Instant::now();
            host_rows(&m.forward_all_batched(&ids(t), base).unwrap());
            batched.push(s.elapsed().as_secs_f64());

            m.trim_kv(base);
            let s = std::time::Instant::now();
            for (r, tok) in t.iter().enumerate() {
                host_rows(&m.forward(&ids(&[*tok]), base + r).unwrap());
            }
            steps.push(s.elapsed().as_secs_f64());
        }
        let (v, b, d) = (median(verify), median(batched), median(steps));
        println!(
            "{tag}: {rows} rows: verify as decode {:.2} ms, batched {:.2} ms, {rows} decode steps {:.2} ms \
             (verify = {:.2} steps)",
            v * 1e3,
            b * 1e3,
            d * 1e3,
            v / (d / rows as f64)
        );
    }
}
