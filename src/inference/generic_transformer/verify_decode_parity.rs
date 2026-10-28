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

/// The decode steps a parity test runs: `LOKEN_PARITY_STEPS`, or 64.
fn parity_steps() -> usize {
    std::env::var("LOKEN_PARITY_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
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

    // Every layer's weights under `DecodeOrder`: a format or a shape whose rows are not the
    // one-row launch is named with its layer.
    let mut bad = Vec::new();
    for (li, l) in m.layers.iter().enumerate() {
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
            let Some((w, q)) = w.and_then(|w| w.qtensor().map(|q| (w, q))) else {
                continue;
            };
            let k = q.shape().dims()[1];
            let x = Tensor::randn(0f32, 1f32, (1, rows, k), &card).unwrap();
            let single: Vec<Tensor> = (0..rows)
                .map(|r| w.forward(&x.narrow(1, r, 1).unwrap()).unwrap())
                .collect();
            let want = Tensor::cat(&single.iter().collect::<Vec<_>>(), 1).unwrap();
            let got = {
                let _order = DecodeOrder::enter();
                w.forward(&x).unwrap()
            };
            let d = max_diff(&got, &want);
            if d > 0.0 {
                bad.push(format!("layer {li} {name} {:?} k={k}: {d:.1e}", q.dtype()));
            }
        }
    }
    println!(
        "{tag}: decode-order rows against one-row launches over every layer: {}",
        if bad.is_empty() {
            "all exact".to_string()
        } else {
            bad.join("; ")
        }
    );

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
    let steps = parity_steps();
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

/// The decode steps of `steps` greedy tokens after a prompt, and the tokens: what the rows of
/// a verify are held against.
fn decoded_steps(
    m: &mut GenericHeteroTransformer,
    card: &Device,
    steps: usize,
) -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    let prompt: Vec<u32> = (0..24u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let ids = |t: &[u32]| Tensor::new(t, card).unwrap().unsqueeze(0).unwrap();
    let first = host_rows(&m.forward(&ids(&prompt), 0).unwrap());
    let base = prompt.len();
    let mut tokens = vec![argmax(&first[0]) as u32];
    let mut decoded = Vec::with_capacity(steps);
    for i in 0..steps {
        let row = host_rows(&m.forward(&ids(&tokens[i..=i]), base + i).unwrap()).remove(0);
        tokens.push(argmax(&row) as u32);
        decoded.push(row);
    }
    (tokens, base, decoded)
}

fn row_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The same tokens through recorded verify windows of every width a draft can take: each
/// window run eagerly as the kernels a graph records, then recorded (once per width) and
/// replayed at every later window of that width. Every row has to be its decode step, and
/// every replay its eager run.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn recorded_window_rows_match_decode() {
    let steps = parity_steps();
    let Some((tag, mut m, card)) = load(4 * (steps + 64)) else {
        return;
    };
    let (tokens, base, decoded) = decoded_steps(&mut m, &card, steps);
    if let Some(why) = m.verify_window_block() {
        panic!("{tag}: no recorded window: {why}");
    }
    let most = m.verify_window_rows();
    println!("{tag}: recorded windows of up to {most} rows");
    let stream = crate::tensor::cuda_ext::stream_of(&card).unwrap();
    let mut graphs: HashMap<usize, (crate::tensor::cuda_ext::CapturedGraph, Tensor)> =
        HashMap::new();
    m.trim_kv(base);
    let (mut eager_differing, mut replay_differing, mut flipped) = (0usize, 0usize, 0usize);
    let mut worst = 0f32;
    let mut at = 0usize;
    let mut width = 2usize;
    while at + 1 < steps {
        let rows = width.min(steps - at).min(most);
        let window = &tokens[at..at + rows];
        let pos = base + at;
        // Eagerly: the kernels of the recorded window, run at once.
        m.verify_window_prepare(pos, window).unwrap();
        let eager = host_rows(&m.forward_verify_window(rows).unwrap());
        stream.synchronize().unwrap();
        for (r, row) in eager.iter().enumerate() {
            let d = row_diff(row, &decoded[at + r]);
            if d > 0.0 {
                eager_differing += 1;
            }
            if argmax(row) != argmax(&decoded[at + r]) {
                flipped += 1;
            }
            worst = worst.max(d);
        }
        // Back to before the rows; the window of this width recorded or replayed.
        m.verify_window_settle(0).unwrap();
        m.verify_window_prepare(pos, window).unwrap();
        stream.synchronize().unwrap();
        let logits = match graphs.get(&rows) {
            Some((g, logits)) => {
                g.launch().unwrap();
                m.verify_window_replayed();
                logits.clone()
            }
            None => {
                crate::tensor::cuda_ext::begin_capture(&stream).unwrap();
                let logits = m.forward_verify_window(rows).unwrap();
                let g = crate::tensor::cuda_ext::end_capture(&stream)
                    .unwrap()
                    .expect("a window records kernels");
                g.launch().unwrap();
                m.verify_window_replayed();
                graphs.insert(rows, (g, logits.clone()));
                logits
            }
        };
        stream.synchronize().unwrap();
        let replayed = host_rows(&logits);
        let replay_worst = replayed
            .iter()
            .zip(&eager)
            .map(|(a, b)| row_diff(a, b))
            .fold(0f32, f32::max);
        replay_differing += replayed
            .iter()
            .zip(&eager)
            .filter(|(a, b)| row_diff(a, b) > 0.0)
            .count();
        println!(
            "{tag}: rows {:>3}..{:>3} (width {rows}): eager window vs steps {:.6}, replay vs eager {:.6}",
            at,
            at + rows,
            eager
                .iter()
                .zip(&decoded[at..])
                .map(|(a, b)| row_diff(a, b))
                .fold(0f32, f32::max),
            replay_worst
        );
        // Every row kept: decoding goes on from the window's last row.
        m.verify_window_settle(rows).unwrap();
        at += rows;
        width = (width % most + 1).max(2);
    }
    m.invalidate_graph_state();
    println!(
        "{tag}: {steps} steps in recorded windows: {eager_differing} eager rows differ from their \
         steps, {flipped} argmax flips, worst logit diff {worst:.6}; {replay_differing} replayed \
         rows differ from their eager run"
    );
    assert_eq!(
        replay_differing, 0,
        "a replayed window differs from its eager run"
    );
    assert_eq!(
        eager_differing, 0,
        "a window's row differs from its decode step"
    );
}

/// Each layer's stages over a verify's rows against the same stages one row at a time, from
/// the one-row path's own inputs, so a stage that is not decode-ordered is named with its
/// layer: the attention block, the post-attention add and norm, the feed-forward, and the
/// whole layer.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn verify_stages_match_decode() {
    use crate::tensor::quantized::DecodeOrder;
    let rows = VERIFY_ROWS;
    let Some((tag, mut m, card)) = load(256) else {
        return;
    };
    let (tokens, base, _) = decoded_steps(&mut m, &card, rows);
    m.trim_kv(base);
    let ids = Tensor::new(&tokens[..rows], &card)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut hidden = match m.embed_table_on(&card) {
        Some(table) => m.embed_rows_on_card(table, &ids).unwrap(),
        None => {
            let dev = m.embeddings.embeddings().device().clone();
            m.embeddings
                .forward(&ids.to_device(&dev).unwrap())
                .unwrap()
                .to_device(&card)
                .unwrap()
        }
    };
    if let Some(scale) = m.config.embed_scale {
        hidden = (hidden * scale).unwrap();
    }
    // The rows one at a time, as the decode steps run them, and at once under `DecodeOrder`.
    let single = |f: &mut dyn FnMut(&Tensor, usize) -> Tensor, x: &Tensor| -> Tensor {
        let parts: Vec<Tensor> = (0..rows)
            .map(|r| f(&x.narrow(1, r, 1).unwrap().contiguous().unwrap(), r))
            .collect();
        Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1).unwrap()
    };
    let n_layers = m.layers.len();
    let mut first_bad: Option<usize> = None;
    for li in 0..n_layers {
        let x = hidden.clone();
        // The attention block from the normed rows.
        m.layers[li].trim_kv_to(base);
        let xn = m.layers[li].attn_norm.forward(&x).unwrap();
        let attn_one = single(
            &mut |row, r| {
                m.layers[li]
                    .forward_attn(row, None, base + r, None, None, None)
                    .unwrap()
            },
            &xn,
        );
        m.layers[li].trim_kv_to(base);
        let attn_rows = {
            let _order = DecodeOrder::enter();
            m.layers[li]
                .forward_attn(&xn, None, base, None, None, None)
                .unwrap()
        };
        let d_attn = max_diff(&attn_one, &attn_rows);
        // The post-attention add and the feed-forward norm, from the one-row attention.
        let fused = m.layers[li].post_attn_norm.is_none();
        let (d_norm, x_norm_one) = match m.layers[li].ffn_norm_weight.clone().filter(|_| fused) {
            Some(w) => {
                let eps = m.layers[li].ffn_norm_eps;
                let mut sums = Vec::new();
                let mut norms = Vec::new();
                for r in 0..rows {
                    let (s, n) = crate::inference::kernel::fused::fused_add_rmsnorm_dual(
                        &attn_one.narrow(1, r, 1).unwrap().contiguous().unwrap(),
                        &x.narrow(1, r, 1).unwrap().contiguous().unwrap(),
                        &w,
                        eps,
                    )
                    .unwrap();
                    sums.push(s);
                    norms.push(n);
                }
                let sum_one = Tensor::cat(&sums.iter().collect::<Vec<_>>(), 1).unwrap();
                let norm_one = Tensor::cat(&norms.iter().collect::<Vec<_>>(), 1).unwrap();
                let (sum_rows, norm_rows) =
                    crate::inference::kernel::fused::fused_add_rmsnorm_dual(&attn_one, &x, &w, eps)
                        .unwrap();
                (
                    max_diff(&sum_one, &sum_rows).max(max_diff(&norm_one, &norm_rows)),
                    norm_one,
                )
            }
            None => (0.0, xn.clone()),
        };
        // The feed-forward from the one-row norm.
        let ffn_one = single(
            &mut |row, _| m.layers[li].forward_ffn(row).unwrap(),
            &x_norm_one,
        );
        let ffn_rows = {
            let _order = DecodeOrder::enter();
            m.layers[li].forward_ffn(&x_norm_one).unwrap()
        };
        let d_ffn = max_diff(&ffn_one, &ffn_rows);
        // The whole layer.
        m.layers[li].trim_kv_to(base);
        let layer_one = single(
            &mut |row, r| {
                m.layers[li]
                    .forward(row, None, base + r, None, None, None, None)
                    .unwrap()
            },
            &x,
        );
        m.layers[li].trim_kv_to(base);
        let layer_rows = {
            let _order = DecodeOrder::enter();
            m.layers[li]
                .forward(&x, None, base, None, None, None, None)
                .unwrap()
        };
        let d_layer = max_diff(&layer_one, &layer_rows);
        if d_attn > 0.0 || d_norm > 0.0 || d_ffn > 0.0 || d_layer > 0.0 {
            println!(
                "{tag} layer {li:>2}: attention {d_attn:.3e}, add+norm {d_norm:.3e}, feed-forward {d_ffn:.3e}, layer {d_layer:.3e}"
            );
            first_bad.get_or_insert(li);
        }
        hidden = layer_one;
    }
    match first_bad {
        Some(li) => println!("{tag}: the rows first differ from the steps at layer {li}"),
        None => println!("{tag}: every layer's rows are their decode steps"),
    }
    assert!(
        first_bad.is_none(),
        "a layer's rows differ from its decode steps"
    );
}
