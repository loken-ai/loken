//! A model's final hidden state with its first layers on a card, against the same state
//! computed wholly on the host. The first layer count at which they part names the layer
//! kind the card computes differently.
//!
//! Ignored by default: it needs the model in the store and a CUDA card. Run with
//!   LOKEN_PARITY_MODEL=olmo2:7b cargo test --release card_and_host_agree -- --ignored --nocapture

use super::*;

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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn card_and_host_agree() {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "olmo2:7b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let n_layers = GenericTransformerConfig::from_gguf(&read(), {
        let c = read();
        &c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone()
    })
    .unwrap()
    .n_layers;
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    // A prompt long enough to take several prefill chunks.
    let ids: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let input = Tensor::new(ids.as_slice(), &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let hidden = |on_card: usize| {
        let plan = HeteroPlan::forced_gpu(n_layers, on_card, 0);
        let mut m = GenericHeteroTransformer::from_gguf(read(), &mmap[..], &cards, &plan).unwrap();
        m.embed_last_hidden(&input).unwrap()
    };
    let host = hidden(0);
    let mut counts = vec![1usize, 2, 4, 8, 16];
    counts.push(n_layers);
    counts.retain(|&n| n <= n_layers);
    for n in counts {
        let h = hidden(n);
        let worst = host
            .iter()
            .zip(&h)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "{tag}: {n:>3} layers on the card: cosine {:.6}, max diff {worst:.4}",
            cosine(&host, &h)
        );
    }
}

/// Layer 0's operations one by one, each fed the same host input on the card and on the
/// host: the first one whose outputs part is the one the card computes differently.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn first_layer_operations_agree() {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "olmo2:7b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card.clone())].into_iter().collect();
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
    let host_m = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 0, 0),
    )
    .unwrap();
    let card_m = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, n_layers, 0),
    )
    .unwrap();
    let layer: usize = std::env::var("LOKEN_PARITY_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (h, c) = (&host_m.layers[layer], &card_m.layers[layer]);
    let dt = |m: Option<&QMatMul>| {
        m.and_then(|q| q.cpu_raw())
            .map(|r| format!("{:?} k={}", r.0, r.1))
            .unwrap_or_default()
    };
    println!(
        "layer {layer}: attn_v {} ffn_down {}",
        dt(h.attn_v.as_ref()),
        dt(Some(&h.ffn_down))
    );
    let ids: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let input = Tensor::new(ids.as_slice(), &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let x = host_m
        .embeddings
        .forward(
            &input
                .to_device(&host_m.embeddings.embeddings().device())
                .unwrap(),
        )
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let report = |what: &str, a: &Tensor, b: &Tensor| {
        let a = a
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!(
            "{what:>14}: cosine {:.7}  max diff {worst:.5}  (max |x| {scale:.3})",
            cosine(&a, &b)
        );
    };
    let on_card = |t: &Tensor| t.to_device(&card).unwrap();
    let pair =
        |f: &dyn Fn(&GenericTransformerLayer, &Tensor) -> Tensor| (f(h, &x), f(c, &on_card(&x)));
    // One row is what a decode step feeds the projections: the GEMV kernels, not the GEMM.
    let row = x.narrow(1, 299, 1).unwrap();
    for (name, hp, cp) in [&h.attn_q, &h.attn_k, &h.attn_v, &h.attn_qkv]
        .into_iter()
        .zip([&c.attn_q, &c.attn_k, &c.attn_v, &c.attn_qkv])
        .zip(["attn_q/1", "attn_k/1", "attn_v/1", "attn_qkv/1"])
        .map(|((a, b), n)| (n, a, b))
    {
        if let (Some(hp), Some(cp)) = (hp, cp) {
            let out = (
                hp.forward(&row).unwrap(),
                cp.forward(&on_card(&row)).unwrap(),
            );
            report(name, &out.0, &out.1);
        }
    }
    let up1 = (
        h.ffn_up.forward(&row).unwrap(),
        c.ffn_up.forward(&on_card(&row)).unwrap(),
    );
    report("ffn_up/1", &up1.0, &up1.1);
    // ffn_down reads the FFN width: half a fused gate+up output.
    let d_in = if h.ffn_gate.is_none() {
        up1.0.narrow(2, 0, up1.0.dims()[2] / 2).unwrap()
    } else {
        up1.0.clone()
    };
    let down1 = (
        h.ffn_down.forward(&d_in).unwrap(),
        c.ffn_down.forward(&on_card(&d_in)).unwrap(),
    );
    report("ffn_down/1", &down1.0, &down1.1);
    for (name, hp, cp) in [
        ("attn_q", &h.attn_q, &c.attn_q),
        ("attn_k", &h.attn_k, &c.attn_k),
        ("attn_v", &h.attn_v, &c.attn_v),
        ("attn_qkv", &h.attn_qkv, &c.attn_qkv),
    ] {
        if let (Some(hp), Some(cp)) = (hp, cp) {
            let out = (hp.forward(&x).unwrap(), cp.forward(&on_card(&x)).unwrap());
            report(name, &out.0, &out.1);
            if name == "attn_q" {
                if let (Some(hn), Some(cn)) = (&h.attn_q_norm, &c.attn_q_norm) {
                    let qn = (
                        hn.forward(&out.0).unwrap(),
                        cn.forward(&on_card(&out.0)).unwrap(),
                    );
                    report("q_norm", &qn.0, &qn.1);
                }
            }
        }
    }
    let up = pair(&|l, t| l.ffn_up.forward(t).unwrap());
    report("ffn_up", &up.0, &up.1);
    // A fused gate+up projection carries the gate in its first half.
    let ffn_in = match &h.ffn_gate {
        Some(g) => g.forward(&x).unwrap().silu().unwrap().mul(&up.0).unwrap(),
        None => {
            let half = up.0.dims()[2] / 2;
            let g = up.0.narrow(2, 0, half).unwrap();
            let u = up.0.narrow(2, half, half).unwrap();
            g.silu().unwrap().mul(&u).unwrap()
        }
    };
    let down = (
        h.ffn_down.forward(&ffn_in).unwrap(),
        c.ffn_down.forward(&on_card(&ffn_in)).unwrap(),
    );
    report("ffn_down", &down.0, &down.1);
    let attn_in = Tensor::randn(0f32, 1f32, (1, 300, h.n_head * h.head_dim), &Device::Cpu).unwrap();
    let o = (
        h.attn_output.forward(&attn_in).unwrap(),
        c.attn_output.forward(&on_card(&attn_in)).unwrap(),
    );
    report("attn_output", &o.0, &o.1);
    if let (Some(hn), Some(cn)) = (&h.post_attn_norm, &c.post_attn_norm) {
        let pn = (
            hn.forward(&o.0).unwrap(),
            cn.forward(&on_card(&o.0)).unwrap(),
        );
        report("post_attn_norm", &pn.0, &pn.1);
    }
    let mut hl = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 0, 0),
    )
    .unwrap();
    let mut cl = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, n_layers, 0),
    )
    .unwrap();
    let mask = hl.make_mask(300, 0, None).ok();
    let whole = (
        hl.layers[layer]
            .forward(&x, mask.as_ref(), 0, None, None, None, None)
            .unwrap(),
        cl.layers[layer]
            .forward(
                &on_card(&x),
                mask.as_ref().map(on_card).as_ref(),
                0,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
    );
    report("whole layer", &whole.0, &whole.1);
}

/// Greedy decode after the same prompt, wholly on the host and wholly on the card: the step
/// at which the tokens part, and how close the logits were when they did.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn card_and_host_decode_alike() {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "olmo2:7b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
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
    let prompt: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let run = |on_card: usize| -> Vec<(u32, Vec<f32>)> {
        let plan = HeteroPlan::forced_gpu(n_layers, on_card, 0);
        let mut m = GenericHeteroTransformer::from_gguf(read(), &mmap[..], &cards, &plan).unwrap();
        let mut out = Vec::new();
        let mut input = Tensor::new(prompt.as_slice(), &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut pos = 0;
        for _ in 0..40 {
            let n = input.dims()[1];
            let logits = m.forward(&input, pos).unwrap();
            pos += n;
            let l = logits
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let t = l
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0 as u32;
            out.push((t, l));
            input = Tensor::new(&[t], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
        }
        out
    };
    let host = run(0);
    for n in [n_layers] {
        let on_card = run(n);
        println!("{tag}: {n} layers on the card");
        for (i, ((ht, hl), (ct, cl))) in host.iter().zip(&on_card).enumerate() {
            let mut sorted = hl.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            println!(
            "{tag} step {i:>2}: host {ht:>6} card {ct:>6}  logits cosine {:.6}  host margin {:.3}",
            cosine(hl, cl),
            sorted[0] - sorted[1]
        );
            if ht != ct {
                break;
            }
        }
    }
}

/// Layer 0 alone: a prefill, then one decoded token, on the card and on the host from the
/// same inputs. Separates the decode path from the prefill path.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn first_layer_decode_agrees() {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "olmo2:7b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card.clone())].into_iter().collect();
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
    let mut hm = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 0, 0),
    )
    .unwrap();
    let mut cm = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 1, 0),
    )
    .unwrap();
    let embed = |m: &GenericHeteroTransformer, ids: &[u32]| {
        let input = Tensor::new(ids, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        m.embeddings
            .forward(
                &input
                    .to_device(&m.embeddings.embeddings().device())
                    .unwrap(),
            )
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    };
    let prompt: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let x = embed(&hm, &prompt);
    let mask = hm.make_mask(300, 0, None).unwrap();
    let hp = hm.layers[0]
        .forward(&x, Some(&mask), 0, None, None, None, None)
        .unwrap();
    let cp = cm.layers[0]
        .forward(
            &x.to_device(&card).unwrap(),
            Some(&mask.to_device(&card).unwrap()),
            0,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    let flat = |t: &Tensor| {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    println!(
        "{tag} prefill: cosine {:.7}",
        cosine(&flat(&hp), &flat(&cp))
    );
    for (step, tok) in [4815u32, 1075, 2000].into_iter().enumerate() {
        let x1 = embed(&hm, &[tok]);
        let pos = 300 + step;
        let hd = hm.layers[0]
            .forward(&x1, None, pos, None, None, None, None)
            .unwrap();
        let cd = cm.layers[0]
            .forward(
                &x1.to_device(&card).unwrap(),
                None,
                pos,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let (a, b) = (flat(&hd), flat(&cd));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!(
            "{tag} decode {step}: cosine {:.7}  max diff {worst:.5}",
            cosine(&a, &b)
        );
    }
}

/// Every layer fed the host's own input for it, on the card and on the host, through a
/// prefill and one decoded token: which layers the card decodes differently, in isolation
/// from the error the layers before them carry.
#[test]
#[ignore = "needs a model in the store and a CUDA card"]
fn each_layer_decodes_alike() {
    let tag = std::env::var("LOKEN_PARITY_MODEL").unwrap_or_else(|_| "olmo2:7b".into());
    let path = model_blob(&tag);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card.clone())].into_iter().collect();
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
    let mut hm = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 0, 0),
    )
    .unwrap();
    let mut cm = GenericHeteroTransformer::from_gguf(
        read(),
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(n_layers, n_layers, 0),
    )
    .unwrap();
    let embed = |m: &GenericHeteroTransformer, ids: &[u32]| {
        let input = Tensor::new(ids, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        m.embeddings
            .forward(
                &input
                    .to_device(&m.embeddings.embeddings().device())
                    .unwrap(),
            )
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    };
    let flat = |t: &Tensor| {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let prompt: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let mask = hm.make_mask(300, 0, None).unwrap();
    let card_mask = mask.to_device(&card).unwrap();
    let mut xp = embed(&hm, &prompt);
    let mut xd = embed(&hm, &[4815u32]);
    for l in 0..n_layers {
        let hp = hm.layers[l]
            .forward(&xp, Some(&mask), 0, None, None, None, None)
            .unwrap();
        let cp = cm.layers[l]
            .forward(
                &xp.to_device(&card).unwrap(),
                Some(&card_mask),
                0,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let hd = hm.layers[l]
            .forward(&xd, None, 300, None, None, None, None)
            .unwrap();
        let cd = cm.layers[l]
            .forward(
                &xd.to_device(&card).unwrap(),
                None,
                300,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let (a, b) = (flat(&hd), flat(&cd));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!(
            "{tag} layer {l:>2}: prefill cosine {:.7}  decode cosine {:.7}  decode max diff {worst:.4} of {scale:.1}",
            cosine(&flat(&hp), &flat(&cp)),
            cosine(&a, &b)
        );
        xp = hp;
        xd = hd;
    }
}
