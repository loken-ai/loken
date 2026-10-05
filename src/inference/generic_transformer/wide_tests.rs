//! The wide-residual model at toy size against the reference that produced its weights.
//!
//! `tests/vectors/qwen4exp/toy.gguf` holds a seeded eight-layer model with every structural
//! feature on (two attention layers behind an indexer whose budget drops blocks at 16 tokens,
//! an n-gram layer, GDN layers, a routed mixture with a shared expert); `vectors.safetensors`
//! the reference's logits for two prompts, one inside the indexer's budget and one past it.

use super::*;
use crate::inference::engine::llm_engine::KvQuant;
use safetensors::SafeTensors;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/qwen4exp")
}

fn toy() -> GenericHeteroTransformer {
    toy_on(None)
}

/// The toy placed on card `ordinal`, or on the host.
fn toy_on(ordinal: Option<usize>) -> GenericHeteroTransformer {
    let file = std::fs::File::open(dir().join("toy.gguf")).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
    let mut content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
    let layers = content.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    let cards: HashMap<usize, Device> = match ordinal {
        #[cfg(feature = "cuda")]
        Some(o) => [(o, Device::new_cuda(o).expect("a card for the card test"))]
            .into_iter()
            .collect(),
        _ => HashMap::new(),
    };
    let on_card = if ordinal.is_some() { layers } else { 0 };
    GenericHeteroTransformer::from_gguf_with_kv_quant(
        content,
        &mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(layers, on_card, ordinal.unwrap_or(0)),
        KvQuant::Off,
        None,
    )
    .unwrap()
}

fn vectors() -> Vec<u8> {
    std::fs::read(dir().join("vectors.safetensors")).unwrap()
}

fn f32s(v: &SafeTensors, name: &str) -> Vec<f32> {
    v.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn ids(v: &SafeTensors, name: &str) -> Vec<u32> {
    v.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The largest magnitude among `x`: the scale errors are judged on.
fn scale_of(x: &[f32]) -> f32 {
    x.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// Cosine similarity of two vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb): (f32, f32) = (a.iter().map(|x| x * x).sum(), b.iter().map(|x| x * x).sum());
    dot / (na.sqrt() * nb.sqrt())
}

/// The weights of the toy are float, so the loader requantises its projections and the
/// matmuls quantise their activations: a layer built from them departs from the dense
/// reference by the rounding of those formats, a few percent of the output's scale, and the
/// eight layers compound it. A wrong head order, gate or norm departs by the scale itself.
const QUANTISED_SLACK: f32 = 0.12;

/// The direction the logits keep. Two vectors whose difference is a fraction `e` of their norm
/// and is orthogonal to them have cosine `1 - e^2 / 2`. The host's budget is
/// `QUANTISED_SLACK`; a card adds the rounding of its half-precision activations, an
/// independent error of at most the same size, so the budget doubles. A wrong head order,
/// gate or norm departs by `e` near 1, which this bound rejects.
const MIN_COSINE: f32 = 1.0 - (2.0 * QUANTISED_SLACK) * (2.0 * QUANTISED_SLACK) / 2.0;

/// Logits agree within the rounding, with the same direction, and with the same best token
/// wherever the reference's best leads its runner-up by more than the rounding.
fn within_rounding(got: &[f32], want: &[f32], scale: f32, vocab: usize, what: &str) {
    let d = max_diff(got, want);
    assert!(
        d < QUANTISED_SLACK * scale,
        "{what}: differs by {d} on a scale of {scale}"
    );
    for (row, (g, w)) in got.chunks(vocab).zip(want.chunks(vocab)).enumerate() {
        assert!(
            cosine(g, w) > MIN_COSINE,
            "{what} row {row}: cosine {}",
            cosine(g, w)
        );
        let best = |x: &[f32]| (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap();
        let mut sorted = w.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted[0] - sorted[1] > QUANTISED_SLACK * scale {
            assert_eq!(best(g), best(w), "{what} row {row}: best token");
        }
    }
}

#[test]
fn the_toy_logits_match_the_reference_for_a_prompt_inside_and_past_the_budget() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    for tag in ["a", "b"] {
        let tokens = ids(&v, &format!("{tag}.ids"));
        let want = f32s(&v, &format!("{tag}.logits"));
        let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &Device::Cpu).unwrap();
        let got = model.forward_all(&input, 0).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        within_rounding(
            &got,
            &want,
            scale_of(&want),
            want.len() / tokens.len(),
            &format!("prompt {tag} logits"),
        );
    }
}

#[test]
fn decoding_one_token_at_a_time_gives_the_logits_of_the_whole_prompt() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "b.ids");
    let want = f32s(&v, "b.logits");
    let vocab = want.len() / tokens.len();
    for (i, t) in tokens.iter().enumerate() {
        let input = Tensor::from_vec(vec![*t], (1, 1), &Device::Cpu).unwrap();
        let got = model.forward(&input, i).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        within_rounding(
            &got,
            &want[i * vocab..(i + 1) * vocab],
            scale_of(&want),
            vocab,
            &format!("position {i}"),
        );
    }
}

#[test]
fn a_chunked_prefill_gives_the_logits_of_the_whole_prompt() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "b.ids");
    let want = f32s(&v, "b.logits");
    let vocab = want.len() / tokens.len();
    let mut at = 0;
    for chunk in tokens.chunks(7) {
        let input = Tensor::from_vec(chunk.to_vec(), (1, chunk.len()), &Device::Cpu).unwrap();
        let got = model.forward_all(&input, at).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        within_rounding(
            &got,
            &want[at * vocab..(at + chunk.len()) * vocab],
            scale_of(&want),
            vocab,
            &format!("chunk at {at}"),
        );
        at += chunk.len();
    }
}

#[test]
fn the_mixers_of_the_wide_residual_match_the_reference_exactly() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let model = toy();
    let n = ids(&v, "a.ids").len();
    let t = |name: &str, width: usize| {
        Tensor::from_vec(f32s(&v, name), (1, n, width), &Device::Cpu).unwrap()
    };
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let wide = super::wide::widen_embedding(&t("a.embed", 64), 4).unwrap();
    let (mixed, inj) = model.layers[0]
        .hyper
        .as_ref()
        .unwrap()
        .attn
        .mix(&wide)
        .unwrap();
    assert!(max_diff(&flat(&mixed), &f32s(&v, "a.layer0_attn_hc_mixed")) < 1e-5);
    assert!(max_diff(&flat(&inj.unwrap()), &f32s(&v, "a.layer0_attn_hc_inj")) < 1e-5);
}

#[test]
fn the_ngram_features_match_the_reference() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "a.ids");
    let n = tokens.len();
    let ng = model.ngram.as_mut().unwrap();
    let rows = model.layers[1]
        .ngram
        .as_mut()
        .unwrap()
        .history
        .rows(&ng.meta, &tokens);
    let emb = ng.table.gather(&rows).unwrap();
    let emb =
        Tensor::from_vec(emb, (1, n, ng.meta.heads() * ng.meta.row_dim), &Device::Cpu).unwrap();
    let r = Tensor::from_vec(f32s(&v, "a.layer0_out"), (1, n, 256), &Device::Cpu).unwrap();
    let out = model.layers[1]
        .ngram
        .as_mut()
        .unwrap()
        .forward(&r, &emb, 0)
        .unwrap();
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(max_diff(&got, &f32s(&v, "a.layer1_ple")) < 1e-4);
}

#[test]
fn the_indexer_keeps_the_tokens_the_reference_keeps() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let n = ids(&v, "b.ids").len();
    let x = Tensor::from_vec(f32s(&v, "b.layer3_attn_hc_mixed"), (1, n, 64), &Device::Cpu).unwrap();
    // Blocks that score alike are an arbitrary pick: two selections agree when every block
    // they disagree on scores the same as the last block kept.
    let mut all_scores: Vec<Vec<f32>> = Vec::new();
    let q_rows = {
        let sp = model.layers[3].sparse.as_mut().unwrap();
        sp.observe_scored(&x, 0, |_, s| all_scores.push(s.to_vec()))
            .unwrap()
            .unwrap()
    };
    let want = f32s(&v, "b.layer3_selected");
    for (i, kept) in q_rows.iter().enumerate() {
        let mut row = vec![0f32; n];
        for t in kept {
            row[*t as usize] = 1.0;
        }
        let reference = &want[i * n..(i + 1) * n];
        if row == reference {
            continue;
        }
        let differing: Vec<usize> = (0..n).filter(|&t| row[t] != reference[t]).collect();
        let scores = &all_scores[i];
        let block = model.layers[3].sparse.as_ref().unwrap().meta.block;
        let tie = differing
            .iter()
            .filter(|&&t| t / block < scores.len())
            .all(|&t| scores[t / block] == scores[differing[0] / block]);
        assert!(
            tie,
            "query {i}: selections differ outside a tie, at {differing:?}"
        );
    }
}

#[test]
fn the_selected_attention_matches_the_reference_block() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let n = ids(&v, "b.ids").len();
    let x = Tensor::from_vec(f32s(&v, "b.layer3_attn_hc_mixed"), (1, n, 64), &Device::Cpu).unwrap();
    let want = f32s(&v, "b.layer3_attn");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let selected = model.layers[3].wide_attention(&x, None, 0).unwrap();
    let d_sel = max_diff(&flat(&selected), &want);
    // The same block with the budget lifted reads every token and must read as dense.
    let mask = model.make_mask(n, 0, None).ok();
    model.layers[3].sparse = None;
    let dense = model.layers[3]
        .wide_attention(&x, mask.as_ref(), 0)
        .unwrap();
    let d_dense = max_diff(&flat(&dense), &want);
    eprintln!("selected {d_sel} dense {d_dense} scale {scale}");
    assert!(
        d_sel < 0.03 * scale,
        "selected attention differs by {d_sel} on a scale of {scale}"
    );
    assert!(
        d_dense > 2.0 * d_sel,
        "the block cannot tell a selection from dense attention"
    );
}

#[test]
fn a_low_rank_wide_residual_is_not_the_deepseek_port() {
    let file = std::fs::File::open(dir().join("toy.gguf")).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let content = gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..])).unwrap();
    assert!(!crate::inference::model::deepseek_v41::claims_gguf(
        &content
    ));
}

#[test]
fn a_prefix_snapshot_resumes_to_the_same_logits() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "b.ids");
    let split = 24;
    let row = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let logits = model.forward(&row(&tokens[..split]), 0).unwrap();
    model.snapshot_prefix(&tokens[..split], &logits).unwrap();
    let first = flat(&model.forward_all(&row(&tokens[split..]), split).unwrap());
    let (_, length) = model.restore_kv_snapshot(0).unwrap();
    assert_eq!(length, split);
    let again = flat(&model.forward_all(&row(&tokens[split..]), split).unwrap());
    assert_eq!(first, again);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn the_toy_on_a_card_gives_the_logits_of_the_host() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut host = toy();
    let mut card = toy_on(Some(0));
    for tag in ["a", "b"] {
        let tokens = ids(&v, &format!("{tag}.ids"));
        let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &Device::Cpu).unwrap();
        let on_host = host
            .forward_all(&input, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let input_card = input.to_device(&card.layers[0].cos.device()).unwrap();
        let on_card = card
            .forward_all(&input_card, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = f32s(&v, &format!("{tag}.logits"));
        within_rounding(
            &on_card,
            &want,
            scale_of(&want),
            want.len() / tokens.len(),
            &format!("card {tag} against the reference"),
        );
        eprintln!(
            "prompt {tag}: card against host {}",
            max_diff(&on_card, &on_host)
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "diagnostic on a real file named by LOKEN_Q38_FILE, needs an idle CUDA card"]
fn real_layers_on_a_card_against_the_host() {
    let path = std::env::var("LOKEN_Q38_FILE").expect("LOKEN_Q38_FILE names the file");
    let open = || {
        let file = std::fs::File::open(&path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        let mut content =
            gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
                .unwrap();
        content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
        (content, mmap)
    };
    let layers_on_card = 4;
    let (c1, m1) = open();
    let n = c1.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    let mut host = GenericHeteroTransformer::from_gguf_with_kv_quant(
        c1,
        &m1[..],
        &HashMap::new(),
        &HeteroPlan::forced_gpu(n, 0, 0),
        KvQuant::Off,
        Some(512),
    )
    .unwrap();
    let (c2, m2) = open();
    let card = Device::new_cuda(0).unwrap();
    let cards: HashMap<usize, Device> = [(0, card.clone())].into_iter().collect();
    let mut onc = GenericHeteroTransformer::from_gguf_with_kv_quant(
        c2,
        &m2[..],
        &cards,
        &HeteroPlan::forced_gpu(n, layers_on_card, 0),
        KvQuant::Off,
        Some(512),
    )
    .unwrap();
    let tokens: Vec<u32> = vec![760, 6511, 314, 9338, 369, 1234, 5678, 910];
    let seq = tokens.len();
    let ids = Tensor::from_vec(tokens.clone(), (1, seq), &Device::Cpu).unwrap();
    let embed = host.embeddings.forward(&ids).unwrap();
    let mut h = super::wide::widen_embedding(&embed, 4).unwrap();
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
    let report = |what: &str, a: &Tensor, b: &Tensor| {
        let (x, y) = (flat(a), flat(b));
        eprintln!(
            "    {what:<26} max {:>10.3e}  scale {:>10.3e}  cosine {:.5}",
            max_diff(&x, &y),
            scale_of(&x),
            cosine(&x, &y)
        );
    };
    let to_card = |t: &Tensor| t.to_device(&card).unwrap();
    // Fill the card's free memory with NaN and give it back: whatever the allocator hands out
    // next is stale, and a kernel that reads a byte nobody wrote shows it.
    let poison = |gib: usize| {
        let held: Vec<Tensor> = (0..gib)
            .filter_map(|_| Tensor::full(f32::NAN, 256usize << 20, &card).ok())
            .collect();
        drop(held);
    };
    let poisoned = std::env::var("LOKEN_Q38_POISON").is_ok();
    for i in 0..layers_on_card {
        if poisoned {
            poison(6);
        }
        eprintln!(
            "layer {i} (poisoned: {poisoned}): {}",
            if matches!(host.layers[i].mixer, mixer::Mixer::Attention) {
                "attention"
            } else {
                "delta rule"
            }
        );
        let mask_h = host.make_mask(seq, 0, None).ok();
        let mask_c = onc.make_mask(seq, 0, Some(0)).ok();
        // The attention-side mixer, every intermediate.
        let mut trace_h: Vec<(String, Tensor)> = Vec::new();
        let mut trace_c: Vec<(String, Tensor)> = Vec::new();
        let (mixed_h, inj_h) = host.layers[i]
            .hyper
            .as_ref()
            .unwrap()
            .attn
            .mix_traced(&h, |k, t| trace_h.push((k.to_string(), t.clone())))
            .unwrap();
        let (mixed_c, _) = onc.layers[i]
            .hyper
            .as_ref()
            .unwrap()
            .attn
            .mix_traced(&to_card(&h), |k, t| {
                trace_c.push((k.to_string(), t.clone()))
            })
            .unwrap();
        eprintln!("  hyper mixer (attention side):");
        for ((k, a), (_, b)) in trace_h.iter().zip(&trace_c) {
            report(k, a, b);
        }
        let x_h = mixed_h.to_dtype(DType::F32).unwrap();
        let x_c16 = mixed_c.to_dtype(DType::F16).unwrap();
        let x_c32 = mixed_c.to_dtype(DType::F32).unwrap();
        let out_h;
        if let mixer::Mixer::DeltaNet(_) = &host.layers[i].mixer {
            // The projections.
            let (ph, pc) = match (&host.layers[i].mixer, &onc.layers[i].mixer) {
                (mixer::Mixer::DeltaNet(a), mixer::Mixer::DeltaNet(b)) => {
                    (a.project(&x_h).unwrap(), b.project(&x_c16).unwrap())
                }
                _ => unreachable!(),
            };
            eprintln!("  projections (host f32 against card f16 input):");
            report("qkv", &ph.0, &pc.0);
            report("z", &ph.1, &pc.1);
            report("alpha", &ph.2, &pc.2);
            report("beta", &ph.3, &pc.3);
            // The recurrence.
            host.layers[i].mixer.reset();
            let core_h = host.layers[i].mixer.forward(&x_h, 0).unwrap();
            let run_card = |layer: &mut GenericTransformerLayer, x: &Tensor, rows: bool| {
                layer.mixer.reset();
                match &mut layer.mixer {
                    mixer::Mixer::DeltaNet(g) => {
                        if rows {
                            g.forward_rows(x, 1, seq).unwrap()
                        } else {
                            g.forward_card(x, seq).unwrap()
                        }
                    }
                    _ => unreachable!(),
                }
            };
            eprintln!("  delta-rule core against the host:");
            for call in 0..3 {
                report(
                    &format!("card fused, f16, call {call}"),
                    &core_h,
                    &run_card(&mut onc.layers[i], &x_c16, false),
                );
            }
            report(
                "card fused, f32 input",
                &core_h,
                &run_card(&mut onc.layers[i], &x_c32, false),
            );
            report(
                "card per-row ops, f16",
                &core_h,
                &run_card(&mut onc.layers[i], &x_c16, true),
            );
            report(
                "card per-row ops, f32",
                &core_h,
                &run_card(&mut onc.layers[i], &x_c32, true),
            );
            let o_h = host.layers[i].attn_output.forward(&core_h).unwrap();
            let core_c = run_card(&mut onc.layers[i], &x_c16, false);
            let o_c = onc.layers[i].attn_output.forward(&core_c).unwrap();
            report("output projection", &o_h, &o_c);
            out_h = o_h;
        } else {
            let a_h = host.layers[i]
                .wide_attention(&x_h, mask_h.as_ref(), 0)
                .unwrap();
            let a_c = onc.layers[i]
                .wide_attention(&x_c16, mask_c.as_ref(), 0)
                .unwrap();
            eprintln!("  attention block:");
            report("block output", &a_h, &a_c);
            out_h = a_h;
        }
        let r1 = super::hyper::combine(&h, &out_h, &inj_h.unwrap()).unwrap();
        // The feed-forward side.
        let mut fh: Vec<(String, Tensor)> = Vec::new();
        let mut fc: Vec<(String, Tensor)> = Vec::new();
        let (fm_h, _) = host.layers[i]
            .hyper
            .as_ref()
            .unwrap()
            .ffn
            .mix_traced(&r1, |k, t| fh.push((k.to_string(), t.clone())))
            .unwrap();
        let (fm_c, _) = onc.layers[i]
            .hyper
            .as_ref()
            .unwrap()
            .ffn
            .mix_traced(&to_card(&r1), |k, t| fc.push((k.to_string(), t.clone())))
            .unwrap();
        eprintln!("  hyper mixer (feed-forward side):");
        for ((k, a), (_, b)) in fh.iter().zip(&fc) {
            report(k, a, b);
        }
        let moe_h = host.layers[i].moe.as_ref().unwrap();
        let moe_c = onc.layers[i].moe.as_ref().unwrap();
        let xf_h = fm_h.to_dtype(DType::F32).unwrap();
        let xf_c = fm_c.to_dtype(DType::F16).unwrap();
        eprintln!("  mixture:");
        report(
            "router logits",
            &moe_h.gate_inp.forward(&xf_h).unwrap(),
            &moe_c
                .gate_inp
                .forward(&xf_c.to_dtype(DType::F32).unwrap())
                .unwrap(),
        );
        let eh = moe_h
            .forward_rows(
                &xf_h,
                &xf_h,
                host.layers[i].embedding_length_for_moe,
                host.layers[i].ffn_norm_eps,
            )
            .unwrap();
        // The card twice: a result that moves between two identical calls reads memory nobody wrote.
        for call in 0..3 {
            let ec = moe_c
                .forward_rows(
                    &xf_c,
                    &xf_c,
                    onc.layers[i].embedding_length_for_moe,
                    onc.layers[i].ffn_norm_eps,
                )
                .unwrap();
            report(&format!("whole mixture, call {call}"), &eh, &ec);
        }

        h = host.layers[i]
            .forward_wide(&h, mask_h.as_ref(), 0, None)
            .unwrap();
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "diagnostic on a real file named by LOKEN_Q38_FILE, needs both cards idle"]
fn real_model_layer_by_layer_on_the_cards_against_the_host() {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};
    let path = std::env::var("LOKEN_Q38_FILE").expect("LOKEN_Q38_FILE names the file");
    let open = || {
        let file = std::fs::File::open(&path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        let mut content =
            gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
                .unwrap();
        content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
        (content, mmap)
    };
    let (c1, m1) = open();
    let n = c1.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    let mut host = GenericHeteroTransformer::from_gguf_with_kv_quant(
        c1,
        &m1[..],
        &HashMap::new(),
        &HeteroPlan::forced_gpu(n, 0, 0),
        KvQuant::Off,
        Some(512),
    )
    .unwrap();
    let (c2, m2) = open();
    let cards: HashMap<usize, Device> = [
        (0, Device::new_cuda(0).unwrap()),
        (1, Device::new_cuda(1).unwrap()),
    ]
    .into_iter()
    .collect();
    let split = n / 2;
    let plan = HeteroPlan {
        segments: vec![
            HeteroSegment {
                kind: DeviceKind::Cuda(0),
                layer_start: 0,
                layer_end: split,
                free_memory_bytes: u64::MAX,
            },
            HeteroSegment {
                kind: DeviceKind::Cuda(1),
                layer_start: split,
                layer_end: n,
                free_memory_bytes: u64::MAX,
            },
        ],
        total_layers: n,
        host_experts_from: None,
    };
    let mut onc = GenericHeteroTransformer::from_gguf_with_kv_quant(
        c2,
        &m2[..],
        &cards,
        &plan,
        KvQuant::Off,
        Some(512),
    )
    .unwrap();
    let prompt =
        std::env::var("LOKEN_Q38_TOKENS").unwrap_or_else(|_| "785,6722,314,9338,369".into());
    let tokens: Vec<u32> = prompt.split(',').map(|t| t.parse().unwrap()).collect();
    let seq = tokens.len();
    let ids = Tensor::from_vec(tokens, (1, seq), &Device::Cpu).unwrap();
    let embed = host.embeddings.forward(&ids).unwrap();
    let mut hh = super::wide::widen_embedding(&embed, 4).unwrap();
    let mut hc = hh.clone();
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
    for i in 0..n {
        let dev = match onc.layer_devs[i] {
            LayerDevice::Cuda(k) => (k, onc.cuda_devices[&k].clone()),
            LayerDevice::Cpu => unreachable!(),
        };
        let mask_h = host.make_mask(seq, 0, None).ok();
        let mask_c = onc.make_mask(seq, 0, Some(dev.0)).ok();
        hh = host.layers[i]
            .forward_wide(&hh, mask_h.as_ref(), 0, None)
            .unwrap();
        hc = onc.layers[i]
            .forward_wide(&hc.to_device(&dev.1).unwrap(), mask_c.as_ref(), 0, None)
            .unwrap();
        let (a, b) = (flat(&hh), flat(&hc));
        // The last row alone: the one the head reads.
        let w = a.len() / seq;
        let (la, lb) = (&a[a.len() - w..], &b[b.len() - w..]);
        eprintln!(
            "layer {i:2}: cosine {:.5}  last row cosine {:.5}  scale {:>9.3e}  last row max diff {:>9.3e}",
            cosine(&a, &b), cosine(la, lb), scale_of(la), max_diff(la, lb)
        );
        if i < 5 {
            let per_row: Vec<String> = (0..seq)
                .map(|r| {
                    format!(
                        "{:.4}/{:.2e}",
                        cosine(&a[r * w..(r + 1) * w], &b[r * w..(r + 1) * w]),
                        scale_of(&a[r * w..(r + 1) * w])
                    )
                })
                .collect();
            eprintln!("          rows (cosine/scale): {}", per_row.join("  "));
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "diagnostic on a real file named by LOKEN_Q38_FILE, needs idle cards"]
fn the_decode_step_by_stage() {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};
    let path = std::env::var("LOKEN_Q38_FILE").expect("LOKEN_Q38_FILE names the file");
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
    let mut content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
    let n = content.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    let cards: HashMap<usize, Device> = [
        (0, Device::new_cuda(0).unwrap()),
        (1, Device::new_cuda(1).unwrap()),
    ]
    .into_iter()
    .collect();
    let split = n * 26 / 48;
    let seg = |k: usize, a: usize, b: usize| HeteroSegment {
        kind: DeviceKind::Cuda(k),
        layer_start: a,
        layer_end: b,
        free_memory_bytes: u64::MAX,
    };
    let plan = HeteroPlan {
        segments: vec![seg(0, 0, split), seg(1, split, n)],
        total_layers: n,
        host_experts_from: None,
    };
    let mut model = GenericHeteroTransformer::from_gguf_with_kv_quant(
        content,
        &mmap[..],
        &cards,
        &plan,
        KvQuant::Q8,
        Some(512),
    )
    .unwrap();
    let prompt: Vec<u32> = (0..64).map(|i| 1000 + (i * 37) % 5000).collect();
    let ids = Tensor::from_vec(prompt.clone(), (1, 64), &Device::Cpu).unwrap();
    let first = model.forward(&ids, 0).unwrap();
    let argmax = |t: &Tensor| {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let mut tok = argmax(&first);
    let mut pos = 64;
    let step = |model: &mut GenericHeteroTransformer, tok: u32, pos: usize| {
        let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu).unwrap();
        let l = model.forward(&t, pos).unwrap();
        argmax(&l)
    };
    for _ in 0..8 {
        tok = step(&mut model, tok, pos);
        pos += 1;
    }
    let t = std::time::Instant::now();
    for _ in 0..16 {
        tok = step(&mut model, tok, pos);
        pos += 1;
    }
    let plain = t.elapsed().as_secs_f64() / 16.0;
    eprintln!(
        "decode step, no profile: {:.2} ms ({:.1} tok/s)",
        plain * 1e3,
        1.0 / plain
    );
    crate::inference::offload::stage_prof_enable(true);
    for _ in 0..16 {
        tok = step(&mut model, tok, pos);
        pos += 1;
    }
    let mut stages = crate::inference::offload::stage_prof_snapshot();
    crate::inference::offload::stage_prof_enable(false);
    stages.sort_by_key(|s| std::cmp::Reverse(s.1));
    let total: u64 = stages.iter().map(|s| s.1).sum();
    for (name, ns, calls) in stages {
        eprintln!(
            "  {name:<12} {:>7.3} ms per step  {:>5.1}%  ({} calls per step)",
            ns as f64 / 16.0 / 1e6,
            100.0 * ns as f64 / total as f64,
            calls / 16
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn the_fused_card_stages_match_the_tensor_ops() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let model = toy_on(Some(0));
    let card = model.layers[0].cos.device();
    let n = ids(&v, "b.ids").len();
    let r = Tensor::from_vec(f32s(&v, "b.layer1_out"), (1, n, 256), &Device::Cpu)
        .unwrap()
        .to_device(&card)
        .unwrap();
    let hc = model.layers[1].hyper.as_ref().unwrap();
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
    let (mixed_ref, inj_ref) = hc.attn.mix(&r).unwrap();
    let xn = hc.attn.normed_on_card(&r).unwrap();
    let (mixed, inj_raw) = hc.attn.mix_on_card(&xn, false).unwrap();
    assert!(
        max_diff(&flat(&mixed), &flat(&mixed_ref)) < 1e-4,
        "the sublayer input"
    );
    let (s, d) = (hc.attn.streams(), hc.attn.dim());
    let weights = (crate::tensor::ops::sigmoid(&(inj_raw.clone().unwrap() / s as f64).unwrap())
        .unwrap()
        * 2.0)
        .unwrap();
    assert!(
        max_diff(&flat(&weights), &flat(&inj_ref.clone().unwrap())) < 1e-5,
        "the write-back weights"
    );
    // the write-back and the next norm, against the tensor ops
    let out = (mixed_ref.clone() * 0.5).unwrap();
    let want = super::hyper::combine(&r, &out, &inj_ref.unwrap()).unwrap();
    let want_norm = super::hyper::grouped_rms(&want, hc.ffn.norm_params().0, s, d, 1e-6).unwrap();
    let (got, got_norm) =
        super::hyper::combine_on_card(&r, &out, &inj_raw.unwrap(), Some(&hc.ffn), s, d).unwrap();
    assert!(max_diff(&flat(&got), &flat(&want)) < 1e-5, "the write-back");
    assert!(
        max_diff(&flat(&got_norm.unwrap()), &flat(&want_norm)) < 1e-4,
        "the next norm"
    );
    // an F16 sublayer output
    let (f16_in, _) = hc.attn.mix_on_card(&xn, true).unwrap();
    assert_eq!(f16_in.dtype(), DType::F16);
    assert!(
        max_diff(&flat(&f16_in), &flat(&mixed_ref)) < 5e-3,
        "the F16 sublayer input"
    );
}

/// A mixer handed more rows than its block works on them block by block and gives, row for row,
/// what it gives when it takes them all at once: every operation of the mixer is per row.
#[test]
fn a_mixer_blocked_over_its_rows_matches_the_whole() {
    let m = toy();
    let hc = m.layers[0].hyper.as_ref().expect("a wide layer");
    let wide = m.config.hyper.unwrap().streams * m.config.embedding_length;
    let rows = 2 * hyper::mixer_block_rows() + 37;
    let data: Vec<f32> = (0..rows * wide)
        .map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let r = Tensor::from_vec(data, (1, rows, wide), &Device::Cpu).unwrap();
    let (blocked, inj_b) = hc.attn.mix(&r).unwrap();
    let (whole, inj_w) = hc.attn.mix_traced(&r, |_, _| {}).unwrap();
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(flat(&blocked), flat(&whole));
    assert_eq!(flat(&inj_b.unwrap()), flat(&inj_w.unwrap()));
}

/// A wide model's prefill footprint grows with the streams and its wanted chunk with the experts,
/// within the ladder's top and the ceiling; a model without a routed mixture asks for no chunk.
#[cfg(feature = "cuda")]
#[test]
fn the_wide_prefill_footprint_follows_the_model() {
    let a = hyper::prefill_footprint(4, 2560, Some((160, 10)), 640);
    let b = hyper::prefill_footprint(4, 2560, Some((512, 10)), 640);
    assert!(a.bytes_per_token > 0 && a.fixed_bytes > 0);
    assert!(a.target_chunk >= 2048 && b.target_chunk > a.target_chunk);
    assert!(b.target_chunk <= 8192);
    let plain = hyper::prefill_footprint(4, 2560, None, 0);
    assert_eq!(plain.target_chunk, 0);
    assert_eq!(
        hyper::prefill_footprint(8, 2560, None, 0).bytes_per_token,
        2 * plain.bytes_per_token
    );
    // The routed experts that run on the card add their own activations per routed pair.
    assert!(a.bytes_per_token > plain.bytes_per_token);
}

/// A prefill pinned at the whole prompt, at a divisor of its length and at a size that leaves a
/// short last chunk gives the logits of the unchunked reference at every position.
#[test]
fn a_prefill_at_any_pinned_chunk_gives_the_unchunked_logits() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let want = f32s(&v, "b.logits");
    let n = tokens.len();
    let vocab = want.len() / n;
    let divisor = (2..n)
        .find(|d| n.is_multiple_of(*d) && n / d > 1)
        .unwrap_or(1);
    let ragged = (2..n)
        .rev()
        .find(|c| !n.is_multiple_of(*c) && n.div_ceil(*c) > 1)
        .unwrap_or(1);
    for chunk in [n, n / divisor, ragged] {
        let mut model = toy();
        let mut at = 0;
        while at < n {
            let len = chunk.min(n - at);
            let input =
                Tensor::from_vec(tokens[at..at + len].to_vec(), (1, len), &Device::Cpu).unwrap();
            let got = model.forward_all(&input, at).unwrap();
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            within_rounding(
                &got,
                &want[at * vocab..(at + len) * vocab],
                scale_of(&want),
                vocab,
                &format!("chunk {chunk} at {at}"),
            );
            at += len;
        }
    }
}
