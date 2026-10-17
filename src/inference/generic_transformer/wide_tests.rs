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
    toy_with(ordinal, KvQuant::Off)
}

/// The toy placed on card `ordinal`, or on the host, with its caches in `kv`.
fn toy_with(ordinal: Option<usize>, kv: KvQuant) -> GenericHeteroTransformer {
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
        kv,
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

/// The convolution's rows are rewritten where they lie, by a forward and by a restore: a step
/// recorded against them reads what a later token leaves.
#[test]
fn the_ngram_rows_stay_where_they_lie_through_a_forward_and_a_restore() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "a.ids");
    let embs: Vec<Tensor> = tokens
        .iter()
        .take(4)
        .map(|t| {
            let ng = model.ngram.as_ref().unwrap();
            let rows = vec![t % 7; ng.meta.heads()];
            let width = ng.meta.heads() * ng.meta.row_dim;
            Tensor::from_vec(ng.table.gather(&rows).unwrap(), (1, 1, width), &Device::Cpu).unwrap()
        })
        .collect();
    let residual = |i: usize| {
        let row: Vec<f32> = (0..256)
            .map(|c| ((c + 7 * i) % 13) as f32 / 13.0 - 0.5)
            .collect();
        Tensor::from_vec(row, (1, 1, 256), &Device::Cpu).unwrap()
    };
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let ple = model.layers[1].ngram.as_mut().unwrap();
    for (i, e) in embs.iter().take(3).enumerate() {
        ple.forward(&residual(i), e, i).unwrap();
    }
    let held = ple.carried().unwrap().clone();
    let saved = ple.snapshot().unwrap();
    let before = flat(&held);
    ple.forward(&residual(3), &embs[3], 3).unwrap();
    assert!(
        held.shares_storage(ple.carried().unwrap()),
        "a forward moved the rows"
    );
    assert_ne!(
        flat(ple.carried().unwrap()),
        before,
        "the token left no trace"
    );
    ple.restore(&saved).unwrap();
    assert!(
        held.shares_storage(ple.carried().unwrap()),
        "a restore moved the rows"
    );
    assert_eq!(flat(ple.carried().unwrap()), before);
    let moves = ple.buffer_moves();
    ple.reset();
    assert!(
        held.shares_storage(ple.carried().unwrap()),
        "a reset moved the rows"
    );
    assert!(flat(ple.carried().unwrap()).iter().all(|&v| v == 0.0));
    assert_eq!(ple.buffer_moves(), moves);
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

/// A card decodes the prompt one token at a time, its indexed layers on the card, and gives the
/// reference's logits at every position.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_card_decoding_one_token_at_a_time_gives_the_logits_of_the_whole_prompt() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy_on(Some(0));
    let card = model.layers[0].cos.device();
    let tokens = ids(&v, "b.ids");
    let want = f32s(&v, "b.logits");
    let vocab = want.len() / tokens.len();
    assert!(model.layers[3].indexed_on_card());
    for (i, t) in tokens.iter().enumerate() {
        let input = Tensor::from_vec(vec![*t], (1, 1), &card).unwrap();
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

/// A dense attention whose rotation covers part of the head decodes on a card, its cache in Q8,
/// as it does on the host.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_card_decoding_a_partly_rotated_dense_attention_matches_the_host() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let dense = |mut m: GenericHeteroTransformer| {
        for l in &mut m.layers {
            l.sparse = None;
        }
        m
    };
    let mut host = dense(toy());
    let mut card_model = dense(toy_with(Some(0), KvQuant::Q8));
    let card = card_model.layers[0].cos.device();
    let rotated = card_model.layers.iter().find(|l| l.attends()).unwrap();
    assert!(rotated.rope_dim > 0 && rotated.rope_dim < rotated.head_dim);
    let tokens = ids(&v, "b.ids");
    for (i, t) in tokens.iter().enumerate() {
        let want = host
            .forward(
                &Tensor::from_vec(vec![*t], (1, 1), &Device::Cpu).unwrap(),
                i,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = card_model
            .forward(&Tensor::from_vec(vec![*t], (1, 1), &card).unwrap(), i)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        within_rounding(
            &got,
            &want,
            scale_of(&want),
            want.len(),
            &format!("position {i}"),
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

/// The mixer of a decode step runs as two launches for one row and for a short batch, and gives
/// what the tensor ops give; a longer chunk takes the stage-by-stage chain and gives it too.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn the_fused_mixer_matches_the_tensor_ops_for_a_decode_step() {
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
    for rows in [1, 3, n] {
        let r = r.narrow(1, 0, rows).unwrap();
        let (want, want_inj) = hc.attn.mix(&r).unwrap();
        let xn = hc.attn.normed_on_card(&r).unwrap();
        let (got, got_inj) = hc.attn.mix_on_card(&xn, false).unwrap();
        assert!(
            max_diff(&flat(&got), &flat(&want)) < 1e-4,
            "{rows} rows: input"
        );
        let s = hc.attn.streams() as f64;
        let weights =
            (crate::tensor::ops::sigmoid(&(got_inj.unwrap() / s).unwrap()).unwrap() * 2.0).unwrap();
        assert!(
            max_diff(&flat(&weights), &flat(&want_inj.unwrap())) < 1e-5,
            "{rows} rows: write-back weights"
        );
    }
    let (down, up) = (hc.attn.down_dense().unwrap(), hc.attn.up_dense().unwrap());
    let (down, up) = (&down, &up);
    let xn = hc.attn.normed_on_card(&r.narrow(1, 0, 1).unwrap()).unwrap();
    let fused = crate::inference::moe_cuda::hc_mix_fused(
        &xn,
        down,
        up,
        None,
        hc.attn.streams(),
        hc.attn.dim(),
        false,
    )
    .unwrap();
    assert!(
        fused.is_some_and(|(_, inj)| inj.is_none()),
        "one row takes the launches"
    );
    let long = hc.attn.normed_on_card(&r).unwrap();
    let past = crate::inference::moe_cuda::hc_mix_fused(
        &long,
        down,
        up,
        None,
        hc.attn.streams(),
        hc.attn.dim(),
        false,
    )
    .unwrap();
    assert_eq!(
        past.is_some(),
        n <= crate::inference::moe_cuda::FUSED_MIX_MAX_ROWS
    );
}

/// The mixer launches read half-precision weights and give what they give on the same weights
/// held in single precision, when the values are exact in the narrower format.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn the_fused_mixer_reads_half_weights_like_single_ones() {
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
    let streams = hc.attn.streams();
    let dim = hc.attn.dim();
    let round = |t: &Tensor| {
        let half = t.to_dtype(DType::F16).unwrap();
        (half.to_dtype(DType::F32).unwrap(), half)
    };
    let (down32, down16) = round(&hc.attn.down_dense().unwrap());
    let (up32, up16) = round(&hc.attn.up_dense().unwrap());
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
    for rows in [1, 3] {
        let xn = hc
            .attn
            .normed_on_card(&r.narrow(1, 0, rows).unwrap())
            .unwrap();
        let mix = |down: &Tensor, up: &Tensor| {
            crate::inference::moe_cuda::hc_mix_fused(&xn, down, up, None, streams, dim, true)
                .unwrap()
                .expect("the launches take these weights")
                .0
        };
        let want = flat(&mix(&down32, &up32));
        let got = flat(&mix(&down16, &up16));
        assert!(max_diff(&got, &want) < 1e-3, "{rows} rows");
    }
}

/// The mixer launches read int8 rows with a scale for each group of values and give what they
/// give on the same weights widened to single precision.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn the_fused_mixer_reads_int8_rows_like_the_widened_weights() {
    use crate::inference::moe_cuda::{hc_mix_fused_over, MixWeight};
    let Ok(card) = Device::new_cuda(0) else {
        return;
    };
    let (streams, dim, rank) = (2usize, 64usize, 64usize);
    let k = streams * dim;
    // Weights of a mixer's shapes and a spread of magnitudes, from a fixed generator.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut matrix = |n: usize, m: usize| {
        let data: Vec<f32> = (0..n * m)
            .map(|i| next() * (1.0 + (i % 7) as f32))
            .collect();
        Tensor::from_vec(data, (n, m), &Device::Cpu)
            .unwrap()
            .to_device(&card)
            .unwrap()
    };
    let (down, up) = (matrix(rank, k), matrix(k, rank));
    let (qd, sd) = hyper::int8_rows(&down, &down.device())
        .unwrap()
        .expect("within the tolerance");
    let (qu, su) = hyper::int8_rows(&up, &up.device())
        .unwrap()
        .expect("within the tolerance");
    // What the rows widen to is what the launches must reproduce.
    let widen =
        |q: &Tensor, s: &Tensor| crate::inference::moe_cuda::hc_dequant(q, s, true).unwrap();
    let (down_w, up_w) = (widen(&qd, &sd), widen(&qu, &su));
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
    for rows in [1usize, 3] {
        let data: Vec<f32> = (0..rows * k).map(|_| next()).collect();
        let xn = Tensor::from_vec(data, (rows, k), &Device::Cpu)
            .unwrap()
            .to_device(&card)
            .unwrap();
        let want = hc_mix_fused_over(
            &xn,
            MixWeight::Dense(&down_w),
            MixWeight::Dense(&up_w),
            None,
            streams,
            dim,
            false,
        )
        .unwrap()
        .expect("the launches take these weights")
        .0;
        let got = hc_mix_fused_over(
            &xn,
            MixWeight::Int8 { q: &qd, scale: &sd },
            MixWeight::Int8 { q: &qu, scale: &su },
            None,
            streams,
            dim,
            false,
        )
        .unwrap()
        .expect("the launches take int8 rows")
        .0;
        assert!(max_diff(&flat(&got), &flat(&want)) < 1e-4, "{rows} rows");
    }
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

/// A wide model's prefill footprint grows with the streams and plans the card for the same chunk
/// whatever the experts; a model without a routed mixture asks for no chunk.
#[cfg(feature = "cuda")]
#[test]
fn the_wide_prefill_footprint_follows_the_model() {
    let a = hyper::prefill_footprint(4, 2560, Some((160, 10)), 640);
    let b = hyper::prefill_footprint(4, 2560, Some((512, 10)), 640);
    assert!(a.bytes_per_token > 0 && a.fixed_bytes > 0);
    assert_eq!(a.target_chunk, hyper::PLANNED_CHUNK_ROWS);
    assert_eq!(b.target_chunk, a.target_chunk);
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

/// The host's selected attention of an indexed layer for the decoded token `x` at `pos`: the
/// indexer's selection on the host, the attention over the kept tokens.
#[cfg(feature = "cuda")]
fn host_attention(layer: &mut GenericTransformerLayer, x: &Tensor, pos: usize) -> Tensor {
    let selection = layer
        .sparse
        .as_mut()
        .unwrap()
        .observe(x, pos)
        .unwrap()
        .expect("past the indexer's budget");
    layer.attend_selected(x, pos, &selection).unwrap()
}

/// A decoded token of an indexed layer on a card runs on the card, eagerly: it gives what the host's
/// selected attention gives at every position, and leaves the host's counters where the host's
/// own steps would.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn an_indexed_layer_decodes_on_the_card_as_the_host_does() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let mixed = f32s(&v, "b.layer3_attn_hc_mixed");
    let width = mixed.len() / tokens.len();
    let split = tokens.len() - 6;
    let mut host = toy_with(Some(0), KvQuant::Q8);
    let mut card = toy_with(Some(0), KvQuant::Q8);
    assert!(card.layers[3].indexed_on_card());
    let dev = host.layers[0].cos.device();
    let row = |p: usize| {
        Tensor::from_vec(
            mixed[p * width..(p + 1) * width].to_vec(),
            (1, 1, width),
            &dev,
        )
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
    let prefill = Tensor::from_vec(tokens[..split].to_vec(), (1, split), &dev).unwrap();
    host.forward(&prefill, 0).unwrap();
    card.forward(&prefill, 0).unwrap();
    for p in split..tokens.len() {
        let want = flat(&host_attention(&mut host.layers[3], &row(p), p));
        let got = flat(&card.layers[3].wide_attention(&row(p), None, p).unwrap());
        let d = max_diff(&got, &want);
        assert!(
            d < 1e-2 * scale_of(&want),
            "position {p}: differs by {d} on a scale of {}",
            scale_of(&want)
        );
        assert_eq!(card.layers[3].kv_cache.current_seq_len(), p + 1);
    }
    // The host takes over from the card where the card made the tokens.
    let next = tokens.len();
    let x = row(next - 1);
    let from_host = flat(&host_attention(&mut host.layers[3], &x, next));
    let sparse = card.layers[3].sparse.as_mut().unwrap();
    assert!(sparse.observe(&x, next).unwrap().is_some());
    assert_eq!(sparse.len(), next + 1);
    assert!(from_host.iter().all(|v| v.is_finite()));
}

/// The attention of an indexed layer, recorded once against the buffers of its decode step, gives
/// at every later position what the host's selected attention gives: the indexer on the card
/// holds the tokens, selects and attends, and the caches the two keep agree afterwards.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_recorded_indexed_attention_replays_at_every_position() {
    use crate::tensor::cuda_ext;
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let mixed = f32s(&v, "b.layer3_attn_hc_mixed");
    let width = mixed.len() / tokens.len();
    let split = tokens.len() - 6;
    let mut eager = toy_with(Some(0), KvQuant::Q8);
    let mut stepped = toy_with(Some(0), KvQuant::Q8);
    assert!(
        stepped.layer_captured(3),
        "{:?}",
        stepped.layer_eager_reason(3)
    );
    let card = eager.layers[0].cos.device();
    let row = |p: usize| {
        Tensor::from_vec(
            mixed[p * width..(p + 1) * width].to_vec(),
            (1, 1, width),
            &card,
        )
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
    let prefill = Tensor::from_vec(tokens[..split].to_vec(), (1, split), &card).unwrap();
    eager.forward(&prefill, 0).unwrap();
    stepped.forward(&prefill, 0).unwrap();

    let input = row(split).force_contiguous().unwrap();
    let out = Tensor::zeros_on((1, 1, width), DType::F32, &card).unwrap();
    let stream = stepped.cuda_stream().unwrap();
    let mut graph: Option<cuda_ext::CapturedGraph> = None;
    for p in split..tokens.len() {
        let want = flat(&host_attention(&mut eager.layers[3], &row(p), p));
        stepped.update_graph_state(p).unwrap();
        input.slice_set(&row(p), 0, 0).unwrap();
        if graph.is_none() {
            stream.synchronize().unwrap();
            cuda_ext::begin_capture(&stream).unwrap();
            {
                // What the step makes is freed inside the recording, the answer kept apart.
                let _step = graph::GraphStep::enter();
                let y = stepped.layers[3].forward_sparse_graph(&input).unwrap();
                out.slice_set(&y, 0, 0).unwrap();
            }
            graph = Some(cuda_ext::end_capture(&stream).unwrap().unwrap());
            stream.synchronize().unwrap();
            crate::tensor::cuda::trim_pool(stream.context().ordinal());
            graph.as_ref().unwrap().upload().unwrap();
            stepped.sync_kv_len_for_graph(p);
        }
        graph.as_ref().unwrap().launch().unwrap();
        stream.synchronize().unwrap();
        stepped.sync_kv_len_for_graph(p + 1);
        let got = flat(&out);
        let d = max_diff(&got, &want);
        assert!(
            d < 1e-2 * scale_of(&want),
            "position {p}: differs by {d} on a scale of {}",
            scale_of(&want)
        );
    }
    let on_card = stepped.layers[3]
        .sparse
        .as_ref()
        .unwrap()
        .snapshot_cache()
        .unwrap();
    let on_host = eager.layers[3]
        .sparse
        .as_ref()
        .unwrap()
        .snapshot_cache()
        .unwrap();
    assert_eq!(on_card.len(), tokens.len());
    assert_eq!(on_card.len(), on_host.len());
}

/// A verify of drafted rows that keeps only some of them leaves the model where decoding the kept
/// rows one at a time would: the logits of what follows are the reference's, whether the verify
/// kept every row, some, or just the first.
#[test]
fn a_verify_cut_back_to_the_rows_kept_leaves_the_state_of_decoding_them() {
    verify_cut_back(None, KvQuant::Off);
}

/// [`a_verify_cut_back_to_the_rows_kept_leaves_the_state_of_decoding_them`] on a card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_verify_cut_back_on_a_card_leaves_the_state_of_decoding_the_kept_rows() {
    verify_cut_back(Some(0), KvQuant::Off);
}

/// The same with the quantised cache a served model gets.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_verify_cut_back_on_a_card_with_a_quantised_cache_leaves_the_decoded_state() {
    verify_cut_back(Some(0), KvQuant::Q8);
}

/// Several verifies in a row, each keeping a different number of its rows, leave the model where
/// decoding the kept tokens one at a time would: the cut-back of one round is not undone, or
/// compounded, by the next.
#[test]
fn verifies_that_follow_each_other_keep_the_state_of_decoding() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let dev = Device::Cpu;
    let step = |m: &mut GenericHeteroTransformer, t: u32, at: usize| -> Vec<f32> {
        let input = Tensor::from_vec(vec![t], (1, 1), &dev).unwrap();
        m.forward(&input, at)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let mut reference = toy();
    let want: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| step(&mut reference, *t, i))
        .collect();
    let scale = want.iter().map(|r| scale_of(r)).fold(0f32, f32::max);
    let vocab = want[0].len();
    let mut m = toy();
    for (i, t) in tokens[..2].iter().enumerate() {
        step(&mut m, *t, i);
    }
    let mut pos = 2usize;
    for (rows, kept) in [(3usize, 2usize), (3, 1), (2, 2), (3, 3), (2, 1)] {
        if pos + rows >= tokens.len() {
            break;
        }
        let batch = Tensor::from_vec(tokens[pos..pos + rows].to_vec(), (1, rows), &dev).unwrap();
        let got = m.forward_all(&batch, pos).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for r in 0..rows {
            let d = max_diff(&got[r * vocab..(r + 1) * vocab], &want[pos + r]);
            assert!(d < 1e-3 * scale, "pos {pos}: verify row {r} differs by {d}");
        }
        if kept < rows {
            m.trim_kv(pos + kept);
        }
        pos += kept;
    }
    assert!(pos > 6, "the chain ran through several rounds");
}

/// A verify of a few rows through the recorded window gives the logits of decoding them one
/// by one, eagerly and replayed from a graph, and a cut back to the rows kept leaves the state
/// of decoding those rows alone.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_recorded_verify_window_matches_the_decoded_rows_and_cuts_back() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let toy = || toy_on(Some(0));
    let dev = toy().layers[0].cos.device();
    let step = |m: &mut GenericHeteroTransformer, t: u32, at: usize| -> Vec<f32> {
        let input = Tensor::from_vec(vec![t], (1, 1), &dev).unwrap();
        m.forward(&input, at)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let mut reference = toy();
    let want: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| step(&mut reference, *t, i))
        .collect();
    let scale = want.iter().map(|r| scale_of(r)).fold(0f32, f32::max);
    let vocab = want[0].len();
    let top = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    let rows_of = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let stream = crate::tensor::cuda_ext::stream_of(&dev).unwrap();
    for (prefix, rows, kept) in [(3usize, 4usize, 2usize), (2, 3, 1), (4, 3, 3)] {
        let mut m = toy();
        for (i, t) in tokens[..prefix].iter().enumerate() {
            step(&mut m, *t, i);
        }
        if let Some(why) = m.verify_window_block() {
            eprintln!("skipping: the toy records no verify ({why})");
            return;
        }
        assert!(
            m.verify_window_rows() >= rows,
            "the toy on a card records a verify of {rows} rows"
        );
        let window = tokens[prefix..prefix + rows].to_vec();
        // Eagerly: the kernels of the recorded window, run at once.
        m.verify_window_prepare(prefix, &window).unwrap();
        let eager = rows_of(&m.forward_verify_window(rows).unwrap());
        stream.synchronize().unwrap();
        for r in 0..rows {
            let row = &eager[r * vocab..(r + 1) * vocab];
            let d = max_diff(row, &want[prefix + r]);
            assert!(d < QUANTISED_SLACK * scale, "eager row {r} differs by {d}");
            assert_eq!(
                top(row),
                top(&want[prefix + r]),
                "eager row {r} names another token"
            );
        }
        // Back to before the rows, recorded, and replayed: the same logits.
        m.verify_window_settle(0).unwrap();
        m.verify_window_prepare(prefix, &window).unwrap();
        stream.synchronize().unwrap();
        crate::tensor::cuda_ext::begin_capture(&stream).unwrap();
        let logits = m.forward_verify_window(rows).unwrap();
        let graph = crate::tensor::cuda_ext::end_capture(&stream)
            .unwrap()
            .expect("a window records kernels");
        graph.launch().unwrap();
        stream.synchronize().unwrap();
        let replayed = rows_of(&logits);
        for r in 0..rows {
            let d = max_diff(
                &replayed[r * vocab..(r + 1) * vocab],
                &eager[r * vocab..(r + 1) * vocab],
            );
            assert!(
                d < 1e-3 * scale,
                "replayed row {r} differs from the eager one by {d}"
            );
        }
        // Cut back to the rows kept: decoding goes on as if those rows alone had been decoded.
        m.verify_window_settle(kept).unwrap();
        let mut alone = toy();
        for (i, t) in tokens[..prefix + kept].iter().enumerate() {
            step(&mut alone, *t, i);
        }
        for i in prefix + kept..prefix + kept + 2 {
            let (got, want_alone) = (step(&mut m, tokens[i], i), step(&mut alone, tokens[i], i));
            let d = max_diff(&got, &want_alone);
            assert!(
                d < QUANTISED_SLACK * scale,
                "prefix {prefix}, {rows} rows, {kept} kept: position {i} differs by {d}"
            );
            assert_eq!(
                top(&got),
                top(&want_alone),
                "position {i} names another token"
            );
        }
    }
}

fn verify_cut_back(ordinal: Option<usize>, kv: KvQuant) {
    let toy = || toy_with(ordinal, kv);
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let tokens = ids(&v, "b.ids");
    let dev = toy().layers[0].cos.device();
    let step = |m: &mut GenericHeteroTransformer, t: u32, at: usize| -> Vec<f32> {
        let input = Tensor::from_vec(vec![t], (1, 1), &dev).unwrap();
        m.forward(&input, at)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let mut reference = toy();
    assert!(reference.is_recurrent(), "the toy has a recurrent layer");
    let want: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| step(&mut reference, *t, i))
        .collect();
    let scale = want.iter().map(|r| scale_of(r)).fold(0f32, f32::max);
    // A card's rows of a verify come from the batched kernels, so they agree with the decoded
    // rows to the rounding of the half-width carriers, not to the bit. What is held to the bit
    // there is the state the cut leaves: it is the one a verify of the kept rows alone makes.
    let verify_slack = if ordinal.is_some() {
        QUANTISED_SLACK
    } else {
        1e-3
    };
    let n = tokens.len();
    // A verify taken back whole, none of its rows kept, leaves the state before it.
    for (prefix, rows, kept) in [
        (3usize, 4usize, 1usize),
        (3, 4, 2),
        (2, 5, 4),
        (4, 3, 3),
        (3, 4, 0),
    ] {
        assert!(prefix + rows + 1 < n, "the toy's prompt is long enough");
        let mut m = toy();
        for (i, t) in tokens[..prefix].iter().enumerate() {
            step(&mut m, *t, i);
        }
        let batch =
            Tensor::from_vec(tokens[prefix..prefix + rows].to_vec(), (1, rows), &dev).unwrap();
        let got = m.forward_all(&batch, prefix).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vocab = want[0].len();
        let mut noise = 0f32;
        for r in 0..rows {
            let d = max_diff(&got[r * vocab..(r + 1) * vocab], &want[prefix + r]);
            assert!(d < verify_slack * scale, "verify row {r} differs by {d}");
            noise = noise.max(d);
            let top = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
            };
            assert_eq!(
                top(&got[r * vocab..(r + 1) * vocab]),
                top(&want[prefix + r]),
                "verify row {r} names another token than the decoded row"
            );
        }
        if kept < rows {
            m.trim_kv(prefix + kept);
        }
        // What decoding goes on to give: the decoded rows' own logits on the host, and on a
        // card those after a verify of the kept rows alone.
        let follow: Vec<Vec<f32>> = if ordinal.is_some() {
            let mut alone = toy();
            for (i, t) in tokens[..prefix].iter().enumerate() {
                step(&mut alone, *t, i);
            }
            if kept > 0 {
                let kept_rows =
                    Tensor::from_vec(tokens[prefix..prefix + kept].to_vec(), (1, kept), &dev)
                        .unwrap();
                alone.forward_all(&kept_rows, prefix).unwrap();
            }
            (prefix + kept..prefix + kept + 2)
                .map(|i| step(&mut alone, tokens[i], i))
                .collect()
        } else {
            want[prefix + kept..prefix + kept + 2].to_vec()
        };
        for (j, i) in (prefix + kept..prefix + kept + 2).enumerate() {
            let d = max_diff(&step(&mut m, tokens[i], i), &follow[j]);
            // The kept rows' keys and values came from the batched kernels: the state they leave
            // is as far from the alone one as the batched rows are from the decoded ones.
            assert!(
                d < (1e-3 * scale).max(1.5 * noise),
                "prefix {prefix}, {rows} rows, {kept} kept: position {i} differs by {d} (noise {noise})"
            );
        }
    }
}

/// The toy with a draft head: its layer the indexed attention layer of a second copy, the
/// projections fixed smooth matrices, the output mixer the second copy's.
pub(super) fn toy_with_head() -> GenericHeteroTransformer {
    use crate::tensor::layer::RmsNorm;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    let mut m = toy();
    let mut donor = toy();
    let at = donor
        .layers
        .iter()
        .position(|l| l.sparse.is_some())
        .unwrap();
    let layer = donor.layers.remove(at);
    let layer_dev = donor.layer_devs.remove(at);
    let mixer = donor.output_hyper.take().unwrap();
    let (streams, dim) = (mixer.streams(), mixer.dim());
    let eps = m.config.rms_norm_eps;
    let dev = Device::Cpu;
    let proj = |seed: f32| {
        let w: Vec<f32> = (0..dim * dim)
            .map(|i| (i as f32 * 0.37 + seed).sin() * 0.2)
            .collect();
        let t = Tensor::from_vec(w, (dim, dim), &dev).unwrap();
        QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::F32).unwrap()).unwrap()
    };
    m.draft = Some(draft_head::DraftHead {
        layer,
        layer_dev,
        norm_embedding: RmsNorm::from_tensor(Tensor::ones(dim, DType::F32, &dev).unwrap(), eps),
        hidden_gamma: Tensor::ones(streams * dim, DType::F32, &dev).unwrap(),
        fc_embedding: proj(1.0),
        fc_hidden: proj(2.0),
        mixer,
        streams,
        dim,
        eps,
        len: 0,
        committed: 0,
        carried: None,
        fit: (0, 0),
        indexer_mark: None,
        vocab: None,
        #[cfg(feature = "cuda")]
        step: None,
    });
    m
}

/// What the head guesses after `pairs` tokens, the pairs given in the calls `calls` cut
/// (each call is `(first, count)` of `pairs`): the head reads the later calls after the guesses
/// of the earlier ones were cut off.
fn head_guesses(calls: &[(usize, usize)], k: usize) -> Vec<u32> {
    let mut m = toy_with_head();
    let wide = {
        let d = m.draft.as_ref().unwrap();
        d.streams * d.dim
    };
    let vocab = m.config.vocab_size as u32;
    let tokens: Vec<u32> = (0..40u32).map(|i| (i * 5 + 1) % vocab).collect();
    let feature = |from: usize, n: usize| {
        let v: Vec<f32> = (from * wide..(from + n) * wide)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        Tensor::from_vec(v, (1, n, wide), &Device::Cpu).unwrap()
    };
    let mut out = Vec::new();
    for &(from, n) in calls {
        m.last_feature = Some(feature(from, n));
        out = m.draft_tokens(&tokens[from..from + n], k, 0.5).unwrap();
    }
    out
}

#[test]
fn the_draft_head_cuts_its_guesses_from_the_indexer_as_from_its_cache() {
    // 36 pairs read in one call, or in four with guesses made in between (the indexer's
    // budget drops blocks at 16 tokens, so the later pairs are past it).
    let whole = head_guesses(&[(0, 36)], 3);
    let cut = head_guesses(&[(0, 12), (12, 9), (21, 7), (28, 8)], 3);
    assert_eq!(whole, cut);
}

/// What the head guesses after a prompt read as chunks of `chunk` rows, layer by layer or chunk
/// by chunk, and one decode step on.
fn head_guesses_after_prompt(layer_major: bool, chunk: usize) -> (Vec<f32>, Vec<u32>) {
    let mut m = toy_with_head();
    m.set_capture_feature(true);
    let vocab = m.config.vocab_size as u32;
    let prompt: Vec<u32> = (0..26u32).map(|i| (i * 7 + 2) % vocab).collect();
    let ids = |v: &[u32]| Tensor::from_vec(v.to_vec(), (1usize, v.len()), &Device::Cpu).unwrap();
    let logits = if layer_major {
        m.forward_prefill_layer_major(&ids(&prompt), 0, chunk)
            .unwrap()
    } else {
        let mut last = None;
        let mut off = 0;
        while off < prompt.len() {
            let n = chunk.min(prompt.len() - off);
            last = Some(m.forward(&ids(&prompt[off..off + n]), off).unwrap());
            off += n;
        }
        last.unwrap()
    };
    let row = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let next = (row
        .iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0) as u32;
    m.forward(&ids(&[next]), prompt.len()).unwrap();
    let guesses = m.draft_tokens(&[next], 3, 0.5).unwrap();
    (row, guesses)
}

#[test]
fn the_draft_head_reads_a_prompt_layer_by_layer_as_chunk_by_chunk() {
    for chunk in [5usize, 9] {
        let by_chunk = head_guesses_after_prompt(false, chunk);
        let by_layer = head_guesses_after_prompt(true, chunk);
        assert_eq!(by_layer, by_chunk, "chunk {chunk}");
    }
}

/// A forward that failed after its n-gram rows were gathered runs again from the same position:
/// the gather finds the history where the first one did and gathers the same rows.
#[test]
fn a_forward_run_again_gathers_the_same_ngram_rows() {
    let bytes = vectors();
    let v = SafeTensors::deserialize(&bytes).unwrap();
    let mut model = toy();
    let tokens = ids(&v, "a.ids");
    let as_ids = |t: &[u32]| Tensor::from_vec(t.to_vec(), (1, t.len()), &Device::Cpu).unwrap();
    let (head, tail) = tokens.split_at(tokens.len() / 2);
    let pending = |m: &mut GenericHeteroTransformer| {
        let p = m.ngram.as_mut().unwrap().pending.take().unwrap();
        p.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    model.gather_ngram_rows(&as_ids(head), 0).unwrap();
    model.gather_ngram_rows(&as_ids(tail), head.len()).unwrap();
    let first = pending(&mut model);
    model.gather_ngram_rows(&as_ids(tail), head.len()).unwrap();
    assert_eq!(pending(&mut model), first);
    assert!(
        model
            .gather_ngram_rows(&as_ids(tail), head.len() + 1)
            .is_err(),
        "a position the history never stood at is still refused"
    );
}
