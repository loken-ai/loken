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
    let file = std::fs::File::open(dir().join("toy.gguf")).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
    let mut content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
    let layers = content.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    GenericHeteroTransformer::from_gguf_with_kv_quant(
        content,
        &mmap[..],
        &HashMap::new(),
        &HeteroPlan::forced_gpu(layers, 0, 0),
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
            cosine(g, w) > 0.99,
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
    let rows = ng.history.rows(&ng.meta, &tokens);
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
