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
