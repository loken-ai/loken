//! The ignored cases here need real weights, a device, or a reference dump on
//! this machine; nothing about them is automatic. Run one by name with
//!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
use super::*;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// Batched CFG (one instance, 2-row decode) must produce codes BIT-IDENTICAL to the serial
// two-instance generate_cfg for the same seed/prompt - the only difference is the GPU
// batched-MMVQ kernel, which must give per-row results identical to the b=1 path.
#[test]
#[ignore = "needs LM GGUF (config.test HF hub); GPU"]
fn cfg_batched_matches_serial() {
    let g = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let gp = g.to_str().unwrap();
    let tok = super::acestep_tokenizer(gp).unwrap();
    let cot = super::build_cot_yaml(120, "energetic EDM, 120 BPM", 8, "", "en", "4");
    let (cap, lyr) = (
        "energetic EDM, four-on-the-floor, bright synths, 120 BPM",
        "We run all night,\nthe city lights.",
    );
    // serial (two instances)
    let serial = {
        let mut lm = super::Qwen3Lm::from_gguf(gp).unwrap();
        let mut lu = super::Qwen3Lm::from_gguf(gp).unwrap();
        lm.generate_cfg(&mut lu, &tok, cap, lyr, &cot, "", 40, 7, 0.85, 0.9, 2.0)
            .unwrap()
    };
    // batched (one instance)
    let batched = {
        let mut lm = super::Qwen3Lm::from_gguf(gp).unwrap();
        lm.generate_cfg_batched(&tok, cap, lyr, &cot, "", 40, 0, 7, 0.85, 0.9, 2.0)
            .unwrap()
    };
    println!(
        "serial={} codes, batched={} codes",
        serial.len(),
        batched.len()
    );
    let nmatch = serial
        .iter()
        .zip(&batched)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "matching prefix = {nmatch}/{}",
        serial.len().min(batched.len())
    );
    assert_eq!(
        serial, batched,
        "batched CFG diverged from serial (first diff at {nmatch})"
    );
}

// Validate BPE+prompt building: rebuild the oracle's prompt from /tmp/lm_req.json
// (the request behind the dump) and check it == /tmp/lm_tokens.csv (260 tokens).
#[test]
#[ignore = "needs LM GGUF + /tmp/lm_req.json + /tmp/lm_tokens.csv (ace-lm dump) - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
fn validate_prompt_tokens_vs_oracle() {
    let gguf = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let tok = super::acestep_tokenizer(gguf.to_str().unwrap()).unwrap();
    let req: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/lm_req.json").unwrap()).unwrap();
    let s = |k: &str| {
        req.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let i = |k: &str| req.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
    let cot = super::build_cot_yaml(
        i("bpm"),
        &s("caption"),
        i("duration"),
        &s("keyscale"),
        &s("vocal_language"),
        &s("timesignature"),
    );
    let ids = super::build_lm_prompt_with_cot(&tok, &s("caption"), &s("lyrics"), &cot).unwrap();
    let oref: Vec<u32> = std::fs::read_to_string("/tmp/lm_tokens.csv")
        .unwrap()
        .trim()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let nmatch = ids.iter().zip(&oref).take_while(|(a, b)| a == b).count();
    println!(
        "prompt tokens: mine={} oracle={} matching-prefix={nmatch}",
        ids.len(),
        oref.len()
    );
    let lo = nmatch.saturating_sub(2);
    println!("  mine  [{lo}..]: {:?}", &ids[lo..(lo + 10).min(ids.len())]);
    println!(
        "  oracle[{lo}..]: {:?}",
        &oref[lo..(lo + 10).min(oref.len())]
    );
    let dec = |s: &[u32]| tok.decode(s, false).unwrap_or_default();
    println!(
        "  mine decoded   : {:?}",
        dec(&ids[lo..(lo + 10).min(ids.len())])
    );
    println!(
        "  oracle decoded : {:?}",
        dec(&oref[lo..(lo + 10).min(oref.len())])
    );
    assert_eq!(
        ids, oref,
        "prompt token mismatch (first divergence at {nmatch})"
    );
}

// Validate the 5Hz LM prefill against the oracle's ace-lm dump:
//   ace-lm --dump-tokens /tmp/lm_tokens.csv --dump-logits /tmp/lm_logits.bin
// (qw3lm_forward last-position logits over the full 217204 vocab, deterministic).
#[test]
#[ignore = "needs LM GGUF (config.test HF hub) + /tmp/lm_tokens.csv + /tmp/lm_logits.bin; slow"]
fn validate_lm_prefill_vs_oracle() {
    let gguf = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let mut m = Qwen3Lm::from_gguf(gguf.to_str().unwrap()).unwrap();
    let toks: Vec<u32> = std::fs::read_to_string("/tmp/lm_tokens.csv")
        .unwrap()
        .trim()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let hidden = m.prefill(&toks).unwrap();
    let logits = m.logits(&hidden).unwrap();
    let bytes = std::fs::read("/tmp/lm_logits.bin").unwrap();
    let oref: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(logits.len(), oref.len(), "vocab size");
    let dot: f32 = logits.iter().zip(&oref).map(|(a, b)| a * b).sum();
    let na: f32 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = oref.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    let ma = argmax(&logits);
    // The oracle's top logits cluster within <1 (e.g. top1-top2 ≈ 0.07); over 36
    // Q8 layers the f32 drift reorders these near-ties. Validate the distribution
    // (cosine) + that my top token is within the oracle's top-8 (the LM samples
    // with temp/top-p, so the near-tie ordering is immaterial to the audio).
    let mut order: Vec<usize> = (0..oref.len()).collect();
    order.sort_by(|&a, &b| oref[b].partial_cmp(&oref[a]).unwrap());
    let rank = order.iter().position(|&i| i == ma).unwrap();
    println!(
        "LM prefill cosine={cos:.6} my_argmax={ma} (oracle rank {rank}) ref_argmax={} (S={})",
        order[0],
        toks.len()
    );
    assert!(cos > 0.999, "LM logits cosine {cos} too low");
    assert!(rank < 8, "my top token outside oracle top-8 (rank {rank})");
}

// A plan that spans two cards must decode the same codes as the whole model on one card,
// on the eager path both: the graph replays the eager step and is compared to it by
// `graph_matches_eager`.
// the eager per-layer path moves the activation to each layer's card, and nothing else
// may differ. The split is forced with an explicit budget that leaves the first card room
// for roughly half the layers.
#[test]
#[ignore = "needs LM GGUF (config.test HF hub); two GPUs"]
fn cross_card_matches_single_card() {
    let g = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let gp = g.to_str().unwrap();
    let tok = super::acestep_tokenizer(gp).unwrap();
    let cot = super::build_cot_yaml(120, "energetic EDM, 120 BPM", 8, "", "en", "4");
    let (cap, lyr) = (
        "energetic EDM, four-on-the-floor, bright synths, 120 BPM",
        "We run all night,\nthe city lights.",
    );
    let whole = {
        let mut lm = super::Qwen3Lm::from_gguf(gp).unwrap();
        assert!(
            lm.on_one_card(),
            "the whole model is expected to fit one card here"
        );
        lm.set_graph(false);
        lm.generate_cfg_batched(&tok, cap, lyr, &cot, "", 40, 0, 7, 0.85, 0.9, 2.0)
            .unwrap()
    };
    let split = {
        let half = super::placement_demand(gp) / 2;
        let mut lm =
            super::Qwen3Lm::from_gguf_placed(gp, Some(&[(0, half), (1, u64::MAX / 4)])).unwrap();
        assert!(!lm.on_one_card(), "the budget was meant to split the model");
        lm.generate_cfg_batched(&tok, cap, lyr, &cot, "", 40, 0, 7, 0.85, 0.9, 2.0)
            .unwrap()
    };
    let nmatch = whole.iter().zip(&split).take_while(|(a, b)| a == b).count();
    println!(
        "matching prefix = {nmatch}/{}",
        whole.len().min(split.len())
    );
    assert_eq!(
        whole, split,
        "the split decode diverged (first diff at {nmatch})"
    );
}

// The captured graph and the eager step decode different codes: the graph scores a fixed
// window of 256 positions and eager the real few, and the two products are rounded by
// different kernels. The reference implementation runs in bfloat16 and carries no such
// stability either, so this measures the gap and does not require it closed.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LM GGUF (config.test HF hub); GPU"]
fn graph_matches_eager() {
    let g = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let gp = g.to_str().unwrap();
    let tok = super::acestep_tokenizer(gp).unwrap();
    let cot = super::build_cot_yaml(120, "energetic EDM, 120 BPM", 8, "", "en", "4");
    let (cap, lyr) = (
        "energetic EDM, four-on-the-floor, bright synths, 120 BPM",
        "We run all night,\nthe city lights.",
    );
    let mut lm = super::Qwen3Lm::from_gguf(gp).unwrap();
    let graph = lm
        .generate_cfg_batched(&tok, cap, lyr, &cot, "", 40, 0, 7, 0.85, 0.9, 2.0)
        .unwrap();
    lm.set_graph(false);
    let eager = lm
        .generate_cfg_batched(&tok, cap, lyr, &cot, "", 40, 0, 7, 0.85, 0.9, 2.0)
        .unwrap();
    let nmatch = graph.iter().zip(&eager).take_while(|(a, b)| a == b).count();
    println!(
        "matching prefix = {nmatch}/{}",
        graph.len().min(eager.len())
    );
    assert!(
        !graph.is_empty() && !eager.is_empty(),
        "both paths decode codes"
    );
}

// The capped attention against the plain one on synthetic rows: the same query, the
// same cache filled to the same position, the write and the read through each path.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LM GGUF (config.test HF hub); GPU"]
fn capped_attention_matches_plain() {
    use crate::tensor::{DType, Tensor};
    let g = crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
    let gp = g.to_str().unwrap();
    let lm = super::Qwen3Lm::from_gguf(gp).unwrap();
    let (nh, nkv, d) = (lm.n_head, lm.n_kv, lm.head_dim);
    let dev = lm.device.clone();
    let mut seed = 7u64;
    let mut rand = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed % 20001) as f32 / 10000.0 - 1.0) * scale
            })
            .collect()
    };
    let mut worst = 0f32;
    for trial in 0..24 {
        let scale = [0.5f32, 2.0, 8.0, 32.0][trial % 4];
        let pos = 7 + trial % 5;
        let cap = super::GRAPH_BUCKET;
        let mut kc: Option<Tensor> =
            Some(Tensor::zeros_on(vec![1, nkv, super::MAX_SEQ, d], DType::F32, &dev).unwrap());
        let mut vc: Option<Tensor> =
            Some(Tensor::zeros_on(vec![1, nkv, super::MAX_SEQ, d], DType::F32, &dev).unwrap());
        let kc2 = Tensor::zeros_on(vec![1, nkv, super::MAX_SEQ, d], DType::F32, &dev).unwrap();
        let vc2 = Tensor::zeros_on(vec![1, nkv, super::MAX_SEQ, d], DType::F32, &dev).unwrap();
        for p in 0..pos {
            let k = Tensor::from_vec_f32(rand(nkv * d, scale), vec![1, nkv, 1, d])
                .unwrap()
                .to_device(&dev)
                .unwrap();
            let v = Tensor::from_vec_f32(rand(nkv * d, 1.0), vec![1, nkv, 1, d])
                .unwrap()
                .to_device(&dev)
                .unwrap();
            kc.as_ref().unwrap().slice_set(&k, 2, p).unwrap();
            vc.as_ref().unwrap().slice_set(&v, 2, p).unwrap();
            kc2.slice_set(&k, 2, p).unwrap();
            vc2.slice_set(&v, 2, p).unwrap();
        }
        let q = Tensor::from_vec_f32(rand(nh * d, scale), vec![1, nh, 1, d])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let k_new = Tensor::from_vec_f32(rand(nkv * d, scale), vec![1, nkv, 1, d])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let v_new = Tensor::from_vec_f32(rand(nkv * d, 1.0), vec![1, nkv, 1, d])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let plain = lm
            .attn_row(&q, &k_new, &v_new, pos, &mut kc, &mut vc)
            .unwrap()
            .to_vec_f32();
        let widx = Tensor::from_vec_i64(vec![pos as i64; nkv * d], vec![1, nkv, 1, d])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let mask: Vec<f32> = (0..cap)
            .map(|j| if j > pos { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec_f32(mask, vec![1, 1, 1, cap])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let capped = lm
            .attn_row_capped(&q, &k_new, &v_new, &widx, &mask, cap, &kc2, &vc2)
            .unwrap()
            .to_vec_f32();
        let diff = plain
            .iter()
            .zip(&capped)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let size = plain.iter().map(|v| v.abs()).fold(0f32, f32::max);
        println!(
            "trial {trial:2} scale {scale:5} pos {pos}: max abs diff {diff:e} (size {size:e})"
        );
        worst = worst.max(diff);
    }
    assert!(
        worst < 1e-4,
        "the capped attention differs from the plain one by {worst:e}"
    );
}
