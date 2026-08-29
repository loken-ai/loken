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
