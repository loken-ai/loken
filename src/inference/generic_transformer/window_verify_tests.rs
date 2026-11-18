//! A draft verify after a windowed layer's cache has slid, against the decode steps it stands
//! for. The windowed cache holds fewer rows than the sequence has positions once it slides: the
//! verify must still run at the sequence's position, its mask must cover the keys of the global
//! layer, and a cut after it must keep the rows of the tokens accepted.

use super::tiny_model;
use super::*;
use crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS;
use crate::tensor::quantized::gguf_file::Value;

const LAYERS: usize = 2;

/// Keys a verify of this many rows carries: the token decoded and its drafts.
const ROWS: usize = 6;

/// Keys the windowed layer reads.
const WINDOW: usize = 4;

/// The header keys that window the first layer, leave the second global, and give the model
/// room for a prompt of a whole prefill chunk and its decode.
fn windowed_first_layer() -> Vec<(String, Value)> {
    vec![
        (
            "llama.attention.sliding_window".into(),
            Value::U32(WINDOW as u32),
        ),
        (
            "llama.attention.sliding_window_pattern".into(),
            Value::Array(vec![Value::U32(1), Value::U32(0)]),
        ),
        (
            "llama.context_length".into(),
            Value::U32(2 * PREFILL_CHUNK_TOKENS as u32),
        ),
    ]
}

/// The header keys that window both layers and make the second read the first one's cache, with
/// room for a prompt of a whole prefill chunk and its decode.
fn windowed_shared_layer() -> Vec<(String, Value)> {
    vec![
        (
            "llama.attention.sliding_window".into(),
            Value::U32(WINDOW as u32),
        ),
        (
            "llama.attention.sliding_window_pattern".into(),
            Value::Array(vec![Value::U32(1), Value::U32(1)]),
        ),
        ("llama.attention.shared_kv_layers".into(), Value::U32(1)),
        (
            "llama.context_length".into(),
            Value::U32(2 * PREFILL_CHUNK_TOKENS as u32),
        ),
    ]
}

/// A prompt that fills the windowed cache's buffer but for its window.
/// The first verify after it makes the cache slide.
fn prompt() -> Vec<u32> {
    let vocab = tiny_model::VOCAB as u32;
    (0..PREFILL_CHUNK_TOKENS as u32)
        .map(|i| (i * 7 + 3) % vocab)
        .collect()
}

/// The tokens decoded after the prompt.
fn next() -> Vec<u32> {
    let vocab = tiny_model::VOCAB as u32;
    (0..3 * ROWS as u32).map(|i| (i * 5 + 1) % vocab).collect()
}

fn ids(toks: &[u32], dev: &Device) -> Tensor {
    Tensor::new(toks, dev).unwrap().unsqueeze(0).unwrap()
}

fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    let t = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let vocab = t.dims().last().copied().unwrap();
    let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    flat.chunks(vocab).map(<[f32]>::to_vec).collect()
}

/// Largest logit difference a row of the host may show against its step, over the range of
/// the step's logits, when the second layer reads the windowed first layer's cache.
/// The reading layer attends a verify's rows over the windowed donor together, not as its
/// steps.
const SHARED_HOST_SHARE: f32 = 1e-4;

/// How a row must match its step: bit for bit, or within a share of the step's range.
#[derive(Clone, Copy)]
enum Match {
    Bits,
    Share(f32),
}

/// Why `got` does not match `want` as `how` asks, `None` when it does.
fn mismatch(want: &[f32], got: &[f32], how: Match) -> Option<String> {
    assert_eq!(want.len(), got.len(), "the rows differ in width");
    match how {
        Match::Bits => want
            .iter()
            .zip(got)
            .position(|(a, b)| a.to_bits() != b.to_bits())
            .map(|i| format!("at logit {i}: {} against {}", want[i], got[i])),
        Match::Share(share) => {
            let range = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let gap = worst / range.max(f32::MIN_POSITIVE);
            (gap > share).then(|| format!("by {gap:.3e} of the range"))
        }
    }
}

/// The logits of each token of `next` decoded alone after the prompt.
fn decode_steps(m: &mut GenericHeteroTransformer, dev: &Device) -> Vec<Vec<f32>> {
    let prompt = prompt();
    m.forward(&ids(&prompt, dev), 0).unwrap();
    next()
        .iter()
        .enumerate()
        .map(|(i, &t)| rows(&m.forward(&ids(&[t], dev), prompt.len() + i).unwrap()).remove(0))
        .collect()
}

/// 2 verifies, each keeping 2 of its rows, then plain steps: every row and every step gives
/// the logits of its decode step as `how` asks, and the model's length follows the tokens kept.
fn verifies_follow_the_steps(
    m: &mut GenericHeteroTransformer,
    want: &[Vec<f32>],
    dev: &Device,
    how: Match,
) {
    let prompt = prompt();
    let next = next();
    m.forward(&ids(&prompt, dev), 0).unwrap();
    let mut kept = 0;
    for _ in 0..2 {
        let at = m.kv_len().unwrap();
        assert_eq!(at, prompt.len() + kept, "the length is not the position");
        let got = rows(
            &m.forward_all(&ids(&next[kept..kept + ROWS], dev), at)
                .unwrap(),
        );
        for (r, row) in got.iter().enumerate() {
            if let Some(why) = mismatch(&want[kept + r], row, how) {
                panic!("row {r} of the verify at {at} differs from its step {why}");
            }
        }
        assert_eq!(m.kv_len(), Some(at + ROWS));
        kept += 2;
        m.trim_kv(prompt.len() + kept);
        assert_eq!(m.kv_len(), Some(prompt.len() + kept));
    }
    assert!(
        m.layers[0].kv_cache.current_seq_len() < prompt.len(),
        "the windowed cache did not slide"
    );
    for (i, &t) in next.iter().enumerate().skip(kept) {
        let got = rows(&m.forward(&ids(&[t], dev), prompt.len() + i).unwrap()).remove(0);
        if let Some(why) = mismatch(&want[i], &got, how) {
            panic!("step {i} after the verifies differs {why}");
        }
    }
}

/// On the host.
#[test]
fn a_verify_after_the_window_slides_gives_the_decode_steps_on_the_host() {
    let path = tiny_model::write_wide_with("verify-slide", LAYERS, false, windowed_first_layer());
    let mut stepped = tiny_model::load(&path, LAYERS);
    assert!(stepped.layers[0].kv_cache.is_windowed());
    assert!(!stepped.layers[1].kv_cache.is_windowed());
    let want = decode_steps(&mut stepped, &Device::Cpu);
    let mut verified = tiny_model::load(&path, LAYERS);
    verifies_follow_the_steps(&mut verified, &want, &Device::Cpu, Match::Bits);
    tiny_model::remove(&path);
}

/// On card 0, where a decode step appends to the windowed cache and slides it.
#[cfg(feature = "gpu")]
#[test]
fn a_verify_after_the_window_slides_gives_the_decode_steps_on_a_card() {
    let Ok(dev) = Device::new_cuda(0) else {
        eprintln!("no card: the card variant checked nothing");
        return;
    };
    let path =
        tiny_model::write_wide_with("verify-slide-card", LAYERS, false, windowed_first_layer());
    let mut stepped = tiny_model::load_on_card(&path, LAYERS);
    let want = decode_steps(&mut stepped, &dev);
    let mut verified = tiny_model::load_on_card(&path, LAYERS);
    verifies_follow_the_steps(&mut verified, &want, &dev, Match::Bits);
    tiny_model::remove(&path);
}

/// On the host, a layer reading a windowed donor's cache: the donor appends the verify's rows one
/// by one and slides between them, and the reader's mask covers the rows the donor holds.
#[test]
fn a_verify_reading_a_slid_donor_gives_the_decode_steps_on_the_host() {
    let path = tiny_model::write_wide_with(
        "verify-slide-shared",
        LAYERS,
        false,
        windowed_shared_layer(),
    );
    let mut stepped = tiny_model::load(&path, LAYERS);
    assert_eq!(stepped.config.shared_kv_layers, 1);
    let want = decode_steps(&mut stepped, &Device::Cpu);
    let mut verified = tiny_model::load(&path, LAYERS);
    verifies_follow_the_steps(
        &mut verified,
        &want,
        &Device::Cpu,
        Match::Share(SHARED_HOST_SHARE),
    );
    tiny_model::remove(&path);
}

/// On card 0, a layer reading a windowed donor's cache.
#[cfg(feature = "gpu")]
#[test]
fn a_verify_reading_a_slid_donor_gives_the_decode_steps_on_a_card() {
    let Ok(dev) = Device::new_cuda(0) else {
        eprintln!("no card: the card variant checked nothing");
        return;
    };
    let path = tiny_model::write_wide_with(
        "verify-slide-shared-card",
        LAYERS,
        false,
        windowed_shared_layer(),
    );
    let mut stepped = tiny_model::load_on_card(&path, LAYERS);
    let want = decode_steps(&mut stepped, &dev);
    let mut verified = tiny_model::load_on_card(&path, LAYERS);
    verifies_follow_the_steps(&mut verified, &want, &dev, Match::Bits);
    tiny_model::remove(&path);
}
