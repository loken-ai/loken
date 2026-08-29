//! Kyutai `tts-1.6b-en_fr` - the DSM generation loop (components 7b + 7c of the port).
//!
//! Ties the validated pieces together into text->audio synthesis:
//!   conditioner (voice+cfg) -> per-step delayed-streams stepping of the Helium LM ->
//!   text sampling + StateMachine word scheduling -> autoregressive Depformer (32 audio
//!   codebooks) -> Mimi decode -> 24 kHz waveform.
//!
//! The model was trained with CFG DISTILLATION (`cfg_coef == 1.0` at the LM), so there is
//! NO batch-doubled classifier-free guidance here - the guidance is baked into the `cfg`
//! conditioning. Delays: text/semantic 0, acoustic 2 (`max_delay` 2); a global
//! `delay_steps` (16) warm-up keeps the audio silent while the text stream primes.

use std::collections::VecDeque;

use crate::inference::model::kyutai::cond::KyutaiConditioner;
use crate::inference::model::kyutai::depformer::KyutaiDepformer;
use crate::inference::model::kyutai::lm::KyutaiLm;
use crate::inference::model::kyutai::mimi::KyutaiMimiDecoder;
use crate::tensor::{Device, Result, Tensor};
use rand::SeedableRng;

// -- TokenIds --
const CARD: i64 = 8001; // multiplexing modulus (= text_card)
const NEW_WORD: i64 = 0;
const PAD: i64 = 3;
const ZERO: i64 = -1; // -> all-zeros embedding
const UNGENERATED: i64 = -2;
const TEXT_INITIAL: i64 = 8000;
const AUDIO_INITIAL: i64 = 2048;

const N_CB: usize = 33; // 1 text + 32 audio
const DEP_Q: usize = 32;
const AUDIO_OFFSET: usize = 1;
const MAX_DELAY: usize = 2;
const CT: usize = MAX_DELAY + 2; // cyclic cache length = 4
const DELAY_STEPS: usize = 16; // audio warm-up (audio_delay 1.28s x 12.5)
const FINAL_PADDING: usize = 4;
const MAX_GEN: usize = 30000;

const SECOND_STREAM_AHEAD: usize = 2;
const MAX_PADDING: i64 = 8;
const INITIAL_PADDING: i64 = 2;

const TEXT_TOPK: usize = 25;
const AUDIO_TOPK: usize = 250;

/// delays[k] for the 33 codebooks: text/semantic 0, acoustic 2.
fn delay(k: usize) -> usize {
    if k < 2 {
        0
    } else {
        2
    }
}

/// A prepared word: its text tokens (with an optional leading speaker token) and how many
/// pad steps to force after it.
#[derive(Clone)]
pub struct Entry {
    pub tokens: Vec<i64>,
    pub padding: i64,
}

const MAIN_SPEAKER: i64 = 1;
const PADDING_BETWEEN: i64 = 1;

/// Build word entries for a single-turn script (mirrors moshi `script_to_entries`):
/// SentencePiece-encode each whitespace-separated word, prepend the main-speaker token
/// to the first, and force `padding_between + len - 1` pad steps after each word.
pub fn prepare_entries(
    spm: &crate::inference::token::sentencepiece::SentencePiece,
    text: &str,
) -> Vec<Entry> {
    let clean = text
        .replace('\u{2019}', "'")
        .replace(':', " ")
        .replace(['(', ')'], "");
    let mut entries = Vec::new();
    let mut first = true;
    for word in clean.split_whitespace() {
        let mut tokens: Vec<i64> = spm.encode(word).iter().map(|&t| t as i64).collect();
        if tokens.is_empty() {
            continue;
        }
        if first {
            tokens.insert(0, MAIN_SPEAKER);
            first = false;
        }
        let padding = (PADDING_BETWEEN + tokens.len() as i64 - 1).max(0);
        entries.push(Entry { tokens, padding });
    }
    entries
}

/// StateMachine state (per utterance).
struct State {
    entries: VecDeque<Entry>,
    remaining_padding: i64,
    forced_padding: i64,
    queued: VecDeque<i64>,
    lookahead_queued: VecDeque<i64>,
    end_step: Option<usize>,
}
impl State {
    fn new(entries: Vec<Entry>) -> Self {
        State {
            entries: entries.into(),
            remaining_padding: INITIAL_PADDING,
            forced_padding: INITIAL_PADDING,
            queued: VecDeque::new(),
            lookahead_queued: VecDeque::new(),
            end_step: None,
        }
    }
    /// The tokens of the `lookahead`-th upcoming word that carries tokens.
    fn tokens_ahead(&self, lookahead: usize) -> Vec<i64> {
        let mut n = lookahead;
        for e in &self.entries {
            if !e.tokens.is_empty() {
                n -= 1;
                if n == 0 {
                    return e.tokens.clone();
                }
            }
        }
        Vec::new()
    }
}

/// Port of `StateMachine.process`: given the model's sampled text token, decide the actual
/// (multiplexed) text token to feed next, driving word timing + the second lookahead stream.
fn process(step: usize, st: &mut State, sampled: i64) -> i64 {
    let mut token = sampled;
    if token != NEW_WORD && token != PAD {
        token = PAD;
    }
    if !st.queued.is_empty() {
        token = PAD;
    } else if st.forced_padding > 0 {
        token = PAD;
    } else if st.remaining_padding <= 0 {
        token = NEW_WORD;
    }

    if token == NEW_WORD {
        if let Some(entry) = st.entries.pop_front() {
            if !entry.tokens.is_empty() {
                st.queued.extend(entry.tokens.iter().copied());
                let ahead = st.tokens_ahead(SECOND_STREAM_AHEAD);
                st.lookahead_queued.extend(ahead);
                st.remaining_padding = MAX_PADDING;
            } else {
                token = PAD;
            }
            st.forced_padding = entry.padding;
        } else {
            token = PAD;
            if st.end_step.is_none() {
                token = NEW_WORD;
            }
            if st.end_step.is_none() {
                st.end_step = Some(step);
            }
        }
    }

    let mut output: i64;
    if token == PAD {
        if st.remaining_padding > 0 {
            st.remaining_padding -= 1;
        }
        if st.forced_padding > 0 {
            st.forced_padding -= 1;
        }
        output = st.queued.pop_front().unwrap_or(PAD);
    } else {
        // token == NEW_WORD
        output = NEW_WORD;
    }

    // second_stream_ahead: multiplex two text streams into one id.
    let mut second: i64 = -1;
    if output == NEW_WORD {
        second = NEW_WORD;
        output = st.queued.pop_front().unwrap_or(PAD);
    } else if let Some(t) = st.lookahead_queued.pop_front() {
        second = t;
    }
    (second + 1) * CARD + output
}

/// Simple seeded top-k multinomial sampler over logits (softmax at `temp`, keep top-k).
struct Sampler {
    rng: rand::rngs::StdRng,
}
impl Sampler {
    fn new(seed: u64) -> Self {
        Sampler {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
    fn sample(&mut self, logits: &[f32], temp: f32, top_k: usize) -> u32 {
        use rand::distr::{weighted::WeightedIndex, Distribution};
        if temp <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
        }
        // softmax(logits/temp)
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|&x| ((x - m) / temp).exp()).collect();
        let s: f32 = p.iter().sum();
        for v in &mut p {
            *v /= s;
        }
        // keep top-k
        if top_k < p.len() {
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.select_nth_unstable_by(top_k, |&i, &j| p[j].total_cmp(&p[i]));
            let thresh = p[idx[top_k]];
            for v in p.iter_mut() {
                if *v < thresh {
                    *v = 0.0;
                }
            }
        }
        WeightedIndex::new(&p).unwrap().sample(&mut self.rng) as u32
    }
}

/// Full generation: `entries` (from the text/tokenizer front-end), a `voice` tensor
/// `[512, T]`, and a `cfg_coef` conditioning value -> 24 kHz mono waveform.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    lm: &KyutaiLm,
    depformer: &KyutaiDepformer,
    cond: &KyutaiConditioner,
    mimi: &KyutaiMimiDecoder,
    entries: Vec<Entry>,
    voice: &Tensor,
    cfg_coef: f32,
    temp: f32,
    seed: u64,
    dev: &Device,
) -> Result<Vec<f32>> {
    let sum = cond.condition_sum(cfg_coef)?; // [1, DIM]
    let cross = cond.condition_cross(voice)?; // [Sc, DIM]
    let mut st = State::new(entries);
    let mut sampler = Sampler::new(seed);

    // cyclic cache [N_CB, CT]
    let mut cache = vec![vec![UNGENERATED; CT]; N_CB];
    let initial: Vec<i64> = std::iter::once(TEXT_INITIAL)
        .chain(std::iter::repeat(AUDIO_INITIAL).take(DEP_Q))
        .collect();

    let mut frames: Vec<[i64; N_CB]> = Vec::new();
    // Streaming KV cache: the Helium attends the accumulated K/V (O(1) per step) instead
    // of re-running the whole history each step, and reuses the constant cross-attn K/V.
    let mut kv = lm.new_cache(&cross)?;
    let timing = std::env::var("KYUTAI_DEBUG").is_ok();
    let (mut t_lm, mut t_dep) = (0u128, 0u128);
    for offset in 0..MAX_GEN {
        if let Some(end) = st.end_step {
            if offset >= end + DELAY_STEPS + FINAL_PADDING {
                break;
            }
        }
        // Build the LM input frame [33]: cache at offset%CT, or initial while offset<=delay.
        let pos = offset % CT;
        let mut input = [0i64; N_CB];
        for k in 0..N_CB {
            input[k] = if offset <= delay(k) {
                initial[k]
            } else {
                cache[k][pos]
            };
        }
        let seq_u: Vec<u32> = input.iter().map(|&x| x as u32).collect();
        let seq = Tensor::from_vec_u32(seq_u, vec![N_CB, 1])?.to_device(dev)?;
        let tl0 = if timing {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (transformer_out, text_logits) = lm.forward_text_step(&seq, &sum, &mut kv)?;
        let tl = text_logits.flatten_all()?.to_vec1_f32()?;
        if let Some(t) = tl0 {
            t_lm += t.elapsed().as_micros();
        }
        let sampled_text = sampler.sample(&tl, temp, TEXT_TOPK) as i64;
        let text_token = process(offset, &mut st, sampled_text);

        // Depformer: silent (zeros) during the warm-up, else autoregressive sampling with a
        // per-step KV cache (each codebook is O(1) in the prior ones).
        let mut audio = [ZERO; DEP_Q];
        if offset >= DELAY_STEPS {
            let td0 = if timing {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let mut dcache = depformer.new_cache();
            let mut prev = text_token as u32; // input token for the next codebook position
            for cb in 0..DEP_Q {
                let logits = depformer.forward_step(&transformer_out, cb, prev, &mut dcache)?;
                let tok = sampler.sample(&logits, temp, AUDIO_TOPK);
                audio[cb] = tok as i64;
                prev = tok;
            }
            if let Some(t) = td0 {
                t_dep += t.elapsed().as_micros();
            }
        }
        // on_audio_hook: force zero for codebooks still inside their delay window.
        for q in 0..DEP_Q {
            if offset < delay(q + AUDIO_OFFSET) + DELAY_STEPS {
                audio[q] = ZERO;
            }
        }

        // Write generated tokens at (offset+1)%CT.
        let pos_new = (offset + 1) % CT;
        cache[0][pos_new] = text_token;
        for q in 0..DEP_Q {
            cache[1 + q][pos_new] = audio[q];
        }

        // Emit a delay-aligned frame once past the warm-up window.
        let off_new = offset + 1;
        if off_new > MAX_DELAY {
            let mut frame = [0i64; N_CB];
            for k in 0..N_CB {
                let idx = (off_new - MAX_DELAY + delay(k)) % CT;
                frame[k] = cache[k][idx];
            }
            frames.push(frame);
        }
    }

    // Assemble the 32 audio codebooks (drop codebook 0 = text), skip warm-up silence
    // (leading frames whose acoustic tokens are still zero/ungenerated).
    let start = frames
        .iter()
        .position(|f| f[1..].iter().all(|&t| t >= 0))
        .unwrap_or(0);
    let valid = &frames[start..];
    let t = valid.len();
    if t == 0 {
        return Ok(Vec::new());
    }
    let mut codes = vec![0u32; DEP_Q * t];
    for (ti, f) in valid.iter().enumerate() {
        for q in 0..DEP_Q {
            let v = f[1 + q];
            codes[q * t + ti] = if v < 0 { 0 } else { v as u32 };
        }
    }
    if timing {
        eprintln!(
            "[gen] LM {:.2}s + Depformer {:.2}s over {} frames",
            t_lm as f64 / 1e6,
            t_dep as f64 / 1e6,
            frames.len()
        );
    }
    let codes_t = Tensor::from_vec_u32(codes, vec![DEP_Q, t])?.to_device(dev)?;
    mimi.decode(&codes_t)
}
